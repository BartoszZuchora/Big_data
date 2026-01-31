import os
from typing import List, Tuple

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

BOOTSTRAP = "localhost:29092"

TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"
TOPIC_METRICS = "tmdb_stream_metrics"

MODEL_DIR = "models"

CHECKPOINT_OUT = "checkpoints/tmdb_stream_out"
CHECKPOINT_METRICS = "checkpoints/tmdb_stream_metrics"

MAX_POPULARITY = 1e7
MAX_VOTE_COUNT = 1e8

schema = T.StructType(
    [
        T.StructField("id", T.StringType(), True),
        T.StructField("event_time", T.StringType(), True),

        T.StructField("tmdb_id", T.StringType(), True),
        T.StructField("title", T.StringType(), True),

        T.StructField("popularity", T.DoubleType(), True),
        T.StructField("vote_count", T.DoubleType(), True),
        T.StructField("release_year", T.IntegerType(), True),
        T.StructField("original_language", T.StringType(), True),
        T.StructField("adult", T.BooleanType(), True),
        T.StructField("video", T.BooleanType(), True),
        T.StructField("genre_count", T.IntegerType(), True),
    ]
)


def clamp_nonneg(col: str, maxv: float):
    return (
        F.when(F.col(col).isNull(), F.lit(None).cast("double"))
        .when(F.col(col) < 0, F.lit(None).cast("double"))
        .when(F.col(col) > F.lit(maxv), F.lit(maxv).cast("double"))
        .otherwise(F.col(col).cast("double"))
    )


def safe_log1p(col: str):
    return F.when(F.col(col).isNull(), F.lit(None).cast("double")).otherwise(F.log1p(F.col(col)))


def list_model_paths(model_dir: str) -> List[Tuple[str, str]]:
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"MODEL_DIR nie istnieje lub nie jest katalogiem: {model_dir}")

    items = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if os.path.isdir(path) and name.startswith("tmdb_"):
            items.append((name, path))

    if not items:
        raise RuntimeError(f"Nie znaleziono żadnych modeli w {model_dir} (tmdb_*)")

    return items


def load_models(model_dir: str) -> List[Tuple[str, PipelineModel]]:
    models = []
    for name, path in list_model_paths(model_dir):
        models.append((name, PipelineModel.load(path)))
    return models


spark = SparkSession.builder.appName("TMDB-Streaming-Predict-AllModels").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

os.makedirs(CHECKPOINT_OUT, exist_ok=True)
os.makedirs(CHECKPOINT_METRICS, exist_ok=True)

models = load_models(MODEL_DIR)
print("Loaded models:")
for n, _ in models:
    print(" -", n)

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("subscribe", TOPIC_IN)
    .option("startingOffsets", "latest")
    .option("failOnDataLoss", "false")
    .load()
)

json_df = raw.select(F.col("value").cast("string").alias("json_str"))
parsed = json_df.select(F.from_json("json_str", schema).alias("data")).select("data.*")

# Feature engineering MUST match training pipeline inputs:
features = (
    parsed
    .withColumn("event_ts", F.to_timestamp("event_time"))
    .withColumn("processed_ts", F.current_timestamp())

    .withColumn("original_language", F.coalesce(F.col("original_language"), F.lit("unknown")))
    .withColumn("release_year", F.col("release_year").cast("int"))
    .withColumn("genre_count", F.col("genre_count").cast("int"))

    # bool casts
    .withColumn("adult", F.col("adult").cast("boolean"))
    .withColumn("video", F.col("video").cast("boolean"))

    # IMPORTANT: create adult_num/video_num like in batch training
    .withColumn("adult_num", F.when(F.col("adult") == True, F.lit(1.0)).otherwise(F.lit(0.0)))
    .withColumn("video_num", F.when(F.col("video") == True, F.lit(1.0)).otherwise(F.lit(0.0)))

    # clean numeric
    .withColumn("popularity", clamp_nonneg("popularity", MAX_POPULARITY))
    .withColumn("vote_count", clamp_nonneg("vote_count", MAX_VOTE_COUNT))

    # log features
    .withColumn("popularity_log", safe_log1p("popularity"))
    .withColumn("vote_count_log", safe_log1p("vote_count"))

    # state control for aggregations
    .withWatermark("event_ts", "2 minutes")
)

# Predict with all models, union, bundle
pred_dfs = []
for model_name, m in models:
    p = m.transform(features)

    score_col = F.lit(None).cast("double").alias("score")
    if "probability" in p.columns:
        score_col = vector_to_array(F.col("probability")).getItem(1).cast("double").alias("score")

    one = (
        p.select(
            "id",
            "event_ts",
            "processed_ts",
            F.lit(model_name).alias("model"),
            F.col("prediction").cast("int").alias("prediction"),
            score_col,
        )
        .withColumn("pred_struct", F.struct("model", "prediction", "score"))
        .select("id", "event_ts", "processed_ts", "pred_struct")
    )
    pred_dfs.append(one)

union_preds = pred_dfs[0]
for d in pred_dfs[1:]:
    union_preds = union_preds.unionByName(d)

bundled = (
    union_preds
    .groupBy("id", "event_ts", "processed_ts")
    .agg(F.collect_list("pred_struct").alias("predictions"))
    .withColumn("latency_ms", (F.col("processed_ts").cast("long") - F.col("event_ts").cast("long")) * 1000)
    .select(
        F.to_json(
            F.struct(
                F.col("id"),
                F.date_format(F.col("event_ts"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").alias("event_time"),
                F.date_format(F.col("processed_ts"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").alias("processed_time"),
                F.col("latency_ms"),
                F.col("predictions"),
            )
        ).alias("value")
    )
)

query_out = (
    bundled.writeStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("topic", TOPIC_OUT)
    .option("checkpointLocation", CHECKPOINT_OUT)
    .outputMode("update")
    .trigger(processingTime="2 seconds")
    .start()
)

# Metrics stream: throughput + latency percentiles (p50/p95) per 10s
metrics = (
    features
    .groupBy(F.window("processed_ts", "10 seconds").alias("w"))
    .agg(
        F.count("*").alias("records"),
        F.expr("percentile_approx((processed_ts.cast('long')-event_ts.cast('long'))*1000, 0.5)").alias("p50_latency_ms"),
        F.expr("percentile_approx((processed_ts.cast('long')-event_ts.cast('long'))*1000, 0.95)").alias("p95_latency_ms"),
    )
    .select(
        F.to_json(
            F.struct(
                F.date_format(F.col("w.start"), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("window_start"),
                F.date_format(F.col("w.end"), "yyyy-MM-dd'T'HH:mm:ss'Z'").alias("window_end"),
                "records",
                "p50_latency_ms",
                "p95_latency_ms",
            )
        ).alias("value")
    )
)

query_metrics = (
    metrics.writeStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("topic", TOPIC_METRICS)
    .option("checkpointLocation", CHECKPOINT_METRICS)
    .outputMode("update")
    .trigger(processingTime="10 seconds")
    .start()
)

spark.streams.awaitAnyTermination()
