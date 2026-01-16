# tmdb_stream_predict.py
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

BOOTSTRAP = "localhost:29092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"
MODEL_PATH = "models/tmdb_lr"

# Te same sanity limity co w treningu
MAX_RUNTIME = 400.0
MAX_BUDGET = 5e8
MAX_POPULARITY = 1e6
MAX_VOTE_COUNT = 5e6

schema = T.StructType([
    T.StructField("id", T.StringType(), True),
    T.StructField("budget", T.DoubleType(), True),
    T.StructField("runtime", T.DoubleType(), True),
    T.StructField("popularity", T.DoubleType(), True),
    T.StructField("vote_count", T.DoubleType(), True),
    T.StructField("release_year", T.IntegerType(), True),
    T.StructField("original_language", T.StringType(), True),
])

def clean_nonneg_unknown0(col: str, maxv: float):
    # 0 / <=0 traktujemy jako missing (jak w treningu)
    return (
        F.when(F.col(col).isNull(), None)
        .when(F.col(col) <= 0, None)
        .when(F.col(col) > F.lit(maxv), F.lit(maxv))
        .otherwise(F.col(col))
    )

spark = SparkSession.builder.appName("TMDB-Streaming-Predict").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

model = PipelineModel.load(MODEL_PATH)

raw = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("subscribe", TOPIC_IN)
    .option("startingOffsets", "latest")
    .load()
)

# Kafka value -> JSON -> kolumny
json_df = raw.select(F.col("value").cast("string").alias("json_str"))
parsed = (
    json_df
    .select(F.from_json("json_str", schema).alias("data"))
    .select("data.*")
)

# Upewnij się o typach (czasem JSON przyjdzie jako int -> cast)
features = (
    parsed
    .withColumn("budget", F.col("budget").cast("double"))
    .withColumn("runtime", F.col("runtime").cast("double"))
    .withColumn("popularity", F.col("popularity").cast("double"))
    .withColumn("vote_count", F.col("vote_count").cast("double"))
    .withColumn("release_year", F.col("release_year").cast("int"))
    .withColumn("original_language", F.coalesce(F.col("original_language"), F.lit("unknown")))
)

# Czyszczenie jak w treningu
features = (
    features
    .withColumn("budget_log", F.log1p(F.col("budget")))
    .withColumn("runtime_log", F.log1p(F.col("runtime")))
    .withColumn("popularity_log", F.log1p(F.col("popularity")))
    .withColumn("vote_count_log", F.log1p(F.col("vote_count")))
)

# Predykcja
pred = model.transform(features)

# Output: prediction + P(label=1)
out = pred.select(
    F.col("id"),
    F.col("prediction").cast("int").alias("prediction"),
    vector_to_array(F.col("probability")).getItem(1).cast("double").alias("probability"),
)

out_json = out.select(F.to_json(F.struct(*out.columns)).alias("value"))

query = (
    out_json.writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("topic", TOPIC_OUT)
    .option("checkpointLocation", "checkpoints/tmdb_stream_predict")
    .outputMode("append")
    .start()
)

query.awaitTermination()