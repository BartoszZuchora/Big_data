import os
from typing import Dict, List, Tuple

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

BOOTSTRAP = "localhost:29092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"

# Katalog z modelami zapisanymi przez batch-trening
MODEL_DIR = "models"

CHECKPOINT_DIR = "checkpoints/tmdb_stream_predict_all_models"

# sanity caps (opcjonalnie)
MAX_RUNTIME = 400.0
MAX_BUDGET = 5e8
MAX_POPULARITY = 1e6
MAX_VOTE_COUNT = 5e6

schema = T.StructType(
    [
        T.StructField("id", T.StringType(), True),
        T.StructField("budget", T.DoubleType(), True),
        T.StructField("runtime", T.DoubleType(), True),
        T.StructField("popularity", T.DoubleType(), True),
        T.StructField("vote_count", T.DoubleType(), True),
        T.StructField("release_year", T.IntegerType(), True),
        T.StructField("original_language", T.StringType(), True),
    ]
)


def clamp_nonneg(col: str, maxv: float):
    """
    - null -> null
    - <=0 -> null
    - >maxv -> maxv
    - else -> value
    """
    return (
        F.when(F.col(col).isNull(), F.lit(None).cast("double"))
        .when(F.col(col) <= 0, F.lit(None).cast("double"))
        .when(F.col(col) > F.lit(maxv), F.lit(maxv).cast("double"))
        .otherwise(F.col(col).cast("double"))
    )


def safe_log1p(col: str):
    """
    log1p tylko dla wartości > -1; w praktyce zakładamy nieujemne po clamp.
    null zostaje null.
    """
    return F.when(F.col(col).isNull(), F.lit(None).cast("double")).otherwise(
        F.log1p(F.col(col))
    )


def list_model_paths(model_dir: str) -> List[Tuple[str, str]]:
    """
    Zwraca listę (model_name, model_path) dla podkatalogów w MODEL_DIR.
    Filtr: katalogi zaczynające się od 'tmdb_' (zgodnie z treningiem).
    """
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"MODEL_DIR nie istnieje lub nie jest katalogiem: {model_dir}")

    items = []
    for name in sorted(os.listdir(model_dir)):
        path = os.path.join(model_dir, name)
        if os.path.isdir(path) and name.startswith("tmdb_"):
            items.append((name, path))

    if not items:
        raise RuntimeError(
            f"Nie znaleziono żadnych modeli w {model_dir}. "
            f"Oczekuję katalogów np. models/tmdb_logisticregression"
        )

    return items


def load_models(model_dir: str) -> List[Tuple[str, PipelineModel]]:
    models = []
    for name, path in list_model_paths(model_dir):
        m = PipelineModel.load(path)
        models.append((name, m))
    return models


spark = SparkSession.builder.appName("TMDB-Streaming-Predict-AllModels").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

models = load_models(MODEL_DIR)
print("Loaded models:")
for name, _ in models:
    print(" -", name)

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("subscribe", TOPIC_IN)
    .option("startingOffsets", "latest")
    .load()
)

json_df = raw.select(F.col("value").cast("string").alias("json_str"))
parsed = json_df.select(F.from_json("json_str", schema).alias("data")).select("data.*")

# ---- casting + lekkie czyszczenie ----
features = (
    parsed.withColumn("budget", F.col("budget").cast("double"))
    .withColumn("runtime", F.col("runtime").cast("double"))
    .withColumn("popularity", F.col("popularity").cast("double"))
    .withColumn("vote_count", F.col("vote_count").cast("double"))
    .withColumn("release_year", F.col("release_year").cast("int"))
    .withColumn(
        "original_language",
        F.coalesce(F.col("original_language"), F.lit("unknown")),
    )
)

# Caps + nonneg
features = (
    features.withColumn("budget", clamp_nonneg("budget", MAX_BUDGET))
    .withColumn("runtime", clamp_nonneg("runtime", MAX_RUNTIME))
    .withColumn("popularity", clamp_nonneg("popularity", MAX_POPULARITY))
    .withColumn("vote_count", clamp_nonneg("vote_count", MAX_VOTE_COUNT))
)

# Log cechy (opcjonalne). Uwaga: jeśli model nie był trenowany z *_log,
# to PipelineModel i tak ich nie użyje (nie zaszkodzi mieć dodatkowych kolumn).
features = (
    features.withColumn("budget_log", safe_log1p("budget"))
    .withColumn("runtime_log", safe_log1p("runtime"))
    .withColumn("popularity_log", safe_log1p("popularity"))
    .withColumn("vote_count_log", safe_log1p("vote_count"))
)

# ---- Predykcje dla wszystkich modeli ----
# Zrobimy osobne transform() dla każdego modelu i potem złączymy do "bundle"
# (jedna wiadomość na id).

pred_dfs = []
for model_name, m in models:
    p = m.transform(features)

    # prediction zawsze jest
    pred_col = F.col("prediction").cast("int").alias("prediction")

    # score:
    # - dla modeli z probability: bierzemy P(1)
    # - jeśli brak probability: score = null
    score_col = F.lit(None).cast("double").alias("score")
    if "probability" in p.columns:
        score_col = (
            vector_to_array(F.col("probability"))
            .getItem(1)
            .cast("double")
            .alias("score")
        )

    one = (
        p.select(
            F.col("id"),
            F.lit(model_name).alias("model"),
            pred_col,
            score_col,
        )
        .withColumn(
            "pred_struct",
            F.struct(
                F.col("model"),
                F.col("prediction"),
                F.col("score"),
            ),
        )
        .select("id", "pred_struct")
    )

    pred_dfs.append(one)

# union all
union_preds = pred_dfs[0]
for d in pred_dfs[1:]:
    union_preds = union_preds.unionByName(d)

# group into list per id
bundled = (
    union_preds.groupBy("id")
    .agg(F.collect_list("pred_struct").alias("predictions"))
    .select(
        F.to_json(F.struct(F.col("id"), F.col("predictions"))).alias("value")
    )
)

query = (
    bundled.writeStream.format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP)
    .option("topic", TOPIC_OUT)
    .option("checkpointLocation", CHECKPOINT_DIR)
    .outputMode("update")  # update, bo robimy agregację groupBy
    .start()
)

query.awaitTermination()