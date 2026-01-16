# tmdb_spark_ml.py
# Poprawiona wersja:
# - spójna z pipeline streamingowym (te same feature'y co w stream_predict.py)
# - log1p na skośnych rozkładach
# - StandardScaler dla LR
# - wagi klas (class imbalance)
# - AUTOMATYCZNE wyrzucanie numeric feature'ów, które na TRAIN są całe NULL/NaN (fix dla Imputer crash)
# - opcjonalnie: traktowanie 0 jako brak danych dla budget/runtime (częste w TMDB)

from __future__ import annotations

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
    StandardScaler,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# === KONFIG ===
MODEL_PATH = "models/tmdb_lr"

# MUSI PASOWAĆ DO STREAMINGU
STREAM_NUMERIC = ["budget", "runtime", "popularity", "vote_count", "release_year"]
STREAM_CATEGORICAL = ["original_language"]

LABEL_THRESHOLD = 7.0  # vote_average >= 7.0 => label=1

# sanity limity (TMDB bywa brudne / outliery)
MAX_RUNTIME = 400.0
MAX_BUDGET = 5e8
MAX_POPULARITY = 1e6
MAX_VOTE_COUNT = 5e6


def build_spark(app_name: str = "TMDB-ETL-ML") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .enableHiveSupport()  # jak nie masz hive -> usuń tę linię
        .getOrCreate()
    )


def normalize_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    return df


def clean_nonneg_unknown0(col: str, maxv: float):
    """
    Czyści kolumny numeryczne:
    - NULL zostaje NULL
    - wartości <= 0 traktujemy jako brak danych (często w TMDB 0 = unknown)
    - outliery tniemy do maxv
    """
    return (
        F.when(F.col(col).isNull(), None)
        .when(F.col(col) <= 0, None)
        .when(F.col(col) > F.lit(maxv), F.lit(maxv))
        .otherwise(F.col(col))
    )


def keep_nonempty_numeric(df, cols: list[str]) -> list[str]:
    """
    Zwraca tylko te kolumny numeryczne, które mają na df >=1 wartości nie-NULL i nie-NaN.
    Fix dla:
      SparkException: surrogate cannot be computed. All the values ... are Null, Nan ...
    """
    if not cols:
        return []

    agg_exprs = [
        F.count(F.when(F.col(c).isNotNull() & (~F.isnan(F.col(c))), 1)).alias(c)
        for c in cols
    ]
    counts = df.agg(*agg_exprs).collect()[0].asDict()
    kept = [c for c in cols if counts.get(c, 0) > 0]
    return kept


def main(
    input_csv: str,
    output_parquet: str = "tmdb_clean.parquet",
    hive_db: str = "bigdata",
    hive_table: str = "tmdb_movies_clean",
):
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1) Wczytanie CSV
    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .csv(input_csv)
    )
    df = normalize_columns(df)

    # 2) Minimalny zestaw kolumn (nie wszystko w TMDB ma sens do ML)
    needed = [
        "release_date",
        "budget",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "original_language",
        "id",
        "title",
    ]
    existing = [c for c in needed if c in df.columns]
    df_sel = df.select(*existing)

    # 3) Cast liczb
    for c in ["budget", "runtime", "popularity", "vote_average", "vote_count"]:
        if c in df_sel.columns:
            df_sel = df_sel.withColumn(c, F.col(c).cast(DoubleType()))

    # release_year
    if "release_date" in df_sel.columns:
        df_sel = df_sel.withColumn("release_year", F.year(F.to_date("release_date")))
    else:
        df_sel = df_sel.withColumn("release_year", F.lit(None).cast("int"))

    # 4) Czyszczenie: outliery + 0 jako brak danych dla budget/runtime/votes/popularity
    if "runtime" in df_sel.columns:
        df_sel = df_sel.withColumn("runtime", clean_nonneg_unknown0("runtime", MAX_RUNTIME))
    if "budget" in df_sel.columns:
        df_sel = df_sel.withColumn("budget", clean_nonneg_unknown0("budget", MAX_BUDGET))
    if "popularity" in df_sel.columns:
        df_sel = df_sel.withColumn("popularity", clean_nonneg_unknown0("popularity", MAX_POPULARITY))
    if "vote_count" in df_sel.columns:
        df_sel = df_sel.withColumn("vote_count", clean_nonneg_unknown0("vote_count", MAX_VOTE_COUNT))

    # 5) Label
    df_ml = df_sel.filter(F.col("vote_average").isNotNull())
    df_ml = df_ml.withColumn(
        "label",
        F.when(F.col("vote_average") >= F.lit(LABEL_THRESHOLD), F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    # 6) Zapewnij spójność ze streamingiem (kolumny muszą istnieć)
    for c in STREAM_NUMERIC:
        if c not in df_ml.columns:
            df_ml = df_ml.withColumn(c, F.lit(None).cast(DoubleType()))
    for c in STREAM_CATEGORICAL:
        if c not in df_ml.columns:
            df_ml = df_ml.withColumn(c, F.lit("unknown"))

    # 7) Log1p features (dla skośnych rozkładów)
    # log1p tylko dla >=0; NULL zostaje NULL
    df_ml = df_ml.withColumn("budget_log", F.log1p(F.col("budget")))
    df_ml = df_ml.withColumn("runtime_log", F.log1p(F.col("runtime")))
    df_ml = df_ml.withColumn("popularity_log", F.log1p(F.col("popularity")))
    df_ml = df_ml.withColumn("vote_count_log", F.log1p(F.col("vote_count")))

    numeric_features = ["budget_log", "runtime_log", "popularity_log", "vote_count_log", "release_year"]
    categorical_features = ["original_language"]

    # 8) Split
    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

    # 9) Wagi klas (imbalance)
    counts = train.groupBy("label").count().collect()
    cdict = {float(r["label"]): float(r["count"]) for r in counts}
    n0 = cdict.get(0.0, 1.0)
    n1 = cdict.get(1.0, 1.0)
    pos_weight = (n0 / n1) if n1 > 0 else 1.0

    train = train.withColumn("weight", F.when(F.col("label") == 1.0, F.lit(pos_weight)).otherwise(F.lit(1.0)))
    test = test.withColumn("weight", F.lit(1.0))

    # 10) FIX: usuń featury numeryczne, które na TRAIN są całe NULL/NaN (Imputer inaczej wywala)
    numeric_features = keep_nonempty_numeric(train, numeric_features)
    if not numeric_features:
        raise RuntimeError(
            "Brak sensownych numeric features po czyszczeniu (wszystko NULL/NaN). "
            "Sprawdź dane wejściowe / casty kolumn."
        )

    print("Numeric features kept:", numeric_features)

    # 11) Pipeline
    stages = []

    # Imputer (tylko na tych co zostały)
    imp_out = [f"{c}_imp" for c in numeric_features]
    imputer = Imputer(inputCols=numeric_features, outputCols=imp_out)
    stages.append(imputer)

    # Vector + StandardScaler
    num_vec = VectorAssembler(inputCols=imp_out, outputCol="num_vec", handleInvalid="keep")
    scaler = StandardScaler(inputCol="num_vec", outputCol="num_scaled", withMean=True, withStd=True)
    stages += [num_vec, scaler]

    # Kategoria -> index -> onehot
    # (jeśli kiedyś dodasz więcej kategorii, zrób pętlę; tu tylko original_language)
    idx = StringIndexer(
        inputCol="original_language",
        outputCol="original_language_idx",
        handleInvalid="keep",
    )
    ohe = OneHotEncoder(
        inputCols=["original_language_idx"],
        outputCols=["original_language_ohe"],
    )
    stages += [idx, ohe]

    # Final features
    features = VectorAssembler(
        inputCols=["num_scaled", "original_language_ohe"],
        outputCol="features",
        handleInvalid="keep",
    )
    stages.append(features)

    # LR
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxIter=200,
        regParam=0.03,
        elasticNetParam=0.0,  # L2
    )
    stages.append(lr)

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train)

    # 12) zapis pipeline model (pod streaming)
    model.write().overwrite().save(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

    preds = model.transform(test).cache()

    # 13) Ewaluacja
    eval_roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    eval_pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )

    auc_roc = eval_roc.evaluate(preds)
    auc_pr = eval_pr.evaluate(preds)

    print(f"\nAUC ROC = {auc_roc:.4f}")
    print(f"AUC PR  = {auc_pr:.4f}\n")

    # 14) Prosty sweep progu pod F1 (żeby nie fiksować się na 0.5)
    thresholds = [i / 100.0 for i in range(10, 91, 5)]
    best = (0.0, None, None, None)  # f1, thr, prec, rec

    scored = preds.select(
        F.col("label").cast("double").alias("label"),
        vector_to_array(F.col("probability")).getItem(1).alias("p1"),
    ).cache()

    for thr in thresholds:
        yhat = scored.withColumn("pred", F.when(F.col("p1") >= F.lit(thr), 1.0).otherwise(0.0))
        tp = yhat.filter((F.col("pred") == 1.0) & (F.col("label") == 1.0)).count()
        fp = yhat.filter((F.col("pred") == 1.0) & (F.col("label") == 0.0)).count()
        fn = yhat.filter((F.col("pred") == 0.0) & (F.col("label") == 1.0)).count()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        if f1 > best[0]:
            best = (f1, thr, prec, rec)

    f1, thr, prec, rec = best
    print(f"Best threshold (by F1 sweep) = {thr:.2f}")
    print(f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}\n")

    # 15) Zapis ETL (parquet + hive)
    df_sel.write.mode("overwrite").parquet(output_parquet)

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {hive_db}")
    spark.sql(f"USE {hive_db}")

    (
        df_sel.write.mode("overwrite")
        .format("parquet")
        .saveAsTable(f"{hive_db}.{hive_table}")
    )

    # 16) Szybka analiza
    spark.sql(f"""
        SELECT release_year, COUNT(*) AS n, AVG(vote_average) AS avg_rating
        FROM {hive_db}.{hive_table}
        WHERE release_year IS NOT NULL
        GROUP BY release_year
        ORDER BY release_year DESC
        LIMIT 20
    """).show(20, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main(input_csv="data/tmdb_10000_movies.csv")
