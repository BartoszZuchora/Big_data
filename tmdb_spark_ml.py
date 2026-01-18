import os

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    Imputer,
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
    LinearSVC,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# ====== DOPASUJ ŚCIEŻKI ======
INPUT_CSV = "data/tmdb_10000_movies.csv"
MODEL_DIR = "models"  # katalog, do którego zapiszą się wszystkie modele


def build_spark(app_name: str = "TMDB-Batch-ETL-ML"):
    return SparkSession.builder.appName(app_name).getOrCreate()


def normalize_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    return df


def safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s)


def build_preprocess_stages(numeric_features, categorical_features):
    stages = []

    # Imputer dla numerycznych
    numeric_imp = []
    if numeric_features:
        imputer = Imputer(
            inputCols=numeric_features,
            outputCols=[f"{c}_imp" for c in numeric_features],
        )
        stages.append(imputer)
        numeric_imp = [f"{c}_imp" for c in numeric_features]

    # OneHot dla kategorii
    ohe_outputs = []
    for c in categorical_features:
        idx = StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep",
        )
        ohe = OneHotEncoder(
            inputCols=[f"{c}_idx"],
            outputCols=[f"{c}_ohe"],
        )
        stages += [idx, ohe]
        ohe_outputs.append(f"{c}_ohe")

    feature_cols = numeric_imp + ohe_outputs
    if not feature_cols:
        raise RuntimeError("Brak cech po preprocessingu (feature_cols puste).")

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )
    stages.append(assembler)

    return stages


def confusion_and_metrics(preds):
    cm = (
        preds.groupBy("label", "prediction")
        .count()
        .orderBy("label", "prediction")
    )
    cm.show(truncate=False)

    tp = preds.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
    tn = preds.filter((F.col("label") == 0) & (F.col("prediction") == 0)).count()
    fp = preds.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
    fn = preds.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(name, model, test):
    preds = model.transform(test)

    auc = None
    try:
        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )
        auc = evaluator.evaluate(preds)
    except Exception as e:
        print(f"[{name}] AUC niepoliczalne (np. LinearSVC): {e}")

    print(f"\n=== {name} ===")
    if auc is not None:
        print(f"AUC ROC: {auc:.4f}")

    print("Confusion matrix (label, prediction, count):")
    metrics = confusion_and_metrics(preds)

    print(
        "Accuracy : {accuracy:.4f}\n"
        "Precision: {precision:.4f}\n"
        "Recall   : {recall:.4f}\n"
        "F1-score : {f1:.4f}\n".format(**metrics)
    )

    return auc, metrics


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    df = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .csv(INPUT_CSV)
    )
    df = normalize_columns(df)

    expected = [
        "title",
        "release_date",
        "budget",
        "revenue",
        "runtime",
        "popularity",
        "vote_average",
        "vote_count",
        "original_language",
    ]
    existing = [c for c in expected if c in df.columns]
    df_sel = df.select(*existing)

    # Casty numeryczne
    for c in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]:
        if c in df_sel.columns:
            df_sel = df_sel.withColumn(c, F.col(c).cast(DoubleType()))

    # Feature engineering
    if "release_date" in df_sel.columns:
        df_sel = df_sel.withColumn(
            "release_year",
            F.year(F.to_date("release_date")),
        )

    if "budget" in df_sel.columns and "revenue" in df_sel.columns:
        df_sel = df_sel.withColumn("profit", F.col("revenue") - F.col("budget"))
        df_sel = df_sel.withColumn(
            "roi",
            F.when(F.col("budget") > 0, F.col("profit") / F.col("budget")).otherwise(
                F.lit(None)
            ),
        )

    # ====== LABEL (hit/non-hit) ======
    if "vote_average" not in df_sel.columns:
        raise RuntimeError("Brak kolumny vote_average – nie mogę zbudować label.")

    df_ml = df_sel.filter(F.col("vote_average").isNotNull())
    df_ml = df_ml.withColumn(
        "label",
        F.when(F.col("vote_average") >= F.lit(7.0), 1.0).otherwise(0.0),
    )

    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

    categorical = []
    if "original_language" in df_ml.columns:
        categorical.append("original_language")

    numeric_features = [
        c
        for c in [
            "budget",
            "revenue",
            "runtime",
            "popularity",
            "vote_count",
            "release_year",
            "profit",
            "roi",
        ]
        if c in df_ml.columns
    ]

    preprocess_stages = build_preprocess_stages(numeric_features, categorical)

    # ====== MODELE ======
    models = [
        (
            "LogisticRegression",
            LogisticRegression(
                featuresCol="features",
                labelCol="label",
                maxIter=100,
                regParam=0.1,
                elasticNetParam=0.0,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                numTrees=300,
                maxDepth=10,
                seed=42,
            ),
        ),
        (
            "GBTClassifier",
            GBTClassifier(
                featuresCol="features",
                labelCol="label",
                maxIter=150,
                maxDepth=5,
                seed=42,
            ),
        ),
        (
            "LinearSVC",
            LinearSVC(
                featuresCol="features",
                labelCol="label",
                maxIter=200,
                regParam=0.1,
            ),
        ),
    ]

    results = []

    for name, estimator in models:
        pipeline = Pipeline(stages=preprocess_stages + [estimator])
        fitted = pipeline.fit(train)

        # Zapis każdego modelu
        model_path = os.path.join(MODEL_DIR, f"tmdb_{safe_name(name)}")
        fitted.write().overwrite().save(model_path)
        print(f"Saved model {name} to: {model_path}")

        # Ewaluacja
        auc, metrics = evaluate_model(name, fitted, test)
        results.append(
            {
                "model": name,
                "model_path": model_path,
                "auc": float(auc) if auc is not None else None,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    # Zapis tabeli wyników
    results_path = os.path.join(MODEL_DIR, "tmdb_model_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nSaved results table to: {results_path}\n")

    # (opcjonalnie) zapis danych po ETL
    df_sel.write.mode("overwrite").parquet("tmdb_clean.parquet")
    print("Saved cleaned data to tmdb_clean.parquet")

    spark.stop()


if __name__ == "__main__":
    main()