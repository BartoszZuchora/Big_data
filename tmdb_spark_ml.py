import os
import ast
import pandas as pd

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, IntegerType, BooleanType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer,
    VectorSlicer
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

INPUT_CSV = "data/tmdb_10000_movies.csv"
MODEL_DIR = "models"
OUT_CLEAN = "tmdb_clean.parquet"

LABEL_THRESHOLD = 7.0

TOP_N_FEATURES = 30


def build_spark(app_name="TMDB-Batch-ETL-ML"):
    return SparkSession.builder.appName(app_name).getOrCreate()


def normalize_columns(df):
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))
    return df


def safe_name(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s)


@F.udf(IntegerType())
def genre_count_udf(s):
    if s is None:
        return None
    try:
        arr = ast.literal_eval(s)
        return int(len(arr)) if isinstance(arr, list) else None
    except Exception:
        return None


def evaluate_predictions(preds, name: str):
    cm = preds.groupBy("label", "prediction").count().orderBy("label", "prediction")
    print(f"\n[{name}] Confusion matrix (label, prediction, count):")
    cm.show(truncate=False)

    acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    pr_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    rc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

    accuracy = acc_eval.evaluate(preds)
    f1 = f1_eval.evaluate(preds)
    precision = pr_eval.evaluate(preds)
    recall = rc_eval.evaluate(preds)

    auc = None
    try:
        auc_eval = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC"
        )
        auc = auc_eval.evaluate(preds)
    except Exception as e:
        print(f"[{name}] AUC niepoliczalne: {e}")

    out = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
    print(f"[{name}] metrics: {out}")
    return out


def get_feature_names_from_metadata(df_with_features, feature_col="features"):
    try:
        md = df_with_features.schema[feature_col].metadata
        attrs = md["ml_attr"]["attrs"]
        names = []
        for k in ["binary", "numeric"]:
            if k in attrs:
                names += [a["name"] for a in attrs[k]]
        return names
    except Exception:
        return []


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

    cols = [
        "id", "title", "overview", "genre_ids",
        "adult", "video",
        "original_language",
        "popularity", "vote_average", "vote_count",
        "release_date",
    ]
    cols = [c for c in cols if c in df.columns]
    df_sel = df.select(*cols)

    for c in ["popularity", "vote_average", "vote_count"]:
        if c in df_sel.columns:
            df_sel = df_sel.withColumn(c, F.col(c).cast(DoubleType()))

    df_sel = df_sel.withColumn("release_year", F.year(F.to_date("release_date")))
    df_sel = df_sel.withColumn("genre_count", genre_count_udf(F.col("genre_ids")))

    df_sel = df_sel.withColumn("adult", F.col("adult").cast(BooleanType()))
    df_sel = df_sel.withColumn("video", F.col("video").cast(BooleanType()))

    df_sel = df_sel.withColumn(
        "popularity_log",
        F.when(F.col("popularity").isNull(), None).otherwise(F.log1p("popularity"))
    )
    df_sel = df_sel.withColumn(
        "vote_count_log",
        F.when(F.col("vote_count").isNull(), None).otherwise(F.log1p("vote_count"))
    )

    df_ml = df_sel.filter(F.col("vote_average").isNotNull())
    df_ml = df_ml.withColumn("label", F.when(F.col("vote_average") >= F.lit(LABEL_THRESHOLD), 1.0).otherwise(0.0))

    df_ml = df_ml.withColumn("adult_num", F.when(F.col("adult") == True, 1.0).otherwise(0.0))
    df_ml = df_ml.withColumn("video_num", F.when(F.col("video") == True, 1.0).otherwise(0.0))

    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

    categorical = ["original_language"] if "original_language" in df_ml.columns else []
    numeric = [
        c for c in [
            "popularity", "vote_count", "release_year", "genre_count",
            "popularity_log", "vote_count_log",
            "adult_num", "video_num"
        ] if c in df_ml.columns
    ]

    stages = []

    imputer = Imputer(inputCols=numeric, outputCols=[f"{c}_imp" for c in numeric])
    stages.append(imputer)
    numeric_imp = [f"{c}_imp" for c in numeric]

    ohe_out = []
    for c in categorical:
        idx = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"])
        stages += [idx, ohe]
        ohe_out.append(f"{c}_ohe")

    feature_cols = numeric_imp + ohe_out
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1, elasticNetParam=0.0)
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, maxDepth=10, seed=42)
    gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=100, maxDepth=5, seed=42)

    baseline_models = [("Baseline_LR", lr), ("Baseline_RF", rf), ("Baseline_GBT", gbt)]

    results = []

    for name, est in baseline_models:
        pipe = Pipeline(stages=stages + [est])
        model = pipe.fit(train)
        preds = model.transform(test)
        metrics = evaluate_predictions(preds, name)

        path = os.path.join(MODEL_DIR, f"tmdb_{safe_name(name)}")
        model.write().overwrite().save(path)
        results.append({"model": name, "model_path": path, **metrics})

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    lr2 = LogisticRegression(featuresCol="features", labelCol="label")

    lr_pipe = Pipeline(stages=stages + [lr2])
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr2.regParam, [0.0, 0.01, 0.1])
        .addGrid(lr2.elasticNetParam, [0.0, 0.5, 1.0])
        .addGrid(lr2.maxIter, [50, 100])
        .build()
    )

    lr_cv = CrossValidator(
        estimator=lr_pipe,
        estimatorParamMaps=lr_grid,
        evaluator=evaluator_auc,
        numFolds=3,
        parallelism=1,
        seed=42
    )

    lr_cv_model = lr_cv.fit(train)
    lr_preds = lr_cv_model.transform(test)
    lr_metrics = evaluate_predictions(lr_preds, "Tuned_LR")

    lr_path = os.path.join(MODEL_DIR, "tmdb_tuned_lr")
    lr_cv_model.bestModel.write().overwrite().save(lr_path)
    results.append({"model": "Tuned_LR", "model_path": lr_path, **lr_metrics})

    rf2 = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
    rf_pipe = Pipeline(stages=stages + [rf2])

    rf_grid = (
        ParamGridBuilder()
        .addGrid(rf2.numTrees, [100, 300])
        .addGrid(rf2.maxDepth, [8, 12])
        .addGrid(rf2.maxBins, [32, 64])
        .build()
    )

    rf_cv = CrossValidator(
        estimator=rf_pipe,
        estimatorParamMaps=rf_grid,
        evaluator=evaluator_auc,
        numFolds=3,
        parallelism=1,
        seed=42
    )

    rf_cv_model = rf_cv.fit(train)
    rf_preds = rf_cv_model.transform(test)
    rf_metrics = evaluate_predictions(rf_preds, "Tuned_RF")

    rf_path = os.path.join(MODEL_DIR, "tmdb_tuned_rf")
    rf_cv_model.bestModel.write().overwrite().save(rf_path)
    results.append({"model": "Tuned_RF", "model_path": rf_path, **rf_metrics})

    best_rf_model = rf_cv_model.bestModel.stages[-1]
    importances = best_rf_model.featureImportances

    tmp = rf_cv_model.bestModel.transform(train.limit(2000))
    feature_names = get_feature_names_from_metadata(tmp, feature_col="features")

    fi = []
    for i, v in enumerate(importances):
        fname = feature_names[i] if i < len(feature_names) else f"f_{i}"
        fi.append((fname, float(v), i))

    fi_sorted = sorted(fi, key=lambda x: x[1], reverse=True)
    top = fi_sorted[:TOP_N_FEATURES]

    fi_path = os.path.join(MODEL_DIR, "feature_importance_top30.csv")
    pd.DataFrame([(n, imp) for n, imp, _ in top], columns=["feature", "importance"]).to_csv(fi_path, index=False)
    print("Saved feature importance:", fi_path)

    top_indices = [idx for _, _, idx in top]

    slicer = VectorSlicer(inputCol="features", outputCol="features_reduced", indices=top_indices)
    lr_red = LogisticRegression(featuresCol="features_reduced", labelCol="label", maxIter=100, regParam=0.01)

    reduced_pipe = Pipeline(stages=stages + [slicer, lr_red])
    reduced_model = reduced_pipe.fit(train)
    reduced_preds = reduced_model.transform(test)
    reduced_metrics = evaluate_predictions(reduced_preds, f"LR_ReducedTop{TOP_N_FEATURES}")

    red_path = os.path.join(MODEL_DIR, f"tmdb_lr_reduced_top{TOP_N_FEATURES}")
    reduced_model.write().overwrite().save(red_path)
    results.append({"model": f"LR_ReducedTop{TOP_N_FEATURES}", "model_path": red_path, **reduced_metrics})

    results_path = os.path.join(MODEL_DIR, "tmdb_model_results.csv")
    pd.DataFrame(results).to_csv(results_path, index=False)
    print("Saved results:", results_path)

    df_sel.write.mode("overwrite").parquet(OUT_CLEAN)
    print("Saved cleaned data:", OUT_CLEAN)

    spark.stop()


if __name__ == "__main__":
    main()
