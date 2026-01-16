# tmdb_spark_ml.py
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, Imputer
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

def build_spark(app_name: str = "TMDB-ETL-ML"):
    return (
        SparkSession.builder
        .appName(app_name)
        # Jeśli masz Hive:
        .enableHiveSupport()
        .getOrCreate()
    )

def safe_col(df, name):
    return name in df.columns

def main(input_csv: str,
         output_parquet: str = "tmdb_clean.parquet",
         hive_db: str = "bigdata",
         hive_table: str = "tmdb_movies_clean"):
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    # 1) Wczytanie CSV
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")
        .option("escape", "\"")
        .csv(input_csv)
    )

    # 2) Normalizacja nazw kolumn (opcjonalnie)
    # Ułatwia życie w Spark SQL
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lower().replace(" ", "_"))

    # 3) Wybór i czyszczenie (dopasuj do faktycznych kolumn)
    # Typowe kolumny TMDB: title, release_date, budget, revenue, runtime, popularity,
    # vote_average, vote_count, original_language, genres
    needed = ["title", "release_date", "budget", "revenue", "runtime",
              "popularity", "vote_average", "vote_count", "original_language"]
    existing = [c for c in needed if safe_col(df, c)]

    df_sel = df.select(*existing)

    # 4) Podstawowe czyszczenie
    # - usuwamy rekordy bez vote_average
    if safe_col(df_sel, "vote_average"):
        df_sel = df_sel.filter(F.col("vote_average").isNotNull())

    # - rzutowania do liczb (na wypadek stringów)
    num_cols = [c for c in ["budget", "revenue", "runtime", "popularity", "vote_count", "vote_average"]
                if safe_col(df_sel, c)]
    for c in num_cols:
        df_sel = df_sel.withColumn(c, F.col(c).cast(DoubleType()))

    # - proste feature engineering
    # rok z release_date
    if safe_col(df_sel, "release_date"):
        df_sel = df_sel.withColumn("release_year", F.year(F.to_date("release_date")))

    # profit i ROI
    if safe_col(df_sel, "budget") and safe_col(df_sel, "revenue"):
        df_sel = df_sel.withColumn("profit", F.col("revenue") - F.col("budget"))
        df_sel = df_sel.withColumn(
            "roi",
            F.when(F.col("budget") > 0, F.col("profit") / F.col("budget")).otherwise(F.lit(None))
        )

    # 5) Target (hit/non-hit)
    df_ml = df_sel.withColumn("label", F.when(F.col("vote_average") >= F.lit(7.0), 1.0).otherwise(0.0))

    # 6) Podział train/test
    train, test = df_ml.randomSplit([0.8, 0.2], seed=42)

    # 7) Pipeline MLlib
    categorical = []
    if safe_col(df_ml, "original_language"):
        categorical.append("original_language")

    numeric_features = [c for c in ["budget", "revenue", "runtime", "popularity", "vote_count", "release_year", "profit", "roi"]
                        if safe_col(df_ml, c)]

    stages = []

    # Imputer dla numerycznych
    if numeric_features:
        imputer = Imputer(inputCols=numeric_features,
                          outputCols=[f"{c}_imp" for c in numeric_features])
        stages.append(imputer)
        numeric_features_imp = [f"{c}_imp" for c in numeric_features]
    else:
        numeric_features_imp = []

    # OneHot dla kategorii
    ohe_outputs = []
    for c in categorical:
        idx = StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        ohe = OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_ohe"])
        stages += [idx, ohe]
        ohe_outputs.append(f"{c}_ohe")

    # VectorAssembler
    feature_cols = numeric_features_imp + ohe_outputs
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    # Model
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50, regParam=0.1)
    stages.append(lr)

    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(train)

    preds = model.transform(test)

    # 8) Ewaluacja
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(preds)

    print(f"\nAUC ROC = {auc:.4f}\n")

    # 9) Zapis wyników ETL
    # Parquet (pod Hive i generalnie Big Data-friendly)
    (
        df_sel.write
        .mode("overwrite")
        .parquet(output_parquet)
    )

    # 10) Zapis do Hive (jeśli masz skonfigurowane)
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {hive_db}")
    spark.sql(f"USE {hive_db}")

    (
        df_sel.write
        .mode("overwrite")
        .format("parquet")
        .saveAsTable(f"{hive_db}.{hive_table}")
    )

    # 11) Kilka przykładowych analiz Spark SQL
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
    # Podmień ścieżkę:
    # - lokalnie: "data/tmdb_10000.csv"
    # - HDFS: "hdfs:///data/tmdb/tmdb_10000.csv"
    main(input_csv="data/tmdb_10000_movies.csv")
