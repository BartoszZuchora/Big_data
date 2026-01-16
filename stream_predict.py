from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel

BOOTSTRAP = "localhost:9092"
TOPIC_IN = "tmdb_features_in"
TOPIC_OUT = "tmdb_predictions_out"

MODEL_PATH = "models/tmdb_lr"

schema = T.StructType([
    T.StructField("id", T.StringType(), True),
    T.StructField("budget", T.DoubleType(), True),
    T.StructField("runtime", T.DoubleType(), True),
    T.StructField("popularity", T.DoubleType(), True),
    T.StructField("vote_count", T.DoubleType(), True),
    T.StructField("release_year", T.IntegerType(), True),
    T.StructField("original_language", T.StringType(), True),
])

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

json_df = raw.select(F.col("value").cast("string").alias("json_str"))
parsed = json_df.select(F.from_json("json_str", schema).alias("data")).select("data.*")

pred = model.transform(parsed)

out = pred.select(
    "id",
    F.col("prediction").cast("int").alias("prediction"),
    F.col("probability").getItem(1).alias("probability")
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
