from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Arabic AI Streaming Simulation") \
    .getOrCreate()

print("=== Spark Structured Streaming Started ===")

schema = "text STRING, label DOUBLE, source_model STRING, generation_method STRING, clean_text STRING, word_count DOUBLE, avg_word_length DOUBLE"

stream_df = spark.readStream \
    .schema(schema) \
    .option("header", True) \
    .csv("stream_input")

result = stream_df.select(
    "clean_text",
    "label",
    "word_count",
    "avg_word_length"
)

query = result.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", True) \
    .start()

query.awaitTermination()
