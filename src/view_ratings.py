from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("View Rating Predictions") \
    .getOrCreate()

# Read and display
df = spark.read.parquet("hdfs://namenode:8020/user/root/output/rating_predictions")
df.show(10, truncate=False)

spark.stop()