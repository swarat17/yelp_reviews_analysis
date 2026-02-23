from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("View User Segments") \
    .getOrCreate()

# Read and display
df = spark.read.parquet("hdfs://namenode:8020/user/root/output/user_segments")
df.show(10, truncate=False)

spark.stop()
