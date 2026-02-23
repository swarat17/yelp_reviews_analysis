from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("Data Cleaning Yelp Dataset") \
    .getOrCreate()

# Step 2: Read raw data
df = spark.read.json("hdfs://namenode:8020/input/yelp_academic_dataset_review.json")

# Step 3: Handle Missing Values
df = df.dropna(subset=["text", "stars"])  # Essential fields
df = df.fillna({"useful": 0, "funny": 0, "cool": 0})

# Step 4: Review Length Column
df = df.withColumn('review_length', length('text'))

# Step 5: Outlier Handling
# Example: Cap review_length at 99th percentile
quantiles = df.approxQuantile("review_length", [0.99], 0.05)
upper_cap = quantiles[0]
df = df.withColumn('review_length', when(col('review_length') > upper_cap, upper_cap).otherwise(col('review_length')))

# Step 6: Data Type Corrections
df = df.withColumn("stars", col("stars").cast("integer"))
df = df.withColumn("useful", col("useful").cast("integer"))
df = df.withColumn("funny", col("funny").cast("integer"))
df = df.withColumn("cool", col("cool").cast("integer"))

# Step 7: Save Cleaned Data to HDFS
df = df.repartition(100)  # or higher depending on your cluster size
df.write.mode('overwrite').parquet("hdfs://namenode:8020/user/root/output/cleaned/yelp_reviews_cleaned.parquet")

spark.stop()