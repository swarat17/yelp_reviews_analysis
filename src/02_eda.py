from pyspark.sql import SparkSession
from pyspark.sql.functions import year, length
import matplotlib.pyplot as plt
import os

# Create output directories if not exist
os.makedirs('output/plots', exist_ok=True)

# Step 1: Create Spark session
spark = SparkSession.builder \
    .appName("EDA Yelp Dataset") \
    .getOrCreate()

# Step 2: Read data from HDFS
df = spark.read.json("hdfs://namenode:8020/input/yelp_academic_dataset_review.json")

# Step 3: Print Schema and first rows   
df.printSchema()
df.show(5)

# Step 4: Basic Summary Statistics
df.describe(['stars', 'useful', 'funny', 'cool']).show()

# Step 5: Review Length Analysis
df = df.withColumn('review_length', length('text'))
df.select('stars', 'review_length').show(5)

# Step 6: Temporal Trends (Year Extraction)
df = df.withColumn('year', year('date'))
yearly_review_counts = df.groupBy('year').count().orderBy('year')
yearly_review_counts.show()

# Step 7: Year wise Reviews
pandas_yearly = yearly_review_counts.toPandas()
pandas_yearly.plot(x='year', y='count', kind='bar')
plt.title('Number of Reviews per Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig('output/plots/reviews_per_year.png')

# Step 8: Star Ratings Distribution
stars_df = df.groupBy('stars').count().orderBy('stars')
pandas_stars = stars_df.toPandas()
pandas_stars.plot(x='stars', y='count', kind='bar')
plt.title('Star Ratings Distribution')
plt.xlabel('Star Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('output/plots/star_distribution.png')

spark.stop()