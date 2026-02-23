from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count as spark_count
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Yelp User Segmentation") \
        .getOrCreate()

    reviews = spark.read.parquet(
        "hdfs://namenode:8020/user/root/output/cleaned/yelp_reviews_cleaned.parquet"
    )

    user_stats = reviews.groupBy("user_id") \
        .agg(
            spark_count("*").alias("review_count"),
            avg("stars").alias("avg_rating_given"),
            avg("review_length").alias("avg_review_length")
        )

    user_stats.cache()

    assembler = VectorAssembler(
        inputCols=["review_count", "avg_rating_given", "avg_review_length"],
        outputCol="features"
    )

    kmeans = KMeans(featuresCol="features", predictionCol="segment", seed=42)

    pipeline = Pipeline(stages=[assembler, kmeans])

    paramGrid = ParamGridBuilder() \
        .addGrid(kmeans.k, [3, 5, 8]) \
        .build()

    evaluator = ClusteringEvaluator(
        featuresCol="features", predictionCol="segment",
        metricName="silhouette", distanceMeasure="squaredEuclidean"
    )

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8
    ).fit(user_stats)

    best = tvs.bestModel
    optimal_k = best.stages[-1].getK()
    print(f"✅ Optimal number of clusters selected: k = {optimal_k}")

    segmented = best.transform(user_stats)
    segmented.select("user_id", "segment") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/user/root/output/user_segments")

    best.write() \
        .overwrite() \
        .save("hdfs://namenode:8020/user/root/models/user_segmentation")

    spark.stop()