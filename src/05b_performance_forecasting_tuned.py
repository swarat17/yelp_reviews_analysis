from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, avg, count, lag
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Yelp Performance Forecasting with Tuning") \
        .config("spark.shuffle.io.retryWait", "60s") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

    reviews = spark.read.parquet(
        "hdfs://namenode:8020/user/root/output/cleaned/yelp_reviews_cleaned.parquet"
    )

    df = reviews.withColumn("year", year("date")) \
                .withColumn("month", month("date"))

    monthly = df.groupBy("business_id", "year", "month") \
                .agg(
                    avg("stars").alias("avg_rating"),
                    count("*").alias("review_count")
                )

    w = Window.partitionBy("business_id").orderBy("year", "month")
    monthly = monthly \
        .withColumn("lag_avg1", lag("avg_rating", 1).over(w)) \
        .withColumn("lag_cnt1", lag("review_count", 1).over(w)) \
        .na.drop(subset=["lag_avg1", "lag_cnt1"])

    # Cache before training
    monthly.cache()

    assembler = VectorAssembler(
        inputCols=["lag_avg1", "lag_cnt1", "year", "month"],
        outputCol="features"
    )

    rf = RandomForestRegressor(labelCol="avg_rating", featuresCol="features")

    pipeline = Pipeline(stages=[assembler, rf])

    train, test = monthly.randomSplit([0.8, 0.2], seed=42)

    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    evaluator = RegressionEvaluator(
        labelCol="avg_rating", predictionCol="prediction", metricName="rmse"
    )

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8,
        parallelism=2  # optional: tune more models at once if enough CPUs
    )

    model = tvs.fit(train)

    preds = model.transform(test)

    rmse = evaluator.evaluate(preds)
    print(f"📉 Test RMSE after tuning = {rmse:.4f}")

    model.bestModel.write() \
        .overwrite() \
        .save("hdfs://namenode:8020/user/root/output/models/performance_forecasting_tuned")

    preds.select("business_id", "year", "month", "prediction", "avg_rating") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/user/root/output/forecasting_predictions_tuned")

    spark.stop()