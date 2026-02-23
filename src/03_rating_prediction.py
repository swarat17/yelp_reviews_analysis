from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("Yelp Star Rating Prediction") \
        .config("spark.shuffle.io.retryWait", "60s") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.network.timeout", "800s") \
        .config("spark.executor.heartbeatInterval", "60s") \
        .getOrCreate()

    # Load cleaned reviews (must include 'text' & 'stars')
    reviews = spark.read.parquet(
        "hdfs://namenode:8020/user/root/output/cleaned/yelp_reviews_cleaned.parquet"
    )

    reviews = reviews.sample(fraction=0.05, seed=42)  # Only 5% data

    # 1. Pipeline stages for text → features
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    remover   = StopWordsRemover(inputCol="words", outputCol="filtered")
    cv        = CountVectorizer(inputCol="filtered", outputCol="rawFeatures",
                                vocabSize=1000, minDF=5)
    idf       = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=5)
    label_idx = StringIndexer(inputCol="stars", outputCol="label")

    # 2. Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")

    pipeline = Pipeline(stages=[
        tokenizer, remover, cv, idf, label_idx, rf
    ])

    # 3. Train/Test split
    #reviews.cache()
    train, test = reviews.randomSplit([0.8, 0.2], seed=42)

    # 4. Hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(cv.vocabSize, [1000]) \
        .addGrid(rf.numTrees, [20]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )

    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        trainRatio=0.8
    ).fit(train)

    # 5. Evaluate on test set
    preds = tvs.transform(test)
    acc = evaluator.evaluate(preds)
    print(f"⭐ Test Accuracy = {acc:.4f}")

    # 6. Persist model & predictions
    tvs.bestModel.write() \
        .overwrite() \
        .save("hdfs://namenode:8020/user/root/models/rating_prediction")

    preds.select("review_id", "prediction", "label") \
        .write.mode("overwrite") \
        .parquet("hdfs://namenode:8020/user/root/output/rating_predictions")

    spark.stop()