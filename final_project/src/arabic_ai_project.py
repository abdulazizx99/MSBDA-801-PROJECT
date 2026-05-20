from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Arabic AI Text Detection Project").getOrCreate()

print("=== Spark Session Started ===")

df = spark.read.csv(
    "hdfs://localhost:9000/user/abdulaziz99/arabic_ai_project/raw/arabic_ai_balanced_dataset.csv",
    header=True,
    inferSchema=True
)

print("=== Dataset Loaded From HDFS ===")
print("Total Rows:", df.count())

df = df.withColumn("clean_text", regexp_replace(col("clean_text"), "[ًٌٍَُِّْـ]", "")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "[إأآا]", "ا")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "ى", "ي")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "ؤ", "و")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "ئ", "ي")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "[^\\u0600-\\u06FF\\s]", " ")) \
       .withColumn("clean_text", regexp_replace(col("clean_text"), "\\s+", " ")) \
       .withColumn("clean_text", trim(col("clean_text")))

df = df.withColumn("word_count", col("word_count").cast("double")) \
       .withColumn("avg_word_length", col("avg_word_length").cast("double")) \
       .withColumn("label", col("label").cast("double"))

clean_df = df.dropna(subset=["label", "word_count", "avg_word_length", "clean_text"])

print("=== Cleaned Dataset Count ===")
print("Clean Rows:", clean_df.count())

print("=== Label Distribution ===")
clean_df.groupBy("label").count().show()

clean_df.write.mode("overwrite").parquet(
    "hdfs://localhost:9000/user/abdulaziz99/arabic_ai_project/processed/parquet"
)

print("=== Processed Data Saved as Parquet in HDFS ===")

arabic_stopwords = ["من", "في", "على", "الى", "إلى", "عن", "أن", "ان", "كان", "كانت", "هذا", "هذه", "هو", "هي", "و", "او", "أو", "مع"]

tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="tokens", pattern="\\s+")
tokenized_df = tokenizer.transform(clean_df)

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=arabic_stopwords)
filtered_df = remover.transform(tokenized_df)

hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="tf_features", numFeatures=4096)
tf_df = hashing_tf.transform(filtered_df)

idf = IDF(inputCol="tf_features", outputCol="tfidf_features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

assembler = VectorAssembler(
    inputCols=["tfidf_features", "word_count", "avg_word_length"],
    outputCol="features"
)

final_df = assembler.transform(tfidf_df).select("features", "label")

print("=== TF-IDF Features Ready ===")
final_df.show(5, truncate=False)

train_data, validation_data, test_data = final_df.randomSplit([0.7, 0.15, 0.15], seed=42)

print("=== Train / Validation / Test Split ===")
print("Train:", train_data.count())
print("Validation:", validation_data.count())
print("Test:", test_data.count())

accuracy_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
roc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

models = {
    "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label", maxIter=20),
    "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=30, seed=42),
    "Linear SVM": LinearSVC(featuresCol="features", labelCol="label", maxIter=20)
}

print("=== Model Training and Evaluation ===")

for name, clf in models.items():
    print("\n---", name, "---")
    model = clf.fit(train_data)
    predictions = model.transform(test_data)

    accuracy = accuracy_eval.evaluate(predictions)
    f1 = f1_eval.evaluate(predictions)
    roc_auc = roc_eval.evaluate(predictions)

    print("Accuracy =", accuracy)
    print("F1 Score =", f1)
    print("ROC-AUC =", roc_auc)

    print("Confusion Matrix:")
    predictions.groupBy("label", "prediction").count().show()

spark.stop()
print("=== Spark Session Stopped ===")
