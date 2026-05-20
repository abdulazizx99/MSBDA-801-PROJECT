# Arabic AI Text Detection Using Hadoop, Apache Spark, and Machine Learning

## Project Overview
This project builds a scalable Arabic AI-generated text detection system using Hadoop HDFS, Apache Spark, PySpark MLlib, and machine learning models.

## Dataset
The dataset contains Arabic text samples labeled as human-written or AI-generated. The data was stored in HDFS and processed using Apache Spark.

## Technologies Used
- Hadoop HDFS
- Apache Spark
- PySpark
- Spark MLlib
- TF-IDF
- Logistic Regression
- Random Forest
- Linear SVM
- Spark Structured Streaming

## Pipeline
1. Store dataset in HDFS
2. Load data using Spark
3. Clean Arabic text
4. Extract TF-IDF features
5. Train machine learning models
6. Evaluate models using Accuracy, F1-Score, and ROC-AUC
7. Simulate real-time processing using Spark Structured Streaming

## Results
Linear SVM achieved the best performance:

- Accuracy: 93.67%
- F1-Score: 93.67%
- ROC-AUC: 98.22%

## Screenshots
Project screenshots are available in:

`reports/figures/`

They include:
- HDFS storage verification
- Spark preprocessing output
- Model evaluation results
- Confusion matrix results
- Streaming output
- Project directory structure

## How to Run
```bash
spark-submit src/arabic_ai_project.py
