# Arabic AI Text Detection Project

## Project Overview
This project detects whether Arabic text is AI-generated or human-written using Apache Spark and Machine Learning techniques.

## Technologies Used
- Apache Spark
- PySpark MLlib
- HDFS
- Parquet Storage
- Structured Streaming
- Python

## Machine Learning Models
- Logistic Regression
- Random Forest
- Linear SVM

## Features Used
- TF-IDF text features
- Word Count
- Average Word Length

## Dataset
Arabic balanced dataset containing AI-generated and human-written Arabic text.

## Data Pipeline
1. Load dataset from HDFS
2. Clean and preprocess Arabic text
3. Apply tokenization and stopword removal
4. Extract TF-IDF features
5. Train ML models
6. Evaluate using Accuracy, F1 Score, ROC-AUC
7. Simulate streaming using Spark Structured Streaming

## Results

### Logistic Regression
- Accuracy: 91%
- F1 Score: 91%
- ROC-AUC: 97%

### Random Forest
- Accuracy: 86%
- F1 Score: 86%
- ROC-AUC: 95%

### Linear SVM
- Accuracy: 93%
- F1 Score: 93%
- ROC-AUC: 98%

## Project Structure

final_project/
├── data/
├── src/
├── results/
├── reports/

## Authors
Big Data Project Team
