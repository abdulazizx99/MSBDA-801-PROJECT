# Arabic AI Text Detection Using Hadoop, Apache Spark, and Machine Learning

## Project Overview
This project implements a scalable Big Data pipeline for detecting AI-generated Arabic text using Hadoop HDFS, Apache Spark, PySpark MLlib, TF-IDF, and machine learning models.

## Repository Structure
- data/raw: raw Arabic dataset
- data/processed: processed and parquet data
- src: source code for preprocessing, modeling, and streaming
- results: final evaluation outputs
- reports/figures: screenshots and figures
- Documentation: final report and presentation
- notebooks: EDA and experimentation notebooks

## Source Code
- arabic_ai_project.py
- streaming_pipeline.py
- create_figures_results.py

## Models
- Logistic Regression
- Random Forest
- Linear SVM

## Results
Linear SVM achieved the best performance:
- Accuracy: 93.67%
- F1-Score: 93.67%
- ROC-AUC: 98.22%

## Execution Order
1. Load dataset into HDFS
2. Run preprocessing and feature extraction
3. Train and evaluate machine learning models
4. Run Spark Structured Streaming simulation
5. Review results and figures

## Technologies
Hadoop HDFS, Apache Spark, PySpark MLlib, TF-IDF, Spark Structured Streaming, Python.
