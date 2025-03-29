# Credit Card Approval Prediction

## Overview

This project aims to predict credit card approval decisions using machine learning techniques. By analyzing various applicant features such as income, education, and family status, the model assists in effective customer evaluation, thereby minimizing the risk of non-repayment.

## Table of Contents

1. [Project Information](#project-information)
2. [Initial Hypotheses](#initial-hypotheses)
3. [Data Analysis Approach](#data-analysis-approach)
4. [Exploratory Data Analysis (EDA) Findings](#exploratory-data-analysis-eda-findings)
5. [Machine Learning Approach](#machine-learning-approach)
6. [Model Comparison](#model-comparison)

## Project Information

### Importance of the Proposal

- **Digital Financial Transactions:** With the rise in digital financial activities, efficient credit assessment has become crucial to handle complex data and ensure secure transactions.&#8203;:contentReference[oaicite:0]{index=0}

- **Risk Mitigation:** :contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}

### Impact on the Banking Sector

- **Customer Acquisition:** :contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}

- **Risk Management:** :contentReference[oaicite:5]{index=5}&#8203;:contentReference[oaicite:6]{index=6}

- **Customer Satisfaction:** :contentReference[oaicite:7]{index=7}&#8203;:contentReference[oaicite:8]{index=8}

## Initial Hypotheses

1. **Income Type Influence:** :contentReference[oaicite:9]{index=9}&#8203;:contentReference[oaicite:10]{index=10}

2. **Education Level Impact:** :contentReference[oaicite:11]{index=11}&#8203;:contentReference[oaicite:12]{index=12}

3. **Family Size Consideration:** :contentReference[oaicite:13]{index=13}&#8203;:contentReference[oaicite:14]{index=14}

4. **Housing Situation:** :contentReference[oaicite:15]{index=15}&#8203;:contentReference[oaicite:16]{index=16}

## Data Analysis Approach

1. **Data Cleaning:** :contentReference[oaicite:17]{index=17}&#8203;:contentReference[oaicite:18]{index=18}

2. **Feature Engineering:** :contentReference[oaicite:19]{index=19}&#8203;:contentReference[oaicite:20]{index=20}

3. **Encoding Categorical Variables:** :contentReference[oaicite:21]{index=21}&#8203;:contentReference[oaicite:22]{index=22}

4. **Credit Scoring:** :contentReference[oaicite:23]{index=23}&#8203;:contentReference[oaicite:24]{index=24}

5. **Target Variable Creation:** :contentReference[oaicite:25]{index=25}&#8203;:contentReference[oaicite:26]{index=26}

## Exploratory Data Analysis (EDA) Findings

- **Income Distribution:** :contentReference[oaicite:27]{index=27}&#8203;:contentReference[oaicite:28]{index=28}

- **Education Levels:** :contentReference[oaicite:29]{index=29}&#8203;:contentReference[oaicite:30]{index=30}

- **Family Size:** :contentReference[oaicite:31]{index=31}&#8203;:contentReference[oaicite:32]{index=32}

- **Housing Type:** :contentReference[oaicite:33]{index=33}&#8203;:contentReference[oaicite:34]{index=34}

## Machine Learning Approach

1. **Data Splitting:** :contentReference[oaicite:35]{index=35}&#8203;:contentReference[oaicite:36]{index=36}

2. **Model Selection:** :contentReference[oaicite:37]{index=37}&#8203;:contentReference[oaicite:38]{index=38}

3. **Evaluation Metrics:** :contentReference[oaicite:39]{index=39}&#8203;:contentReference[oaicite:40]{index=40}

4. **Model Interpretation:** :contentReference[oaicite:41]{index=41}&#8203;:contentReference[oaicite:42]{index=42}

## Model Comparison

:contentReference[oaicite:43]{index=43}&#8203;:contentReference[oaicite:44]{index=44}

| Model                               | Accuracy | AUC   | Recall | Precision | F1 Score | Kappa  | MCC    | Training Time (Sec) |
|-------------------------------------|----------|-------|--------|-----------|----------|--------|--------|---------------------|
| **CatBoost Classifier**             | 0.8482   | 0.8144| 0.9325 | 0.8483    | 0.8884   | 0.6527 | 0.6598 | 24.010              |
| **LightGBM**                        | 0.8475   | 0.8134| 0.9323 | 0.8476    | 0.8879   | 0.6510 | 0.6582 | 1.901               |
| **XGBoost**                         | 0.8473   | 0.8143| 0.9324 | 0.8474    | 0.8878   | 0.6506 | 0.6579 | 2.609               |
| **Gradient Boosting Classifier**    | 0.8360   | 0.8134| 0.9340 | 0.8331    | 0.8807   | 0.6211 | 0.6316 | 13.490              |
| **Random Forest Classifier**        | 0.8101   | 0.8039| 0.8860 | 0.8319    | 0.8581   | 0.5721 | 0.5747 | 12.926              |
| **AdaBoost Classifier**             | 0.8081   | 0.8099| 0.9282 | 0.8054    | 0.8624   | 0.5506 | 0.5660 | 4.443               |
| **Logistic Regression**             | 0.7895   | 0.7967| 0.9209 | 0.7894    | 0.8501   | 0.5038 | 0.5210 | 11.703              |
| **Extra Trees Classifier**          | 0.7862   | 0.7828| 0.8398 | 0.8318    | 0.8358   | 0.5294 | 0.5294 | 13.506              |
| **K Neighbors Classifier**          | 
::contentReference[oaicite:45]{index=45}
 
