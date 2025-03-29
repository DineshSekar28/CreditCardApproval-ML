# 💳 Credit Card Approval Prediction

This project uses supervised machine learning techniques to predict whether a credit card application should be approved based on applicant information such as income, education, employment, family status, and more. The pipeline includes end-to-end preprocessing, custom credit scoring, AutoML modeling with PyCaret, and evaluation.

---

## 📌 Project Objective

Credit card issuers need robust, fast, and explainable systems to assess the creditworthiness of applicants. This project simulates that process using rule-based credit scoring and machine learning to predict approval (`TARGET = 1`) or rejection (`TARGET = 0`).

---

## 🧪 Initial Hypotheses

- Applicants with **higher income**, **stable job types**, and **higher education levels** are more likely to be approved.
- Larger families (more dependents) may indicate higher risk.
- Housing type and property ownership may influence approval odds.

---

## 🗂 Dataset Features

| Feature               | Description                                 |
|----------------------|---------------------------------------------|
| `AMT_INCOME_TOTAL`    | Applicant's annual income                   |
| `NAME_INCOME_TYPE`    | Type of income (e.g., Working, Pensioner)   |
| `NAME_EDUCATION_TYPE` | Education level                             |
| `NAME_HOUSING_TYPE`   | Housing status                              |
| `DAYS_BIRTH`          | Negative days since birth (used to calculate age) |
| `DAYS_EMPLOYED`       | Negative days employed (used to calculate work experience) |
| `FLAG_OWN_CAR`        | Owns a car (Y/N)                            |
| `FLAG_OWN_REALTY`     | Owns real estate (Y/N)                      |
| `CNT_CHILDREN`        | Number of children                          |
| `CNT_FAM_MEMBERS`     | Total family size                           |
| `OCCUPATION_TYPE`     | Applicant's profession                      |

---

## 🧠 Data Processing Pipeline

### ✅ Preprocessing
- Converted birth and employment days into `AGE` and `WORK_EXPERIENCE` in years
- Replaced placeholder employment values (e.g., `365243`) with the median of negative values
- Filled missing `OCCUPATION_TYPE` with `'NA'`

### ✅ Custom Credit Scoring
A scoring system was created based on:

- **Income** (bucketed every 100k with max score = 5)
- **Education** (Academic degree = 4 → Lower secondary = 0)
- **Income Type**
- **Housing Type**
- **Family Size** (Penalty beyond 4 members)
- **Asset Ownership (Car/Realty)**
- **Work Experience and Age Brackets**

> The final score was stored in the `CREDIT_SCORE` column.  
> Top 60% were labeled `TARGET = 1` (approved); rest as `TARGET = 0`.  
> Random label noise was introduced to simulate real-world inconsistencies.

---

## 📊 Exploratory Data Analysis (EDA)

- **Income** and **education level** show strong positive correlation with approval
- Most applicants are between **25–60 years old**
- **Pensioners** tend to have unrealistic employment durations (cleaned)
- Applicants living in **rented** or **office apartments** tend to have lower approval scores

---

## 📊 Data Visualizations

- **Income** and **Age** correlation with approval of credit card visualized.
- The Bar-graph along with trendline displays the correlation.

![Screenshot 2025-03-29 141112](https://github.com/user-attachments/assets/07f1e0b9-6ded-476e-b665-e5c71b201ac0)

![Screenshot 2025-03-29 163459](https://github.com/user-attachments/assets/b8f18b2c-369c-42b9-b79f-220ac952dd1f)


---

## 🤖 Machine Learning Approach

### ✔ Tools Used
- **PyCaret** (for automated model training and comparison)
- **scikit-learn** (for metrics and testing)
- **SHAP** (for model interpretability)

### ✔ AutoML Setup
- Target: `TARGET`
- Ignored columns: `ID`, `CREDIT_SCORE`, `RISK_LEVEL`
- 3-fold cross-validation
- Feature selection and multicollinearity removal enabled
- Top models tuned using `tune_model()`

---

## 📈 Model Comparison

| Model                         | Accuracy | AUC   | Recall | Precision | F1 Score | MCC   | Kappa | Time (s) |
|------------------------------|----------|-------|--------|-----------|----------|-------|--------|----------|
| CatBoost Classifier          | **0.8482** | 0.814 | **0.9325** | 0.8483    | **0.8884** | 0.6598 | 0.6527 | 24.01    |
| LightGBM                     | 0.8475   | 0.813 | 0.9323 | 0.8476    | 0.8879   | 0.6582 | 0.6510 | 1.90     |
| XGBoost                      | 0.8473   | 0.814 | 0.9324 | 0.8474    | 0.8878   | 0.6579 | 0.6506 | 2.60     |
| Gradient Boosting            | 0.8360   | 0.813 | 0.9340 | 0.8331    | 0.8807   | 0.6316 | 0.6211 | 13.49    |
| Random Forest                | 0.8101   | 0.803 | 0.8860 | 0.8319    | 0.8581   | 0.5747 | 0.5721 | 12.93    |
| Logistic Regression          | 0.7895   | 0.796 | 0.9209 | 0.7894    | 0.8501   | 0.5210 | 0.5038 | 11.70    |
| Dummy Classifier (baseline)  | 0.6480   | 0.500 | 1.0000 | 0.6480    | 0.7864   | 0.0000 | 0.0000 | 1.17     |

✅ **CatBoost** performed best overall.  
✅ **LightGBM** was nearly as good and faster.  
✅ Dummy classifier confirms class imbalance (always predicts majority class).

---
