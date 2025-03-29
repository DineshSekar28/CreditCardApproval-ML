# 💳 Credit Card Approval Prediction

A comprehensive project to build a credit card approval prediction system using machine learning, domain-driven credit scoring, and AutoML. The solution includes complete data preprocessing, synthetic target generation, exploratory data analysis, model selection via PyCaret, and performance evaluation.

---

## 📚 Table of Contents

1. [Project Objective](#project-objective)
2. [Initial Hypotheses](#initial-hypotheses)
3. [Dataset Overview](#dataset-overview)
4. [Data Preprocessing](#data-preprocessing)
5. [Custom Credit Scoring](#custom-credit-scoring)
6. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
7. [Data Visualization (Tableau)](#data-visualization-tableau)
8. [Machine Learning Workflow](#machine-learning-workflow)
9. [Model Performance](#model-performance)
10. [Technologies Used](#technologies-used)

---

## 🎯 Project Objective

Credit institutions need efficient and reliable ways to assess credit card applications. This project builds a predictive system using custom scoring logic, domain features, and AutoML to simulate approval decisions (`TARGET = 1`) or rejections (`TARGET = 0`).

---

## 🧪 Initial Hypotheses

- Higher income and higher education positively affect approval chances.
- Owning property and having fewer dependents signals financial stability.
- Unusual employment patterns or housing types (e.g., office apartments) may indicate higher risk.

---

## 📂 Dataset Overview

The dataset includes demographic, financial, and employment-related attributes of credit card applicants.

| Feature               | Description                                      |
|----------------------|--------------------------------------------------|
| `AMT_INCOME_TOTAL`    | Annual income                                   |
| `NAME_INCOME_TYPE`    | Income type (Working, State servant, etc.)       |
| `NAME_EDUCATION_TYPE` | Level of education                              |
| `NAME_HOUSING_TYPE`   | Housing status                                   |
| `DAYS_BIRTH`          | Days since birth (negative)                     |
| `DAYS_EMPLOYED`       | Days employed (negative; capped at 365243)      |
| `CNT_CHILDREN`        | Number of children                              |
| `CNT_FAM_MEMBERS`     | Family size                                     |
| `FLAG_OWN_CAR`        | Owns a car (Y/N)                                |
| `FLAG_OWN_REALTY`     | Owns real estate (Y/N)                          |
| `OCCUPATION_TYPE`     | Job type                                       |

---

## 🔧 Data Preprocessing

- Converted `DAYS_BIRTH` and `DAYS_EMPLOYED` to `AGE` and `WORK_EXPERIENCE` (years)
- Replaced invalid employment durations (365243) with median of negative values
- Filled missing `OCCUPATION_TYPE` with "NA"
- Cleaned categorical values by removing special characters and normalizing spaces (for encoding compatibility)

---

## 🧠 Custom Credit Scoring

A rule-based scoring engine was developed using domain knowledge.

**Scoring Components:**

- **Income Score**: Buckets of 100K up to 500K (score capped at 5)
- **Education Score**: 0 to 4 scale (Lower secondary to Academic degree)
- **Income Type Score**: Highest for "State servant", lowest for "Student"
- **Housing Type Score**: Based on property ownership or stability
- **Family Size Score**: Score drops if family size > 4
- **Work Experience & Age Bracket Score**: Age and experience grouped into brackets
- **Asset Ownership**: Bonus points for car/realty owners

The final `CREDIT_SCORE` was computed by summing all component scores.

**Target Assignment:**
- Top 60% of scores labeled `TARGET = 1`
- Bottom 40% labeled `TARGET = 0`
- Minor noise introduced to simulate approval anomalies

---

## 📊 Exploratory Data Analysis (EDA)

- Income distribution skewed toward lower buckets (<200K)
- Majority of applicants fall between ages 25 and 60
- Pensioners showed inflated employment days (cleaned)
- Education and income positively correlated with approval
- Office apartment dwellers and large families show lower approval likelihood

---

## 📉 Data Visualization (Tableau)

Interactive visualizations built with Tableau to explore key patterns:

- Approval breakdown by income bucket (side-by-side bar chart)
- Approval vs. Age/Work Experience (scatter & heatmaps)
- Education level and occupation type vs approval status
- Percent approval trends by housing and income type

> ![Income vs Approval Chart](https://github.com/user-attachments/assets/07f1e0b9-6ded-476e-b665-e5c71b201ac0)

> ![Trend by Age and Income](https://github.com/user-attachments/assets/b8f18b2c-369c-42b9-b79f-220ac952dd1f)

---

## 🤖 Machine Learning Workflow

- Setup done via **PyCaret**
- Ignored: `ID`, `CREDIT_SCORE`
- Enabled: Feature selection, multicollinearity removal
- Models trained: LightGBM, CatBoost, XGBoost, Logistic Regression, etc.
- Tuning: Used `tune_model()` to optimize hyperparameters (F1 Score)
- Final model: User-selected model (not necessarily AutoML winner)
- Evaluation: Predictions made on hold-out test set

---

## 🧪 Model Performance

| Model                         | Accuracy | AUC   | Recall | Precision | F1 Score | MCC   | Kappa | Time (s) |
|------------------------------|----------|-------|--------|-----------|----------|-------|--------|----------|
| **CatBoost Classifier**      | **0.8482** | 0.814 | **0.9325** | 0.8483    | **0.8884** | 0.6598 | 0.6527 | 24.01    |
| LightGBM                     | 0.8475   | 0.813 | 0.9323 | 0.8476    | 0.8879   | 0.6582 | 0.6510 | 1.90     |
| XGBoost                      | 0.8473   | 0.814 | 0.9324 | 0.8474    | 0.8878   | 0.6579 | 0.6506 | 2.60     |
| Gradient Boosting            | 0.8360   | 0.813 | 0.9340 | 0.8331    | 0.8807   | 0.6316 | 0.6211 | 13.49    |
| Random Forest                | 0.8101   | 0.803 | 0.8860 | 0.8319    | 0.8581   | 0.5747 | 0.5721 | 12.93    |
| Logistic Regression          | 0.7895   | 0.796 | 0.9209 | 0.7894    | 0.8501   | 0.5210 | 0.5038 | 11.70    |
| Dummy Classifier (baseline)  | 0.6480   | 0.500 | 1.0000 | 0.6480    | 0.7864   | 0.0000 | 0.0000 | 1.17     |

---

## 🛠 Technologies Used

- **Python 3.10**
- **PyCaret 3.x**
- **scikit-learn**
- **pandas / numpy**
- **Tableau Public** (for dashboarding)
- **LightGBM / CatBoost / XGBoost**

---

## 📌 Conclusion

This project simulates a real-world credit approval pipeline using domain logic, clean preprocessing, robust visualization, and AutoML. By building both a **credit scoring engine** and a **predictive model**, the project ensures interpretability and accuracy — both critical in financial services.

**Next Steps:**
- Deploy model as REST API or Streamlit App
- Integrate dashboard for real-time analysis
- Monitor model performance over time using SHAP and drift detection
