import pandas as pd
import numpy as np
from scipy.stats import zscore


# Load dataset

def summarize_data(df):
    print("Dataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary statistics:")
    summary_stats = df.describe()
    summary_stats.loc["median"] = df.median(numeric_only=True)
    print("\nMissing values:")
    print(df.isnull().sum())

    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    print("\nrepetitive values of columns")
    for column in df.columns:
        if df[column].dtype == "object":
            most_common = df[column].mode()[0]
            count = df[column].value_counts().max()
            print(f"Column '{column}': Most repeated value = '{most_common}' (appears {count} times)")
        else:
            most_common = df[column].mode()[0]
            count = df[column].value_counts().max()
            print(f"Column '{column}': Most repeated value = '{most_common}' (appears {count} times)")



    # Select only numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).copy()

    # Compute standard deviation to filter low-variance columns
    std_devs = numerical_columns.std()
    variance_threshold = 1e-8  # Avoid division by near-zero variance
    valid_columns = std_devs[std_devs > variance_threshold].index
    filtered_numerical_columns = numerical_columns[valid_columns]

    print("\n📌 Outlier count per column (Z-Score > 3):")

    for column in filtered_numerical_columns.columns:
        z_scores = np.abs(zscore(filtered_numerical_columns[column], nan_policy='omit'))
        outlier_count = (z_scores > 3).sum()
        outlier_values = filtered_numerical_columns[column][z_scores > 3]
        top_outliers = outlier_values.iloc[np.argsort(-z_scores[outlier_values.index])].head(10)
        print(f"🔹 {column}: {outlier_count} outliers")
        if outlier_count > 0:
            print(f"   ⚠️ Sample Outliers: {top_outliers.tolist()}\n")