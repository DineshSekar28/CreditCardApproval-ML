import pandas as pd
import numpy as np
# Load dataset
file_path = "../Data/CreditCardApproval.csv"  # Update the filename if necessary
df = pd.read_csv(file_path)

# Display basic information
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
    if df[column].dtype =="object":
        most_common = df[column].mode()[0]
        count = df[column].value_counts().max()
        print(f"Column '{column}': Most repeated value = '{most_common}' (appears {count} times)")
    else:
        most_common = df[column].mode()[0]
        count = df[column].value_counts().max()
        print(f"Column '{column}': Most repeated value = '{most_common}' (appears {count} times)")
