import pandas as pd
import numpy as np
from scipy.stats import zscore

def data_cleaning(df):
    # Find and clean duplicates from unique ID column


    # Handling duplicate IDs
    duplicate_ids = df.duplicated(subset="ID").sum()
    if duplicate_ids > 0:
        df = df.drop_duplicates(subset="ID", keep="first")
        print(f"Removed {duplicate_ids} duplicate IDs.")

    # Ensure a copy is made to avoid SettingWithCopyWarning
    df = df.copy()

    # Handling missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    print("Missing values replaced with median.")



    return df
