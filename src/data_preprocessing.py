import pandas as pd
import numpy as np
from scipy.stats import zscore
from src.custom_scoring import custom_scoring



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

    #if Occupation_Type is empty, Fill NA
    df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("NA")

    # replace invalid positive values for DAYS_EMPLOYED Column with median
    # Identify valid (negative) employment durations and calculate the median
    valid_employment = df["DAYS_EMPLOYED"] < 0
    median_days_employed = df.loc[valid_employment, "DAYS_EMPLOYED"].median()

    # Replace positive values with the median
    df.loc[df["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = median_days_employed

    # Create two new columns in df as Age and work experience.
    # Age calculates DATE_BIRTH value against today's date to give answer in years round off.
    # Work experience is calculated the same way against the field DAYS_EMPLOYED

    age = (-df['DAYS_BIRTH'] / 365).round().astype(int)
    work_exp = (-df['DAYS_EMPLOYED'] / 365).round().astype(int)

    # Insert AGE after DAYS_BIRTH
    birth_index = df.columns.get_loc('DAYS_BIRTH')
    df.insert(birth_index + 1, 'AGE', age)

    # Insert WORK_EXPERIENCE after DAYS_EMPLOYED
    employed_index = df.columns.get_loc('DAYS_EMPLOYED')
    df.insert(employed_index + 1, 'WORK_EXPERIENCE', work_exp)

    df = custom_scoring(df)

    return df
