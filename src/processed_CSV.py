import pandas as pd

def processed_CSV(df):
    output_path = "../Data/processed_credit_card_approval.csv"
    df.to_csv(output_path, index=False)
    print("✅ File saved as 'processed_credit_card_approval.csv'")