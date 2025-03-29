import pandas as pd
from src.source_validation import load_data
from src.explore_data import summarize_data
from src.data_preprocessing import data_cleaning
from src.processed_CSV import processed_CSV
from src.train_model import train_credit_card_model
def main():
    file_path = "../Data/CreditCardApproval.csv"  # File path for datasource

    df = load_data(file_path)
    if df is None:
        return
    else:
        summarize_data(df)
        df = data_cleaning(df)
    processed_CSV(df)
    print("Columns in DF:", df.columns.tolist())

    #  Train AutoML model
    train_credit_card_model(df)

# Run the service
if __name__ == "__main__":
    main()
