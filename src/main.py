import pandas as pd
from src.source_validation import load_data
from src.explore_data import summarize_data
from src.data_preprocessing import data_cleaning
def main():
    file_path = "../Data/CreditCardApproval.csv"  # Update this if needed

    df = load_data(file_path)
    if df is None:
        return
    else:
        summarize_data(df)
        df = data_cleaning(df)


# Run the service
if __name__ == "__main__":
    main()
