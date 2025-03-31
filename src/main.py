import pandas as pd
import os
from pathlib import Path
from src.source_validation import load_data
from src.explore_data import summarize_data
from src.data_preprocessing import data_cleaning
from src.processed_CSV import processed_CSV
from src.train_model import train_credit_card_model

def main():
    print("\n🔹 Choose an option:")
    print("1️⃣  Run AutoML training")
    print("2️⃣  Launch Application")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        file_path = "../Data/CreditCardApproval.csv"  # Update if needed
        df = load_data(file_path)

        if df is None:
            print(" Failed to load data.")
            return

        summarize_data(df)
        df = data_cleaning(df)
        processed_CSV(df)
        print(" Columns in DF:", df.columns.tolist())

        # Train AutoML model
        train_credit_card_model(df)

    elif choice == "2":
        print(" Launching GUI...")
        gui_path = Path("interface.py").resolve()
        os.system(f'streamlit run "{gui_path}"')

    else:
        print(" Invalid choice. Please enter 1 or 2.")

# Run the service
if __name__ == "__main__":
    main()
