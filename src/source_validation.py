import pandas as pd


def load_data(file_path):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("✅ Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print("❌ Error: File not found.")
        return None