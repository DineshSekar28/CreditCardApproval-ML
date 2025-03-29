# src/train_model.py

from pycaret.classification import setup, compare_models, finalize_model, save_model
import pandas as pd

def train_credit_card_model(df):
    from sklearn.model_selection import train_test_split

    # Split your full dataset into 80% training and 20% testing
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['TARGET'])

    # Setup the AutoML environment with PyCaret
    setup(
        data=train_df,
        target='TARGET',
        ignore_features=['ID','CREDIT_SCORE'],
        session_id=123,
        verbose=False
    )

    # Train and compare multiple models
    best_model = compare_models()

    # Finalize the best model on full data
    final_model = finalize_model(best_model)

    # Save the trained model
    save_model(final_model, 'credit_card_approval_model')

    print(" Model trained and saved successfully.")
