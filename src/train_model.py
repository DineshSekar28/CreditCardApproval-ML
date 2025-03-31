from pycaret.classification import setup, compare_models, finalize_model, save_model, create_model, tune_model, plot_model, predict_model
import pandas as pd

def train_credit_card_model(df):
    from sklearn.model_selection import train_test_split

    # Clean categorical text to avoid encoding mismatches
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['TARGET'])

    print("Model trained with columns:", train_df.columns.tolist())

    # Setup PyCaret
    setup(
        data=train_df,
        target='TARGET',
        ignore_features=['ID', 'CREDIT_SCORE'],
        features=[
        'AGE', 'WORK_EXPERIENCE', 'AMT_INCOME_TOTAL', 'CNT_CHILDREN',
        'CNT_FAM_MEMBERS', 'NAME_EDUCATION_TYPE',
        'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE',
        'OCCUPATION_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY'],
        session_id=123,
        fold=3,
        feature_selection=True,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.95,
        html=False,
        verbose=False
    )

    # AutoML comparison
    print("🔍 Comparing models...")
    leaderboard = compare_models(n_select=5, sort='F1', turbo=True, verbose=False)
    print("\n✅ Top models displayed above.\n")

    # Manual model selection
    print("💬 Choose a model to use:")
    print("  lightgbm | catboost | xgboost | rf | lr | dt | ada | nb | knn | svm | et")
    chosen_model = input("👉 Enter model name: ").strip().lower()

    # Create and tune
    model = create_model(chosen_model)
    tuned_model = tune_model(model, optimize='F1')

    # Finalize and save the tuned model
    final_model = finalize_model(tuned_model)
    save_model(final_model, f'credit_card_model_{chosen_model}')
    print(f"✅ Model '{chosen_model}' trained and saved successfully.")

    # Predictions
    print("📈 Predicting on test data...")
    predictions = predict_model(final_model, data=test_df)
    print("📦 Available columns in predictions:", predictions.columns.tolist())

    # Save relevant columns
    columns_to_export = [col for col in ['TARGET', 'Label', 'prediction_label'] if col in predictions.columns]
    predictions[columns_to_export].to_csv("test_predictions_lgbm.csv", index=False)
    print("📝 Predictions saved to test_predictions_lgbm.csv")

    # Plot important visuals
    plot_model(final_model, plot='feature')
    plot_model(final_model, plot='confusion_matrix')
