import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Set page title and icon
st.set_page_config(page_title="Jenius Bank Credit Approval", page_icon="🏦")

# Load the trained model
model = load_model('credit_card_model_lightgbm')

# 🏦 Branding
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🏦 Bank Credit Card Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("#### Enter your details below to check your eligibility.")
st.write("---")

# Input form
with st.form("credit_form"):
    income = st.number_input("Annual Income", min_value=10000, max_value=1000000, step=5000)
    education = st.selectbox("Education Level", [
        "Academic_degree", "Higher_education", "Incomplete_higher",
        "Secondary__secondary_special", "Lower_secondary"
    ])
    income_type = st.selectbox("Income Type", [
        "Working", "Commercial_associate", "State_servant", "Pensioner", "Student"
    ])
    housing = st.selectbox("Housing Type", [
        "House__apartment", "Rented_apartment", "Municipal_apartment",
        "With_parents", "Coop_apartment", "Office_apartment"
    ])
    occupation = st.selectbox("Occupation Type", [
        "Laborers", "Sales_staff", "Managers", "Core_staff", "Drivers", "NA"
    ])
    cnt_children = st.number_input("Number of Children", min_value=0, max_value=10, step=1)
    cnt_fam = st.number_input("Total Family Members", min_value=1, max_value=10, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    work_exp = st.number_input("Work Experience (Years)", min_value=0, max_value=80, step=1)
    own_car = st.selectbox("Owns Car?", ["Y", "N"])
    own_realty = st.selectbox("Owns Realty?", ["Y", "N"])

    submit = st.form_submit_button("🔍 Predict Approval Status")

# Prediction
if submit:
    user_input = pd.DataFrame([{
        "AMT_INCOME_TOTAL": income,
        "NAME_EDUCATION_TYPE": education,
        "NAME_INCOME_TYPE": income_type,
        "NAME_HOUSING_TYPE": housing,
        "OCCUPATION_TYPE": occupation,
        "CNT_CHILDREN": cnt_children,
        "CNT_FAM_MEMBERS": cnt_fam,
        "AGE": age,
        "WORK_EXPERIENCE": work_exp,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty
    }])

    # Reindex columns to match model's training schema
    expected_features = model.feature_names_in_
    user_input = user_input.reindex(columns=expected_features, fill_value=0)

    prediction = predict_model(model, data=user_input)

    # ✅ Fix ambiguous Series issue here
    if 'prediction_label' in prediction.columns:
        prediction_label = prediction['prediction_label'].iloc[0]
    elif 'Label' in prediction.columns:
        prediction_label = prediction['Label'].iloc[0]
    else:
        st.error("❌ Prediction column not found.")
        st.stop()

    if prediction_label == 1:
        st.success("🎉 Congrats! Your Application is likely to be Approved.")
    else:
        st.error("❌ Sorry! Your Application is likely to be Rejected.")
