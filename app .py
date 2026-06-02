import streamlit as st
import pandas as pd
import numpy as np
import pickle

# CRITICAL FIX: Must be the absolute first Streamlit command!
st.set_page_config(page_title="Churn Predictor", layout="centered")

# 1. Load resources safely
@st.cache_resource
def load_resources():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_resources()
except FileNotFoundError:
    st.error("Error: 'model.pkl' or 'scaler.pkl' not found in workspace.")
    st.stop()

# 2. Set up the UI
st.title("📊 Customer Churn Prediction Dashboard")
st.write("Fill out these 7 key metrics to calculate the customer's churn risk instantly.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Account & Billing Details")
    tenure = st.slider("Tenure Months", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.subheader("🌐 Services Utilized")
    internet = st.selectbox("Internet Service Type", ["Fiber optic", "DSL", "No"])
    security = st.selectbox("Online Security Feature", ["No", "Yes"])
    tech_support = st.selectbox("Technical Support Feature", ["No", "Yes"])

st.markdown("---")

# 3. Prediction Pipeline
if st.button("🔮 Calculate Churn Probability", type="primary"):
    total_charges_estimated = tenure * monthly_charges
    
    input_data = {
        'Tenure Months': tenure,
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges_estimated,
        'Online Security': 1 if security == "Yes" else 0,
        'Internet Service_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'Internet Service_No': 1 if internet == "No" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'Payment Method_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'Payment Method_Electronic check': 1 if payment == "Electronic check" else 0,
        'Payment Method_Mailed check': 1 if payment == "Mailed check" else 0,
        'Tech Support_Yes': 1 if tech_support == "Yes" else 0,

        # Hidden constants to satisfy structural 27-feature requirements
        'CLTV': 4000, 
        'Senior Citizen': 0,
        'Dependents': 0,
        'Gender': 0,  
        'Partner': 0,  
        'Phone Service': 1,  
        'Multiple Lines': 0,  
        'Streaming TV': 0,
        'Streaming Movies': 0,
        'Paperless Billing': 1,
        'Online Backup_No internet service': 1 if internet == "No" else 0,
        'Online Backup_Yes': 0,
        'Device Protection_No internet service': 1 if internet == "No" else 0,
        'Device Protection_Yes': 0,
        'Tech Support_No internet service': 1 if internet == "No" else 0
    }

    expected_order = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
        'Phone Service', 'Multiple Lines', 'Online Security', 'Streaming TV',
        'Streaming Movies', 'Paperless Billing', 'Monthly Charges', 'Total Charges',
        'CLTV', 'Internet Service_Fiber optic', 'Internet Service_No',
        'Online Backup_No internet service', 'Online Backup_Yes',
        'Device Protection_No internet service', 'Device Protection_Yes',
        'Tech Support_No internet service', 'Tech Support_Yes', 'Contract_One year',
        'Contract_Two year', 'Payment Method_Credit card (automatic)',
        'Payment Method_Electronic check', 'Payment Method_Mailed check'
    ]
    
    input_df = pd.DataFrame([input_data])[expected_order]
    input_scaled = scaler.transform(input_df)
    churn_proba = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Result Evaluation:")
    if churn_proba > 0.5:
        st.error(f"🚨 **High Risk Profile!** Churn probability is **{churn_proba * 100:.1f}%**")
    else:
        st.success(f"✅ **Stable Profile.** Churn probability is **{churn_proba * 100:.1f}%**")
    st.progress(float(churn_proba))
