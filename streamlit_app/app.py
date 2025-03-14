import streamlit as st
import pandas as pd
import joblib
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import mlflow.sklearn

# Load trained models
log_model = joblib.load("models/logistic_regression.pkl")
rf_model = joblib.load("models/random_forest.pkl")
dt_model = joblib.load("models/decision_tree.pkl")

# Define input fields for user
st.title("💰 Loan Approval Prediction App")
st.markdown("This app predicts **loan approvals** based on user-provided financial details.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Bank Balance", min_value=-5000, max_value=100000, value=2000)
duration = st.number_input("Last Contact Duration (seconds)", min_value=1, max_value=5000, value=300)
campaign = st.number_input("Number of Contacts", min_value=1, max_value=50, value=2)
pdays = st.number_input("Days Since Last Contact (-1 if never contacted)", min_value=-1, max_value=500, value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0, max_value=50, value=1)

# Categorical Inputs
default = st.selectbox("Has Credit in Default?", ["no", "yes"])
housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
loan = st.selectbox("Has Personal Loan?", ["no", "yes"])

# Convert categorical inputs to 0/1
default = 1 if default == "yes" else 0
housing = 1 if housing == "yes" else 0
loan = 1 if loan == "yes" else 0

# Create a DataFrame from inputs
user_input = pd.DataFrame([[age, balance, duration, campaign, pdays, previous, default, housing, loan]],
                          columns=['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'default', 'housing', 'loan'])

# Model Selection
model_choice = st.selectbox("Choose Model for Prediction", ["Logistic Regression", "Random Forest", "Decision Tree"])

# Load Selected Model
if model_choice == "Logistic Regression":
    model = log_model
elif model_choice == "Random Forest":
    model = rf_model
else:
    model = dt_model

# Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(user_input)[0]
    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.success("✅ Approved for Loan!")
    else:
        st.error("❌ Loan Not Approved.")

    # Explainability using SHAP
    st.subheader("📊 SHAP (Feature Importance)")
    explainer = shap.Explainer(model, user_input)
    shap_values = explainer(user_input)

    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0], max_display=5, show=False)
    st.pyplot(fig)

    # Explainability using LIME
    st.subheader("🔍 LIME Explanation")
    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(user_input),
                                                       feature_names=user_input.columns,
                                                       class_names=["Not Approved", "Approved"],
                                                       mode="classification")
    exp = explainer.explain_instance(np.array(user_input.iloc[0]), model.predict_proba)
    st.pyplot(exp.as_pyplot_figure())

# MLflow Experiment Tracking
st.sidebar.header("📊 MLflow Experiment Tracking")
with st.sidebar:
    st.write("Tracking MLflow experiments:")
    st.write(f"Model: **{model_choice}**")
    st.write("Experiment: **Loan Prediction**")
