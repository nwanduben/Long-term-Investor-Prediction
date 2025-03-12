import streamlit as st
import joblib
import pandas as pd

st.title("📈 Long-Term Investor Prediction")

# Load trained model
try:
    model = joblib.load("models/random_forest.pkl")
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Test user input form
balance = st.number_input("Balance", min_value=-5000, max_value=100000, value=1000, step=100)
duration = st.number_input("Duration", min_value=0, max_value=5000, value=300, step=10)
pdays = st.number_input("Days Passed", min_value=-1, max_value=1000, value=10, step=1)
previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=2, step=1)

if st.button("Predict"):
    try:
        user_input = [[balance, duration, pdays, previous]]
        prediction = model.predict(user_input)
        result = "✅ Likely to Invest" if prediction[0] == 1 else "❌ Not Likely to Invest"
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
