import pandas as pd
import mlflow.pyfunc
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/EliteBank_Model_Tracking/1")  

# ✅ Load encoders & scaler
label_encoders = joblib.load("artifacts/label_encoders.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

# ✅ Define numerical columns
num_cols = ["balance", "duration", "campaign", "pdays", "previous"]

# ✅ Load feature order from processed dataset
processed_df = pd.read_csv("data/processed/processed_data.csv")
feature_order = processed_df.drop(columns=["deposit"]).columns.tolist()

# 🔷 Long-Term Investor Test Case
long_term_input = {
    "age": 32,
    "job": label_encoders["job"].transform(["admin."])[0],
    "marital": label_encoders["marital"].transform(["married"])[0],
    "education": label_encoders["education"].transform(["tertiary"])[0],
    "default": label_encoders["default"].transform(["no"])[0],
    "balance": 4500,
    "housing": label_encoders["housing"].transform(["yes"])[0],
    "loan": label_encoders["loan"].transform(["no"])[0],
    "contact": label_encoders["contact"].transform(["cellular"])[0],
    "day": 15,
    "month": label_encoders["month"].transform(["may"])[0],
    "duration": 800,
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": label_encoders["poutcome"].transform(["unknown"])[0],
}

# 🔶 Short-Term Investor Test Case (More Extreme)
short_term_input = {
    "age": 65,
    "job": label_encoders["job"].transform(["blue-collar"])[0],
    "marital": label_encoders["marital"].transform(["single"])[0],
    "education": label_encoders["education"].transform(["primary"])[0],
    "default": label_encoders["default"].transform(["yes"])[0],
    "balance": -7000,
    "housing": label_encoders["housing"].transform(["no"])[0],
    "loan": label_encoders["loan"].transform(["yes"])[0],
    "contact": label_encoders["contact"].transform(["unknown"])[0],
    "day": 1,
    "month": label_encoders["month"].transform(["jan"])[0],
    "duration": 2,
    "campaign": 40,
    "pdays": 1300,
    "previous": 25,
    "poutcome": label_encoders["poutcome"].transform(["failure"])[0],
}

# ✅ Convert dictionaries to DataFrames
df_long_raw = pd.DataFrame([long_term_input])
df_short_raw = pd.DataFrame([short_term_input])

# ✅ Apply scaling to numerical features
df_long_raw[num_cols] = scaler.transform(df_long_raw[num_cols])
df_short_raw[num_cols] = scaler.transform(df_short_raw[num_cols])

# ✅ Reorder columns to match training order
df_long = df_long_raw[feature_order]
df_short = df_short_raw[feature_order]

# ✅ Debugging: Print the scaled test inputs
print("\n🔹 Scaled Long-Term Input:\n", df_long)
print("\n🔹 Scaled Short-Term Input:\n", df_short)

# ✅ Set Decision Threshold
threshold = 0.15

# ✅ Make Predictions and print confidence scores
for label, df in zip(["Long-Term", "Short-Term"], [df_long, df_short]):
    prob_long = model.predict_proba(df)[0][1]  # Probability for long-term investor
    if prob_long > threshold:
        result = "✅ Long-Term Investor"
        confidence = prob_long
    else:
        result = "⚠️ Short-Term Investor"
        confidence = 1 - prob_long

    print(f"\n🔍 {label} Prospect:")
    print(df)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
