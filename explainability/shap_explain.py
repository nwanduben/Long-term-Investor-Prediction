import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ✅ Load trained model
model = joblib.load("artifacts/best_model.pkl")

# ✅ Load dataset (drop the target column)
df = pd.read_csv("data/processed/processed_data.csv")
X = df.drop(columns=["deposit"])  # Ensure feature set matches training

# ✅ Ensure feature names are in correct order
model_features = model.feature_names_in_  # Get feature names from the model
X = X[model_features]  # Reorder columns to match training

# ✅ Sample 100 rows to avoid SHAP errors
X_sample = X.sample(n=100, random_state=42)

# ✅ Initialize SHAP explainer for Gradient Boosting models
explainer = shap.Explainer(model)
shap_values = explainer(X_sample)

# ✅ Handle SHAP value format for Gradient Boosting models
if isinstance(shap_values, list):  
    shap_values = shap_values[1]  # Select class 1 SHAP values if multi-class

# ✅ Generate SHAP summary plot
shap.summary_plot(shap_values, X_sample)
