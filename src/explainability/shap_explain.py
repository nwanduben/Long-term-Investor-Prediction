import shap
import pandas as pd
import joblib
from src.data.preprocess import preprocess_data

# Load the processed data for explanation
df = pd.read_csv("data/processed/bank_marketing_data_cleaned.csv")
df = preprocess_data(df)

# Define features
features = ['balance', 'duration', 'pdays', 'previous']
X = df[features]

# Load one of your models, e.g., the random forest model
model = joblib.load("models/random_forest.pkl")

# Create a SHAP explainer (for tree-based models, TreeExplainer is appropriate)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Generate a summary plot
shap.summary_plot(shap_values, X)
