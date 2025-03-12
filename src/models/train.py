# src/models/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from src.data.preprocess import preprocess_data

# Load and preprocess data
df = pd.read_csv("data/processed/bank_marketing_data_cleaned.csv")
df = preprocess_data(df)

# Define features and target variable
features = ['balance', 'duration', 'pdays', 'previous']
target = 'deposit'

X = df[features]
y = df[target]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Investor_Prediction_Experiment")

with mlflow.start_run():
    # Train Logistic Regression model
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, y_pred_log)

    # Log model and accuracy
    mlflow.sklearn.log_model(log_model, "logistic_regression")
    mlflow.log_metric("logistic_regression_accuracy", log_acc)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    # Log model and accuracy
    mlflow.sklearn.log_model(rf_model, "random_forest")
    mlflow.log_metric("random_forest_accuracy", rf_acc)

    print(f"Logistic Regression Accuracy: {log_acc:.4f}")
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    # Save models
    joblib.dump(log_model, "models/logistic_regression.pkl")
    joblib.dump(rf_model, "models/random_forest.pkl")
