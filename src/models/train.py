import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from src.data.preprocess import preprocess_data
import os

# ✅ Define the processed file path
PROCESSED_DATA_PATH = "data/processed/bank_marketing_cleaned.csv"

def train_models(max_iter, n_estimators, max_depth, dt_max_depth):
    # ✅ Preprocess data before training
    print("🔄 Running data preprocessing...")
    preprocess_data()  # No arguments needed

    # ✅ Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {PROCESSED_DATA_PATH}. Check preprocessing.")

    # ✅ Load preprocessed dataset
    print(f"📥 Loading preprocessed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # ✅ Define initial feature list (excluding categorical variables that are one-hot encoded)
    features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 
                'default', 'housing', 'loan']  # Binary categorical variables

    # ✅ Automatically add one-hot encoded categorical columns
    for col in ['job', 'marital', 'education', 'contact', 'poutcome']:
        one_hot_cols = [c for c in df.columns if col in c]  # Get all one-hot encoded columns
        features.extend(one_hot_cols)  # ✅ Ensure all one-hot encoded columns are included

    target = 'deposit'
    X = df[features]
    y = df[target]

    # ✅ Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # ✅ Set up MLflow experiment
    mlflow.set_experiment("Investor_Prediction_Experiment")
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("rf_max_depth", max_depth)
        mlflow.log_param("dt_max_depth", dt_max_depth)

        # Prepare input example for MLflow
        input_example = X_train.iloc[:1].to_dict(orient="records")
        
        # 🔹 Logistic Regression
        log_model = LogisticRegression(max_iter=max_iter)
        log_model.fit(X_train, y_train)
        log_val_acc = accuracy_score(y_val, log_model.predict(X_val))
        mlflow.log_metric("logistic_regression_val_accuracy", log_val_acc)
        mlflow.sklearn.log_model(log_model, "logistic_regression", input_example=input_example, signature=infer_signature(X_train, log_model.predict(X_train)))

        # 🔹 Random Forest
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val))
        mlflow.log_metric("random_forest_val_accuracy", rf_val_acc)
        mlflow.sklearn.log_model(rf_model, "random_forest", input_example=input_example, signature=infer_signature(X_train, rf_model.predict(X_train)))

        # 🔹 Decision Tree
        dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_val_acc = accuracy_score(y_val, dt_model.predict(X_val))
        mlflow.log_metric("decision_tree_val_accuracy", dt_val_acc)
        mlflow.sklearn.log_model(dt_model, "decision_tree", input_example=input_example, signature=infer_signature(X_train, dt_model.predict(X_train)))

        # ✅ Print accuracy scores
        print(f"📊 Logistic Regression Validation Accuracy: {log_val_acc:.4f}")
        print(f"🌲 Random Forest Validation Accuracy: {rf_val_acc:.4f}")
        print(f"🧩 Decision Tree Validation Accuracy: {dt_val_acc:.4f}")

        # ✅ Save models locally
        joblib.dump(log_model, "models/logistic_regression.pkl")
        joblib.dump(rf_model, "models/random_forest.pkl")
        joblib.dump(dt_model, "models/decision_tree.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for Investor Prediction")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for Logistic Regression")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in Random Forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth for Random Forest trees")
    parser.add_argument("--dt_max_depth", type=int, default=5, help="Max depth for Decision Tree")
    args = parser.parse_args()
    
    train_models(args.max_iter, args.n_estimators, args.max_depth, args.dt_max_depth)
