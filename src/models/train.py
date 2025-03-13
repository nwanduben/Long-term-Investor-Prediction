# src/models/train.py
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

def train_models(max_iter, n_estimators, max_depth, dt_max_depth):
    # Load and preprocess data
    df = pd.read_csv("data/processed/bank_marketing_data_cleaned.csv")
    df = preprocess_data(df)

    # Define features and target variable
    features = ['balance', 'duration', 'pdays', 'previous']
    target = 'deposit'
    X = df[features]
    y = df[target]

    # Split data: 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25*0.8 = 0.2

    # Set up MLflow experiment
    mlflow.set_experiment("Investor_Prediction_Experiment")
    with mlflow.start_run():
        # Log hyperparameters common to models
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("rf_max_depth", max_depth)
        mlflow.log_param("dt_max_depth", dt_max_depth)

        # Prepare input example and signatures for reproducibility
        input_example = X_train.iloc[:1].to_dict(orient="records")
        
        # ----- Logistic Regression -----
        log_model = LogisticRegression(max_iter=max_iter)
        log_model.fit(X_train, y_train)
        y_pred_log_val = log_model.predict(X_val)
        log_val_acc = accuracy_score(y_val, y_pred_log_val)
        mlflow.log_metric("logistic_regression_val_accuracy", log_val_acc)
        signature_log = infer_signature(X_train, log_model.predict(X_train))
        mlflow.sklearn.log_model(
            log_model, "logistic_regression",
            input_example=input_example, signature=signature_log
        )
        
        # ----- Random Forest -----
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf_val = rf_model.predict(X_val)
        rf_val_acc = accuracy_score(y_val, y_pred_rf_val)
        mlflow.log_metric("random_forest_val_accuracy", rf_val_acc)
        signature_rf = infer_signature(X_train, rf_model.predict(X_train))
        mlflow.sklearn.log_model(
            rf_model, "random_forest",
            input_example=input_example, signature=signature_rf
        )
        
        # ----- Decision Tree -----
        dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred_dt_val = dt_model.predict(X_val)
        dt_val_acc = accuracy_score(y_val, y_pred_dt_val)
        mlflow.log_metric("decision_tree_val_accuracy", dt_val_acc)
        signature_dt = infer_signature(X_train, dt_model.predict(X_train))
        mlflow.sklearn.log_model(
            dt_model, "decision_tree",
            input_example=input_example, signature=signature_dt
        )

        print(f"Logistic Regression Validation Accuracy: {log_val_acc:.4f}")
        print(f"Random Forest Validation Accuracy: {rf_val_acc:.4f}")
        print(f"Decision Tree Validation Accuracy: {dt_val_acc:.4f}")

        # Optionally, evaluate on test set and log metrics
        y_pred_log_test = log_model.predict(X_test)
        log_test_acc = accuracy_score(y_test, y_pred_log_test)
        mlflow.log_metric("logistic_regression_test_accuracy", log_test_acc)
        
        y_pred_rf_test = rf_model.predict(X_test)
        rf_test_acc = accuracy_score(y_test, y_pred_rf_test)
        mlflow.log_metric("random_forest_test_accuracy", rf_test_acc)
        
        y_pred_dt_test = dt_model.predict(X_test)
        dt_test_acc = accuracy_score(y_test, y_pred_dt_test)
        mlflow.log_metric("decision_tree_test_accuracy", dt_test_acc)
        
        print(f"Logistic Regression Test Accuracy: {log_test_acc:.4f}")
        print(f"Random Forest Test Accuracy: {rf_test_acc:.4f}")
        print(f"Decision Tree Test Accuracy: {dt_test_acc:.4f}")

        # Save models locally
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
