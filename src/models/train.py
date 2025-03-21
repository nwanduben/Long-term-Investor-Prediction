import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from src.data.preprocess import preprocess_data  # Ensure preprocessing is correctly imported

def train_multiple_models():
    # ✅ Set MLflow Tracking URI (before logging runs)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure MLflow tracking server is running
    mlflow.set_experiment("EliteBank_Model_Tracking")

    # ✅ Load and preprocess data
    df = pd.read_csv('data/raw/Elite_bank.csv')
    df_processed = preprocess_data(df)

    X = df_processed.drop('deposit', axis=1)
    y = df_processed['deposit']

    # ✅ Split data into train & test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # ✅ Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05),
    }

    best_model = None
    best_score = 0
    best_model_name = None  # ✅ Track best model name

    for name, model in models.items():
        with mlflow.start_run(run_name=name):  # ✅ No run_id, starts a new run
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            specificity = confusion_matrix(y_test, y_pred)[0, 0] / sum(confusion_matrix(y_test, y_pred)[0])

            # ✅ Log model parameters & metrics to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_prob))
            mlflow.log_metric("specificity", specificity)

            # ✅ Save model to MLflow
            mlflow.sklearn.log_model(model, "model")

            # ✅ Track best model
            if roc_auc_score(y_test, y_prob) > best_score:
                best_score = roc_auc_score(y_test, y_prob)
                best_model = model
                best_model_name = name  # ✅ Save best model name

    # ✅ Register the best model in MLflow
    model_name = "EliteBank_Model_Tracking"

    if best_model:
        with mlflow.start_run():  # ✅ Starts a new run (no old run_id)
            mlflow.sklearn.log_model(best_model, model_name, registered_model_name=model_name)
        print(f"✅ Model Registered in MLflow: {model_name}")

        # ✅ Save best model locally for deployment
        joblib.dump(best_model, "artifacts/best_model.pkl")

        print(f"🏆 Best Model: {best_model_name} - ROC-AUC: {best_score:.4f}")
    else:
        print("⚠️ No model was selected as the best.")

if __name__ == "__main__":
    train_multiple_models()
