import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.data.preprocess import preprocess_data
import os

# ✅ Define processed data path
PROCESSED_DATA_PATH = "data/processed/bank_marketing_cleaned.csv"

def tune_and_train_model():
    # ✅ Ensure data preprocessing runs before training
    print("🔄 Running data preprocessing...")
    preprocess_data()  # Ensure preprocess runs before loading

    # ✅ Check if processed data exists
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"❌ Processed data not found at {PROCESSED_DATA_PATH}. Check preprocessing.")

    # ✅ Load the preprocessed dataset
    print(f"📥 Loading preprocessed data from {PROCESSED_DATA_PATH}...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # ✅ Convert target column if necessary
    if df['deposit'].dtype == 'object':
        df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

    # ✅ Ensure categorical variables are properly encoded
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"⚠️ Categorical columns found: {categorical_columns} → Encoding...")
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # ✅ Check for missing values in deposit
    print(f"🔍 Checking for NaNs in 'deposit' before splitting: {df['deposit'].isnull().sum()} missing values")
    
    if df['deposit'].isnull().sum() > 0:
        df = df.dropna(subset=['deposit'])
    
    # ✅ Define features and target
    features = [col for col in df.columns if col != 'deposit']
    target = 'deposit'
    X = df[features]
    y = df[target]

    # ✅ Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Hyperparameter tuning
    param_grid = {
         'n_estimators': [50, 100, 200],
         'max_depth': [None, 5, 10, 15],
         'min_samples_split': [2, 5, 10]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    print("✅ Best Parameters:", best_params)
    print("✅ Best CV Score:", best_cv_score)

    # ✅ Evaluate the best model
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("✅ Test Accuracy:", test_accuracy)
    print("✅ Classification Report:\n", classification_report(y_test, y_pred))

    # ✅ Log experiment
    mlflow.set_experiment("Investor_Prediction_Experiment_Tuning")
    with mlflow.start_run():
         mlflow.log_params(best_params)
         mlflow.log_metric("best_cv_accuracy", best_cv_score)
         mlflow.log_metric("test_accuracy", test_accuracy)
         signature = infer_signature(X_train, best_model.predict(X_train))
         mlflow.sklearn.log_model(best_model, "best_random_forest_model", input_example=X_train.iloc[:1].to_dict(orient="list"), signature=signature)
    
    # ✅ Save the best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_random_forest_model.pkl")
    print("🚀 Best model saved to models/best_random_forest_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Investor Prediction")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    args = parser.parse_args()
    
    tune_and_train_model()
