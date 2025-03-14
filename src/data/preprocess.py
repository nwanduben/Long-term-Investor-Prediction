import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ✅ Define file paths
RAW_DATA_PATH = "data/raw/bank_marketing_data.csv"
PROCESSED_DATA_PATH = "data/processed/bank_marketing_cleaned.csv"

def preprocess_data():
    """Loads raw data, preprocesses it, and saves the cleaned version."""
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"❌ Error: {RAW_DATA_PATH} not found!")

    print(f"📥 Loading raw data from: {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # ✅ Drop 'log_balance' column if present
    df = df.drop(columns=['log_balance'], errors='ignore')

    # ✅ Handle missing values
    df.fillna(df.mode().iloc[0], inplace=True)

    # ✅ Convert binary categorical columns to 1s and 0s
    binary_columns = ['default', 'housing', 'loan', 'deposit']
    df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

    # ✅ One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['job', 'marital', 'education'], drop_first=True)

    # ✅ Normalize numerical columns
    numeric_cols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ✅ Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"✅ Preprocessing complete. Data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    preprocess_data()
