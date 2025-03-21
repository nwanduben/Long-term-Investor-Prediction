import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df, save_path='data/processed/processed_data.csv'):
    # Ensure necessary directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    label_encoders = {}
    df_processed = df.copy()

    # Encode target variable
    df_processed['deposit'] = df_processed['deposit'].map({'yes': 1, 'no': 0})

    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include='object').columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # ✅ Cap `duration` at 300 seconds to limit its influence
    df_processed["duration"] = df_processed["duration"].clip(upper=300)

    # ✅ Apply log transformation to `pdays` to reduce skewness
    df_processed["pdays"] = df_processed["pdays"].apply(lambda x: np.log1p(x) if x > 0 else 0)

    # ✅ Scale numerical variables to normalize feature influence
    scaler = StandardScaler()
    num_cols = ["balance", "duration", "campaign", "pdays", "previous"]
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

    # ✅ Save encoders and scaler for later use in inference
    joblib.dump(label_encoders, 'artifacts/label_encoders.pkl')
    joblib.dump(scaler, 'artifacts/scaler.pkl')

    # ✅ Save processed CSV
    df_processed.to_csv(save_path, index=False)

    return df_processed

# Optional standalone runner
if __name__ == "__main__":
    df = pd.read_csv("data/raw/Elite_bank.csv")
    preprocess_data(df)
    print("✅ Preprocessing complete.")
