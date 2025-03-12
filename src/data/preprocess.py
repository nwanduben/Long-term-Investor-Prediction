# src/data/preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    """
    Preprocesses the given DataFrame:
    - Handles missing values.
    - Encodes categorical variables.
    - Standardizes numerical features.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Processed dataset.
    """


    # Convert categorical variables to numerical
    label_encoders = {}
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'deposit']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Standardize numerical columns
    numerical_cols = ['balance', 'duration', 'pdays', 'previous']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

# Test the function
if __name__ == "__main__":
    df = pd.read_csv("data/raw/bank_marketing_data.csv")
    processed_df = preprocess_data(df)
    processed_df.to_csv("data/processed/bank_marketing_data_cleaned.csv", index=False)
    print("Preprocessing complete. Cleaned data saved!")
