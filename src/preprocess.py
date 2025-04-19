import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

RAW_DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = "../data/processed_data.csv"

def load_data(path):
    """Load dataset from CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def clean_data(df):
    """Clean and prepare raw data."""
    df = df.copy()

    # Drop customerID - not useful for modeling
    df.drop('customerID', axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df


def encode_features(df):
    """Encode categorical features."""
    df = df.copy()

    # Identify categorical columns
    cat_cols = df.select_dtypes(include='object').columns

    # Label Encoding for binary features, One-Hot for the rest
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2]
    multi_cols = [col for col in cat_cols if col not in binary_cols]

    for col in binary_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def scale_features(df):
    """Scale numeric features."""
    df = df.copy()
    scaler = StandardScaler()

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def preprocess_and_save():
    """Main runner: load, clean, encode, scale, save."""
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"[✔] Preprocessing complete. Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_and_save()
