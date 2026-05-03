"""
Data Preprocessing Utilities
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def clean_telco_data(df):
    """Clean and preprocess the Telco dataset"""
    
    # Make a copy
    df = df.copy()
    
    # Drop customerID (not useful for prediction)
    df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric (handle empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values with median
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Convert SeniorCitizen to string for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    print("✅ Data cleaned!")
    return df

def encode_features(df):
    """Encode categorical features"""
    
    df = df.copy()
    
    # Columns to encode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from encoding list
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    # Label encoding
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print("✅ Features encoded!")
    return df, label_encoders

def split_features_target(df):
    """Split features and target"""
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return X, y

if __name__ == "__main__":
    # Test
    from data_loader import load_telco_data
    df = load_telco_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = clean_telco_data(df)
    df, encoders = encode_features(df)
    print(df.head())