"""
Data Loading Utility
"""
import pandas as pd

def load_telco_data(filepath):
    """Load the Telco Customer Churn dataset"""
    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    # Test loading
    df = load_telco_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(df.head())