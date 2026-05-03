"""
Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda(df, output_dir='../images'):
    """Perform comprehensive EDA"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Basic Info
    print("=" * 50)
    print("📊 DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total Customers: {len(df)}")
    print(f"Total Features: {df.shape[1]}")
    
    # 2. Churn Distribution
    print("\n📉 Churn Distribution:")
    churn_counts = df['Churn'].value_counts()
    for churn, count in churn_counts.items():
        pct = count / len(df) * 100
        print(f"  {churn}: {count} ({pct:.1f}%)")
    
    # 3. Churn by Contract Type
    print("\n📈 Churn Rate by Contract Type:")
    if 'Contract' in df.columns:
        contract_churn = df.groupby('Contract')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        )
        for contract, rate in contract_churn.items():
            print(f"  {contract}: {rate:.1f}%")
    
    # 4. Save visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Churn Distribution', fontsize=14)
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/churn_distribution.png', dpi=150)
    plt.close()
    
    # 5. Churn by tenure
    plt.figure(figsize=(10, 6))
    df[df['Churn'] == 'Yes']['tenure'].hist(bins=30, alpha=0.7, label='Churned', color='red')
    df[df['Churn'] == 'No']['tenure'].hist(bins=30, alpha=0.5, label='Stayed', color='blue')
    plt.title('Churn by Tenure (Months)', fontsize=14)
    plt.xlabel('Tenure (Months)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'{output_dir}/churn_by_tenure.png', dpi=150)
    plt.close()
    
    print(f"\n✅ EDA complete! Saved {len(os.listdir(output_dir))} plots.")
    return {
        'total_customers': len(df),
        'churn_rate': (df['Churn'] == 'Yes').sum() / len(df) * 100
    }

if __name__ == "__main__":
    from data_loader import load_telco_data
    df = load_telco_data('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    stats = perform_eda(df)