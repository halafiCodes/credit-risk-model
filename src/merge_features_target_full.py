# src/merge_features_target_full.py

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def extract_time_features(df):
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    df['year'] = df['TransactionStartTime'].dt.year
    return df

def aggregate_customer_features(df):
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('TransactionId', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    agg_df['std_amount'] = agg_df['std_amount'].fillna(0)
    return agg_df

def log_transform_features(df, features):
    for feature in features:
        df[f'{feature}_log'] = np.log1p(df[feature].fillna(0))
    return df

def preprocess_transactions(df):
    df = extract_time_features(df)
    agg_df = aggregate_customer_features(df)
    df = df.merge(agg_df, on='CustomerId', how='left')

    skewed_features = ['Amount', 'total_amount', 'avg_amount', 'std_amount']
    df = log_transform_features(df, skewed_features)

    return df

def merge_features_target(processed_df, target_df):
    # Merge on CustomerId
    merged_df = processed_df.merge(target_df[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')
    return merged_df

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    # Paths
    raw_data_path = os.path.join(base_dir, "../data/raw/data.csv")         # Task 3 raw transactions
    target_path = os.path.join(base_dir, "../data/processed/target.csv")    # Task 4 target
    processed_dir = os.path.join(base_dir, "../data/processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Load raw data and target
    df_raw = pd.read_csv(raw_data_path)
    df_target = pd.read_csv(target_path)
    
    # Preprocess
    df_processed = preprocess_transactions(df_raw)
    
    # Merge with target
    df_merged = merge_features_target(df_processed, df_target)
    
    # Save merged dataset
    merged_csv_path = os.path.join(processed_dir, "merged_dataset.csv")
    df_merged.to_csv(merged_csv_path, index=False)
    
    print(f"[INFO] Merged dataset saved at: {merged_csv_path}")
    print(f"[INFO] Columns in merged dataset: {df_merged.columns.tolist()}")
