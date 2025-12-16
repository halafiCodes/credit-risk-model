import os
import pandas as pd
import numpy as np
import joblib
import logging
import sys
import warnings
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

    # Save only numeric and aggregated features (no one-hot)
    features_to_save = ['CustomerId', 'Amount', 'Value', 'hour', 'day', 'month', 'year',
                        'total_amount', 'avg_amount', 'transaction_count', 'std_amount',
                        'Amount_log', 'total_amount_log', 'avg_amount_log', 'std_amount_log']
    df_features = df[features_to_save]

    return df_features

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(base_dir, "../data/raw/data.csv")
    processed_dir = os.path.join(base_dir, "../data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    df = pd.read_csv(raw_data_path)
    df_features = preprocess_transactions(df)

    processed_csv_path = os.path.join(processed_dir, "processed_transactions.csv")
    df_features.to_csv(processed_csv_path, index=False)
    print(f"[INFO] Processed features saved at: {processed_csv_path}")
