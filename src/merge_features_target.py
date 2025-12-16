import pandas as pd
import os

# Paths
base_dir = os.path.dirname(__file__)
features_path = os.path.join(base_dir, "../data/processed/processed_transactions.csv")
target_path = os.path.join(base_dir, "../data/processed/target.csv")
merged_path = os.path.join(base_dir, "../data/processed/merged_dataset.csv")

# Load files
df_features = pd.read_csv(features_path)
df_target = pd.read_csv(target_path)

# Merge on CustomerId
df_merged = df_features.merge(df_target[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# Save merged dataset
df_merged.to_csv(merged_path, index=False)
print(f"[INFO] Merged dataset saved at: {merged_path}")
