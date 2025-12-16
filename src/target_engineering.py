import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_high_risk_target(df):
    # Convert TransactionStartTime to datetime and remove timezone
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime']).dt.tz_localize(None)

    # Compute RFM metrics per customer
    snapshot_date = pd.to_datetime("2019-01-01")  # tz-naive
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'Monetary'
    }).reset_index()

    # Scale RFM
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Identify high-risk cluster (lowest engagement)
    cluster_means = rfm.groupby('cluster')[['Recency','Frequency','Monetary']].mean()
    high_risk_cluster = cluster_means.sort_values(['Frequency','Monetary'], ascending=[True, True]).index[0]
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    return rfm[['CustomerId', 'is_high_risk']]

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, "../data/processed")
    raw_data_path = os.path.join(base_dir, "../data/raw/data.csv")

    df = pd.read_csv(raw_data_path)
    df_target = create_high_risk_target(df)

    target_csv_path = os.path.join(processed_dir, "target.csv")
    df_target.to_csv(target_csv_path, index=False)
    print(f"[INFO] Target variable saved at: {target_csv_path}")
