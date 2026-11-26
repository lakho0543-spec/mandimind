"""
pipeline/cluster.py — Phase 3C: Cluster districts by risk profile.

Groups districts using K-Means based on average price and volatility.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def build_district_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates price data by district to build a profile for clustering.
    """
    df = df.copy()
    
    # Group by district and calculate mean price and volatility (std dev)
    profiles = df.groupby('admin2_name').agg(
        avg_price=('price_pkr_per_40kg', 'mean'),
        price_volatility=('price_pkr_per_40kg', 'std'),
        total_records=('price_pkr_per_40kg', 'count')
    ).reset_index()
    
    # Drop districts with too few records (outliers)
    profiles = profiles[profiles['total_records'] > 10].copy()
    
    # Fill any NaNs in volatility with 0
    profiles['price_volatility'] = profiles['price_volatility'].fillna(0)
    
    return profiles

def cluster_districts(profiles: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """
    Runs K-Means on the district profiles.
    """
    df = profiles.copy()
    
    # 1. Select features for clustering
    features = ['avg_price', 'price_volatility']
    X = df[features]
    
    # 2. Scale the features (K-Means is sensitive to scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(X_scaled)
    
    # 4. Create human-readable cluster names based on centroids
    # We'll map cluster IDs to names like "Stable", "Volatile", "High Price"
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Simple rule-based naming
    cluster_names = {}
    for i, center in enumerate(centers):
        avg_p, vol = center
        if vol > profiles['price_volatility'].median():
            cluster_names[i] = "Volatile / High Risk"
        elif avg_p > profiles['avg_price'].median():
            cluster_names[i] = "High Price / Premium"
        else:
            cluster_names[i] = "Stable / Normal"
            
    df['cluster_name'] = df['cluster_id'].map(cluster_names)
    
    return df

if __name__ == "__main__":
    from pipeline.ingest import load_wfp_prices
    from pipeline.clean import clean_prices
    
    print("Loading and cleaning all district prices...")
    raw = load_wfp_prices()
    cleaned = clean_prices(raw)
    
    print("Building district profiles...")
    profiles = build_district_features(cleaned)
    
    print("Clustering districts...")
    clustered = cluster_districts(profiles)
    
    print("\nTop 5 districts in each cluster:")
    print(clustered[['admin2_name', 'avg_price', 'price_volatility', 'cluster_name']].head(10))
    
    out_path = Path("data/processed/district_profiles.parquet")
    clustered.to_parquet(out_path)
    print(f"\nSaved to {out_path}")
