"""
pipeline/anomaly.py — Phase 3B: Detect price shocks.

Uses Isolation Forest to flag abnormal price spikes/drops.
Generates a plain-language reason for each anomaly.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pathlib import Path

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits an Isolation Forest to find anomalous rows.
    """
    df = df.copy()
    df = df.sort_values(by=['commodity_name', 'date']).reset_index(drop=True)

    # 1. Select features for the model to look at
    # We only give it price and weather. We don't give it dates, or it will flag every Ramadan.
    features = ['price_pkr_per_40kg', 'precipitation_sum']
    
    # Fill NaNs just in case, then select
    model_data = df[features].fillna(0)

    # 2. Train Isolation Forest
    # contamination=0.05 means we expect about 5% of our data to be anomalies
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_score'] = iso.fit_predict(model_data)
    
    # Convert -1 (anomaly) to boolean True, 1 (normal) to False
    df['is_anomaly'] = df['anomaly_score'] == -1

    # 3. Generate plain-language reasons
    # We need the percentage change from the previous row for each commodity
    df['price_change_pct'] = df.groupby('commodity_name')['price_pkr_per_40kg'].pct_change() * 100

    reasons = []
    for idx, row in df.iterrows():
        if not row['is_anomaly']:
            reasons.append("")
            continue
        
        # Build the reason string
        reason_parts = []
        
        # 1. Price change
        pct = row['price_change_pct']
        if pd.notna(pct):
            direction = "spike" if pct > 0 else "drop"
            reason_parts.append(f"Price {direction} ({pct:+.1f}%)")
        else:
            reason_parts.append("Price anomaly")
            
        # 2. Weather context
        rain = row['precipitation_sum']
        if pd.notna(rain) and rain > 10: # > 10mm weekly is heavy
            reason_parts.append(f"Heavy rain ({rain:.0f}mm)")
            
        reasons.append(". ".join(reason_parts))
        
    df['anomaly_reason'] = reasons
    return df

def save_anomalies(df: pd.DataFrame, out_path: str | Path = "data/processed/anomalies.parquet"):
    """Save only the anomalous rows for the app to use."""
    df_anomalies = df[df['is_anomaly']].copy()
    df_anomalies.to_parquet(out_path)
    print(f"Saved {len(df_anomalies)} anomalies to {out_path}")

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_parquet("data/processed/analysis_table.parquet")
    
    print("Detecting anomalies...")
    anomalies_df = detect_anomalies(df)
    
    print("\nTop 5 anomalies found:")
    print(anomalies_df[anomalies_df['is_anomaly']][['date', 'commodity_name', 'price_pkr_per_40kg', 'anomaly_reason']].head())
    
    save_anomalies(anomalies_df)