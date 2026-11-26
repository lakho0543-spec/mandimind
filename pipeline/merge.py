"""
pipeline/merge.py — Phase 2C: Merge prices and weather.

Combines the cleaned price data with weather data to create the single
analysis table used by all downstream models.

Key logic:
  - Filter prices to the city we have weather for (Lahore).
  - Resample daily weather to weekly averages.
  - Join on the nearest date.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

def resample_weather_to_weekly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather is daily. We resample it to weekly frequency to match prices.
    We use 'W-MON' (weeks ending on Monday) to align with typical market weeks.
    """
    df = weather_df.copy()
    df = df.set_index('date')
    
    # Resample to weekly, taking the mean for temperature and sum for rain
    weekly = df.resample('W-MON').agg({
        'precipitation_sum': 'sum',
        'temperature_2m_max': 'mean',
        'temperature_2m_min': 'mean'
    }).reset_index()
    
    return weekly

def merge_prices_weather(prices_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge prices and weather for Lahore only.
    """
    # 1. Filter prices to Lahore
    # Note: WFP uses 'Lahore' in market_name
    lahore_prices = prices_df[prices_df['market_name'].str.contains('Lahore', case=False, na=False)].copy()
    
    if lahore_prices.empty:
        raise ValueError("No price data found for Lahore. Check the raw CSV.")
    
    # 2. Resample weather
    weekly_weather = resample_weather_to_weekly(weather_df)
    
    # 3. Merge on date (using pd.merge_asof for nearest date matching)
    # We sort both dataframes first (required by merge_asof)
    lahore_prices = lahore_prices.sort_values('date')
    weekly_weather = weekly_weather.sort_values('date')
    
    merged = pd.merge_asof(
        lahore_prices,
        weekly_weather,
        on='date',
        direction='backward' # Match the most recent weather leading up to the price date
    )
    
    return merged

def build_analysis_table(prices_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: merges and saves the final analysis table.
    """
    merged = merge_prices_weather(prices_df, weather_df)
    
    # Drop rows where price is NaN (should be rare)
    merged = merged.dropna(subset=['price_pkr_per_40kg'])
    
    return merged

if __name__ == "__main__":
    # This allows us to run `python -m pipeline.merge` to test it manually
    from pipeline.ingest import load_raw
    from pipeline.clean import clean_prices
    
    print("Loading raw data...")
    raw_prices, raw_weather = load_raw()
    
    print("Cleaning prices...")
    cleaned_prices = clean_prices(raw_prices)
    
    print("Merging...")
    final_df = build_analysis_table(cleaned_prices, raw_weather)
    
    print(f"Merged table shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    
    # Save to parquet (requires pyarrow, which is in requirements)
    out_path = Path("data/processed/analysis_table.parquet")
    out_path.parent.mkdir(exist_ok=True)
    final_df.to_parquet(out_path)
    print(f"Saved to {out_path}")