"""
pipeline/features.py — Phase 2D: Build machine learning features.

Takes the analysis table and adds lags, rolling statistics, 
seasonality flags, and the target variable for forecasting.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features (t-1, t-4) and rolling means (window=4).
    We group by commodity so we don't mix wheat prices with tomato prices.
    """
    df = df.copy()
    df = df.sort_values(by=['commodity_name', 'date'])
    
    # Group by commodity to calculate history for each crop separately
    grouped = df.groupby('commodity_name')
    
    # 1. Lags: price 1 week ago, price 4 weeks ago
    df['price_lag_1'] = grouped['price_pkr_per_40kg'].shift(1)
    df['price_lag_4'] = grouped['price_pkr_per_40kg'].shift(4)
    
    # 2. Rolling stats (mean and volatility over last 4 periods)
    # We shift by 1 so the rolling mean doesn't include the current price
    df['price_rolling_mean_4'] = grouped['price_pkr_per_40kg'].shift(1).rolling(window=4).mean()
    df['price_rolling_std_4'] = grouped['price_pkr_per_40kg'].shift(1).rolling(window=4).std()
    
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and differencing features.
    These capture whether prices are accelerating up or down.
    """
    df = df.copy()
    df = df.sort_values(by=['commodity_name', 'date'])
    grouped = df.groupby('commodity_name')

    # Price change over last week (1-week return)
    df['price_pct_change_1'] = grouped['price_pkr_per_40kg'].pct_change(1)
    
    # Price change over last month (4-week return)
    df['price_pct_change_4'] = grouped['price_pkr_per_40kg'].pct_change(4)
    
    # Difference between current and rolling mean (is price above/below trend?)
    df['price_vs_rolling_mean'] = df['price_pkr_per_40kg'] - df['price_rolling_mean_4']

    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic seasonality flags.
    """
    df = df.copy()
    
    # 1. Month
    df['month'] = df['date'].dt.month
    
    # 2. Ramadan flag (approximate for MVP; in a real app, use Islamic calendar lib)
    # Ramadan 2023: March 23 - April 20
    # For this MVP, we will just flag March and April as high-demand months
    df['is_ramadan_season'] = df['month'].isin([3, 4]).astype(int)
    
    # 3. Monsoon/Flood season (July, August, September in Pakistan)
    df['is_monsoon_season'] = df['month'].isin([7, 8, 9]).astype(int)
    
    return df

def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    The target is the price in the NEXT time period.
    We shift by -1 (move data up by one row).
    """
    df = df.copy()
    df = df.sort_values(by=['commodity_name', 'date'])
    
    grouped = df.groupby('commodity_name')
    df['target_next_price'] = grouped['price_pkr_per_40kg'].shift(-1)
    
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lags_and_rolling(df)
    df = add_momentum_features(df)          # <- NEW
    df = add_calendar_features(df)
    df = add_target_variable(df)
    
    ml_columns = ['price_lag_1', 'price_lag_4', 'target_next_price']
    df = df.dropna(subset=ml_columns)
    return df