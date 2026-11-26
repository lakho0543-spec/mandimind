"""
pipeline/clean.py — Phase 2B: Clean and standardize raw data.

Transforms the raw WFP prices CSV into a standardized analysis table.
Rules (spec Section 4):
  - standardize units to PKR per 40kg (maund) where possible
  - flag missing/forward-dated rows, never delete
  - flag outliers (Z-score > 3), never delete
"""

from __future__ import annotations

import pandas as pd
import numpy as np

def standardize_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts prices to a standard unit: PKR per 40kg.
    - KG: multiply by 40.
    - L: keep as is, but flag.
    - DOZEN: divide by 12.
    """
    df = df.copy()
    
    # Initialize standardized price column
    df['price_pkr_per_40kg'] = np.nan
    
    # 1. KG -> 40 kg (Maund)
    kg_mask = df['unit'].str.upper() == 'KG'
    df.loc[kg_mask, 'price_pkr_per_40kg'] = df.loc[kg_mask, 'price'] * 40
    
    # 2. L -> keep as is (and mark unit as 'L')
    l_mask = df['unit'].str.upper() == 'L'
    df.loc[l_mask, 'price_pkr_per_40kg'] = df.loc[l_mask, 'price']
    
    # 3. DOZEN -> per piece
    dozen_mask = df['unit'].str.upper() == 'DOZEN'
    df.loc[dozen_mask, 'price_pkr_per_40kg'] = df.loc[dozen_mask, 'price'] / 12.0
    
    # Tag the standard unit for clarity
    df['std_unit'] = df['unit'].str.upper().map({
        'KG': 'per_40kg', 'L': 'per_liter', 'DOZEN': 'per_piece'
    })
    
    return df

def flag_forward_dated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flags rows where the reference_period_end is in the future.
    HDX publishes rolling windows; these aren't errors, just flags.
    """
    df = df.copy()
    today = pd.Timestamp.now().normalize()
    # Convert to datetime if not already
    df['reference_period_end'] = pd.to_datetime(df['reference_period_end'])
    df['is_forward_dated'] = df['reference_period_end'] > today
    return df

def flag_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Flags price outliers using a rolling Z-score per commodity and market.
    We use a window of 12 periods (approx 1 year) to account for inflation.
    """
    df = df.copy()
    
    # Fill NaNs to avoid errors in groupby
    df = df.sort_values('date')
    
    # Calculate rolling mean and std per commodity
    rolling_mean = df.groupby('commodity_name')['price_pkr_per_40kg'].transform(
        lambda x: x.rolling(window=12, min_periods=3).mean()
    )
    rolling_std = df.groupby('commodity_name')['price_pkr_per_40kg'].transform(
        lambda x: x.rolling(window=12, min_periods=3).std()
    )
    
    # Z-score = (price - mean) / std
    z_scores = (df['price_pkr_per_40kg'] - rolling_mean) / rolling_std
    
    # Flag if absolute Z > threshold
    df['is_outlier'] = (z_scores.abs() > z_thresh) & (rolling_std > 0)
    
    return df

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function: applies all cleaning steps.
    """
    df = df.copy()
    
    # 1. Standardize dates
    df['date'] = pd.to_datetime(df['reference_period_start'])
    
    # 2. Standardize units
    df = standardize_units(df)
    
    # 3. Forward-dated flag
    df = flag_forward_dated(df)
    
    # 4. Outlier flag
    df = flag_outliers(df)
    
    return df