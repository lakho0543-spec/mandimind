"""
pipeline/model.py — Phase 3A (revised): Train forecasting model.

Trains XGBoost on LOG RETURNS instead of absolute prices.
Log returns are stationary, which makes ML models far more reliable for
time series than absolute prices (which trend upward).

Compares against a Seasonal Naive baseline (predict next = current).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib


def load_and_prepare_data(parquet_path: str | Path = "data/processed/analysis_table.parquet") -> pd.DataFrame:
    from pipeline.features import build_features
    df = pd.read_parquet(parquet_path)
    df = build_features(df)
    return df


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Features: lags, rolling stats, weather, calendar.
    Target: log(price_next / price_now).
    """
    df = df.copy()

    # Log return target — stationary, so ML models can generalize across years.
    df['log_return'] = np.log(df['target_next_price'] / df['price_pkr_per_40kg'])
    df = df[df['log_return'].notna() & np.isfinite(df['log_return'])].copy()

    feature_cols = [
        'price_lag_1', 'price_lag_4',
        'price_rolling_mean_4', 'price_rolling_std_4',
        'temperature_2m_max', 'precipitation_sum',
        'month', 'is_ramadan_season', 'is_monsoon_season',
    ]

    X = df[feature_cols]
    y = df['log_return']
    return X, y, feature_cols


def train_and_evaluate(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, dict]:
    """
    Train XGBoost on log returns, evaluate on the ORIGINAL price scale
    (so the comparison against the naive baseline is apples-to-apples).
    """
    df = df.copy()
    X, y, feature_cols = get_features_and_target(df)

    # --- Time-based split ---
    train_mask = df['date'] < '2024-01-01'
    test_mask = df['date'] >= '2024-01-01'

    # Use the SAME mask applied to X / y via index alignment
    X_train, y_train = X[train_mask.values], y[train_mask.values]
    X_test, y_test = X[test_mask.values], y[test_mask.values]
    df_test = df[test_mask.values].copy()

    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

    # --- Train XGBoost on log returns ---
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror',
    )
    model.fit(X_train, y_train)

    # --- Predict ---
    preds_return = model.predict(X_test)

    # Convert log return predictions back to price predictions
    # price_next = price_now * exp(log_return)
    preds_price_xgb = df_test['price_pkr_per_40kg'].values * np.exp(preds_return)

    # --- Naive baseline: next = current ---
    preds_price_naive = df_test['price_pkr_per_40kg'].values

    # --- Evaluate on the PRICE scale (PKR per 40 kg) ---
    y_true_price = df_test['target_next_price'].values

    mae_xgb = mean_absolute_error(y_true_price, preds_price_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_true_price, preds_price_xgb))
    mae_naive = mean_absolute_error(y_true_price, preds_price_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_true_price, preds_price_naive))

    metrics = {
        'mae_xgb': mae_xgb,
        'rmse_xgb': rmse_xgb,
        'mae_naive': mae_naive,
        'rmse_naive': rmse_naive,
        'improvement_pct': ((mae_naive - mae_xgb) / mae_naive) * 100,
    }
    return model, metrics


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare_data()

    print("Training and evaluating model...")
    model, metrics = train_and_evaluate(df)

    print("\n--- RESULTS ---")
    print(f"Naive Baseline MAE: {metrics['mae_naive']:.2f} PKR")
    print(f"XGBoost MAE:        {metrics['mae_xgb']:.2f} PKR")
    print(f"Improvement:        {metrics['improvement_pct']:.1f}%")

    out_path = Path("models/xgboost_forecast.pkl")
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nModel saved to {out_path}")