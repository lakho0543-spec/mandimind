"""
pipeline/model.py — Phase 3A: Train forecasting model.

Trains an XGBoost model to predict next week's price.
Compares against a Naive Baseline to ensure the model is actually useful.
Uses time-based splits (Train: <= 2023, Test: 2024).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def load_and_prepare_data(parquet_path: str | Path = "data/processed/analysis_table.parquet") -> pd.DataFrame:
    """
    Loads the analysis table and builds features.
    """
    from pipeline.features import build_features
    df = pd.read_parquet(parquet_path)
    df = build_features(df)
    return df

def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Selects the columns used for training.
    """
    # These are the features we will feed to XGBoost
    feature_cols = [
        'price_lag_1', 'price_lag_4', 
        'price_rolling_mean_4', 'price_rolling_std_4',
        'temperature_2m_max', 'precipitation_sum',
        'month', 'is_ramadan_season', 'is_monsoon_season'
    ]
    
    X = df[feature_cols]
    y = df['target_next_price']
    
    return X, y, feature_cols

def train_and_evaluate(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, dict]:
    """
    Splits data by time, trains XGBoost, and evaluates against Naive baseline.
    """
    X, y, feature_cols = get_features_and_target(df)
    
    # 1. TIME-BASED SPLIT (Crucial rule from the spec!)
    # Train on data before 2024, test on 2024 onwards
    train_mask = df['date'] < '2024-01-01'
    test_mask = df['date'] >= '2024-01-01'
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
    
    # 2. TRAIN XGBOOST
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # 3. PREDICTIONS
    preds_xgb = model.predict(X_test)
    
    # 4. NAIVE BASELINE (Predict that next week = current week)
    # The current week's price is price_lag_1 (from the test set perspective)
    preds_naive = X_test['price_lag_1'].values
    
    # 5. EVALUATION METRICS
    mae_xgb = mean_absolute_error(y_test, preds_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
    
    mae_naive = mean_absolute_error(y_test, preds_naive)
    rmse_naive = np.sqrt(mean_squared_error(y_test, preds_naive))
    
    metrics = {
        'mae_xgb': mae_xgb,
        'rmse_xgb': rmse_xgb,
        'mae_naive': mae_naive,
        'rmse_naive': rmse_naive,
        'improvement_pct': ((mae_naive - mae_xgb) / mae_naive) * 100
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
    
    # Save the model
    out_path = Path("models/xgboost_forecast.pkl")
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\nModel saved to {out_path}")