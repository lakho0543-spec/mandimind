"""
pipeline/model.py — Phase 3G: Per-commodity forecasting + metrics export.

Trains ONE XGBoost model per commodity. This is the standard approach
in commodity forecasting because each crop has different seasonal and
volatility dynamics.

Saves:
  - Individual models to models/per_commodity/
  - A metrics.json file summarizing which crops XGBoost can actually beat
    the Naive baseline on, so the Streamlit app knows how to behave.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error

# We don't need mean_squared_error here, but we import it anyway if you want to extend later
# from sklearn.metrics import mean_squared_error


def load_and_prepare_data(
    parquet_path: str | Path = "data/processed/analysis_table.parquet",
) -> pd.DataFrame:
    from pipeline.features import build_features

    df = pd.read_parquet(parquet_path)
    df = build_features(df)
    return df


FEATURE_COLS = [
    "price_lag_1",
    "price_lag_4",
    "price_rolling_mean_4",
    "price_rolling_std_4",
    "price_pct_change_1",
    "price_pct_change_4",
    "price_vs_rolling_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "month",
    "is_ramadan_season",
    "is_monsoon_season",
]


def train_single_commodity(df: pd.DataFrame, commodity: str) -> dict:
    """
    Train and evaluate XGBoost for ONE commodity.
    Returns dict of metrics + the trained model.
    """
    df_c = df[df["commodity_name"] == commodity].copy()
    df_c["log_return"] = np.log(
        df_c["target_next_price"] / df_c["price_pkr_per_40kg"]
    )
    df_c = df_c[df_c["log_return"].notna() & np.isfinite(df_c["log_return"])].copy()

    if len(df_c) < 50:
        return {"commodity": commodity, "skipped": True, "reason": "too few rows"}

    train = df_c[df_c["date"] < "2024-01-01"]
    test = df_c[df_c["date"] >= "2024-01-01"]

    if len(train) < 30 or len(test) < 10:
        return {"commodity": commodity, "skipped": True, "reason": "insufficient split"}

    X_train = train[FEATURE_COLS]
    y_train = train["log_return"]
    X_test = test[FEATURE_COLS]

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds_return = model.predict(X_test)
    preds_price_xgb = test["price_pkr_per_40kg"].values * np.exp(preds_return)
    preds_price_naive = test["price_pkr_per_40kg"].values
    y_true = test["target_next_price"].values

    mae_xgb = mean_absolute_error(y_true, preds_price_xgb)
    mae_naive = mean_absolute_error(y_true, preds_price_naive)

    improvement = ((mae_naive - mae_xgb) / mae_naive) * 100

    return {
        "commodity": commodity,
        "skipped": False,
        "n_train": len(train),
        "n_test": len(test),
        "mae_xgb": mae_xgb,
        "mae_naive": mae_naive,
        "improvement_pct": improvement,
        "model": model,
    }


def train_all_commodities(df: pd.DataFrame) -> dict:
    commodities = sorted(df["commodity_name"].unique())
    results = {}
    for c in commodities:
        results[c] = train_single_commodity(df, c)
    return results


if __name__ == "__main__":
    print("Loading data...")
    df = load_and_prepare_data()

    print(
        f"Training per-commodity models for {df['commodity_name'].nunique()} commodities...\n"
    )
    results = train_all_commodities(df)

    print(
        f"{'Commodity':30} {'XGB MAE':>12} {'Naive MAE':>12} {'Improvement':>14}"
    )
    print("-" * 72)
    improving = 0
    total = 0
    metrics_for_app = {}

    for c, r in results.items():
        if r.get("skipped"):
            print(f"{c[:30]:30} {'SKIPPED':>12} {r.get('reason', ''):>30}")
            continue
        total += 1
        mark = "✅" if r["improvement_pct"] > 0 else "❌"
        if r["improvement_pct"] > 0:
            improving += 1
        print(
            f"{c[:30]:30} {r['mae_xgb']:>10.2f}   {r['mae_naive']:>10.2f}   {r['improvement_pct']:>+6.1f}% {mark}"
        )

        # Store for the app
        metrics_for_app[c] = {
            "mae_xgb": float(r["mae_xgb"]),
            "mae_naive": float(r["mae_naive"]),
            "improvement_pct": float(r["improvement_pct"]),
            "use_ml": bool(r["improvement_pct"] > 0),  # only use ML if it beats naive
        }

    print("\n" + "=" * 72)
    print(f"Improvement on {improving} of {total} commodities")
    print("=" * 72)

    # Save per-commodity models
    out_dir = Path("models/per_commodity")
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, r in results.items():
        if not r.get("skipped"):
            safe_name = (
                c.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")
            )
            joblib.dump(r["model"], out_dir / f"{safe_name}.pkl")
    print(f"\nModels saved to {out_dir}/")

    # Save metrics JSON for the app
    metrics_path = Path("models/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_for_app, f, indent=2)
    print(f"Metrics saved to {metrics_path}")