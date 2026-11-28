"""
pipeline/model.py - Phase 3G: Per-commodity forecasting + metrics export.  (MandiMind v7.0)

Trains ONE XGBoost model per commodity. This is the standard approach
in commodity forecasting because each crop has different seasonal and
volatility dynamics.

Saves:
  - Individual models to models/per_commodity/
  - A metrics.json file summarizing which crops XGBoost can actually beat
    the Naive baseline on, so the Streamlit app knows how to behave.

WHAT CHANGED IN v7.0
--------------------
* Feature list now comes from `pipeline.features.MODEL_FEATURE_COLS` (single source
  of truth shared with the app). `FEATURE_COLS` is kept as an alias, and the list
  itself is unchanged, so model inputs are identical to v6.
* New per-commodity metrics in metrics.json (used by the Excel "Metrics" sheet, the
  Forecast Lab KPIs and the ML next-period estimate):
    mape_xgb, mape_naive, rmse_xgb, rmse_naive, directional_accuracy,
    n_train, n_test, last_train_date, top_features (top-5 importances).
  All v6 keys (mae_xgb, mae_naive, improvement_pct, use_ml) are still written.
* A reserved "__meta__" entry (version, generation time, test split, counts and the
  list of skipped commodities). The app strips it before using the metrics.
* Missing feature columns are tolerated (they become NaN, which XGBoost handles),
  and non-finite metric values are written as null so the JSON is always valid.
* The training logic is wrapped in `main()`; an empty/failed run still writes a
  valid (empty) metrics.json instead of crashing.
* NOTE: because `pipeline/features.py` now builds lags per (commodity, district),
  the MAE numbers will differ from v6. The v6 numbers were partly measuring
  cross-district noise, so treat the new ones as the honest baseline.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from pipeline.features import MODEL_FEATURE_COLS, model_filename

# Alias kept for backward compatibility with v6 imports.
FEATURE_COLS: list[str] = MODEL_FEATURE_COLS

TEST_START: str = "2024-01-01"
METRICS_VERSION: str = "7.0"


def load_and_prepare_data(
    parquet_path: str | Path = "data/processed/analysis_table.parquet",
) -> pd.DataFrame:
    """
    Load the analysis table and build training features.

    Parameters
    ----------
    parquet_path : str or Path
        Location of ``analysis_table.parquet``.

    Returns
    -------
    pd.DataFrame
        Feature table (training rows only, i.e. with a known next-period price).
    """
    from pipeline.features import build_features

    df = pd.read_parquet(parquet_path)
    return build_features(df)


def _finite(value: Any) -> float | None:
    """
    Convert a number to a JSON-safe float.

    Parameters
    ----------
    value : Any
        Numeric-like value.

    Returns
    -------
    float or None
        The float, or None when it is NaN/inf/not numeric.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def train_single_commodity(df: pd.DataFrame, commodity: str) -> dict[str, Any]:
    """
    Train and evaluate XGBoost for ONE commodity.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table from ``build_features``.
    commodity : str
        Commodity to train on.

    Returns
    -------
    dict
        Metrics plus the trained ``model``; or ``{"skipped": True, "reason": ...}``
        when there is too little data.
    """
    df_c = df[df["commodity_name"] == commodity].copy()
    df_c["log_return"] = np.log(df_c["target_next_price"] / df_c["price_pkr_per_40kg"])
    df_c = df_c[df_c["log_return"].notna() & np.isfinite(df_c["log_return"])].copy()

    if len(df_c) < 50:
        return {"commodity": commodity, "skipped": True, "reason": "too few rows"}

    train = df_c[df_c["date"] < TEST_START]
    test = df_c[df_c["date"] >= TEST_START]

    if len(train) < 30 or len(test) < 10:
        return {"commodity": commodity, "skipped": True, "reason": "insufficient split"}

    X_train = train.reindex(columns=FEATURE_COLS)
    y_train = train["log_return"]
    X_test = test.reindex(columns=FEATURE_COLS)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    preds_return = model.predict(X_test)
    price_now = test["price_pkr_per_40kg"].to_numpy(dtype=float)
    preds_price_xgb = price_now * np.exp(preds_return)
    preds_price_naive = price_now
    y_true = test["target_next_price"].to_numpy(dtype=float)

    mae_xgb = mean_absolute_error(y_true, preds_price_xgb)
    mae_naive = mean_absolute_error(y_true, preds_price_naive)
    improvement = ((mae_naive - mae_xgb) / mae_naive) * 100 if mae_naive else float("nan")

    # --- v7.0 extra metrics -------------------------------------------------
    safe_true = np.maximum(np.abs(y_true), 1e-9)
    mape_xgb = float(np.mean(np.abs(y_true - preds_price_xgb) / safe_true) * 100)
    mape_naive = float(np.mean(np.abs(y_true - preds_price_naive) / safe_true) * 100)
    rmse_xgb = float(np.sqrt(np.mean((y_true - preds_price_xgb) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((y_true - preds_price_naive) ** 2)))
    directional = float(np.mean(np.sign(preds_return) == np.sign(test["log_return"].to_numpy())) * 100)

    importances = getattr(model, "feature_importances_", None)
    top_features: list[dict[str, Any]] = []
    if importances is not None:
        order = np.argsort(importances)[::-1][:5]
        top_features = [
            {"feature": FEATURE_COLS[i], "importance": float(importances[i])}
            for i in order
            if float(importances[i]) > 0
        ]

    return {
        "commodity": commodity,
        "skipped": False,
        "n_train": len(train),
        "n_test": len(test),
        "mae_xgb": mae_xgb,
        "mae_naive": mae_naive,
        "improvement_pct": improvement,
        "mape_xgb": mape_xgb,
        "mape_naive": mape_naive,
        "rmse_xgb": rmse_xgb,
        "rmse_naive": rmse_naive,
        "directional_accuracy": directional,
        "last_train_date": str(pd.Timestamp(train["date"].max()).date()),
        "top_features": top_features,
        "model": model,
    }


def train_all_commodities(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Train one model per commodity.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table from ``build_features``.

    Returns
    -------
    dict
        Mapping commodity name to the result of ``train_single_commodity``.
    """
    commodities = sorted(df["commodity_name"].dropna().unique())
    results: dict[str, dict[str, Any]] = {}
    for c in commodities:
        results[c] = train_single_commodity(df, c)
    return results


def build_metrics_for_app(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Convert training results into the JSON structure the Streamlit app reads.

    Parameters
    ----------
    results : dict
        Output of ``train_all_commodities``.

    Returns
    -------
    dict
        One entry per trained commodity plus a reserved ``"__meta__"`` entry.
    """
    metrics: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    improving = 0
    total = 0

    for c, r in results.items():
        if r.get("skipped"):
            skipped[c] = str(r.get("reason", ""))
            continue
        total += 1
        improvement = _finite(r["improvement_pct"])
        use_ml = bool(improvement is not None and improvement > 0)  # only use ML if it beats naive
        improving += int(use_ml)
        metrics[c] = {
            "mae_xgb": _finite(r["mae_xgb"]),
            "mae_naive": _finite(r["mae_naive"]),
            "improvement_pct": improvement,
            "use_ml": use_ml,
            "mape_xgb": _finite(r["mape_xgb"]),
            "mape_naive": _finite(r["mape_naive"]),
            "rmse_xgb": _finite(r["rmse_xgb"]),
            "rmse_naive": _finite(r["rmse_naive"]),
            "directional_accuracy": _finite(r["directional_accuracy"]),
            "n_train": int(r["n_train"]),
            "n_test": int(r["n_test"]),
            "last_train_date": r["last_train_date"],
            "top_features": r["top_features"],
        }

    metrics["__meta__"] = {
        "version": METRICS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_start": TEST_START,
        "n_trained": total,
        "n_improving": improving,
        "skipped": skipped,
    }
    return metrics


def save_models(results: dict[str, dict[str, Any]], out_dir: str | Path = "models/per_commodity") -> Path:
    """
    Persist every trained model with joblib.

    Parameters
    ----------
    results : dict
        Output of ``train_all_commodities``.
    out_dir : str or Path
        Target directory (created if missing).

    Returns
    -------
    Path
        The directory the models were written to.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c, r in results.items():
        if not r.get("skipped"):
            joblib.dump(r["model"], out / f"{model_filename(c)}.pkl")
    return out


def main() -> None:
    """
    Train all commodities, print a leaderboard, and write models + metrics.json.

    Returns
    -------
    None
    """
    print("Loading data...")
    df = load_and_prepare_data()

    print(f"Training per-commodity models for {df['commodity_name'].nunique()} commodities...\n")
    results = train_all_commodities(df)

    print(f"{'Commodity':30} {'XGB MAE':>12} {'Naive MAE':>12} {'Improvement':>14}")
    print("-" * 72)
    improving = 0
    total = 0

    for c, r in results.items():
        if r.get("skipped"):
            print(f"{c[:30]:30} {'SKIPPED':>12} {r.get('reason', ''):>30}")
            continue
        total += 1
        mark = "OK " if r["improvement_pct"] > 0 else "-- "
        if r["improvement_pct"] > 0:
            improving += 1
        print(
            f"{c[:30]:30} {r['mae_xgb']:>10.2f}   {r['mae_naive']:>10.2f}   "
            f"{r['improvement_pct']:>+6.1f}% {mark} dir.acc {r['directional_accuracy']:.0f}%"
        )

    print("\n" + "=" * 72)
    print(f"Improvement on {improving} of {total} commodities")
    print("=" * 72)

    out_dir = save_models(results)
    print(f"\nModels saved to {out_dir}/")

    metrics_path = Path("models/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(build_metrics_for_app(results), f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()