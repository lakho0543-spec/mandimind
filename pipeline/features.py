"""
pipeline/features.py - Phase 2D: Build machine learning features.  (MandiMind v7.0)

Takes the analysis table and adds lags, rolling statistics, seasonality flags and
the target variable for forecasting.

WHAT CHANGED IN v7.0
--------------------
* Single source of truth for the model's feature list (`MODEL_FEATURE_COLS`).
  `pipeline/model.py` and `app/main.py` both import it, so they can never drift.
* Causal-inference features for the Root Cause AI and What-If pages:
    - `is_ramadan_window`   calendar-accurate Ramadan flag (Islamic-calendar start
                            dates, 2 weeks of pre-Ramadan buying + the month itself).
                            The original month-based `is_ramadan_season` is KEPT
                            unchanged so existing models behave exactly as before.
    - `precip_rolling_sum_4`, `temp_rolling_mean_4`   recent weather accumulation.
    - `diesel_price_pkr`, `diesel_pct_change_1`       transport-cost driver (optional).
    - `fx_pkr_usd`, `fx_pct_change_1`                 exchange-rate driver (optional).
  Optional drivers are auto-detected from several candidate column names; when the
  source column does not exist the new columns are created as NaN, so downstream
  code can always rely on them being present.
* CORRECTNESS FIX (district-aware grouping): lags, rolling stats, pct-changes and
  the target are now computed per (commodity, district) instead of per commodity.
  The v6 code shifted a commodity's rows across districts, so `price_lag_1` and
  `target_next_price` could be another district's price on the same date. Set
  `GROUP_BY_DISTRICT = False` to restore the old behaviour.
* CORRECTNESS FIX (rolling windows): the rolling mean/std are now computed inside
  each group. Before, `groupby().shift().rolling()` rolled across group
  boundaries and leaked one crop's prices into the next crop's first rows.
* `build_features(df, for_inference=True)` keeps the most recent period of every
  series (which has no target yet) so the app can show the true latest price and
  score the next period. Default behaviour (training rows only) is unchanged.
* `select_latest_rows()` and `model_filename()` helpers shared with the app.
* Infinite percentage changes (division by a zero price) are converted to NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COL: str = "price_pkr_per_40kg"

# If True, per-series features are computed per (commodity, district) whenever the
# table has an `admin2_name` column. Set False to reproduce v6 behaviour.
GROUP_BY_DISTRICT: bool = True

# The exact feature list the per-commodity XGBoost models are trained on.
MODEL_FEATURE_COLS: list[str] = [
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

# Columns the Root Cause AI / What-If pages draw on (always present after build).
CAUSAL_FEATURES: list[str] = [
    "precipitation_sum",
    "precip_rolling_sum_4",
    "temperature_2m_max",
    "price_pct_change_1",
    "is_ramadan_window",
    "is_monsoon_season",
    "diesel_pct_change_1",
    "fx_pct_change_1",
]

# Candidate names for optional driver columns (matched case-insensitively).
DIESEL_CANDIDATES: tuple[str, ...] = (
    "diesel_price_pkr", "diesel_price", "diesel_pkr_per_litre", "diesel",
    "hsd_price", "fuel_price_pkr", "fuel_price",
)
FX_CANDIDATES: tuple[str, ...] = (
    "fx_pkr_usd", "usd_pkr", "pkr_usd", "exchange_rate", "fx_rate", "usd_pkr_rate",
)

# 1 Ramadan (approximate, +/- 1 day depending on moon sighting).
RAMADAN_START_DATES: tuple[str, ...] = (
    "2015-06-18", "2016-06-06", "2017-05-27", "2018-05-16", "2019-05-06",
    "2020-04-24", "2021-04-13", "2022-04-02", "2023-03-23", "2024-03-11",
    "2025-03-01", "2026-02-18", "2027-02-08", "2028-01-28", "2029-01-16",
    "2030-01-05",
)


# ----------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------
def _group_keys(df: pd.DataFrame) -> list[str]:
    """
    Return the columns that identify one price series.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table.

    Returns
    -------
    list of str
        ``["commodity_name", "admin2_name"]`` when district grouping is enabled
        and available, otherwise ``["commodity_name"]``.
    """
    if GROUP_BY_DISTRICT and "admin2_name" in df.columns:
        return ["commodity_name", "admin2_name"]
    return ["commodity_name"]


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Copy and sort a table by series key and date with a stable sort.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table.

    Returns
    -------
    pd.DataFrame
        Sorted copy with a clean RangeIndex.
    """
    out = df.copy()
    return out.sort_values(_group_keys(out) + ["date"], kind="mergesort").reset_index(drop=True)


def _grouped(df: pd.DataFrame, col: str):
    """
    Group one column by series key (keeping rows whose key is missing).

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table.
    col : str
        Column to select after grouping.

    Returns
    -------
    pandas.core.groupby.SeriesGroupBy
        Grouped column.
    """
    return df.groupby(_group_keys(df), sort=False, dropna=False)[col]


def _clean_inf(s: pd.Series) -> pd.Series:
    """
    Replace +/- infinity with NaN.

    Parameters
    ----------
    s : pd.Series
        Numeric series.

    Returns
    -------
    pd.Series
        Series where infinite values are NaN.
    """
    return s.replace([np.inf, -np.inf], np.nan)


def find_optional_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """
    Find the first candidate column present in a table (case-insensitive).

    Parameters
    ----------
    df : pd.DataFrame
        Table to search.
    candidates : tuple of str
        Candidate column names in priority order.

    Returns
    -------
    str or None
        The actual column name, or None when no candidate exists.
    """
    lowered = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return str(lowered[cand.lower()])
    return None


def model_filename(commodity: str) -> str:
    """
    File stem under which a commodity's model is saved (same rule as v6).

    Parameters
    ----------
    commodity : str
        Commodity name, e.g. ``"Rice (coarse)"``.

    Returns
    -------
    str
        Safe file stem, e.g. ``"Rice_coarse"``.
    """
    return commodity.replace("(", "").replace(")", "").replace(" ", "_").replace(",", "")


# ----------------------------------------------------------------------------
# Feature builders
# ----------------------------------------------------------------------------
def add_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features (t-1, t-4) and rolling statistics (window=4).

    Every statistic is computed inside one price series (commodity, district), so
    different crops or districts are never mixed. Rolling statistics use only
    past prices (shifted by one period), so they never include the current price.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table with ``date`` and ``price_pkr_per_40kg``.

    Returns
    -------
    pd.DataFrame
        Sorted copy with ``price_lag_1``, ``price_lag_4``,
        ``price_rolling_mean_4`` and ``price_rolling_std_4``.
    """
    df = _sorted(df)

    # 1. Lags: price 1 period ago, price 4 periods ago
    df["price_lag_1"] = _grouped(df, PRICE_COL).shift(1)
    df["price_lag_4"] = _grouped(df, PRICE_COL).shift(4)

    # 2. Rolling stats over the previous 4 periods (current price excluded)
    prev = df["price_lag_1"]
    keyed = prev.groupby([df[k] for k in _group_keys(df)], sort=False, dropna=False)
    df["price_rolling_mean_4"] = keyed.transform(lambda s: s.rolling(window=4).mean())
    df["price_rolling_std_4"] = keyed.transform(lambda s: s.rolling(window=4).std())
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum and differencing features.

    These capture whether prices are accelerating up or down.

    Parameters
    ----------
    df : pd.DataFrame
        Table that already contains the rolling features.

    Returns
    -------
    pd.DataFrame
        Copy with ``price_pct_change_1``, ``price_pct_change_4`` and
        ``price_vs_rolling_mean``.
    """
    df = _sorted(df)
    df["price_pct_change_1"] = _clean_inf(_grouped(df, PRICE_COL).pct_change(1))
    df["price_pct_change_4"] = _clean_inf(_grouped(df, PRICE_COL).pct_change(4))
    df["price_vs_rolling_mean"] = df[PRICE_COL] - df["price_rolling_mean_4"]
    return df


def _ramadan_window_flag(dates: pd.Series) -> np.ndarray:
    """
    Flag dates inside a Ramadan buying window.

    The window runs from 14 days before 1 Ramadan (pre-Ramadan stocking) to
    30 days after it.

    Parameters
    ----------
    dates : pd.Series
        Datetime series.

    Returns
    -------
    numpy.ndarray
        Integer array (1 = inside a Ramadan window, 0 = outside).
    """
    stamps = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    flag = np.zeros(len(stamps), dtype=bool)
    for start in RAMADAN_START_DATES:
        s = np.datetime64(start, "D")
        flag |= (stamps >= s - np.timedelta64(14, "D")) & (stamps <= s + np.timedelta64(30, "D"))
    return flag.astype(int)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add seasonality flags.

    ``is_ramadan_season`` (March/April) is the original approximation and is kept
    for model compatibility. ``is_ramadan_window`` is the calendar-accurate flag
    used by the Root Cause AI and the What-If simulator.

    Parameters
    ----------
    df : pd.DataFrame
        Table with a datetime ``date`` column.

    Returns
    -------
    pd.DataFrame
        Copy with ``month``, ``is_ramadan_season``, ``is_ramadan_window`` and
        ``is_monsoon_season``.
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["is_ramadan_season"] = df["month"].isin([3, 4]).astype(int)
    df["is_ramadan_window"] = _ramadan_window_flag(df["date"])
    df["is_monsoon_season"] = df["month"].isin([7, 8, 9]).astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add recent weather accumulation features for causal analysis.

    Missing weather columns are created as NaN so later code never fails.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted table that may contain ``precipitation_sum`` and
        ``temperature_2m_max``.

    Returns
    -------
    pd.DataFrame
        Copy with ``precip_rolling_sum_4`` and ``temp_rolling_mean_4``.
    """
    df = df.copy()
    for col in ("precipitation_sum", "temperature_2m_max"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    keys = [df[k] for k in _group_keys(df)]
    df["precip_rolling_sum_4"] = df["precipitation_sum"].groupby(keys, sort=False, dropna=False).transform(
        lambda s: s.rolling(window=4, min_periods=1).sum()
    )
    df["temp_rolling_mean_4"] = df["temperature_2m_max"].groupby(keys, sort=False, dropna=False).transform(
        lambda s: s.rolling(window=4, min_periods=1).mean()
    )
    return df


def add_cost_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transport-cost (diesel) and exchange-rate (PKR/USD) drivers.

    The source columns are optional and auto-detected. When absent, the standard
    columns are still created (as NaN) so the app can detect "driver unavailable".
    Exchange-rate columns are assumed to be PKR per USD (higher = weaker rupee).

    Parameters
    ----------
    df : pd.DataFrame
        Sorted table.

    Returns
    -------
    pd.DataFrame
        Copy with ``diesel_price_pkr``, ``diesel_pct_change_1``, ``fx_pkr_usd``
        and ``fx_pct_change_1``.
    """
    df = df.copy()
    for src_candidates, level_col, change_col in (
        (DIESEL_CANDIDATES, "diesel_price_pkr", "diesel_pct_change_1"),
        (FX_CANDIDATES, "fx_pkr_usd", "fx_pct_change_1"),
    ):
        src = find_optional_column(df, src_candidates)
        if src is not None:
            df[level_col] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[level_col] = np.nan
        df[change_col] = _clean_inf(_grouped(df, level_col).pct_change(1, fill_method=None))
    return df


def add_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the target: the price in the NEXT time period of the same series.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table.

    Returns
    -------
    pd.DataFrame
        Sorted copy with ``target_next_price`` (NaN for each series' last row).
    """
    df = _sorted(df)
    df["target_next_price"] = _grouped(df, PRICE_COL).shift(-1)
    return df


def build_features(df: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
    """
    Run the full feature pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table (``date``, ``commodity_name``, ``price_pkr_per_40kg`` and
        optionally ``admin2_name``, weather, diesel, FX columns).
    for_inference : bool, default False
        False (training): drop rows without a next-period target, exactly like v6.
        True (app): keep each series' latest row, whose target is still unknown.

    Returns
    -------
    pd.DataFrame
        Feature table sorted by series and date.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_lags_and_rolling(df)
    df = add_momentum_features(df)
    df = add_calendar_features(df)
    df = add_weather_features(df)
    df = add_cost_features(df)
    df = add_target_variable(df)

    required = ["price_lag_1", "price_lag_4"]
    if not for_inference:
        required.append("target_next_price")
    return df.dropna(subset=required)


def select_latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the rows to score for the next period.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``build_features(..., for_inference=True)``.

    Returns
    -------
    pd.DataFrame
        Rows on each commodity's most recent date whose target is unknown.
    """
    if df.empty:
        return df.copy()
    last = df.groupby("commodity_name")["date"].transform("max")
    return df[(df["date"] == last) & df["target_next_price"].isna()].copy()