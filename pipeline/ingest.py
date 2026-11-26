"""
pipeline/ingest.py — Phase 2A: Load raw MandiMind data sources.

Single responsibility: read the two raw files in data/raw/ and return
pandas DataFrames. No cleaning, no filtering, no renaming here.

Callers downstream (clean.py, merge.py, features.py) MUST read from
data/processed/ or from the return values of this module — never from
data/raw/ directly. See spec Section 19: "never edit raw data."

Raises FileNotFoundError with a helpful message if the raw files are
missing (e.g. after a fresh clone where data/raw/* is gitignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# --- Paths -------------------------------------------------------------------
# Resolve from repo root so the module works regardless of cwd.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

WFP_CSV = RAW_DIR / "wfp_food_prices_pakistan.csv"
WEATHER_JSON = RAW_DIR / "weather_lahore_2020_2024.json"


def load_wfp_prices(path: Path = WFP_CSV) -> pd.DataFrame:
    """
    Load the WFP Pakistan food prices CSV.

    Returns the CSV unchanged (raw evidence). Cleaning happens in clean.py.

    Expected columns include:
        admin1_name, admin2_name, market_name, commodity_name,
        unit, price, currency_code, price_type,
        reference_period_start, reference_period_end
    """
    if not path.exists():
        raise FileNotFoundError(
            f"WFP prices file not found: {path}\n"
            f"Re-download via the URL in data/DATA_SOURCES.md, then re-run."
        )
    df = pd.read_csv(path)
    return df


def load_weather(path: Path = WEATHER_JSON) -> pd.DataFrame:
    """
    Load the Open-Meteo daily weather JSON as a tidy DataFrame.

    The JSON has shape:
        { "latitude": ..., "longitude": ...,
          "daily": { "time": [...], "precipitation_sum": [...],
                     "temperature_2m_max": [...], "temperature_2m_min": [...] } }

    We flatten to a DataFrame with columns:
        date, precipitation_sum, temperature_2m_max, temperature_2m_min
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Weather file not found: {path}\n"
            f"Re-download via the URL in data/DATA_SOURCES.md, then re-run."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    daily = data["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "precipitation_sum": daily["precipitation_sum"],
            "temperature_2m_max": daily["temperature_2m_max"],
            "temperature_2m_min": daily["temperature_2m_min"],
        }
    )
    return df


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader: return both frames in one call.

    Returns
    -------
    (prices_df, weather_df)
    """
    return load_wfp_prices(), load_weather()