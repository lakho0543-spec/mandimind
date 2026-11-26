import pandas as pd
import pytest

from pipeline.ingest import load_raw, load_wfp_prices, load_weather, WFP_CSV, WEATHER_JSON

# Skip data-dependent tests if the raw files aren't present (e.g. in CI)
needs_raw_data = pytest.mark.skipif(
    not (WFP_CSV.exists() and WEATHER_JSON.exists()),
    reason="Raw data missing — download via data/DATA_SOURCES.md"
)

@needs_raw_data
def test_load_wfp_prices_returns_dataframe():
    df = load_wfp_prices()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0, "WFP prices CSV is empty"
    required = {
        "admin1_name", "admin2_name", "market_name",
        "commodity_name", "unit", "price", "currency_code",
        "reference_period_start", "reference_period_end",
    }
    missing = required - set(df.columns)
    assert not missing, f"WFP CSV missing columns: {missing}"

@needs_raw_data
def test_load_weather_returns_1827_days():
    df = load_weather()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1827, f"Expected 1827 daily rows, got {len(df)}"
    assert list(df.columns) == [
        "date", "precipitation_sum", "temperature_2m_max", "temperature_2m_min"
    ]
    assert df["date"].min() == pd.Timestamp("2020-01-01")
    assert df["date"].max() == pd.Timestamp("2024-12-31")
    assert df["date"].isna().sum() == 0
    assert df["temperature_2m_max"].isna().sum() == 0
    assert df["temperature_2m_min"].isna().sum() == 0

@needs_raw_data
def test_load_raw_returns_both():
    prices, weather = load_raw()
    assert isinstance(prices, pd.DataFrame)
    assert isinstance(weather, pd.DataFrame)
    assert len(prices) > 0
    assert len(weather) == 1827

def test_missing_file_raises_helpful_error(tmp_path):
    from pipeline.ingest import load_wfp_prices
    fake_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="WFP prices file not found"):
        load_wfp_prices(fake_path)