import pytest
import pandas as pd
from pipeline.merge import resample_weather_to_weekly, merge_prices_weather
from pipeline.ingest import load_raw, WFP_CSV, WEATHER_JSON
from pipeline.clean import clean_prices

needs_raw_data = pytest.mark.skipif(
    not (WFP_CSV.exists() and WEATHER_JSON.exists()),
    reason="Raw data missing"
)

@needs_raw_data
def test_resample_weather():
    _, raw_weather = load_raw()
    weekly = resample_weather_to_weekly(raw_weather)
    
    # 1827 days / 7 = approx 261 weeks
    assert 250 < len(weekly) < 270
    assert 'precipitation_sum' in weekly.columns
    assert 'temperature_2m_max' in weekly.columns

@needs_raw_data
def test_merge_prices_weather():
    raw_prices, raw_weather = load_raw()
    cleaned_prices = clean_prices(raw_prices)
    
    merged = merge_prices_weather(cleaned_prices, raw_weather)
    
    # 1. Must have weather columns now
    assert 'temperature_2m_max' in merged.columns
    assert 'precipitation_sum' in merged.columns
    
    # 2. Must only contain Lahore
    assert all(merged['market_name'].str.contains('Lahore', case=False))
    
    # 3. Row count should match the filtered Lahore prices
    lahore_count = cleaned_prices[cleaned_prices['market_name'].str.contains('Lahore', case=False)].shape[0]
    assert len(merged) == lahore_count