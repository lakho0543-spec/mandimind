import pytest
import pandas as pd
import numpy as np
from pipeline.clean import standardize_units, flag_forward_dated, clean_prices
from pipeline.ingest import load_wfp_prices, WFP_CSV

needs_raw_data = pytest.mark.skipif(
    not WFP_CSV.exists(),
    reason="Raw WFP data missing"
)

@needs_raw_data
def test_standardize_units_kg_to_maund():
    # Create a tiny fake dataframe
    df = pd.DataFrame({
        'unit': ['KG', 'L', 'DOZEN'],
        'price': [10.0, 50.0, 120.0]
    })
    cleaned = standardize_units(df)
    
    # KG * 40 = 400
    assert cleaned.loc[0, 'price_pkr_per_40kg'] == 400.0
    # L stays 50
    assert cleaned.loc[1, 'price_pkr_per_40kg'] == 50.0
    # DOZEN / 12 = 10
    assert cleaned.loc[2, 'price_pkr_per_40kg'] == 10.0
    
    assert cleaned.loc[0, 'std_unit'] == 'per_40kg'
    assert cleaned.loc[1, 'std_unit'] == 'per_liter'
    assert cleaned.loc[2, 'std_unit'] == 'per_piece'

@needs_raw_data
def test_flag_forward_dated():
    df = pd.DataFrame({
        'reference_period_end': [
            '2020-01-01', # past
            '2030-01-01'  # future
        ]
    })
    flagged = flag_forward_dated(df)
    assert flagged.loc[0, 'is_forward_dated'] == False
    assert flagged.loc[1, 'is_forward_dated'] == True

@needs_raw_data
def test_clean_prices_runs_on_real_data():
    # Ensure it runs end to end on the real file without crashing
    raw = load_wfp_prices()
    cleaned = clean_prices(raw)
    
    # Row count must be preserved
    assert len(cleaned) == len(raw)
    
    # New columns must exist
    assert 'price_pkr_per_40kg' in cleaned.columns
    assert 'is_forward_dated' in cleaned.columns
    assert 'is_outlier' in cleaned.columns