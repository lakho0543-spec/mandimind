import pytest
import pandas as pd
from pathlib import Path
from pipeline.model import train_all_commodities, load_and_prepare_data

# Check if the data file exists
DATA_PATH = Path("data/processed/analysis_table.parquet")

@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Analysis table missing — need to run pipeline first"
)
def test_model_trains_and_returns_metrics():
    df = load_and_prepare_data()
    
    # Get just one commodity to test quickly (faster than training all 17)
    sample_crop = df['commodity_name'].unique()[0]
    sample_df = df[df['commodity_name'] == sample_crop]
    
    # This returns a dict of results
    results = train_all_commodities(sample_df)
    
    # Ensure the result has the right keys
    assert sample_crop in results
    assert results[sample_crop]['improvement_pct'] is not None
    
    # Check if the model saved (it should, unless skipped)
    if not results[sample_crop].get('skipped'):
        assert 'model' in results[sample_crop]