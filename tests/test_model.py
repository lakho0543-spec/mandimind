import pytest
import pandas as pd
from pathlib import Path
from pipeline.model import train_and_evaluate, load_and_prepare_data

# Check if the data file exists
DATA_PATH = Path("data/processed/analysis_table.parquet")

@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Analysis table missing — need to run pipeline first"
)
def test_model_trains_and_returns_metrics():
    df = load_and_prepare_data()
    model, metrics = train_and_evaluate(df)
    
    assert 'mae_xgb' in metrics
    assert 'mae_naive' in metrics
    assert metrics['mae_xgb'] > 0
    assert model is not None