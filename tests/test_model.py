import pytest
import pandas as pd
from pipeline.model import train_and_evaluate, load_and_prepare_data

@pytest.mark.skipif(
    not pd.io.common.file_exists("data/processed/analysis_table.parquet"),
    reason="Analysis table missing"
)
def test_model_trains_and_returns_metrics():
    df = load_and_prepare_data()
    model, metrics = train_and_evaluate(df)
    
    assert 'mae_xgb' in metrics
    assert 'mae_naive' in metrics
    assert metrics['mae_xgb'] > 0
    assert model is not None