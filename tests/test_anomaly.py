import pytest
import pandas as pd
from pipeline.anomaly import detect_anomalies

def test_detect_anomalies():
    # Create a tiny fake dataset with one obvious spike
    df = pd.DataFrame({
        'commodity_name': ['Wheat'] * 5,
        'date': pd.date_range(start='2023-01-01', periods=5, freq='W'),
        'price_pkr_per_40kg': [100, 102, 101, 500, 105], # 500 is a massive spike
        'precipitation_sum': [0, 0, 0, 0, 0]
    })
    
    result = detect_anomalies(df)
    
    # Ensure the anomaly column exists
    assert 'is_anomaly' in result.columns
    assert 'anomaly_reason' in result.columns
    
    # The 500 price (index 3) should be flagged as an anomaly
    assert result.loc[3, 'is_anomaly'] == True
    # The reason should contain "spike"
    assert "spike" in result.loc[3, 'anomaly_reason'].lower()