import pytest
import pandas as pd
from pipeline.cluster import build_district_features, cluster_districts

def test_build_district_features():
    # 15 rows per district — enough to pass the > 10 filter (matches real data)
    df = pd.DataFrame({
        'admin2_name': ['Lahore']*15 + ['Quetta']*15,
        'price_pkr_per_40kg': [100]*5 + [110]*5 + [120]*5 + [50]*5 + [60]*5 + [70]*5
    })
    
    profiles = build_district_features(df)
    assert len(profiles) == 2, f"Expected 2 districts, got {len(profiles)}"
    assert 'avg_price' in profiles.columns
    assert 'price_volatility' in profiles.columns
    assert profiles.loc[profiles['admin2_name'] == 'Lahore', 'avg_price'].values[0] == 110.0

def test_cluster_districts():
    profiles = pd.DataFrame({
        'admin2_name': ['A', 'B', 'C', 'D', 'E', 'F'],
        'avg_price': [100, 105, 110, 500, 510, 520],
        'price_volatility': [10, 12, 11, 100, 105, 110],
        'total_records': [50, 50, 50, 50, 50, 50]
    })
    
    clustered = cluster_districts(profiles, n_clusters=2)
    assert 'cluster_id' in clustered.columns
    assert 'cluster_name' in clustered.columns
    # We expect 2 clusters
    assert len(clustered['cluster_id'].unique()) == 2