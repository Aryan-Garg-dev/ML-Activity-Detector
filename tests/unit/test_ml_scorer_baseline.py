"""Unit tests for ML scorer baseline profile generation in small batches."""

import pytest
from core.config import AppConfig
from tools.detection.ml_scorer import fit_predict_ml_anomalies


def test_single_entity_scoring_not_inflated_by_zeros():
    """Test that a single account with moderate transaction volume does not score near 1.0."""
    config = AppConfig()
    
    # Target account with moderate, normal activity (5 txns, $1500 volume)
    target_features = {
        101: {
            "txn_count_7d": 2,
            "txn_count_30d": 5,
            "txn_sum_7d": 500.0,
            "txn_sum_30d": 1500.0,
            "avg_amount": 300.0,
            "amount_std": 50.0,
            "count_near_threshold_30d": 0,
            "round_number_bias": 0.2,
            "distinct_counterparties_30d": 2,
            "fan_in_degree": 1,
            "fan_out_degree": 1,
            "velocity_zscore": 0.1,
            "dwell_time_avg_hours": 12.0,
            "in_out_ratio_30d": 1.0,
            "distinct_countries_30d": 1,
        }
    }

    # Score without explicit background features (should use our new non-zero median profiles)
    results = fit_predict_ml_anomalies(target_features, config, background_features=None)
    
    assert 101 in results
    res = results[101]
    # An ordinary account should not be classified as an extreme outlier by the ensemble models
    assert res["ml_anomaly_score"] < 0.60
    assert res["is_ml_anomalous"] is False
