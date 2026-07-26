"""Edge case tests for robustness: empty datasets, unknown IDs, boundary values."""

import pytest
from tools.features.volume import compute_volume_features
from tools.features.threshold import compute_threshold_features
from tools.features.network import compute_network_features
from tools.features.velocity import compute_velocity_features
from tools.explanation.templates import format_grounded_explanation, verify_numeric_preservation
from tools.detection.rules import evaluate_account_rules
from core.types import PatternType
import pandas as pd


class TestEmptyDatasetHandling:
    """Feature functions must handle empty DataFrames gracefully."""

    def test_volume_features_empty_df(self):
        df = pd.DataFrame(columns=["sender_account_id", "receiver_account_id", "tx_amount", "timestamp"])
        result = compute_volume_features(df)
        assert result == {}

    def test_threshold_features_empty_df(self):
        df = pd.DataFrame(columns=["sender_account_id", "receiver_account_id", "tx_amount", "timestamp"])
        result = compute_threshold_features(df)
        assert result == {}

    def test_network_features_empty_df(self):
        df = pd.DataFrame(columns=["sender_account_id", "receiver_account_id", "tx_amount", "timestamp"])
        result = compute_network_features(df)
        assert result == {}

    def test_velocity_features_empty_df(self):
        df = pd.DataFrame(columns=["sender_account_id", "receiver_account_id", "tx_amount", "timestamp"])
        result = compute_velocity_features(df)
        assert result == {}


class TestSingleRowDataset:
    """Feature functions must handle single-row DataFrames."""

    def _single_row_df(self):
        return pd.DataFrame({
            "sender_account_id": [100],
            "receiver_account_id": [200],
            "tx_amount": [5000.0],
            "timestamp": [10],
        })

    def test_volume_features_single_row(self):
        df = self._single_row_df()
        result = compute_volume_features(df, account_ids=[100])
        assert 100 in result
        assert result[100]["txn_count_30d"] >= 1

    def test_threshold_features_single_row(self):
        df = self._single_row_df()
        result = compute_threshold_features(df, account_ids=[100])
        assert 100 in result

    def test_network_features_single_row(self):
        df = self._single_row_df()
        result = compute_network_features(df, account_ids=[100])
        assert 100 in result
        assert result[100]["fan_out_degree"] == 1

    def test_velocity_features_single_row(self):
        df = self._single_row_df()
        result = compute_velocity_features(df, account_ids=[100])
        assert 100 in result
        assert result[100]["dwell_time_avg_hours"] == 0.0


class TestUnknownAccountHandling:
    """Feature functions must handle unknown account IDs gracefully."""

    def _sample_df(self):
        return pd.DataFrame({
            "sender_account_id": [100, 100, 200],
            "receiver_account_id": [200, 300, 100],
            "tx_amount": [5000.0, 8000.0, 3000.0],
            "timestamp": [10, 15, 20],
        })

    def test_volume_unknown_account(self):
        df = self._sample_df()
        result = compute_volume_features(df, account_ids=[9999])
        assert 9999 in result
        assert result[9999]["txn_count_30d"] == 0

    def test_network_unknown_account(self):
        df = self._sample_df()
        result = compute_network_features(df, account_ids=[9999])
        assert 9999 in result
        assert result[9999]["fan_in_degree"] == 0


class TestBoundaryValues:
    """Tests for boundary and edge-case values in features and rules."""

    def test_zero_amount_transactions(self):
        df = pd.DataFrame({
            "sender_account_id": [100, 100],
            "receiver_account_id": [200, 300],
            "tx_amount": [0.0, 0.0],
            "timestamp": [10, 11],
        })
        result = compute_volume_features(df, account_ids=[100])
        assert result[100]["avg_amount"] == 0.0

    def test_rules_with_all_zero_features(self):
        features = {
            "count_near_threshold_30d": 0,
            "round_number_bias": 0.0,
            "fan_in_degree": 0,
            "fan_out_degree": 0,
            "distinct_counterparties_30d": 0,
            "velocity_zscore": 0.0,
            "dwell_time_avg_hours": 0.0,
            "in_out_ratio_30d": 0.0,
            "distinct_countries_30d": 1,
            "txn_count_7d": 0,
            "txn_count_30d": 0,
            "txn_sum_7d": 0.0,
            "txn_sum_30d": 0.0,
            "avg_amount": 0.0,
            "amount_std": 0.0,
        }
        flags = evaluate_account_rules(features, pattern=PatternType.UNKNOWN)
        assert len(flags) == 0  # No rules should fire with all zeros

    def test_rules_with_missing_features(self):
        """Rules must handle missing feature keys gracefully via .get() defaults."""
        features = {}
        # Should not crash even with empty features
        flags = evaluate_account_rules(features, pattern=PatternType.UNKNOWN)
        assert isinstance(flags, list)


class TestExplanationEdgeCases:
    """Tests for explanation generation edge cases."""

    def test_explanation_with_no_signals(self):
        text = format_grounded_explanation(
            entity_id="999",
            risk_info={"composite_score": 10.0, "risk_level": "low", "triggered_signals": []},
            feature_info={},
        )
        assert "999" in text
        assert "LOW" in text

    def test_explanation_with_unknown_signals(self):
        text = format_grounded_explanation(
            entity_id="123",
            risk_info={"composite_score": 50.0, "risk_level": "medium", "triggered_signals": ["R_UNKNOWN_99"]},
            feature_info={},
        )
        assert "123" in text
        assert "R_UNKNOWN_99" in text

    def test_numeric_preservation_exact_match(self):
        grounded = "Account 123 had 5 transactions totaling $50,000."
        candidate = "Account 123 had 5 transactions totaling $50,000."
        assert verify_numeric_preservation(grounded, candidate) is True

    def test_numeric_preservation_detects_drift(self):
        grounded = "Account 123 had 5 transactions totaling $50,000."
        candidate = "Account 123 had 6 transactions totaling $50,000."
        assert verify_numeric_preservation(grounded, candidate) is False
