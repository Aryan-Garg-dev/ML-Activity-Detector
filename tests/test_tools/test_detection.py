"""Phase 5: Detection tool tests.

Tests for:
- ML scorer (IForest + LOF + HBOS): output format, per-detector flags, edge cases
- Rule engine: signal emission per pattern type
- HBOS scorer: standalone unit tests
- Ensemble combiner: OR-gate logic, signal deduplication, confidence computation
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from core.config import AppConfig
from tools.detection.ensemble import OrGateStrategy
from tools.detection.hbos_scorer import fit_predict_hbos_anomalies
from tools.detection.ml_scorer import fit_predict_ml_anomalies


def _make_config(**kwargs) -> AppConfig:
    """Build a minimal test AppConfig with safe defaults."""
    defaults = {
        "ml_contamination": 0.05,
        "ml_percentile_cutoff": 95.0,
        "ml_random_seed": 42,
        "reporting_threshold": 10000.0,
    }
    defaults.update(kwargs)
    return AppConfig(**defaults)


def _dummy_feature_dict(n: int) -> dict[int, dict]:
    """Generate n dummy account feature dicts for ML testing."""
    import random
    random.seed(42)
    result = {}
    for i in range(1, n + 1):
        result[i] = {
            "txn_count_30d": random.randint(1, 100),
            "txn_sum_30d": random.uniform(100, 500000),
            "txn_count_7d": random.randint(0, 30),
            "avg_amount": random.uniform(50, 20000),
            "velocity_zscore": random.uniform(-2, 10),
            "fan_in_degree": random.randint(1, 30),
            "fan_out_degree": random.randint(1, 30),
        }
    return result


class TestMLScorer:
    config = _make_config()

    def test_output_keys_present(self):
        features = _dummy_feature_dict(30)
        results = fit_predict_ml_anomalies(features, self.config)
        assert len(results) == 30
        sample = next(iter(results.values()))
        assert "ml_anomaly_score" in sample
        assert "is_ml_anomalous" in sample
        assert "iforest_flagged" in sample
        assert "lof_flagged" in sample
        assert "hbos_flagged" in sample

    def test_scores_in_valid_range(self):
        features = _dummy_feature_dict(50)
        results = fit_predict_ml_anomalies(features, self.config)
        for acc, r in results.items():
            assert 0.0 <= r["ml_anomaly_score"] <= 1.0, f"score out of range for {acc}"

    def test_empty_input_returns_empty(self):
        results = fit_predict_ml_anomalies({}, self.config)
        assert results == {}

    def test_single_account_returns_result(self):
        """Small batch (< 20) should still work via baseline augmentation."""
        features = _dummy_feature_dict(1)
        results = fit_predict_ml_anomalies(features, self.config)
        assert 1 in results
        assert "ml_anomaly_score" in results[1]

    def test_per_detector_flags_are_bool(self):
        features = _dummy_feature_dict(25)
        results = fit_predict_ml_anomalies(features, self.config)
        for r in results.values():
            assert isinstance(r["iforest_flagged"], bool)
            assert isinstance(r["lof_flagged"], bool)
            assert isinstance(r["hbos_flagged"], bool)


class TestHBOSScorer:
    config = _make_config()

    def test_output_keys_present(self):
        features = _dummy_feature_dict(30)
        results = fit_predict_hbos_anomalies(features, self.config)
        assert len(results) == 30
        sample = next(iter(results.values()))
        assert "hbos_anomaly_score" in sample
        assert "is_hbos_anomalous" in sample
        assert "hbos_raw_score" in sample

    def test_scores_in_valid_range(self):
        features = _dummy_feature_dict(40)
        results = fit_predict_hbos_anomalies(features, self.config)
        for acc, r in results.items():
            assert 0.0 <= r["hbos_anomaly_score"] <= 1.0

    def test_empty_input_returns_empty(self):
        assert fit_predict_hbos_anomalies({}, self.config) == {}

    def test_small_batch_augmented(self):
        """1 account should still produce a result (augmented with baseline)."""
        features = {1: {"txn_count_30d": 5, "avg_amount": 500.0}}
        results = fit_predict_hbos_anomalies(features, self.config)
        assert 1 in results

    def test_empty_feature_dict_in_batch(self):
        """Accounts with empty feature dicts should get zero-score results."""
        features = {1: {}}
        results = fit_predict_hbos_anomalies(features, self.config)
        assert results[1]["hbos_anomaly_score"] == 0.0


class TestEnsembleCombiner:
    config = _make_config()
    strategy = OrGateStrategy()

    def test_or_gate_with_rule_only(self):
        rule_results = {1: ["R_STRUCT_01"]}
        ml_results = {1: {"ml_anomaly_score": 0.0, "is_ml_anomalous": False, "iforest_flagged": False, "lof_flagged": False, "hbos_flagged": False}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert results[1]["is_flagged"] is True
        assert "R_STRUCT_01" in results[1]["triggered_signals"]

    def test_or_gate_with_ml_only(self):
        rule_results = {2: []}
        ml_results = {2: {"ml_anomaly_score": 0.95, "is_ml_anomalous": True, "iforest_flagged": True, "lof_flagged": False, "hbos_flagged": True}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert results[2]["is_flagged"] is True
        assert "ML_IFOREST" in results[2]["triggered_signals"]
        assert "ML_HBOS" in results[2]["triggered_signals"]
        assert "ML_ANOMALY" in results[2]["triggered_signals"]

    def test_not_flagged_when_all_clean(self):
        rule_results = {3: []}
        ml_results = {3: {"ml_anomaly_score": 0.02, "is_ml_anomalous": False, "iforest_flagged": False, "lof_flagged": False, "hbos_flagged": False}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert results[3]["is_flagged"] is False
        assert results[3]["confidence"] == 0.0

    def test_signal_deduplication(self):
        rule_results = {4: ["R_STRUCT_01", "R_STRUCT_02"]}
        ml_results = {4: {"ml_anomaly_score": 0.95, "is_ml_anomalous": True, "iforest_flagged": True, "lof_flagged": True, "hbos_flagged": True}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        # ML_ANOMALY appears once, not multiple times
        signals = results[4]["triggered_signals"]
        assert signals.count("ML_ANOMALY") == 1

    def test_accounts_in_rule_only(self):
        """Accounts appearing only in rule_results should still be in output."""
        rule_results = {5: ["R_VELOCITY_01"]}
        ml_results = {}  # account 5 not in ml_results
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert 5 in results
        assert results[5]["is_flagged"] is True

    def test_accounts_in_ml_only(self):
        """Accounts appearing only in ml_results should still be in output."""
        rule_results = {}
        ml_results = {6: {"ml_anomaly_score": 0.9, "is_ml_anomalous": True, "iforest_flagged": True, "lof_flagged": False, "hbos_flagged": False}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert 6 in results
        assert results[6]["is_flagged"] is True

    def test_confidence_bounded_to_1(self):
        rule_results = {7: ["R_STRUCT_01", "R_SMURF_01", "R_LAYER_01", "R_CASHOUT_01", "R_VELOCITY_01"]}
        ml_results = {7: {"ml_anomaly_score": 0.99, "is_ml_anomalous": True, "iforest_flagged": True, "lof_flagged": True, "hbos_flagged": True}}
        results = self.strategy.combine_signals(rule_results, ml_results, self.config)
        assert results[7]["confidence"] <= 1.0
