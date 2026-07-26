"""Tests for the expanded AML detection rules: dormant activation, duplicate transfers, large spikes, multi-destination."""

import pytest
from tools.detection.rules import (
    evaluate_dormant_activation_rules,
    evaluate_duplicate_transfer_rules,
    evaluate_large_spike_rules,
    evaluate_multi_destination_rules,
    evaluate_account_rules,
)
from core.types import PatternType


class TestDormantActivationRules:
    """Tests for dormant account reactivation detection."""

    def test_dormant_reactivation_detected(self):
        features = {
            "dwell_time_avg_hours": 1500.0,  # ~62 days
            "txn_count_7d": 5,
            "txn_count_30d": 6,
        }
        flags = evaluate_dormant_activation_rules(features)
        assert "R_DORMANT_01" in flags

    def test_no_dormant_flag_low_dwell(self):
        features = {
            "dwell_time_avg_hours": 24.0,
            "txn_count_7d": 5,
            "txn_count_30d": 20,
        }
        flags = evaluate_dormant_activation_rules(features)
        assert "R_DORMANT_01" not in flags

    def test_recent_burst_detected(self):
        features = {
            "dwell_time_avg_hours": 10.0,
            "txn_count_7d": 8,
            "txn_count_30d": 9,
        }
        flags = evaluate_dormant_activation_rules(features)
        assert "R_DORMANT_02" in flags

    def test_no_burst_low_7d_count(self):
        features = {
            "dwell_time_avg_hours": 10.0,
            "txn_count_7d": 2,
            "txn_count_30d": 50,
        }
        flags = evaluate_dormant_activation_rules(features)
        assert "R_DORMANT_02" not in flags


class TestDuplicateTransferRules:
    """Tests for duplicate/automated transfer detection."""

    def test_high_round_bias_flagged(self):
        features = {
            "round_number_bias": 0.85,
            "txn_count_30d": 10,
        }
        flags = evaluate_duplicate_transfer_rules(features)
        assert "R_DUPLICATE_01" in flags

    def test_normal_round_bias_not_flagged(self):
        features = {
            "round_number_bias": 0.3,
            "txn_count_30d": 10,
        }
        flags = evaluate_duplicate_transfer_rules(features)
        assert "R_DUPLICATE_01" not in flags

    def test_high_bias_low_count_not_flagged(self):
        features = {
            "round_number_bias": 0.9,
            "txn_count_30d": 2,
        }
        flags = evaluate_duplicate_transfer_rules(features)
        assert "R_DUPLICATE_01" not in flags


class TestLargeSpikeRules:
    """Tests for transaction volume spike detection."""

    def test_7d_concentration_spike_flagged(self):
        features = {
            "txn_sum_7d": 80000.0,
            "txn_sum_30d": 100000.0,
            "avg_amount": 5000.0,
            "amount_std": 2000.0,
        }
        flags = evaluate_large_spike_rules(features)
        assert "R_SPIKE_01" in flags

    def test_no_spike_even_distribution(self):
        features = {
            "txn_sum_7d": 20000.0,
            "txn_sum_30d": 100000.0,
            "avg_amount": 5000.0,
            "amount_std": 1000.0,
        }
        flags = evaluate_large_spike_rules(features)
        assert "R_SPIKE_01" not in flags

    def test_erratic_amounts_flagged(self):
        features = {
            "txn_sum_7d": 5000.0,
            "txn_sum_30d": 50000.0,
            "avg_amount": 1000.0,
            "amount_std": 3000.0,  # std > 2x mean
        }
        flags = evaluate_large_spike_rules(features)
        assert "R_SPIKE_02" in flags

    def test_normal_variance_not_flagged(self):
        features = {
            "txn_sum_7d": 5000.0,
            "txn_sum_30d": 50000.0,
            "avg_amount": 5000.0,
            "amount_std": 1000.0,
        }
        flags = evaluate_large_spike_rules(features)
        assert "R_SPIKE_02" not in flags


class TestMultiDestinationRules:
    """Tests for multi-destination fund distribution detection."""

    def test_many_counterparties_flagged(self):
        features = {
            "distinct_counterparties_30d": 15,
            "fan_out_degree": 10,
        }
        flags = evaluate_multi_destination_rules(features)
        assert "R_MULTI_DEST_01" in flags

    def test_high_fanout_flagged(self):
        features = {
            "distinct_counterparties_30d": 7,
            "fan_out_degree": 9,
        }
        flags = evaluate_multi_destination_rules(features)
        assert "R_MULTI_DEST_02" in flags

    def test_normal_counterparties_not_flagged(self):
        features = {
            "distinct_counterparties_30d": 3,
            "fan_out_degree": 2,
        }
        flags = evaluate_multi_destination_rules(features)
        assert len(flags) == 0


class TestExpandedRulesInAccountEvaluation:
    """Tests that new rules are integrated into evaluate_account_rules."""

    def test_expanded_rules_run_in_evaluate_account(self):
        """Ensure new rules are evaluated even with specific pattern filter."""
        features = {
            "count_near_threshold_30d": 0,
            "round_number_bias": 0.85,
            "txn_count_30d": 10,
            "fan_in_degree": 1,
            "fan_out_degree": 1,
            "distinct_counterparties_30d": 2,
            "velocity_zscore": 0.5,
            "dwell_time_avg_hours": 24.0,
            "in_out_ratio_30d": 1.0,
            "distinct_countries_30d": 1,
            "txn_count_7d": 3,
            "txn_sum_7d": 5000.0,
            "txn_sum_30d": 50000.0,
            "avg_amount": 1000.0,
            "amount_std": 500.0,
        }
        flags = evaluate_account_rules(features, pattern="duplicate")
        # Should include R_DUPLICATE_01 from expanded rules
        assert "R_DUPLICATE_01" in flags

    def test_all_rule_families_evaluated_for_unknown_pattern(self):
        """When pattern is UNKNOWN, all rule families including expanded should run."""
        features = {
            "count_near_threshold_30d": 5,
            "round_number_bias": 0.85,
            "txn_count_30d": 10,
            "fan_in_degree": 1,
            "fan_out_degree": 1,
            "distinct_counterparties_30d": 2,
            "velocity_zscore": 0.5,
            "dwell_time_avg_hours": 24.0,
            "in_out_ratio_30d": 1.0,
            "distinct_countries_30d": 1,
            "txn_count_7d": 3,
            "txn_sum_7d": 5000.0,
            "txn_sum_30d": 50000.0,
            "avg_amount": 1000.0,
            "amount_std": 500.0,
        }
        flags = evaluate_account_rules(features, pattern=PatternType.UNKNOWN)
        # At least structuring and duplicate should fire
        assert "R_STRUCT_01" in flags
        assert "R_DUPLICATE_01" in flags
