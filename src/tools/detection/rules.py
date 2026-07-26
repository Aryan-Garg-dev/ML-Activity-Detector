"""Deterministic rule detectors for AML suspicious activity patterns.

Provides rule functions for structuring, smurfing, layering, rapid cash-out,
and velocity patterns based on computed domain features.
"""

from typing import Any
from core.types import PatternType


def evaluate_structuring_rules(
    features: dict[str, Any],
    min_near_threshold_count: int = 3,
) -> list[str]:
    """Evaluate structuring / threshold avoidance rules."""
    flags = []
    near_count = int(features.get("count_near_threshold_30d", 0))
    round_bias = float(features.get("round_number_bias", 0.0))

    if near_count >= min_near_threshold_count:
        flags.append("R_STRUCT_01")  # High volume near reporting threshold
    if round_bias >= 0.5 and near_count > 0:
        flags.append("R_STRUCT_02")  # High round-number bias

    return flags


def evaluate_smurfing_rules(features: dict[str, Any]) -> list[str]:
    """Evaluate smurfing / fan-in fan-out network aggregation rules."""
    flags = []
    fan_in = int(features.get("fan_in_degree", 0))
    fan_out = int(features.get("fan_out_degree", 0))

    if fan_in >= 5:
        flags.append("R_SMURF_01")  # Fan-in gathering point
    if fan_out >= 5:
        flags.append("R_SMURF_02")  # Fan-out distribution point

    return flags


def evaluate_layering_rules(features: dict[str, Any]) -> list[str]:
    """Evaluate layering / rapid pass-through network rules."""
    flags = []
    counterparties = int(features.get("distinct_counterparties_30d", 0))
    ratio = float(features.get("in_out_ratio_30d", 0.0))
    countries = int(features.get("distinct_countries_30d", 1))

    if counterparties >= 5 and 0.8 <= ratio <= 1.2:
        flags.append("R_LAYER_01")  # Multi-counterparty flow-through
    if countries > 1:
        flags.append("R_LAYER_02")  # Cross-jurisdiction movement

    return flags


def evaluate_rapid_cashout_rules(features: dict[str, Any]) -> list[str]:
    """Evaluate rapid cash-out / short holding period rules."""
    flags = []
    dwell_time = float(features.get("dwell_time_avg_hours", 0.0))
    ratio = float(features.get("in_out_ratio_30d", 0.0))
    cnt = int(features.get("txn_count_30d", 0))

    if cnt >= 2 and dwell_time > 0 and dwell_time <= 24.0 and ratio >= 0.9:
        flags.append("R_CASHOUT_01")  # Funds cashed out within 24 hours

    return flags


def evaluate_velocity_rules(features: dict[str, Any]) -> list[str]:
    """Evaluate transaction velocity spike rules."""
    flags = []
    zscore = float(features.get("velocity_zscore", 0.0))

    if zscore >= 2.0:
        flags.append("R_VELOCITY_01")  # Transaction velocity anomaly

    return flags


def evaluate_dormant_activation_rules(
    features: dict[str, Any],
    dormant_days_threshold: int = 60,
) -> list[str]:
    """Detect dormant account reactivation — accounts with no activity for dormant_days_threshold+
    that suddenly resume transacting."""
    flags = []
    dwell_hours = float(features.get("dwell_time_avg_hours", 0.0))
    txn_count_7d = int(features.get("txn_count_7d", 0))
    txn_count_30d = int(features.get("txn_count_30d", 0))

    # High dwell time with sudden recent burst suggests reactivation
    if dwell_hours > dormant_days_threshold * 24 and txn_count_7d >= 3:
        flags.append("R_DORMANT_01")

    # Very low 30d activity but high 7d burst
    if txn_count_30d > 0 and txn_count_7d > 0:
        recent_ratio = txn_count_7d / txn_count_30d
        if recent_ratio > 0.8 and txn_count_7d >= 5:
            flags.append("R_DORMANT_02")

    return flags


def evaluate_duplicate_transfer_rules(
    features: dict[str, Any],
) -> list[str]:
    """Detect repeated exact-amount transfers that may indicate automated layering."""
    flags = []
    round_bias = float(features.get("round_number_bias", 0.0))
    txn_count_30d = int(features.get("txn_count_30d", 0))

    # High round number bias with many transactions suggests automated duplicate transfers
    if round_bias > 0.7 and txn_count_30d >= 5:
        flags.append("R_DUPLICATE_01")

    return flags


def evaluate_large_spike_rules(
    features: dict[str, Any],
) -> list[str]:
    """Detect sudden large transaction amounts compared to historical baseline."""
    flags = []
    avg_amount = float(features.get("avg_amount", 0.0))
    amount_std = float(features.get("amount_std", 0.0))
    txn_sum_7d = float(features.get("txn_sum_7d", 0.0))
    txn_sum_30d = float(features.get("txn_sum_30d", 0.0))

    # 7-day volume spike: more than 60% of 30-day total concentrated in last 7 days
    if txn_sum_30d > 0 and txn_sum_7d > 0:
        concentration = txn_sum_7d / txn_sum_30d
        if concentration > 0.6 and txn_sum_7d > 10000:
            flags.append("R_SPIKE_01")

    # High standard deviation relative to mean indicates erratic amounts
    if avg_amount > 0 and amount_std > avg_amount * 2:
        flags.append("R_SPIKE_02")

    return flags


def evaluate_multi_destination_rules(
    features: dict[str, Any],
    max_counterparties: int = 10,
) -> list[str]:
    """Detect accounts distributing funds to an unusually high number of unique recipients."""
    flags = []
    distinct_counterparties = int(features.get("distinct_counterparties_30d", 0))
    fan_out = int(features.get("fan_out_degree", 0))

    # Many unique counterparties in short window
    if distinct_counterparties >= max_counterparties:
        flags.append("R_MULTI_DEST_01")

    # Very high fan-out with moderate counterparties
    if fan_out >= 8 and distinct_counterparties >= 5:
        flags.append("R_MULTI_DEST_02")

    return flags


def evaluate_account_rules(
    features: dict[str, Any],
    pattern: PatternType | str = PatternType.UNKNOWN,
    min_near_threshold_count: int = 3,
) -> list[str]:
    """Evaluate all applicable rule detectors for an account feature set.

    If pattern is UNKNOWN, evaluates all rule families.
    Otherwise, evaluates rules relevant to the specific pattern.
    """
    pattern_str = str(pattern).lower()
    all_flags = []

    if pattern_str in (PatternType.STRUCTURING, "structuring", PatternType.UNKNOWN):
        all_flags.extend(evaluate_structuring_rules(features, min_near_threshold_count))
    if pattern_str in (PatternType.SMURFING, "smurfing", PatternType.UNKNOWN):
        all_flags.extend(evaluate_smurfing_rules(features))
    if pattern_str in (PatternType.LAYERING, "layering", PatternType.UNKNOWN):
        all_flags.extend(evaluate_layering_rules(features))
    if pattern_str in (PatternType.RAPID_CASHOUT, "rapid_cashout", PatternType.UNKNOWN):
        all_flags.extend(evaluate_rapid_cashout_rules(features))
    if pattern_str in (PatternType.VELOCITY, "velocity", PatternType.UNKNOWN):
        all_flags.extend(evaluate_velocity_rules(features))

    # Evaluate expanded rules when targeted or when running unknown/comprehensive detection
    if pattern_str in ("dormant", "dormant_activation", PatternType.UNKNOWN) or pattern_str not in ("duplicate", "spike", "multi_destination", "structuring", "smurfing", "layering", "rapid_cashout", "velocity"):
        all_flags.extend(evaluate_dormant_activation_rules(features))
    if pattern_str in ("duplicate", "duplicate_transfer", PatternType.UNKNOWN) or pattern_str not in ("dormant", "spike", "multi_destination", "structuring", "smurfing", "layering", "rapid_cashout", "velocity"):
        all_flags.extend(evaluate_duplicate_transfer_rules(features))
    if pattern_str in ("spike", "large_spike", "volume_spike", PatternType.UNKNOWN) or pattern_str not in ("dormant", "duplicate", "multi_destination", "structuring", "smurfing", "layering", "rapid_cashout", "velocity"):
        all_flags.extend(evaluate_large_spike_rules(features))
    if pattern_str in ("multi_destination", "multi_dest", PatternType.UNKNOWN) or pattern_str not in ("dormant", "duplicate", "spike", "structuring", "smurfing", "layering", "rapid_cashout", "velocity"):
        all_flags.extend(evaluate_multi_destination_rules(features))

    return list(dict.fromkeys(all_flags))  # Deduplicate keeping order
