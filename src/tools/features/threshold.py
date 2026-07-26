"""Structuring and threshold-avoidance feature calculations.

Identifies transactions sitting just below mandatory reporting thresholds (e.g. $8,500 - $9,900)
and measures round-number transaction frequency indicative of manual structuring.
"""

from typing import Any
import pandas as pd


def compute_threshold_features(
    df: pd.DataFrame,
    account_ids: list[int] | None = None,
    reporting_threshold: float = 10000.0,
    window_days: int = 30,
) -> dict[int, dict[str, float | int]]:
    """Compute threshold avoidance feature family for target accounts.

    Returns:
        Dict mapping account_id → feature dict containing:
        count_near_threshold_30d, round_number_bias
    """
    if df.empty:
        return {}

    max_ts = int(df["timestamp"].max())
    ts_cutoff = max_ts - window_days

    lower_bound = 0.85 * reporting_threshold
    upper_bound = 0.99 * reporting_threshold

    if account_ids is not None:
        acc_set = set(account_ids)
    else:
        acc_set = set(df["sender_account_id"]).union(set(df["receiver_account_id"]))

    senders = df[["sender_account_id", "tx_amount", "timestamp"]].rename(
        columns={"sender_account_id": "account_id"}
    )
    receivers = df[["receiver_account_id", "tx_amount", "timestamp"]].rename(
        columns={"receiver_account_id": "account_id"}
    )
    tx_long = pd.concat([senders, receivers], ignore_index=True)
    tx_long = tx_long[(tx_long["timestamp"] >= ts_cutoff) & (tx_long["account_id"].isin(acc_set))]

    if tx_long.empty:
        return {acc: _empty_threshold_dict() for acc in acc_set}

    # Flag near-threshold transactions
    tx_long["is_near_threshold"] = (tx_long["tx_amount"] >= lower_bound) & (tx_long["tx_amount"] <= upper_bound)
    # Flag round numbers (e.g. divisible by 100 or 1000 with zero cents)
    tx_long["is_round"] = (tx_long["tx_amount"] % 100 == 0) & (tx_long["tx_amount"] > 0)

    near_counts = tx_long.groupby("account_id")["is_near_threshold"].sum().to_dict()
    total_counts = tx_long.groupby("account_id")["tx_amount"].count().to_dict()
    round_counts = tx_long.groupby("account_id")["is_round"].sum().to_dict()

    results: dict[int, dict[str, float | int]] = {}
    for acc in acc_set:
        total = total_counts.get(acc, 0)
        round_cnt = round_counts.get(acc, 0)
        bias = float(round_cnt / total) if total > 0 else 0.0

        results[acc] = {
            "count_near_threshold_30d": int(near_counts.get(acc, 0)),
            "round_number_bias": float(round(bias, 4)),
        }

    return results


def _empty_threshold_dict() -> dict[str, float | int]:
    return {
        "count_near_threshold_30d": 0,
        "round_number_bias": 0.0,
    }
