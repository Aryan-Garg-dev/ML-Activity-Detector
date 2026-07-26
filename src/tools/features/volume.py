"""Vectorized calculation of volume-based transaction features.

Computes rolling counts, sums, mean, and standard deviation of transaction amounts
over 7-day and 30-day windows per account.
"""

from typing import Any
import pandas as pd
import numpy as np


def compute_volume_features(
    df: pd.DataFrame,
    account_ids: list[int] | None = None,
    window_days: int = 30,
) -> dict[int, dict[str, float | int]]:
    """Compute volume feature family for target accounts.

    Returns:
        Dict mapping account_id → feature dict containing:
        txn_count_7d, txn_count_30d, txn_sum_7d, txn_sum_30d, avg_amount, amount_std
    """
    if df.empty:
        return {}

    max_ts = int(df["timestamp"].max())
    ts_7d_cutoff = max_ts - 7
    ts_30d_cutoff = max_ts - window_days

    # Filter to transactions involving the accounts as sender or receiver
    if account_ids is not None:
        acc_set = set(account_ids)
    else:
        acc_set = set(df["sender_account_id"]).union(set(df["receiver_account_id"]))

    # Prepare long format DataFrame matching account to tx
    senders = df[["sender_account_id", "tx_amount", "timestamp"]].rename(
        columns={"sender_account_id": "account_id"}
    )
    receivers = df[["receiver_account_id", "tx_amount", "timestamp"]].rename(
        columns={"receiver_account_id": "account_id"}
    )
    tx_long = pd.concat([senders, receivers], ignore_index=True)
    tx_long = tx_long[tx_long["account_id"].isin(acc_set)]

    if tx_long.empty:
        return {acc: _empty_volume_dict() for acc in acc_set}

    # Group computations
    tx_30d = tx_long[tx_long["timestamp"] >= ts_30d_cutoff]
    tx_7d = tx_long[tx_long["timestamp"] >= ts_7d_cutoff]

    grp_30d = tx_30d.groupby("account_id")["tx_amount"]
    count_30d = grp_30d.count().to_dict()
    sum_30d = grp_30d.sum().to_dict()
    avg_30d = grp_30d.mean().to_dict()
    std_30d = grp_30d.std(ddof=0).fillna(0.0).to_dict()

    grp_7d = tx_7d.groupby("account_id")["tx_amount"]
    count_7d = grp_7d.count().to_dict()
    sum_7d = grp_7d.sum().to_dict()

    results: dict[int, dict[str, float | int]] = {}
    for acc in acc_set:
        results[acc] = {
            "txn_count_7d": int(count_7d.get(acc, 0)),
            "txn_count_30d": int(count_30d.get(acc, 0)),
            "txn_sum_7d": float(sum_7d.get(acc, 0.0)),
            "txn_sum_30d": float(sum_30d.get(acc, 0.0)),
            "avg_amount": float(avg_30d.get(acc, 0.0)),
            "amount_std": float(std_30d.get(acc, 0.0)),
        }

    return results


def _empty_volume_dict() -> dict[str, float | int]:
    return {
        "txn_count_7d": 0,
        "txn_count_30d": 0,
        "txn_sum_7d": 0.0,
        "txn_sum_30d": 0.0,
        "avg_amount": 0.0,
        "amount_std": 0.0,
    }
