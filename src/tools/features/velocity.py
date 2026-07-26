"""Velocity and rapid cash-out feature calculations.

Computes MAD-based transaction velocity z-score, average dwell/holding time,
in-out volume ratio, and cross-border geographic diversity.
"""

from typing import Any
import pandas as pd
import numpy as np


def compute_velocity_features(
    df: pd.DataFrame,
    accounts_df: pd.DataFrame | None = None,
    account_ids: list[int] | None = None,
    window_days: int = 30,
) -> dict[int, dict[str, float | int]]:
    """Compute velocity and rapid cash-out feature family for target accounts.

    Returns:
        Dict mapping account_id → feature dict containing:
        velocity_zscore, dwell_time_avg_hours, in_out_ratio_30d, distinct_countries_30d
    """
    if df.empty:
        return {}

    max_ts = int(df["timestamp"].max())
    ts_cutoff = max_ts - window_days

    tx_30d = df[df["timestamp"] >= ts_cutoff]

    if account_ids is not None:
        acc_set = set(account_ids)
    else:
        acc_set = set(df["sender_account_id"]).union(set(df["receiver_account_id"]))

    if tx_30d.empty:
        return {acc: _empty_velocity_dict() for acc in acc_set}

    # 1. Population transaction counts for velocity z-score (MAD-based)
    sent_counts = tx_30d.groupby("sender_account_id").size()
    recv_counts = tx_30d.groupby("receiver_account_id").size()
    all_counts = sent_counts.add(recv_counts, fill_value=0)

    count_series = pd.Series([all_counts.get(acc, 0) for acc in acc_set], index=list(acc_set))
    med = float(count_series.median())
    mad = float((count_series - med).abs().median())
    
    if mad > 0:
        z_scores = (0.6745 * (count_series - med) / mad).to_dict()
    else:
        # Fallback to standard z-score or 0.0 if std is 0
        std = float(count_series.std())
        if std > 0:
            z_scores = ((count_series - med) / std).to_dict()
        else:
            z_scores = {acc: 0.0 for acc in acc_set}

    # 2. In/Out volume ratio per account
    in_sums = tx_30d.groupby("receiver_account_id")["tx_amount"].sum().to_dict()
    out_sums = tx_30d.groupby("sender_account_id")["tx_amount"].sum().to_dict()

    # 3. Average inter-transaction interval (dwell time approximation in hours)
    # Timestamps are integer days; 1 day = 24 hours
    dwell_times: dict[int, float] = {}
    senders = tx_30d[["sender_account_id", "timestamp"]].rename(columns={"sender_account_id": "account_id"})
    receivers = tx_30d[["receiver_account_id", "timestamp"]].rename(columns={"receiver_account_id": "account_id"})
    tx_events = pd.concat([senders, receivers], ignore_index=True).sort_values("timestamp")
    
    for acc in acc_set:
        acc_ts = tx_events[tx_events["account_id"] == acc]["timestamp"].values
        if len(acc_ts) > 1:
            diffs = np.diff(acc_ts)
            avg_days = float(np.mean(diffs))
            dwell_times[acc] = float(avg_days * 24.0)
        else:
            dwell_times[acc] = 0.0

    # 4. Distinct countries (from accounts_df if available)
    acc_country_map = {}
    if accounts_df is not None and not accounts_df.empty:
        acc_country_map = accounts_df.set_index("account_id")["country"].to_dict()

    tx_with_countries = tx_30d.copy()
    tx_with_countries["sender_country"] = tx_with_countries["sender_account_id"].map(acc_country_map).fillna("US")
    tx_with_countries["receiver_country"] = tx_with_countries["receiver_account_id"].map(acc_country_map).fillna("US")

    distinct_countries: dict[int, int] = {}
    for acc in acc_set:
        s_countries = set(tx_with_countries[tx_with_countries["sender_account_id"] == acc]["receiver_country"])
        r_countries = set(tx_with_countries[tx_with_countries["receiver_account_id"] == acc]["sender_country"])
        own_country = {acc_country_map.get(acc, "US")}
        distinct_countries[acc] = len(s_countries.union(r_countries).union(own_country))

    results: dict[int, dict[str, float | int]] = {}
    for acc in acc_set:
        in_amt = float(in_sums.get(acc, 0.0))
        out_amt = float(out_sums.get(acc, 0.0))
        ratio = float(in_amt / out_amt) if out_amt > 0 else (float(in_amt) if in_amt > 0 else 0.0)

        results[acc] = {
            "velocity_zscore": float(round(z_scores.get(acc, 0.0), 4)),
            "dwell_time_avg_hours": float(round(dwell_times.get(acc, 0.0), 2)),
            "in_out_ratio_30d": float(round(ratio, 4)),
            "distinct_countries_30d": int(distinct_countries.get(acc, 1)),
        }

    return results


def _empty_velocity_dict() -> dict[str, float | int]:
    return {
        "velocity_zscore": 0.0,
        "dwell_time_avg_hours": 0.0,
        "in_out_ratio_30d": 0.0,
        "distinct_countries_30d": 1,
    }
