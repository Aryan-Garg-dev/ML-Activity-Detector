"""Graph-based transaction network feature calculations using NetworkX.

Constructs a directed graph of transactions over a 30-day window to derive
topological money laundering indicators: fan-in (smurfing gathering point),
fan-out (smurfing distribution point), and distinct counterparty counts.
"""

from typing import Any
import pandas as pd
import networkx as nx


def compute_network_features(
    df: pd.DataFrame,
    account_ids: list[int] | None = None,
    window_days: int = 30,
) -> dict[int, dict[str, int]]:
    """Compute NetworkX directed graph features for target accounts.

    Returns:
        Dict mapping account_id → feature dict containing:
        distinct_counterparties_30d, fan_in_degree, fan_out_degree
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
        return {acc: _empty_network_dict() for acc in acc_set}

    # Build directed graph from sender to receiver
    G = nx.DiGraph()
    for _, row in tx_30d.iterrows():
        src = int(row["sender_account_id"])
        dst = int(row["receiver_account_id"])
        G.add_edge(src, dst)

    results: dict[int, dict[str, int]] = {}
    for acc in acc_set:
        if not G.has_node(acc):
            results[acc] = _empty_network_dict()
            continue

        in_deg = G.in_degree(acc)
        out_deg = G.out_degree(acc)
        in_neighbors = set(G.predecessors(acc))
        out_neighbors = set(G.successors(acc))
        distinct_counterparties = len(in_neighbors.union(out_neighbors))

        results[acc] = {
            "distinct_counterparties_30d": distinct_counterparties,
            "fan_in_degree": in_deg,
            "fan_out_degree": out_deg,
        }

    return results


def _empty_network_dict() -> dict[str, int]:
    return {
        "distinct_counterparties_30d": 0,
        "fan_in_degree": 0,
        "fan_out_degree": 0,
    }
