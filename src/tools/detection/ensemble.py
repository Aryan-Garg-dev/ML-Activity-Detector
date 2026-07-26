"""OR-gate ensemble combiner for hybrid AML detection.

Merges rule detection flags and PyOD ML anomaly scores (IForest, LOF, HBOS) using
an OR-gate logic. Collects unified triggered signals including individual ML detector
signals (ML_IFOREST, ML_LOF, ML_HBOS) so explanations can cite specific methods.
"""

from typing import Any
from core.config import AppConfig


def combine_signals(
    rule_results: dict[int, list[str]],
    ml_results: dict[int, dict[str, Any]],
    config: AppConfig,
    pattern_type: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Combine rule detection flags and ML anomaly scores via OR-gate ensemble.

    Args:
        rule_results: Dict mapping account_id → list of triggered rule IDs
        ml_results: Dict mapping account_id → ML anomaly dict with per-detector flags
        config: Application runtime configuration
        pattern_type: Pattern filter executed (e.g. 'structuring', 'smurfing', 'unknown')

    Returns:
        Dict mapping account_id → ensemble detection summary dict:
            is_flagged: bool (OR-gate: rules triggered OR any ML detector >= threshold)
            triggered_signals: list[str] (e.g. ["R_STRUCT_01", "ML_IFOREST", "ML_HBOS"])
            rule_flags: list[str]
            ml_anomaly_score: float
            confidence: float (0.0 to 1.0)
    """
    all_acc_ids = set(rule_results.keys()).union(set(ml_results.keys()))
    results: dict[int, dict[str, Any]] = {}

    # Total detector families evaluated — used for dynamic confidence denominator
    clean_pattern = str(pattern_type).lower() if pattern_type else "unknown"
    if clean_pattern in ("structuring", "smurfing", "layering", "rapid_cashout", "velocity", "dormant", "duplicate", "spike", "multi_destination"):
        # 1 targeted rule family + 4 expanded rule families + 3 ML models = 8.0 families evaluated
        total_detector_families = 8.0
    else:
        # 5 core rule families + 4 expanded rule families + 3 ML models = 12.0 families evaluated
        total_detector_families = 12.0

    for acc in all_acc_ids:
        r_flags = rule_results.get(acc, [])
        ml_info = ml_results.get(acc, {})
        ml_score = float(ml_info.get("ml_anomaly_score", 0.0))

        triggered_signals = list(r_flags)

        # Surface individual ML detector signals for richer explanations
        if ml_info.get("iforest_flagged", False):
            triggered_signals.append("ML_IFOREST")
        if ml_info.get("lof_flagged", False):
            triggered_signals.append("ML_LOF")
        if ml_info.get("hbos_flagged", False):
            triggered_signals.append("ML_HBOS")

        # Backward-compat: keep ML_ANOMALY signal if any ML detector flagged
        if ml_info.get("is_ml_anomalous", False):
            triggered_signals.append("ML_ANOMALY")

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique_signals: list[str] = []
        for s in triggered_signals:
            if s not in seen:
                seen.add(s)
                unique_signals.append(s)
        triggered_signals = unique_signals

        # OR-gate: flagged if ANY signal triggered
        is_flagged = len(triggered_signals) > 0

        # Confidence = ratio of detector families that agreed
        triggered_families: set[str] = set()
        for flag in r_flags:
            if flag.startswith("R_STRUCT"):
                triggered_families.add("structuring")
            elif flag.startswith("R_SMURF"):
                triggered_families.add("smurfing")
            elif flag.startswith("R_LAYER"):
                triggered_families.add("layering")
            elif flag.startswith("R_CASHOUT"):
                triggered_families.add("rapid_cashout")
            elif flag.startswith("R_VELOCITY"):
                triggered_families.add("velocity")
            elif flag.startswith("R_DORMANT"):
                triggered_families.add("dormant_reactivation")
            elif flag.startswith("R_DUPLICATE"):
                triggered_families.add("duplicate_transfer")
            elif flag.startswith("R_SPIKE"):
                triggered_families.add("volume_spike")
            elif flag.startswith("R_MULTI_DEST"):
                triggered_families.add("multi_destination")

        if ml_info.get("iforest_flagged", False):
            triggered_families.add("ml_iforest")
        if ml_info.get("lof_flagged", False):
            triggered_families.add("ml_lof")
        if ml_info.get("hbos_flagged", False):
            triggered_families.add("ml_hbos")

        confidence = float(round(len(triggered_families) / total_detector_families, 2)) if is_flagged else 0.0

        results[acc] = {
            "is_flagged": is_flagged,
            "triggered_signals": triggered_signals,
            "rule_flags": r_flags,
            "ml_anomaly_score": ml_score,
            "confidence": min(1.0, confidence),
        }

    return results
