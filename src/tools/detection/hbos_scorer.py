"""HBOS (Histogram-Based Outlier Score) anomaly scoring module.

HBOS is a fast, interpretable unsupervised anomaly detection algorithm that
scores each feature independently via histogram density estimation. Unlike
IForest and LOF, HBOS is highly transparent — the score is directly interpretable
as an inverse histogram density. It complements IForest (tree-based) and LOF
(density-based) with a distributional perspective.

Reference: Goldstein & Dengel (2012) "Histogram-based Outlier Score (HBOS)"
"""

from typing import Any
import numpy as np
from loguru import logger
from pyod.models.hbos import HBOS
from sklearn.preprocessing import RobustScaler

from core.config import AppConfig


def fit_predict_hbos_anomalies(
    features_dict: dict[int, dict[str, Any]],
    config: AppConfig,
    background_features: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Fit a HBOS model on feature vectors and return anomaly scores.

    HBOS scores each feature dimension independently using histogram density,
    making it fast and interpretable. Works well for detecting outliers in
    high-dimensional financial feature spaces.

    Args:
        features_dict: Dict mapping account_id → feature dict
        config: Application runtime configuration
        background_features: Optional real historical feature dicts for background population

    Returns:
        Dict mapping account_id → HBOS anomaly result dict:
            hbos_anomaly_score: float (percentile rank 0.0 to 1.0)
            is_hbos_anomalous: bool (True if score >= cutoff percentile)
            hbos_raw_score: float
    """
    if not features_dict:
        return {}

    target_acc_ids = list(features_dict.keys())
    sample_feat_dict = next(iter(features_dict.values()))
    feature_keys = sorted(list(sample_feat_dict.keys()))

    if not feature_keys:
        return {
            acc: {
                "hbos_anomaly_score": 0.0,
                "is_hbos_anomalous": False,
                "hbos_raw_score": 0.0,
            }
            for acc in target_acc_ids
        }

    # Build target feature matrix
    target_matrix = [
        [float(features_dict[acc].get(k, 0.0)) for k in feature_keys]
        for acc in target_acc_ids
    ]
    target_X = np.array(target_matrix, dtype=np.float64)

    # Augment with background if batch is small
    if len(target_acc_ids) < 20:
        if background_features:
            bg_rows = [[float(bg.get(k, 0.0)) for k in feature_keys] for bg in background_features]
            baseline_X = np.array(bg_rows, dtype=np.float64)
        else:
            baseline_X = np.zeros((30, len(feature_keys)), dtype=np.float64)
        X = np.vstack([target_X, baseline_X])
    else:
        X = target_X

    # Clean NaN/inf values before fitting
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features for consistent histogram binning
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    contamination = max(0.01, min(0.5, config.ml_contamination))

    hbos = HBOS(contamination=contamination, n_bins=10)
    try:
        hbos.fit(X_scaled)
        hbos_scores = hbos.decision_scores_
    except Exception as e:
        logger.warning("HBOS fitting failed: {e}", e=e)
        hbos_scores = np.zeros(len(X))

    # Normalize to [0, 1] percentile ranks
    hbos_norm = _percentile_rank_transform(hbos_scores)
    cutoff_threshold = config.ml_percentile_cutoff / 100.0

    results: dict[int, dict[str, Any]] = {}
    for i, acc in enumerate(target_acc_ids):
        score = float(round(hbos_norm[i], 4))
        results[acc] = {
            "hbos_anomaly_score": score,
            "is_hbos_anomalous": bool(score >= cutoff_threshold),
            "hbos_raw_score": float(hbos_scores[i]),
        }

    return results


def _percentile_rank_transform(scores: np.ndarray) -> np.ndarray:
    """Transform raw decision scores to 0.0–1.0 percentile ranks."""
    if len(scores) == 0:
        return scores
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)
