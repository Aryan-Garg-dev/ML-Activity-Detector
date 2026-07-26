"""Ensemble combiners for hybrid AML detection using the Strategy Pattern.

Merges rule detection flags and PyOD ML anomaly scores (IForest, LOF, HBOS).
Supports OR-gate (default), AND-gate, and Weighted strategies.
"""

from typing import Any
from abc import ABC, abstractmethod
from core.config import AppConfig


class EnsembleStrategy(ABC):
    @abstractmethod
    def combine_signals(
        self,
        rule_results: dict[int, list[str]],
        ml_results: dict[int, dict[str, Any]],
        config: AppConfig,
        pattern_type: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        pass


class OrGateStrategy(EnsembleStrategy):
    """OR-gate: flagged if ANY signal (rule or ML) triggered."""
    
    def combine_signals(
        self,
        rule_results: dict[int, list[str]],
        ml_results: dict[int, dict[str, Any]],
        config: AppConfig,
        pattern_type: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        all_acc_ids = set(rule_results.keys()).union(set(ml_results.keys()))
        results: dict[int, dict[str, Any]] = {}
        
        for acc in all_acc_ids:
            r_flags = rule_results.get(acc, [])
            ml_info = ml_results.get(acc, {})
            ml_score = float(ml_info.get("ml_anomaly_score", 0.0))
            
            triggered_signals = list(r_flags)
            if ml_info.get("iforest_flagged", False): triggered_signals.append("ML_IFOREST")
            if ml_info.get("lof_flagged", False): triggered_signals.append("ML_LOF")
            if ml_info.get("hbos_flagged", False): triggered_signals.append("ML_HBOS")
            if ml_info.get("is_ml_anomalous", False): triggered_signals.append("ML_ANOMALY")
            
            seen: set[str] = set()
            unique_signals: list[str] = [s for s in triggered_signals if not (s in seen or seen.add(s))]
            
            is_flagged = len(unique_signals) > 0
            
            confidence = min(len(unique_signals) * 0.2 + (ml_score * 0.5), 1.0) if is_flagged else 0.0
            
            results[acc] = {
                "is_flagged": is_flagged,
                "triggered_signals": unique_signals,
                "rule_flags": r_flags,
                "ml_anomaly_score": ml_score,
                "confidence": round(confidence, 2),
            }
        return results


class AndGateStrategy(EnsembleStrategy):
    """AND-gate: flagged ONLY if BOTH a rule AND an ML model triggered."""
    
    def combine_signals(
        self,
        rule_results: dict[int, list[str]],
        ml_results: dict[int, dict[str, Any]],
        config: AppConfig,
        pattern_type: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        all_acc_ids = set(rule_results.keys()).union(set(ml_results.keys()))
        results: dict[int, dict[str, Any]] = {}
        
        for acc in all_acc_ids:
            r_flags = rule_results.get(acc, [])
            ml_info = ml_results.get(acc, {})
            ml_score = float(ml_info.get("ml_anomaly_score", 0.0))
            
            ml_flags = []
            if ml_info.get("iforest_flagged", False): ml_flags.append("ML_IFOREST")
            if ml_info.get("lof_flagged", False): ml_flags.append("ML_LOF")
            if ml_info.get("hbos_flagged", False): ml_flags.append("ML_HBOS")
            
            has_rule = len(r_flags) > 0
            has_ml = len(ml_flags) > 0
            is_flagged = has_rule and has_ml
            
            triggered_signals = list(r_flags) + ml_flags
            
            confidence = min(len(triggered_signals) * 0.2 + (ml_score * 0.5), 1.0) if is_flagged else 0.0
            
            results[acc] = {
                "is_flagged": is_flagged,
                "triggered_signals": triggered_signals,
                "rule_flags": r_flags,
                "ml_anomaly_score": ml_score,
                "confidence": round(confidence, 2),
            }
        return results


class WeightedStrategy(EnsembleStrategy):
    """Weighted: Combines ml_anomaly_score and rules severity dynamically."""
    
    def combine_signals(
        self,
        rule_results: dict[int, list[str]],
        ml_results: dict[int, dict[str, Any]],
        config: AppConfig,
        pattern_type: str | None = None,
    ) -> dict[int, dict[str, Any]]:
        all_acc_ids = set(rule_results.keys()).union(set(ml_results.keys()))
        results: dict[int, dict[str, Any]] = {}
        
        for acc in all_acc_ids:
            r_flags = rule_results.get(acc, [])
            ml_info = ml_results.get(acc, {})
            ml_score = float(ml_info.get("ml_anomaly_score", 0.0))
            
            rule_weight = len(r_flags) * 0.3
            total_score = (ml_score * 0.6) + rule_weight
            
            is_flagged = total_score >= 0.75
            
            triggered_signals = list(r_flags)
            if ml_score > 0.5: triggered_signals.append("ML_HIGH_RISK")
            
            results[acc] = {
                "is_flagged": is_flagged,
                "triggered_signals": triggered_signals,
                "rule_flags": r_flags,
                "ml_anomaly_score": ml_score,
                "confidence": round(min(total_score, 1.0), 2),
            }
        return results


def get_ensemble_strategy(name: str) -> EnsembleStrategy:
    name = name.lower().strip()
    if name == "and_gate":
        return AndGateStrategy()
    elif name == "weighted":
        return WeightedStrategy()
    return OrGateStrategy()
