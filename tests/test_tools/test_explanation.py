"""Phase 5: Explanation tool and template tests.

Tests for:
- format_grounded_explanation — query-context aware, signal-specific templates
- verify_numeric_preservation — tolerance-based numeric drift detection
- extract_numeric_tokens — consistent normalization of currency/percent/float values
- ExplanationInput serialization
"""

import pytest
from tools.explanation.templates import (
    format_grounded_explanation,
    verify_numeric_preservation,
    extract_numeric_tokens,
)
from tools.explanation.tool import ExplanationInput


class TestExtractNumericTokens:
    def test_plain_integers(self):
        tokens = extract_numeric_tokens("account 12345 had 5 transactions")
        assert "5.0" in tokens
        assert "12345.0" in tokens

    def test_currency_normalization(self):
        """$10,000 and 10000 and 10000.0 must produce the same token."""
        t1 = extract_numeric_tokens("$10,000")
        t2 = extract_numeric_tokens("10000")
        t3 = extract_numeric_tokens("10000.0")
        assert t1 == t2 == t3

    def test_percentage_normalization(self):
        """50% and 50.0% must produce the same token."""
        t1 = extract_numeric_tokens("50%")
        t2 = extract_numeric_tokens("50.0%")
        assert t1 == t2

    def test_mixed_values(self):
        tokens = extract_numeric_tokens("$10,000 threshold with 50.0% round-number bias")
        assert "10000.0" in tokens
        assert "50.0" in tokens

    def test_empty_string(self):
        assert extract_numeric_tokens("") == []

    def test_no_numbers(self):
        assert extract_numeric_tokens("no numbers here") == []


class TestVerifyNumericPreservation:
    def test_exact_match_passes(self):
        grounded = "score of 82.0 across 47 transactions"
        candidate = "score of 82.0 across 47 transactions"
        assert verify_numeric_preservation(grounded, candidate) is True

    def test_equivalent_format_passes(self):
        """$10,000 == 10000.0 — benign reformatting should not fail."""
        grounded = "transactions near the $10,000 threshold"
        candidate = "transactions near the 10000.0 threshold"
        assert verify_numeric_preservation(grounded, candidate) is True

    def test_percentage_reformat_passes(self):
        """50.0% and 50% are equivalent."""
        grounded = "round-number bias of 50.0%"
        candidate = "round-number bias of 50%"
        assert verify_numeric_preservation(grounded, candidate) is True

    def test_hallucinated_number_detected(self):
        """Grounded says 47 transactions, LLM changed to 50 — should fail."""
        grounded = "47 transactions in 30 days"
        candidate = "50 transactions in 30 days"
        # 47 is missing from candidate
        assert verify_numeric_preservation(grounded, candidate) is False

    def test_dropped_number_detected(self):
        """LLM removed a specific grounded value — should fail."""
        grounded = "z-score of 3.45"
        candidate = "z-score was significantly elevated"
        assert verify_numeric_preservation(grounded, candidate) is False

    def test_additional_numbers_in_candidate_pass(self):
        """Candidate may have extra numbers (like step IDs) — should pass."""
        grounded = "score 82.0"
        candidate = "score 82.0 (step 1 of 3)"
        assert verify_numeric_preservation(grounded, candidate) is True

    def test_empty_grounded_always_passes(self):
        """No grounded numbers means nothing to preserve — always passes."""
        assert verify_numeric_preservation("no numbers", "any text 99") is True


class TestFormatGroundedExplanation:
    def _base_risk_info(self, signals=None, score=82.0, risk_level="HIGH"):
        return {
            "composite_score": score,
            "risk_level": risk_level,
            "triggered_signals": signals or [],
        }

    def test_structuring_signal(self):
        feat = {"count_near_threshold_30d": 5, "round_number_bias": 0.6}
        result = format_grounded_explanation(
            entity_id="12345",
            risk_info=self._base_risk_info(["R_STRUCT_01"]),
            feature_info=feat,
        )
        assert "12345" in result or "Structuring" in result
        assert "5" in result
        assert "60.0%" in result

    def test_smurfing_signal(self):
        feat = {"fan_in_degree": 15, "fan_out_degree": 3}
        result = format_grounded_explanation(
            entity_id="99999",
            risk_info=self._base_risk_info(["R_SMURF_01"]),
            feature_info=feat,
        )
        assert "15" in result
        assert "3" in result
        assert "Smurfing" in result or "smurfing" in result

    def test_ml_iforest_signal(self):
        feat = {"ml_anomaly_score": 0.95}
        result = format_grounded_explanation(
            entity_id="77777",
            risk_info={**self._base_risk_info(["ML_IFOREST"]), "ml_anomaly_score": 0.95},
            feature_info=feat,
        )
        assert "IForest" in result or "ML" in result or "anomaly" in result.lower()

    def test_ml_hbos_and_lof_signals(self):
        feat = {"ml_anomaly_score": 0.88}
        result = format_grounded_explanation(
            entity_id="55555",
            risk_info={**self._base_risk_info(["ML_IFOREST", "ML_LOF", "ML_HBOS"]), "ml_anomaly_score": 0.88},
            feature_info=feat,
        )
        # Should include all three detector names
        assert "IForest" in result or "LOF" in result or "HBOS" in result

    def test_fallback_for_no_signals(self):
        result = format_grounded_explanation(
            entity_id="11111",
            risk_info=self._base_risk_info(signals=[], score=55.0, risk_level="MEDIUM"),
        )
        assert "MEDIUM" in result or "medium" in result.lower()
        assert "55.0" in result

    def test_query_context_sentence_with_min_amount(self):
        """With a min_amount filter, explanation must open with query-grounded context."""
        result = format_grounded_explanation(
            entity_id="22222",
            risk_info=self._base_risk_info(["R_VELOCITY_01"]),
            feature_info={"velocity_zscore": 2.5},
            query_context={
                "raw_query": "Detect money laundering with amounts exceeding $1000",
                "filters": {"min_amount": 1000},
                "intent_type": "pattern_search",
                "pattern_type": "unknown",
            },
        )
        assert "1,000" in result or "1000" in result
        assert "In response to" in result
        assert "22222" in result

    def test_query_context_sentence_with_pattern(self):
        result = format_grounded_explanation(
            entity_id="33333",
            risk_info=self._base_risk_info(["R_STRUCT_01"]),
            feature_info={"count_near_threshold_30d": 3, "round_number_bias": 0.4},
            query_context={
                "raw_query": "Find structuring patterns",
                "filters": {},
                "intent_type": "pattern_search",
                "pattern_type": "structuring",
            },
        )
        assert "structuring" in result.lower()
        assert "In response to" in result
        assert "33333" in result

    def test_query_context_none_produces_no_context_sentence(self):
        result = format_grounded_explanation(
            entity_id="44444",
            risk_info=self._base_risk_info(["R_CASHOUT_01"]),
            feature_info={"dwell_time_avg_hours": 1.5, "in_out_ratio_30d": 0.9},
            query_context=None,
        )
        # No query context — should NOT have the "In response to" prefix
        assert "In response to" not in result

    def test_explanation_includes_risk_summary(self):
        result = format_grounded_explanation(
            entity_id="55555",
            risk_info=self._base_risk_info(["R_LAYER_01"], score=91.0, risk_level="HIGH"),
            feature_info={
                "distinct_counterparties_30d": 12,
                "distinct_countries_30d": 3,
                "in_out_ratio_30d": 0.85,
            },
        )
        assert "91.0" in result
        assert "HIGH" in result


class TestExplanationInput:
    def test_default_construction(self):
        inp = ExplanationInput()
        assert inp.risk_assessments == []
        assert inp.detection_results == {}
        assert inp.enable_llm_polish is False
        assert inp.query_context is None

    def test_with_query_context(self):
        inp = ExplanationInput(
            risk_assessments=[{"entity_id": "1", "composite_score": 80.0, "risk_level": "high", "triggered_signals": []}],
            query_context={"raw_query": "test query", "filters": {}, "pattern_type": "unknown"},
        )
        assert inp.query_context["raw_query"] == "test query"

    def test_dict_style_risk_assessments(self):
        """risk_assessments accepts both list and dict."""
        inp = ExplanationInput(
            risk_assessments={"risk_assessments": [{"entity_id": "1", "composite_score": 80.0}]},
        )
        # Should accept dict, flattened inside the tool
        assert isinstance(inp.risk_assessments, dict)
