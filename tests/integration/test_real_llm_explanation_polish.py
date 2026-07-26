"""Integration tests for explanation polishing using real LLM provider.

Validates that real LLM provider responses preserve exact numeric tokens
when refining raw template explanations, or fall back safely to raw explanations if drift occurs.
"""

import pytest
from llm.client import LLMClient
from llm.prompts.explanation_polish import polish_explanation
from tools.explanation.templates import verify_numeric_preservation


class TestRealLLMExplanationPolish:
    def test_polish_explanation_preserves_numbers(self, llm_client: LLMClient):
        raw = (
            "In response to your query for structuring patterns, account 12345 was flagged. "
            "Risk level: HIGH (composite score: 82.0/100). Structuring detected: "
            "5 transactions near the $10,000 reporting threshold with a round-number transaction bias of 60.0%."
        )

        polished = polish_explanation(llm_client, raw)

        # Must verify numeric preservation passes
        assert verify_numeric_preservation(raw, polished) is True
        # Output should be non-empty and retain core identifiers
        assert "12345" in polished
        assert "82.0" in polished

    def test_polish_explanation_fallback_safety(self, llm_client: LLMClient):
        raw = "Risk level: LOW (composite score: 15.0/100). No suspicious signals triggered for account 99999."
        polished = polish_explanation(llm_client, raw)

        assert verify_numeric_preservation(raw, polished) is True
        assert "15.0" in polished
        assert "99999" in polished
