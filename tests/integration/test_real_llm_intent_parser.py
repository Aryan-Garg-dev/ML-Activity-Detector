"""Integration tests for query intent parsing using real LLM provider.

Tests that the active LLM provider (Groq / LM Studio / OpenRouter) correctly parses
natural language AML queries into structured Pydantic QuerySpec objects across
various compliance scenarios.
"""

import pytest
from core.types import IntentType, PatternType
from schemas.contracts import QuerySpec
from llm.client import LLMClient
from llm.prompts.intent_parser import parse_query_intent


class TestRealLLMIntentParser:
    def test_parse_pattern_search_structuring(self, llm_client: LLMClient):
        query = "Find structuring patterns in the last 30 days"
        spec = parse_query_intent(llm_client, query)

        assert isinstance(spec, QuerySpec)
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.STRUCTURING
        assert spec.raw_query == query

    def test_parse_aggregation_query(self, llm_client: LLMClient):
        query = "Which customers made 10+ transactions under $10,000?"
        spec = parse_query_intent(llm_client, query)

        assert isinstance(spec, QuerySpec)
        assert spec.intent_type == IntentType.AGGREGATION_QUERY
        assert spec.raw_query == query

    def test_parse_entity_lookup(self, llm_client: LLMClient):
        query = "Is customer ID 4521 suspicious?"
        spec = parse_query_intent(llm_client, query)

        assert isinstance(spec, QuerySpec)
        assert spec.intent_type == IntentType.ENTITY_LOOKUP
        assert str(spec.target_entity_id) == "4521"
        assert spec.raw_query == query

    def test_parse_broad_eda(self, llm_client: LLMClient):
        query = "Give me an overview of overall transaction volume trends and distributions"
        spec = parse_query_intent(llm_client, query)

        assert isinstance(spec, QuerySpec)
        assert spec.intent_type == IntentType.BROAD_EDA
        assert spec.raw_query == query
