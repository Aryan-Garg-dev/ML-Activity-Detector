"""Tests for the off-topic query rejection guardrail and expanded heuristic parser."""

import pytest
from agent.guardrails import validate_query_relevance
from agent.nodes import _heuristic_query_spec
from core.types import IntentType, PatternType


class TestOffTopicRejection:
    """Tests that off-topic queries are rejected gracefully."""

    @pytest.mark.parametrize("query", [
        "What is the weather today?",
        "Show me a recipe for pasta",
        "Calculate my BMI",
        "Tell me a joke",
        "What is the capital of France?",
        "Write a poem about cats",
        "How to cook chicken?",
        "Best workout routines for fitness",
        "Translate this to Spanish",
        "Tell me about astronomy",
    ])
    def test_off_topic_queries_rejected(self, query: str):
        is_relevant, reason = validate_query_relevance(query)
        assert not is_relevant, f"Expected rejection for: {query}"
        assert len(reason) > 0

    @pytest.mark.parametrize("query", [
        "Find structuring patterns in the last 30 days",
        "Which customers made 10+ transactions under $10,000?",
        "Is customer ID 4521 suspicious?",
        "Show me an overview of transaction trends",
        "Detect smurfing patterns",
        "Find accounts with rapid cash-out behavior",
        "Show me high-risk accounts that need escalation",
        "Analyze this dataset for suspicious activity",
        "What is the fraud rate in the dataset?",
        "Are there any velocity anomalies?",
    ])
    def test_aml_queries_accepted(self, query: str):
        is_relevant, reason = validate_query_relevance(query)
        assert is_relevant, f"Expected acceptance for: {query}, got rejection: {reason}"

    def test_empty_query_rejected(self):
        is_relevant, reason = validate_query_relevance("")
        assert not is_relevant
        assert "Empty query" in reason

    def test_whitespace_only_rejected(self):
        is_relevant, reason = validate_query_relevance("   ")
        assert not is_relevant


class TestExpandedHeuristicParser:
    """Tests for the comprehensive heuristic query spec fallback."""

    # Entity lookup
    def test_entity_lookup_customer_id(self):
        spec = _heuristic_query_spec("Is customer ID 4521 suspicious?")
        assert spec.intent_type == IntentType.ENTITY_LOOKUP
        assert spec.target_entity_id == "4521"

    def test_entity_lookup_account_id(self):
        spec = _heuristic_query_spec("Check account 9876")
        assert spec.intent_type == IntentType.ENTITY_LOOKUP
        assert spec.target_entity_id == "9876"

    def test_entity_lookup_explain_id(self):
        spec = _heuristic_query_spec("Explain why 12345 was flagged")
        assert spec.intent_type == IntentType.ENTITY_LOOKUP
        assert spec.target_entity_id == "12345"

    # Aggregation queries
    def test_aggregation_how_many(self):
        spec = _heuristic_query_spec("How many transactions were under $10,000?")
        assert spec.intent_type == IntentType.AGGREGATION_QUERY

    def test_aggregation_count_over(self):
        spec = _heuristic_query_spec("Count of transactions over $50,000")
        assert spec.intent_type == IntentType.AGGREGATION_QUERY

    def test_aggregation_with_count_threshold(self):
        spec = _heuristic_query_spec("Customers with 10+ transactions under $10,000")
        assert spec.intent_type == IntentType.AGGREGATION_QUERY

    def test_aggregation_top_20(self):
        spec = _heuristic_query_spec("Show top 20 riskiest accounts")
        assert spec.intent_type == IntentType.AGGREGATION_QUERY

    # Broad EDA
    def test_eda_overview(self):
        spec = _heuristic_query_spec("Give me an overview of the dataset")
        assert spec.intent_type == IntentType.BROAD_EDA

    def test_eda_distribution(self):
        spec = _heuristic_query_spec("Show me the amount distribution across accounts")
        assert spec.intent_type == IntentType.BROAD_EDA

    def test_eda_trend(self):
        spec = _heuristic_query_spec("Show volume trends over time")
        assert spec.intent_type == IntentType.BROAD_EDA

    def test_eda_visualize(self):
        spec = _heuristic_query_spec("Visualize outliers in the data")
        assert spec.intent_type == IntentType.BROAD_EDA

    # Pattern search: Structuring
    def test_pattern_structuring(self):
        spec = _heuristic_query_spec("Find structuring patterns in the last 30 days")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.STRUCTURING

    def test_pattern_threshold_avoidance(self):
        spec = _heuristic_query_spec("Detect threshold avoidance behavior")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.STRUCTURING

    # Pattern search: Smurfing
    def test_pattern_smurfing(self):
        spec = _heuristic_query_spec("Detect smurfing patterns")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.SMURFING

    # Pattern search: Velocity
    def test_pattern_velocity(self):
        spec = _heuristic_query_spec("Find velocity anomalies in the last 7 days")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.VELOCITY

    def test_pattern_high_frequency(self):
        spec = _heuristic_query_spec("Detect high frequency trading patterns")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.VELOCITY

    # Pattern search: Rapid cashout
    def test_pattern_rapid_cashout(self):
        spec = _heuristic_query_spec("Find rapid cash-out behavior")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.RAPID_CASHOUT

    # Pattern search: Layering
    def test_pattern_layering(self):
        spec = _heuristic_query_spec("Detect layering through intermediary accounts")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.LAYERING

    def test_pattern_cross_border(self):
        spec = _heuristic_query_spec("Find cross-border suspicious transfers")
        assert spec.intent_type == IntentType.PATTERN_SEARCH
        assert spec.pattern_type == PatternType.LAYERING

    # Risk scoring batch
    def test_risk_scoring_suspicious(self):
        spec = _heuristic_query_spec("Find all suspicious accounts")
        assert spec.intent_type == IntentType.RISK_SCORING_BATCH

    def test_risk_scoring_anomalies(self):
        spec = _heuristic_query_spec("Detect anomalies in the dataset")
        assert spec.intent_type == IntentType.RISK_SCORING_BATCH

    # Time filter extraction
    def test_time_filter_30_days(self):
        spec = _heuristic_query_spec("Find structuring patterns in the last 30 days")
        assert spec.filters.get("days") == 30

    def test_time_filter_7_days(self):
        spec = _heuristic_query_spec("Velocity anomalies in the last 7 days")
        assert spec.filters.get("days") == 7

    def test_time_filter_last_week(self):
        spec = _heuristic_query_spec("Show suspicious activity last week")
        assert spec.filters.get("days") == 7

    def test_time_filter_last_month(self):
        spec = _heuristic_query_spec("Detect anomalies last month")
        assert spec.filters.get("days") == 30

    def test_time_filter_2_weeks(self):
        spec = _heuristic_query_spec("Find patterns in the last 2 weeks")
        assert spec.filters.get("days") == 14

    # Amount filter extraction
    def test_amount_filter_under(self):
        spec = _heuristic_query_spec("Transactions under $10,000")
        assert spec.filters.get("max_amount") == 10000.0

    def test_amount_filter_over(self):
        spec = _heuristic_query_spec("Transactions over $50,000")
        assert spec.filters.get("min_amount") == 50000.0

    # Catch-all
    def test_catch_all_unknown(self):
        spec = _heuristic_query_spec("analyze everything")
        assert spec.intent_type in (IntentType.PATTERN_SEARCH, IntentType.BROAD_EDA, IntentType.RISK_SCORING_BATCH)
