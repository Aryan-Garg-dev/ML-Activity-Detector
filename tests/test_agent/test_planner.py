"""Phase 5: Planner unit tests.

Tests for:
- Tier 1 decision table routing correctness per intent type
- Aggregation query logic — amount-only query uses filtered_lookup, not HAVING
- EDA-only path for visualization queries (no ML)
- Full EDA path for exploration+detection queries
- Step ID sequencing and required terminal step (reporting)
- tools_skipped consistency
"""

import pytest
from unittest.mock import MagicMock

from core.config import AppConfig
from core.types import IntentType, PatternType, ToolName
from schemas.contracts import QuerySpec, ExecutionPlan
from agent.planner import Planner


def _make_config() -> AppConfig:
    return AppConfig(
        reporting_threshold=10000.0,
        ml_contamination=0.05,
        ml_percentile_cutoff=95.0,
    )


def _make_spec(intent: IntentType, pattern=PatternType.UNKNOWN, filters=None, agg_spec=None, target_id=None, raw_query="test") -> QuerySpec:
    return QuerySpec(
        intent_type=intent,
        pattern_type=pattern,
        raw_query=raw_query,
        filters=filters or {},
        aggregation_spec=agg_spec or {},
        target_entity_id=target_id,
    )


@pytest.fixture
def planner():
    config = _make_config()
    registry = MagicMock()
    return Planner(registry=registry)


class TestPlannerRouting:
    def test_pattern_search_plan_structure(self, planner):
        spec = _make_spec(IntentType.PATTERN_SEARCH, PatternType.STRUCTURING, {"days": 30}, raw_query="Find structuring patterns")
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_001")

        assert isinstance(plan, ExecutionPlan)
        tool_names = [str(s.tool_name) for s in plan.steps]
        # Must include detection pipeline
        assert "feature_engineering" in tool_names
        assert "detection" in tool_names
        assert "scoring" in tool_names
        assert "reporting" in tool_names
        # Must skip EDA and data_query
        assert ToolName.EDA in plan.tools_skipped
        assert ToolName.DATA_QUERY in plan.tools_skipped

    def test_entity_lookup_plan_structure(self, planner):
        spec = _make_spec(IntentType.ENTITY_LOOKUP, target_id="4521", raw_query="Is account 4521 suspicious?")
        plan = planner.create_plan(spec, query_id="q_002")

        tool_names = [str(s.tool_name) for s in plan.steps]
        # Must include entity_lookup
        assert "entity_lookup" in tool_names
        # Must end with reporting
        assert "reporting" in tool_names
        # Must NOT run the full ML pipeline (detection/scoring/feature_engineering are not needed)
        assert "detection" not in tool_names
        assert "scoring" not in tool_names
        assert "feature_engineering" not in tool_names
        assert "eda" not in tool_names

    def test_aggregation_amount_only_uses_filtered_lookup(self, planner):
        """Bug fix: amount-only query must use filtered_lookup, not HAVING aggregation."""
        spec = _make_spec(
            IntentType.AGGREGATION_QUERY,
            filters={"min_amount": 1000.0},
            agg_spec={},  # No min_count
            raw_query="Detect money laundering with amounts exceeding $1000",
        )
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_003")

        # Find the data_query step
        dq_steps = [s for s in plan.steps if str(s.tool_name) == "data_query"]
        assert len(dq_steps) == 1, "Expected exactly one data_query step"
        args = dq_steps[0].args

        # Must NOT have group_by or having when only amount filter
        assert "group_by" not in args or args.get("group_by") is None
        assert "having" not in args or args.get("having") is None
        # Must have a tx_amount filter
        assert "filters" in args
        assert "tx_amount" in args["filters"]
        assert args["filters"]["tx_amount"]["value"] == 1000.0

    def test_aggregation_with_count_uses_having(self, planner):
        """10+ transactions query should use GROUP BY + HAVING."""
        spec = _make_spec(
            IntentType.AGGREGATION_QUERY,
            filters={"max_amount": 9999.0},
            agg_spec={"min_count": 10},
            raw_query="Which customers made 10+ transactions under $10,000?",
        )
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_004")

        dq_steps = [s for s in plan.steps if str(s.tool_name) == "data_query"]
        assert len(dq_steps) == 1
        args = dq_steps[0].args
        assert "group_by" in args
        assert "having" in args
        assert args["having"]["value"] == 10

    def test_broad_eda_full_path(self, planner):
        spec = _make_spec(
            IntentType.BROAD_EDA,
            raw_query="Give me an overview of transaction activity",
        )
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_005")

        tool_names = [str(s.tool_name) for s in plan.steps]
        assert "eda" in tool_names
        # Full EDA includes detection
        assert "detection" in tool_names
        assert "reporting" in tool_names

    def test_broad_eda_visualization_only_skips_ml(self, planner):
        """Visualization query without detection keywords should skip ML pipeline."""
        spec = _make_spec(
            IntentType.BROAD_EDA,
            raw_query="Show me the transaction amount distribution",
        )
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_006")

        tool_names = [str(s.tool_name) for s in plan.steps]
        # EDA-only path: EDA + REPORTING only
        assert "eda" in tool_names
        assert "reporting" in tool_names
        # Must skip ML tools
        assert ToolName.DETECTION in plan.tools_skipped
        assert ToolName.SCORING in plan.tools_skipped
        assert ToolName.FEATURE_ENGINEERING in plan.tools_skipped

    def test_risk_scoring_batch_plan(self, planner):
        spec = _make_spec(IntentType.RISK_SCORING_BATCH, raw_query="Score all accounts for risk")
        config = _make_config()
        plan = planner.create_plan(spec, query_id="q_007")

        tool_names = [str(s.tool_name) for s in plan.steps]
        assert "feature_engineering" in tool_names
        assert "detection" in tool_names
        assert "scoring" in tool_names
        assert "reporting" in tool_names

    def test_all_plans_end_with_reporting(self, planner):
        """All routing paths must end with the reporting step."""
        config = _make_config()
        test_specs = [
            _make_spec(IntentType.PATTERN_SEARCH, PatternType.STRUCTURING, {"days": 30}, raw_query="Find structuring"),
            _make_spec(IntentType.AGGREGATION_QUERY, filters={"min_amount": 5000}, raw_query="txns > 5000"),
            _make_spec(IntentType.ENTITY_LOOKUP, target_id="999", raw_query="Account 999"),
            _make_spec(IntentType.BROAD_EDA, raw_query="Overview"),
            _make_spec(IntentType.BROAD_EDA, raw_query="Show distribution chart"),
            _make_spec(IntentType.RISK_SCORING_BATCH, raw_query="Batch score"),
        ]
        for i, spec in enumerate(test_specs):
            plan = planner.create_plan(spec, query_id=f"q_{i:03d}")
            last_step = plan.steps[-1]
            assert str(last_step.tool_name) == "reporting", \
                f"Plan for intent={spec.intent_type} does not end with reporting. Got: {last_step.tool_name}"

    def test_step_ids_sequential(self, planner):
        """Step IDs must be sequential integers starting at 1."""
        config = _make_config()
        spec = _make_spec(IntentType.PATTERN_SEARCH, PatternType.STRUCTURING, {"days": 14}, raw_query="Structuring 14 days")
        plan = planner.create_plan(spec, query_id="q_999")
        ids = [s.step_id for s in plan.steps]
        assert ids == list(range(1, len(ids) + 1))
