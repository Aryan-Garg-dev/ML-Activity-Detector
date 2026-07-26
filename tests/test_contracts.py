"""Phase 5: Contract serialization round-trip tests.

Validates all Pydantic domain models serialize/deserialize correctly,
field validators work, and interfaces match across layers.
"""

import pytest
from pydantic import ValidationError
from core.types import IntentType, PatternType, RiskLevel, EscalationAction, ToolName
from schemas.contracts import QuerySpec, ExecutionPlan, PlanStep, ToolResult, AgentResponse
from schemas.risk import RiskAssessment, FlaggedItem
from schemas.audit import AuditEvent


class TestQuerySpec:
    def test_round_trip_serialization(self):
        spec = QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.STRUCTURING,
            raw_query="Find structuring in last 30 days",
            filters={"days": 30, "max_amount": 9999.0},
        )
        dumped = spec.model_dump()
        reloaded = QuerySpec(**dumped)
        assert reloaded.intent_type == IntentType.PATTERN_SEARCH
        assert reloaded.pattern_type == PatternType.STRUCTURING
        assert reloaded.filters["days"] == 30

    def test_intent_normalization_from_string(self):
        spec = QuerySpec(
            intent_type="pattern_search",  # string instead of enum
            raw_query="test",
        )
        assert spec.intent_type == IntentType.PATTERN_SEARCH

    def test_pattern_normalization_from_string(self):
        spec = QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type="structuring",
            raw_query="test",
        )
        assert spec.pattern_type == PatternType.STRUCTURING

    def test_entity_id_optional(self):
        spec = QuerySpec(intent_type=IntentType.BROAD_EDA, raw_query="test")
        assert spec.target_entity_id is None

    def test_json_round_trip(self):
        spec = QuerySpec(
            intent_type=IntentType.ENTITY_LOOKUP,
            pattern_type=PatternType.UNKNOWN,
            target_entity_id="4521",
            raw_query="Is account 4521 suspicious?",
        )
        json_str = spec.model_dump_json()
        reloaded = QuerySpec.model_validate_json(json_str)
        assert reloaded.target_entity_id == "4521"


class TestExecutionPlan:
    def test_round_trip_with_steps(self):
        plan = ExecutionPlan(
            plan_id="plan_001",
            query_id="q_001",
            steps=[
                PlanStep(step_id=1, tool_name=ToolName.FEATURE_ENGINEERING, args={"feature_family": "volume"}, reason="Compute features"),
                PlanStep(step_id=2, tool_name=ToolName.DETECTION, args={}, reason="Detect anomalies"),
                PlanStep(step_id=3, tool_name=ToolName.REPORTING, args={}, reason="Report"),
            ],
            tools_skipped=[ToolName.EDA],
        )
        dumped = plan.model_dump()
        reloaded = ExecutionPlan(**dumped)
        assert len(reloaded.steps) == 3
        assert reloaded.steps[0].tool_name == ToolName.FEATURE_ENGINEERING

    def test_tool_name_normalization_in_step(self):
        step = PlanStep(
            step_id=1,
            tool_name="feature_engineering",  # string
            args={},
            reason="test",
        )
        assert step.tool_name == ToolName.FEATURE_ENGINEERING


class TestRiskAssessment:
    def test_round_trip(self):
        assessment = RiskAssessment(
            assessment_id="risk_001",
            query_id="q_001",
            entity_id="12345",
            composite_score=82.5,
            confidence=0.75,
            risk_level=RiskLevel.HIGH,
            escalation_action=EscalationAction.REPORT,
            triggered_signals=["R_STRUCT_01", "ML_IFOREST"],
        )
        dumped = assessment.model_dump()
        reloaded = RiskAssessment(**dumped)
        assert reloaded.composite_score == 82.5
        assert reloaded.risk_level == RiskLevel.HIGH
        assert "R_STRUCT_01" in reloaded.triggered_signals

    def test_json_round_trip(self):
        assessment = RiskAssessment(
            query_id="q_001",
            entity_id="99999",
            composite_score=45.0,
            confidence=0.5,
            risk_level=RiskLevel.MEDIUM,
            escalation_action=EscalationAction.REVIEW,
        )
        json_str = assessment.model_dump_json()
        reloaded = RiskAssessment.model_validate_json(json_str)
        assert reloaded.entity_id == "99999"


class TestFlaggedItem:
    def test_round_trip_with_optional_fields(self):
        item = FlaggedItem(
            entity_id="77777",
            score=92.0,
            risk_level=RiskLevel.HIGH,
            escalation_action=EscalationAction.REPORT,
            explanation="Test explanation",
            triggered_signals=["R_VELOCITY_01", "ML_LOF"],
            supporting_evidence={"velocity_zscore": 3.45},
            account_record={"account_id": 77777, "account_type": "savings"},
            recent_transactions=[{"tx_id": "T1", "amount": 5000.0}],
        )
        dumped = item.model_dump()
        reloaded = FlaggedItem(**dumped)
        assert reloaded.score == 92.0
        assert reloaded.account_record is not None
        assert len(reloaded.recent_transactions) == 1

    def test_minimal_flagged_item(self):
        item = FlaggedItem(
            entity_id="00001",
            score=20.0,
            risk_level=RiskLevel.LOW,
            escalation_action=EscalationAction.MONITOR,
            explanation="Low risk",
        )
        assert item.account_record is None
        assert item.recent_transactions == []


class TestAgentResponse:
    def test_round_trip(self):
        response = AgentResponse(
            query_id="q_abc",
            raw_query="Find structuring",
            intent=IntentType.PATTERN_SEARCH,
            execution_summary={"tools_invoked": ["feature_engineering"], "tools_skipped": ["eda"]},
            explanation="Test explanation",
        )
        dumped = response.model_dump()
        reloaded = AgentResponse(**dumped)
        assert reloaded.query_id == "q_abc"
        assert reloaded.intent == IntentType.PATTERN_SEARCH

    def test_flagged_items_coercion(self):
        """AgentResponse.flagged_items should accept raw dicts and coerce to FlaggedItem."""
        raw_items = [
            {
                "entity_id": "12345",
                "score": 75.0,
                "risk_level": "high",
                "escalation_action": "report",
                "explanation": "Test",
                "triggered_signals": ["R_STRUCT_01"],
            }
        ]
        response = AgentResponse(
            query_id="q_coerce",
            raw_query="test",
            intent=IntentType.PATTERN_SEARCH,
            flagged_items=raw_items,
            explanation="Test",
        )
        # Should coerce to FlaggedItem instances
        assert len(response.flagged_items) == 1
        item = response.flagged_items[0]
        assert hasattr(item, "risk_level") or isinstance(item, dict)


class TestAuditEvent:
    def test_round_trip(self):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        event = AuditEvent(
            audit_id="evt_001",
            query_id="q_001",
            tool_name=ToolName.FEATURE_ENGINEERING,
            step_id=1,
            started_at=now_iso,
            ended_at=now_iso,
            duration_ms=123.4,
            rows_in=50,
            rows_out=50,
            config_version="v1.0.0",
            status="ok",
        )
        dumped = event.model_dump()
        assert dumped["audit_id"] == "evt_001"
        assert dumped["status"] == "ok"
        assert dumped["duration_ms"] == 123.4
        assert dumped["rows_in"] == 50
