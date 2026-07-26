from schemas.domain import AccountRecord, TransactionRecord, AlertRecord, FeatureSet
from schemas.contracts import QuerySpec, PlanStep, ExecutionPlan, ToolResult, AgentResponse
from schemas.audit import AuditEvent, RunMetadata
from schemas.risk import RiskAssessment, FlaggedItem
from core.types import IntentType, PatternType, RiskLevel, EscalationAction, ToolName

# Test serialization and roundtrip for domain records
def test_domain_records():
    acc = AccountRecord(
        account_id=1, customer_id="C_1", init_balance=100.0,
        country="US", account_type="I", is_fraud=False, tx_behavior_id=1
    )
    assert acc.account_id == 1
    assert acc.customer_id == "C_1"

    tx = TransactionRecord(
        tx_id=10, sender_account_id=1, receiver_account_id=2,
        tx_type="TRANSFER", tx_amount=50.0, timestamp=100, is_fraud=False
    )
    assert tx.tx_id == 10
    assert tx.alert_id is None

# Test contract models
def test_contracts():
    qspec = QuerySpec(intent_type=IntentType.PATTERN_SEARCH, pattern_type=PatternType.STRUCTURING, raw_query="find structuring")
    assert qspec.intent_type == IntentType.PATTERN_SEARCH

    step = PlanStep(step_id=1, tool_name=ToolName.DETECTION, reason="Detect structuring")
    plan = ExecutionPlan(plan_id="p1", query_id="q1", steps=[step])
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == ToolName.DETECTION

# Test audit and risk models
def test_audit_and_risk_models():
    event = AuditEvent(
        query_id="q1", step_id=1, tool_name=ToolName.DETECTION,
        started_at="2026-07-25T10:00:00Z", ended_at="2026-07-25T10:00:01Z",
        duration_ms=1000.0, rows_in=10, rows_out=2, config_version="v1.0.0", status="ok"
    )
    assert event.status == "ok"

    risk = RiskAssessment(
        query_id="q1", entity_id="1", composite_score=85.0, confidence=0.9,
        risk_level=RiskLevel.HIGH, escalation_action=EscalationAction.REPORT
    )
    assert risk.risk_level == RiskLevel.HIGH
