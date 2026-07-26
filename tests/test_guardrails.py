"""Unit tests for plan validation guardrails and audit logging."""

import pytest
from pydantic import BaseModel, Field

from core.types import ToolName
from schemas.contracts import ExecutionPlan, PlanStep
from tools.registry import ToolRegistry, ToolSpec
from agent.guardrails import validate_execution_plan, audit_write_plan_start


class MockAuditRepo:
    def __init__(self):
        self.events = []

    def save_audit_event(self, event):
        self.events.append(event)

    save_event = save_audit_event



class DummyInput(BaseModel):
    limit: int = Field(default=10)


@pytest.fixture
def dummy_registry():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=ToolName.DATA_QUERY,
            description="Dummy data query",
            input_schema=DummyInput,
            callable=lambda *args: None,
        )
    )
    return registry


def test_guardrail_valid_plan(dummy_registry):
    plan = ExecutionPlan(
        plan_id="p1",
        query_id="q1",
        steps=[PlanStep(step_id=1, tool_name=ToolName.DATA_QUERY, args={"limit": 5}, reason="Query")],
    )
    is_valid, reason = validate_execution_plan(plan, dummy_registry)
    assert is_valid is True
    assert reason is None


def test_guardrail_unregistered_tool(dummy_registry):
    plan = ExecutionPlan(
        plan_id="p1",
        query_id="q1",
        steps=[PlanStep(step_id=1, tool_name=ToolName.EDA, args={}, reason="Unregistered tool")],
    )
    is_valid, reason = validate_execution_plan(plan, dummy_registry)
    assert is_valid is False
    assert "not registered" in reason


def test_guardrail_step_cap(dummy_registry):
    steps = [
        PlanStep(step_id=i, tool_name=ToolName.DATA_QUERY, args={"limit": 1}, reason="Exceed cap")
        for i in range(1, 10)
    ]
    plan = ExecutionPlan(plan_id="p1", query_id="q1", steps=steps)
    is_valid, reason = validate_execution_plan(plan, dummy_registry, max_steps=8)
    assert is_valid is False
    assert "exceeds maximum allowed steps" in reason


def test_guardrail_audit_log_writing():
    audit_repo = MockAuditRepo()
    plan = ExecutionPlan(
        plan_id="p1",
        query_id="q1",
        steps=[PlanStep(step_id=1, tool_name=ToolName.DATA_QUERY, args={}, reason="Test audit")],
    )
    event = audit_write_plan_start(plan, audit_repo)
    assert len(audit_repo.events) == 1
    assert audit_repo.events[0].query_id == "q1"
    assert audit_repo.events[0].rows_out == 1
