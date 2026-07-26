"""Guardrails for plan validation, schema checking, and audit logging."""

import uuid
from datetime import datetime, timezone
from loguru import logger
from pydantic import ValidationError

from core.types import ToolName
from schemas.contracts import ExecutionPlan
from schemas.audit import AuditEvent
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry


class GuardrailValidationError(Exception):
    """Raised when an execution plan fails guardrail validation."""


def validate_execution_plan(
    plan: ExecutionPlan,
    registry: ToolRegistry,
    max_steps: int = 8,
) -> tuple[bool, str | None]:
    """Validate an ExecutionPlan against safety guardrails:
      1. Step count <= max_steps
      2. All tool names belong to the ToolRegistry whitelist
      3. Step arguments conform to tool input schemas

    Returns:
        (is_valid, error_reason)
    """
    if not plan.steps:
        return False, "Execution plan contains no steps"

    if len(plan.steps) > max_steps:
        msg = f"Plan step count ({len(plan.steps)}) exceeds maximum allowed steps ({max_steps})"
        logger.warning(msg)
        return False, msg

    for step in plan.steps:
        # 1. Check tool is registered
        if not registry.has_tool(step.tool_name):
            msg = f"Step {step.step_id} requests tool '{step.tool_name}' which is not registered"
            logger.warning(msg)
            return False, msg

        # 2. Check input args match tool schema
        spec = registry.get(step.tool_name)
        try:
            spec.input_schema(**step.args)
        except (ValidationError, TypeError) as e:
            msg = f"Step {step.step_id} ({step.tool_name}) args failed validation: {e}"
            logger.warning(msg)
            return False, msg

    return True, None


def audit_write_plan_start(
    plan: ExecutionPlan,
    audit_repo: AuditRepo,
    config_version: str = "v1.0.0",
) -> AuditEvent:
    """Write an audit event for the execution plan before tool execution starts (Core Design Rule 8)."""
    event = AuditEvent(
        audit_id=f"audit_plan_{uuid.uuid4().hex[:8]}",
        query_id=plan.query_id,
        tool_name=ToolName.REPORTING,
        step_id=0,
        started_at=datetime.now(timezone.utc).isoformat(),
        ended_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=0.0,
        rows_in=0,
        rows_out=len(plan.steps),
        provider=None,
        model_name=None,
        config_version=config_version,
        status="ok",
        error_summary=None,
    )

    try:
        audit_repo.save_audit_event(event)
        logger.info("Audit log written for ExecutionPlan [{plan_id}]", plan_id=plan.plan_id)

    except Exception as e:
        logger.error("Failed to write plan audit log: {err}", err=e)

    return event


# AML-relevant keywords that indicate a valid financial compliance query
_AML_KEYWORDS = {
    "transaction", "account", "customer", "suspicious", "fraud", "money",
    "laundering", "aml", "structuring", "smurfing", "layering", "cashout",
    "cash out", "velocity", "threshold", "compliance", "risk", "alert",
    "flagged", "pattern", "anomaly", "anomalies", "unusual", "transfer",
    "payment", "deposit", "withdrawal", "counterpart", "counterparty",
    "country", "cross-border", "international", "report", "monitor",
    "review", "escalat", "detect", "score", "network", "graph", "fan-in",
    "fan-out", "eda", "overview", "distribution", "trend", "profile",
    "dataset", "bank", "financial", "audit", "batch", "entity",
}

# Keywords that strongly indicate an off-topic query unrelated to AML
_OFFTOPIC_KEYWORDS = {
    "weather", "recipe", "cook", "bmi", "exercise", "fitness",
    "movie", "music", "song", "game", "sport", "football", "basketball",
    "poem", "poetry", "joke", "riddle", "trivia",
    "translate", "language", "grammar", "spell",
    "diet", "calories", "workout", "meditation",
    "astronomy", "planet", "galaxy", "space travel",
    "homework", "essay", "thesis",
}


def validate_query_relevance(raw_query: str) -> tuple[bool, str]:
    """Check if a query is relevant to AML/financial compliance domain.

    Returns:
        (is_relevant, rejection_reason) — if is_relevant is False, rejection_reason
        explains why the query was rejected.
    """
    if not raw_query or not raw_query.strip():
        return False, "Empty query. Please provide a natural language AML compliance query."

    q_lower = raw_query.lower().strip()
    words = set(q_lower.split())

    # Check for explicit off-topic indicators
    for keyword in _OFFTOPIC_KEYWORDS:
        if keyword in q_lower:
            return False, (
                f"This query appears to be about '{keyword}' which is outside the scope of "
                f"this AML Suspicious Activity Detection system. Please ask a question related "
                f"to financial transactions, account risk, or compliance analysis."
            )

    # Check for at least one AML-relevant keyword
    has_aml_keyword = any(kw in q_lower for kw in _AML_KEYWORDS)

    # Also accept queries with numeric IDs (likely entity lookups)
    import re
    has_numeric_id = bool(re.search(r'\d{3,}', q_lower))

    if has_aml_keyword or has_numeric_id:
        return True, ""

    # No AML keywords and no numeric IDs — reject as irrelevant
    logger.debug("Query has no AML keywords and is not clearly on-topic: '{query}'", query=raw_query)
    return False, (
        "This query does not appear to be related to AML compliance, financial transactions, "
        "or account risk analysis. Please ask a question about suspicious activity detection, "
        "transaction patterns, or account risk assessment."
    )
