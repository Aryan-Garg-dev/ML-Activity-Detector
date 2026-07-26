from typing import Any
from pydantic import BaseModel, Field, field_validator
from core.types import IntentType, PatternType, ToolName


# Intent parsing result containing query intent, pattern, filters, and target entity
class QuerySpec(BaseModel):
    intent_type: IntentType
    pattern_type: PatternType = PatternType.UNKNOWN
    target_entity_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    aggregation_spec: dict[str, Any] = Field(default_factory=dict)
    raw_query: str

    @field_validator("intent_type", mode="before")
    @classmethod
    def _normalize_intent_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for member in IntentType:
                if member.value == v_clean:
                    return member
        return v

    @field_validator("pattern_type", mode="before")
    @classmethod
    def _normalize_pattern_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for member in PatternType:
                if member.value == v_clean:
                    return member
        return v

# Single execution step specification in an agent plan
class PlanStep(BaseModel):
    step_id: int
    tool_name: ToolName
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str

    @field_validator("tool_name", mode="before")
    @classmethod
    def _normalize_tool_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().lower()
            for member in ToolName:
                if member.value == v_clean:
                    return member
        return v


# Execution plan created by the agent planner before tool execution
class ExecutionPlan(BaseModel):
    plan_id: str
    query_id: str
    steps: list[PlanStep]
    tools_skipped: list[ToolName] = Field(default_factory=list)
    skip_reasons: dict[str, str] = Field(default_factory=dict)

# Standard typed output returned by an executed tool
class ToolResult(BaseModel):
    tool_name: ToolName
    step_id: int
    status: str
    data: dict[str, Any] = Field(default_factory=dict)
    rows_count: int = 0
    duration_ms: float = 0.0
    error_summary: str | None = None

# Final structured response returned to user or API client
class AgentResponse(BaseModel):
    query_id: str
    raw_query: str
    intent: IntentType
    execution_summary: dict[str, Any] = Field(default_factory=dict)
    flagged_items: list[Any] = Field(
        default_factory=list,
        description="List of FlaggedItem domain models with account_record and recent_transactions fields.",
    )
    charts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    explanation: str

    @field_validator("flagged_items", mode="before")
    @classmethod
    def _coerce_flagged_items(cls, v: Any) -> Any:
        """Coerce raw dicts to FlaggedItem models for typed domain model access.

        Accepts both FlaggedItem instances and dicts. Dicts are validated against
        FlaggedItem schema to ensure account_record and recent_transactions fields
        are present and properly typed.
        """
        from schemas.risk import FlaggedItem

        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if isinstance(item, dict):
                try:
                    result.append(FlaggedItem.model_validate(item))
                except Exception:
                    # Keep raw dict if it doesn't match FlaggedItem schema (e.g. aggregation results)
                    result.append(item)
            else:
                result.append(item)
        return result
