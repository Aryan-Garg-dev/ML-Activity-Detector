"""AgentState definition for LangGraph state machine."""

from typing import TypedDict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from schemas.contracts import QuerySpec, ExecutionPlan, ToolResult, AgentResponse
from schemas.audit import AuditEvent
from schemas.risk import RiskAssessment, FlaggedItem


class AgentState(TypedDict, total=False):
    """Shared state dictionary passed across all nodes in the LangGraph execution graph.

    Carries both AML domain fields (query_spec, risk_assessments, etc.) and
    the LangGraph message list used by the tool-calling agent loop.
    The `messages` field uses `add_messages` reducer so LangGraph appends
    instead of replacing when nodes return {"messages": [new_message]}.
    """

    # LangGraph tool-calling loop messages (reducer: append, not replace)
    messages: Annotated[list[BaseMessage], add_messages]

    query_id: str
    raw_query: str
    query_spec: QuerySpec | None
    execution_plan: ExecutionPlan | None
    current_step_index: int
    tool_results: list[ToolResult]
    data_query_results: dict[str, Any]
    eda_results: dict[str, Any]
    feature_results: dict[str, Any]
    detection_results: dict[str, Any]
    scoring_results: dict[str, Any]
    explanation_results: dict[str, Any]
    entity_lookup_results: dict[str, Any]
    audit_events: list[AuditEvent]

    risk_assessments: list[RiskAssessment]
    flagged_items: list[FlaggedItem]
    charts: list[str]
    metrics: dict[str, Any]
    explanation: str | None
    agent_response: AgentResponse | None
    replan_count: int
    error: str | None
