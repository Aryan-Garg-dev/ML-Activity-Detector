"""LangGraph agent execution graph compilation.

Two graph implementations are available:

1. `build_agent_graph()` (default) — Tool-calling LLM agent.
   The LLM dynamically decides which tools to invoke based on the query.
   No hardcoded decision table. Provider-neutral via LLMClient adapter.

2. `build_legacy_agent_graph()` — Decision-table planner (Phase 3/4 implementation).
   A Tier-1 decision table maps intent → fixed tool sequence, with LLM fallback.
   Kept for reference, testing, and rollback.

The active graph is controlled by `AppConfig.use_tool_calling_agent` (default: True).
"""

from typing import Any, Literal
from loguru import logger
from langgraph.graph import StateGraph, END

from core.config import AppConfig
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry
from agent.state import AgentState
from agent.planner import Planner
from agent.nodes import (
    parse_intent_node,
    plan_node,
    execute_step_node,
    explain_node,
    report_node,
    replan_node,
)
from agent.tool_agent import build_tool_calling_graph
from llm.client import LLMClient


def build_agent_graph(
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
    audit_repo: AuditRepo,
    llm_client: LLMClient | None = None,
) -> Any:
    """Build and compile the active agent graph.

    Routes to the tool-calling graph (default) or the legacy planner-based graph
    based on `config.use_tool_calling_agent`.
    """
    if config.use_tool_calling_agent:
        logger.info("Building tool-calling LLM agent graph (use_tool_calling_agent=True)")
        return build_tool_calling_graph(
            config=config,
            registry=registry,
            db_client=db_client,
            audit_repo=audit_repo,
            llm_client=llm_client,
        )
    else:
        logger.info("Building legacy planner-based agent graph (use_tool_calling_agent=False)")
        return build_legacy_agent_graph(
            config=config,
            registry=registry,
            db_client=db_client,
            audit_repo=audit_repo,
            llm_client=llm_client,
        )


def build_legacy_agent_graph(
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
    audit_repo: AuditRepo,
    llm_client: LLMClient | None = None,
) -> Any:
    """Build the legacy Tier-1 decision-table + Tier-2 LLM fallback planner graph.

    Preserved for testing, comparison, and rollback.
    Activate via: config.use_tool_calling_agent = False
    """
    planner = Planner(registry=registry, llm_client=llm_client)
    workflow = StateGraph(AgentState)

    def _parse_intent(state: AgentState) -> dict[str, Any]:
        return parse_intent_node(state, config, llm_client or LLMClient(config))

    def _plan(state: AgentState) -> dict[str, Any]:
        return plan_node(state, config, planner, audit_repo)

    def _execute_step(state: AgentState) -> dict[str, Any]:
        return execute_step_node(state, config, registry, db_client)

    def _explain(state: AgentState) -> dict[str, Any]:
        return explain_node(state, config, registry, db_client, llm_client)

    def _report(state: AgentState) -> dict[str, Any]:
        return report_node(state, config, registry, db_client)

    def _replan(state: AgentState) -> dict[str, Any]:
        return replan_node(state, planner)

    def should_continue(state: AgentState) -> Literal["execute_step", "replan", "explain"]:
        plan = state.get("execution_plan")
        step_index = state.get("current_step_index", 0)
        tool_results = state.get("tool_results", [])
        replan_count = state.get("replan_count", 0)
        if tool_results and tool_results[-1].status == "error":
            if replan_count < 1:
                logger.warning("Step execution failed. Routing to replan_node")
                return "replan"
        if plan and step_index < len(plan.steps):
            return "execute_step"
        return "explain"

    workflow.add_node("parse_intent", _parse_intent)
    workflow.add_node("plan", _plan)
    workflow.add_node("execute_step", _execute_step)
    workflow.add_node("explain", _explain)
    from agent.nodes import verify_node
    def _verify(state: AgentState) -> dict[str, Any]:
        return verify_node(state)
    workflow.add_node("verify", _verify)
    workflow.add_node("report", _report)
    workflow.add_node("replan", _replan)

    workflow.set_entry_point("parse_intent")
    workflow.add_edge("parse_intent", "plan")
    workflow.add_edge("plan", "execute_step")
    workflow.add_conditional_edges(
        "execute_step",
        should_continue,
        {"execute_step": "execute_step", "replan": "replan", "explain": "explain"},
    )
    workflow.add_edge("replan", "execute_step")
    workflow.add_edge("explain", "verify")
    workflow.add_edge("verify", "report")
    workflow.add_edge("report", END)

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logger.info("Compiled legacy LangGraph agent workflow graph successfully")
    return app
