"""Unit tests for autonomous graph fallback to fixed-flow deterministic planner."""

import pytest
from unittest.mock import MagicMock
from core.config import AppConfig
from core.types import IntentType, PatternType
from schemas.contracts import QuerySpec
from agent.state import AgentState
from agent.tool_agent import build_tool_calling_graph
from tools.registry import ToolRegistry, build_default_tool_registry
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo


def test_autonomous_fallback_when_undecided(tmp_path):
    """Test that when autonomous tool execution yields no domain data, the graph falls back to fixed flow."""
    db_path = tmp_path / "test.duckdb"
    config = AppConfig(
        db_path=db_path,
        use_tool_calling_agent=True,
    )
    db_client = DuckDBClient(db_path=db_path)
    db_client.init_schema()
    audit_repo = AuditRepo(db_client)
    registry = build_default_tool_registry()

    from langchain_core.messages import AIMessage

    query_spec = QuerySpec(
        intent_type=IntentType.PATTERN_SEARCH,
        pattern_type=PatternType.STRUCTURING,
        raw_query="Find structuring patterns in last 30 days",
    )

    # Mock LLM that returns NO tool calls (undecided) and text string for explanation polish
    mock_llm = MagicMock()
    mock_llm.structured_output.return_value = query_spec
    mock_llm.invoke.return_value = "Explanation: No suspicious structuring patterns found in last 30 days."
    mock_llm.model.bind_tools.return_value.invoke.return_value = AIMessage(content="I am unsure which tool to call.")

    # Compile graph
    graph = build_tool_calling_graph(config, registry, db_client, audit_repo, llm_client=mock_llm)

    initial_state = {
        "messages": [],
        "raw_query": "Find structuring patterns in last 30 days",
        "query_id": "test_fallback_01",
        "query_spec": query_spec,
    }

    # Invoke graph
    res = graph.invoke(initial_state, config={"configurable": {"thread_id": "test"}})

    # Verify that fallback_fixed_flow ran and created an execution plan
    assert "execution_plan" in res
    assert res["execution_plan"] is not None
    assert len(res["execution_plan"].steps) > 0
    # Verify report was generated
    assert res.get("agent_response") is not None
    assert res["agent_response"].execution_summary.get("status") in ("ok", "partial")
