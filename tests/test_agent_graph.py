"""Integration tests for compiled LangGraph agent graph."""

import pytest
from core.config import AppConfig
from core.types import IntentType, ToolName, ProviderName
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry, ToolSpec
from schemas.contracts import ToolResult, AgentResponse, QuerySpec
from agent.graph import build_legacy_agent_graph
from llm.client import LLMClient
from llm.providers.base import BaseProviderAdapter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage


from langchain_core.outputs import ChatResult, ChatGeneration


class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = """{
          "intent_type": "pattern_search",
          "pattern_type": "structuring",
          "target_entity_id": null,
          "filters": {},
          "aggregation_spec": {},
          "raw_query": "Find structuring patterns"
        }"""
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock"


class MockAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return MockChatModel()


@pytest.fixture
def mock_registry():
    registry = ToolRegistry()
    from pydantic import BaseModel

    class GenericInput(BaseModel):
        model_config = {"extra": "allow"}

    def _dummy_tool(input_val, config, db_client, query_id, step_id):
        return ToolResult(
            tool_name=ToolName.DATA_QUERY,
            step_id=step_id,
            status="ok",
            data={"rows_in": 10, "rows_out": 5, "agent_response": {"query_id": query_id, "raw_query": "query", "intent": "pattern_search", "explanation": "Test explanation"}},
            rows_count=5,
        )

    for tool_name in ToolName:
        registry.register(
            ToolSpec(
                name=tool_name,
                description=f"Dummy {tool_name}",
                input_schema=GenericInput,
                callable=_dummy_tool,
            )
        )
    return registry


def test_agent_graph_execution(mock_registry, tmp_path):
    db_path = tmp_path / "test.duckdb"
    config = AppConfig(db_path=db_path)
    db_client = DuckDBClient(db_path=db_path)
    db_client.init_schema()
    audit_repo = AuditRepo(db_client)
    llm_client = LLMClient(config=config, adapter_override=MockAdapter())


    graph = build_legacy_agent_graph(
        config=config,
        registry=mock_registry,
        db_client=db_client,
        audit_repo=audit_repo,
        llm_client=llm_client,
    )

    initial_state = {
        "raw_query": "Find structuring patterns in the last 30 days",
    }

    final_state = graph.invoke(initial_state)
    assert final_state["query_spec"] is not None
    assert final_state["execution_plan"] is not None
    assert len(final_state["tool_results"]) > 0
    assert final_state["agent_response"] is not None


def test_agent_graph_executes_terminal_tools_once(tmp_path):
    db_path = tmp_path / "test_terminal.duckdb"
    config = AppConfig(db_path=db_path)
    db_client = DuckDBClient(db_path=db_path)
    db_client.init_schema()
    audit_repo = AuditRepo(db_client)
    llm_client = LLMClient(config=config, adapter_override=MockAdapter())

    registry = ToolRegistry()
    call_counts: dict[ToolName, int] = {tool_name: 0 for tool_name in ToolName}
    received_inputs: dict[ToolName, dict] = {}

    from pydantic import BaseModel

    class GenericInput(BaseModel):
        model_config = {"extra": "allow"}

    def make_tool(tool_name: ToolName):
        def _tool(input_val, config, db_client, query_id, step_id):
            call_counts[tool_name] += 1
            received_inputs[tool_name] = input_val.model_dump()
            if tool_name == ToolName.FEATURE_ENGINEERING:
                return ToolResult(
                    tool_name=tool_name,
                    step_id=step_id,
                    status="ok",
                    data={"feature_sets": {"101": {"volume_txn_count": 3}}, "rows_in": 1},
                    rows_count=1,
                )
            if tool_name == ToolName.DETECTION:
                return ToolResult(
                    tool_name=tool_name,
                    step_id=step_id,
                    status="ok",
                    data={
                        "detection_results": {"101": {"triggered_signals": ["R_STRUCT_01"], "rule_flags": ["R_STRUCT_01"], "ml_anomaly_score": 0.9, "confidence": 0.8}},
                        "flagged_results": {"101": {}},
                        "flagged_count": 1,
                        "rows_in": 1,
                    },
                    rows_count=1,
                )
            if tool_name == ToolName.SCORING:
                assert input_val.model_dump().get("detection_results")
                return ToolResult(
                    tool_name=tool_name,
                    step_id=step_id,
                    status="ok",
                    data={
                        "risk_assessments": [{"entity_id": "101", "composite_score": 88.0, "confidence": 0.8, "risk_level": "high", "escalation_action": "report", "triggered_signals": ["R_STRUCT_01"]}],
                        "assessed_count": 1,
                        "rows_in": 1,
                    },
                    rows_count=1,
                )
            if tool_name == ToolName.EXPLANATION:
                assert input_val.model_dump().get("risk_assessments")
                assert input_val.model_dump().get("feature_sets")
                return ToolResult(
                    tool_name=tool_name,
                    step_id=step_id,
                    status="ok",
                    data={"explanations": {"101": "Grounded explanation"}, "count": 1, "rows_in": 1},
                    rows_count=1,
                )
            if tool_name == ToolName.REPORTING:
                payload = input_val.model_dump()
                assert payload.get("detection_results")
                assert payload.get("scoring_results")
                assert payload.get("tools_invoked") == [
                    str(ToolName.FEATURE_ENGINEERING),
                    str(ToolName.DETECTION),
                    str(ToolName.SCORING),
                    str(ToolName.EXPLANATION),
                ]
                return ToolResult(
                    tool_name=tool_name,
                    step_id=step_id,
                    status="ok",
                    data={"agent_response": {"query_id": query_id, "raw_query": "query", "intent": "pattern_search", "explanation": "Test explanation"}},
                    rows_count=1,
                )
            return ToolResult(tool_name=tool_name, step_id=step_id, status="ok", data={"rows_in": 1}, rows_count=1)

        return _tool

    for tool_name in ToolName:
        registry.register(
            ToolSpec(
                name=tool_name,
                description=f"Dummy {tool_name}",
                input_schema=GenericInput,
                callable=make_tool(tool_name),
            )
        )

    graph = build_legacy_agent_graph(
        config=config,
        registry=registry,
        db_client=db_client,
        audit_repo=audit_repo,
        llm_client=llm_client,
    )

    final_state = graph.invoke({"raw_query": "Find structuring patterns in the last 30 days"})

    assert call_counts[ToolName.EXPLANATION] == 1
    assert call_counts[ToolName.REPORTING] == 1
    assert final_state["agent_response"] is not None
    assert final_state["scoring_results"]
    assert final_state["detection_results"]

    db_client.close()

    db_client.close()
