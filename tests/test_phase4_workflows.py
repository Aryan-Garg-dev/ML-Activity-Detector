"""Aggressive domain workflow tests for Phase 4 query-aware routing paths and structured outputs."""

import pytest
from core.config import AppConfig
from core.types import IntentType, ToolName
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import build_default_tool_registry
from agent.graph import build_legacy_agent_graph
from llm.client import LLMClient
from llm.providers.base import BaseProviderAdapter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class MockPlannerChatModel(BaseChatModel):
    query_spec_json: str = ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        generation = ChatGeneration(message=AIMessage(content=self.query_spec_json))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock_planner"



class MockPlannerAdapter(BaseProviderAdapter):
    def __init__(self, query_spec_json: str):
        self.query_spec_json = query_spec_json

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return MockPlannerChatModel(query_spec_json=self.query_spec_json)



@pytest.fixture
def test_setup(tmp_path):
    db_path = tmp_path / "test_phase4.duckdb"
    config = AppConfig(db_path=db_path)
    db_client = DuckDBClient(db_path=db_path)
    db_client.init_schema()

    # Load synthetic test rows into accounts and transactions
    db_client.execute(
        "INSERT INTO accounts (account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id) VALUES (4521, 'C999', 50000.0, 'US', 'checking', true, 1)"
    )
    db_client.execute(
        "INSERT INTO accounts (account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id) VALUES (8888, 'C888', 20000.0, 'US', 'checking', false, 1)"
    )
    db_client.execute(
        "INSERT INTO transactions (tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id) VALUES (101, 4521, 8888, 'TRANSFER', 9500.0, 1700000000, true, -1)"
    )


    audit_repo = AuditRepo(db_client)
    registry = build_default_tool_registry()

    yield config, db_client, audit_repo, registry

    db_client.close()


def run_workflow_query(test_setup, raw_query: str, query_spec_json: str):
    config, db_client, audit_repo, registry = test_setup
    llm_client = LLMClient(config=config, adapter_override=MockPlannerAdapter(query_spec_json))

    graph = build_legacy_agent_graph(
        config=config,
        registry=registry,
        db_client=db_client,
        audit_repo=audit_repo,
        llm_client=llm_client,
    )

    final_state = graph.invoke({"raw_query": raw_query}, config={"configurable": {"thread_id": "test"}})
    return final_state


# 1. Pattern Search: Structuring
def test_workflow_pattern_search_structuring(test_setup):
    spec_json = '{"intent_type": "pattern_search", "pattern_type": "structuring", "raw_query": "Find structuring"}'
    state = run_workflow_query(test_setup, "Find structuring patterns in the last 30 days", spec_json)
    resp = state["agent_response"]

    assert resp is not None
    invoked = resp.execution_summary["tools_invoked"]
    assert ToolName.FEATURE_ENGINEERING in invoked
    assert ToolName.DETECTION in invoked
    assert ToolName.SCORING in invoked
    assert ToolName.EDA not in invoked


# 2. Pattern Search: Smurfing
def test_workflow_pattern_search_smurfing(test_setup):
    spec_json = '{"intent_type": "pattern_search", "pattern_type": "smurfing", "raw_query": "Detect smurfing"}'
    state = run_workflow_query(test_setup, "Detect smurfing accounts with high fan-in degree", spec_json)
    resp = state["agent_response"]
    assert resp.intent == IntentType.PATTERN_SEARCH


# 3. Pattern Search: Layering
def test_workflow_pattern_search_layering(test_setup):
    spec_json = '{"intent_type": "pattern_search", "pattern_type": "layering", "raw_query": "Identify layering"}'
    state = run_workflow_query(test_setup, "Identify layering transfers across multiple counterparties", spec_json)
    resp = state["agent_response"]
    assert resp is not None


# 4. Pattern Search: Rapid Cashout
def test_workflow_pattern_search_rapid_cashout(test_setup):
    spec_json = '{"intent_type": "pattern_search", "pattern_type": "rapid_cashout", "raw_query": "Find rapid cashout"}'
    state = run_workflow_query(test_setup, "Find accounts with rapid cashout patterns", spec_json)
    resp = state["agent_response"]
    assert resp is not None


# 5. Pattern Search: Velocity
def test_workflow_pattern_search_velocity(test_setup):
    spec_json = '{"intent_type": "pattern_search", "pattern_type": "velocity", "raw_query": "Flag velocity"}'
    state = run_workflow_query(test_setup, "Flag transaction velocity anomalies", spec_json)
    resp = state["agent_response"]
    assert resp is not None


# 6. Aggregation Query Path
def test_workflow_aggregation_query(test_setup):
    config, db_client, audit_repo, registry = test_setup
    # Insert 11 transactions under $10,000 for account 4521
    for i in range(11):
        db_client.execute(
            f"INSERT INTO transactions (tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id) VALUES ({200 + i}, 4521, 8888, 'TRANSFER', 9500.0, 1700000000 + {i}, false, -1)"
        )

    spec_json = '{"intent_type": "aggregation_query", "pattern_type": "structuring", "filters": {"max_amount": 9999.0}, "aggregation_spec": {"min_count": 10}, "raw_query": "Which customers made 10+ transactions under $10,000?"}'
    state = run_workflow_query(test_setup, "Which customers made 10+ transactions under $10,000?", spec_json)
    resp = state["agent_response"]

    assert resp is not None
    assert resp.intent == IntentType.AGGREGATION_QUERY
    invoked = resp.execution_summary["tools_invoked"]
    assert ToolName.DATA_QUERY in invoked
    assert ToolName.DETECTION not in invoked

    # Assert non-empty flagged items and populated account domain data
    assert len(resp.flagged_items) > 0
    flagged = resp.flagged_items[0]
    assert flagged.entity_id == "4521"
    assert flagged.account_record is not None
    assert flagged.account_record["account_id"] == 4521

    # Assert non-zero assessed and flagged metrics
    assert resp.metrics["total_entities_assessed"] > 0
    assert resp.metrics["flagged_count"] > 0



# 7. Entity Lookup Path & Structured Domain Models
def test_workflow_entity_lookup_structured_models(test_setup):
    spec_json = '{"intent_type": "entity_lookup", "target_entity_id": "4521", "raw_query": "Is customer ID 4521 suspicious?"}'
    state = run_workflow_query(test_setup, "Is customer ID 4521 suspicious?", spec_json)
    resp = state["agent_response"]

    assert resp is not None

    invoked = resp.execution_summary["tools_invoked"]
    assert ToolName.ENTITY_LOOKUP in invoked
    assert len(resp.flagged_items) > 0

    flagged = resp.flagged_items[0]
    assert flagged.entity_id == "4521"
    # Verify structured account_record and recent_transactions models attached!
    assert flagged.account_record is not None
    assert flagged.account_record["account_id"] == 4521
    assert flagged.account_record["country"] == "US"
    assert len(flagged.recent_transactions) > 0
    assert flagged.recent_transactions[0]["tx_id"] == 101



# 8. Broad EDA Path
def test_workflow_broad_eda(test_setup):
    spec_json = '{"intent_type": "broad_eda", "pattern_type": "unknown", "raw_query": "Overview of volume trends"}'
    state = run_workflow_query(test_setup, "Give me an overview of overall transaction volume trends and distributions", spec_json)
    resp = state["agent_response"]

    assert resp is not None
    invoked = resp.execution_summary["tools_invoked"]
    assert ToolName.EDA in invoked


# 9. Batch Risk Scoring Path
def test_workflow_risk_scoring_batch(test_setup):
    spec_json = '{"intent_type": "risk_scoring_batch", "pattern_type": "velocity", "raw_query": "Batch risk scoring"}'
    state = run_workflow_query(test_setup, "Perform batch risk scoring across all accounts", spec_json)
    resp = state["agent_response"]

    assert resp is not None
    invoked = resp.execution_summary["tools_invoked"]
    assert ToolName.FEATURE_ENGINEERING in invoked
    assert ToolName.DETECTION in invoked
    assert ToolName.SCORING in invoked
