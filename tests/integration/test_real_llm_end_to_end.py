"""End-to-end agent graph integration tests using real LLM provider and DuckDB database.

Executes complete natural language compliance queries through build_agent_graph(),
verifying end-to-end orchestration, intent parsing, planning, execution, scoring,
explanation polishing, and final AgentResponse generation.
"""

import pytest
from core.config import AppConfig
from storage.duckdb import get_duckdb_client
from storage.repositories import AuditRepo
from tools.registry import build_default_tool_registry
from llm.client import LLMClient
from agent.graph import build_agent_graph
from schemas.contracts import AgentResponse


class TestRealLLMEndToEnd:
    def test_full_pattern_search_query(self, llm_config: AppConfig, llm_client: LLMClient):
        db_client = get_duckdb_client(llm_config)
        try:
            audit_repo = AuditRepo(db_client)
            registry = build_default_tool_registry()

            graph = build_agent_graph(
                config=llm_config,
                registry=registry,
                db_client=db_client,
                audit_repo=audit_repo,
                llm_client=llm_client,
            )

            raw_query = "Find structuring patterns in the last 30 days"
            final_state = graph.invoke({"raw_query": raw_query})

            agent_response = final_state.get("agent_response")
            assert agent_response is not None
            assert isinstance(agent_response, AgentResponse)
            assert agent_response.raw_query == raw_query
            assert agent_response.explanation is not None
            assert len(agent_response.explanation) > 0

            # Verify audit trail recorded in DB
            events = audit_repo.get_by_query_id(agent_response.query_id)
            assert len(events) > 0
        finally:
            db_client.close()

    def test_full_entity_lookup_query(self, llm_config: AppConfig, llm_client: LLMClient):
        db_client = get_duckdb_client(llm_config)
        try:
            audit_repo = AuditRepo(db_client)
            registry = build_default_tool_registry()

            graph = build_agent_graph(
                config=llm_config,
                registry=registry,
                db_client=db_client,
                audit_repo=audit_repo,
                llm_client=llm_client,
            )

            raw_query = "Is account ID 4521 suspicious?"
            final_state = graph.invoke({"raw_query": raw_query})

            agent_response = final_state.get("agent_response")
            assert agent_response is not None
            assert isinstance(agent_response, AgentResponse)
            assert agent_response.explanation is not None
        finally:
            db_client.close()
