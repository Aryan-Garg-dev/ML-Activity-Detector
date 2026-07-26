"""FastAPI integration tests for /health, /query, /audit/{query_id}, and Swagger OpenAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

from core.config import AppConfig
from storage.duckdb import DuckDBClient
from llm.client import LLMClient
from llm.providers.base import BaseProviderAdapter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class MockApiChatModel(BaseChatModel):
    """Minimal mock model for API integration tests.

    - Supports bind_tools() so it works with the tool-calling graph.
    - Returns a QuerySpec-shaped JSON on the first invocation (parse_intent).
    - Returns a plain AIMessage with no tool_calls on subsequent invocations
      (tool_agent turns) so the graph skips the tool loop and proceeds directly
      to extract_state → explain → report with empty results.
    """

    def bind_tools(self, tools, **kwargs):
        """Return self — the mock handles all invocations uniformly."""
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Check if this is a structured-output / intent-parsing call:
        # Those messages contain the raw user query only (no prior AIMessage).
        # Tool-agent calls contain system + human + possibly prior messages.
        has_prior_ai = any(isinstance(m, AIMessage) for m in messages)

        if not has_prior_ai:
            # parse_intent call — return a valid QuerySpec JSON
            content = (
                '{"intent_type": "pattern_search", '
                '"pattern_type": "structuring", '
                '"target_entity_id": null, '
                '"filters": {}, '
                '"aggregation_spec": {}, '
                '"raw_query": "Find structuring patterns"}'
            )
        else:
            # tool_agent call — return plain text (no tool_calls) so graph
            # routes immediately to extract_state with empty domain state
            content = "Analysis complete. No suspicious activity detected in the test dataset."

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock_api"


class MockApiAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return MockApiChatModel()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api_test.duckdb"
    config = AppConfig(db_path=db_path)
    db_client = DuckDBClient(db_path=db_path)
    db_client.init_schema()
    db_client.close()

    # Monkeypatch AppConfig.load to use temporary database
    monkeypatch.setattr(AppConfig, "load", lambda *args, **kwargs: config)
    # Monkeypatch LLMClient to use MockApiAdapter (replaces all provider calls)
    monkeypatch.setattr(
        LLMClient,
        "__init__",
        lambda self, cfg, adapter_override=None: (
            setattr(self, "config", cfg)
            or setattr(self, "adapter", MockApiAdapter())
            or setattr(self, "model", MockApiChatModel())
        ),
    )

    with TestClient(app) as test_client:
        yield test_client


def test_swagger_openapi_docs_endpoints(client):
    # Verify Swagger UI
    resp_docs = client.get("/docs")
    assert resp_docs.status_code == 200
    assert "swagger-ui" in resp_docs.text.lower()

    # Verify ReDoc UI
    resp_redoc = client.get("/redoc")
    assert resp_redoc.status_code == 200
    assert "redoc" in resp_redoc.text.lower()

    # Verify OpenAPI Schema JSON
    resp_schema = client.get("/openapi.json")
    assert resp_schema.status_code == 200
    schema_json = resp_schema.json()
    assert schema_json["info"]["title"] == "AML Suspicious Activity Detector Agent API"
    assert "/query" in schema_json["paths"]
    assert "/health" in schema_json["paths"]


def test_health_check_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "duckdb_connected" in data
    assert "active_provider" in data


def test_process_query_endpoint(client):
    payload = {
        "query": "Find structuring patterns in the last 30 days",
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()

    assert "query_id" in data
    assert data["raw_query"] == payload["query"]
    assert "execution_summary" in data
    assert "flagged_items" in data
    assert "explanation" in data


def test_process_query_empty_string_bad_request(client):
    payload = {"query": "   "}
    response = client.post("/query", json=payload)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_get_audit_trace_endpoint(client):
    # First execute a query to populate audit trace
    query_resp = client.post("/query", json={"query": "Is customer ID 4521 suspicious?"})
    assert query_resp.status_code == 200, f"Pre-condition failed: {query_resp.text}"
    qid = query_resp.json()["query_id"]

    # Fetch audit trace for qid
    audit_resp = client.get(f"/audit/{qid}")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["query_id"] == qid
    assert audit_data["event_count"] >= 1


def test_ingest_endpoint(client, monkeypatch):
    """Test the /ingest endpoint with dummy CSV files."""
    dummy_csv = b"id,val\n1,A\n2,B"
    files = {
        "accounts": ("accounts.csv", dummy_csv, "text/csv"),
        "transactions": ("transactions.csv", dummy_csv, "text/csv"),
    }
    
    # Mock ingest_csv to avoid actually calling DuckDB
    monkeypatch.setattr("api.main.DuckDBClient.ingest_csv", lambda self, t, p: 2)
    
    response = client.post("/ingest", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["ingested_rows"]["accounts"] == 2
    assert data["ingested_rows"]["transactions"] == 2
