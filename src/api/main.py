"""FastAPI Application for AML Suspicious Activity Detector Agent.

Provides HTTP REST endpoints for query processing, service health, audit retrieval,
and full report downloads. Interactive OpenAPI/Swagger documentation at /docs.

Key design decisions:
- POST /query returns a compact AgentResponse by default (no raw account/transaction dumps)
- POST /query?detailed=true returns full domain records for reviewers
- GET /report/{query_id}/download returns a downloadable JSON or CSV report file
- Smart semantic cache (TF-IDF) is checked before running the agent pipeline
"""

import io
import csv
import json
from typing import Any
from fastapi import FastAPI, HTTPException, Query, status, Path as FastAPIPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ProviderName, IntentType
from core.cache import QueryCache
from storage.duckdb import get_duckdb_client, DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import build_default_tool_registry
from schemas.contracts import AgentResponse
from schemas.audit import AuditEvent
from agent.graph import build_agent_graph
from llm.client import LLMClient


class QueryRequest(BaseModel):
    """Request payload for natural language query execution."""

    query: str = Field(
        ...,
        description="Natural language compliance query string.",
        json_schema_extra={"example": "Find structuring patterns in the last 30 days"},
    )
    provider: ProviderName | None = Field(
        default=None,
        description="Optional LLM provider override.",
        json_schema_extra={"example": ProviderName.GROQ},
    )
    model_id: str | None = Field(
        default=None,
        description="Optional model ID override.",
        json_schema_extra={"example": "llama-3.3-70b-versatile"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find structuring patterns in the last 30 days",
                    "provider": "groq",
                    "model_id": "llama-3.3-70b-versatile",
                },
                {
                    "query": "Which customers made 10+ transactions under $10,000?",
                    "provider": "lmstudio",
                    "model_id": "mistralai/ministral-3-3b",
                },
                {
                    "query": "Is customer ID 4521 suspicious?",
                },
            ]
        }
    }


class AgentResponseSummary(BaseModel):
    """Compact AgentResponse returned by default — omits raw account/transaction dumps.

    Full domain records (account_record, recent_transactions) are only included when
    the client requests detailed=true or calls the /report/{query_id}/download endpoint.
    """

    query_id: str
    raw_query: str
    intent: IntentType
    execution_summary: dict[str, Any] = Field(default_factory=dict)
    flagged_items: list[dict[str, Any]] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class HealthResponse(BaseModel):
    """Service health check response payload."""

    status: str = Field(json_schema_extra={"example": "healthy"})
    duckdb_connected: bool = Field(json_schema_extra={"example": True})
    active_provider: str = Field(json_schema_extra={"example": "groq"})
    config_version: str = Field(json_schema_extra={"example": "v1.0.0"})


class AuditResponse(BaseModel):
    """Execution audit events response payload."""

    query_id: str = Field(json_schema_extra={"example": "q_001"})
    event_count: int = Field(json_schema_extra={"example": 5})
    events: list[AuditEvent] = Field(default_factory=list)


def _strip_bulk_records(flagged_items: list[Any]) -> list[dict[str, Any]]:
    """Remove account_record and recent_transactions from flagged items for compact response.

    These fields can contain MBs of data for batch queries. Clients who need the
    full records should use POST /query?detailed=true or download the full report.
    """
    stripped = []
    for item in flagged_items:
        if isinstance(item, dict):
            compact = {k: v for k, v in item.items() if k not in ("account_record", "recent_transactions")}
        elif hasattr(item, "model_dump"):
            d = item.model_dump()
            compact = {k: v for k, v in d.items() if k not in ("account_record", "recent_transactions")}
        else:
            compact = item
        stripped.append(compact)
    return stripped


app = FastAPI(
    title="AML Suspicious Activity Detector Agent API",
    description=(
        "Autonomous AI-powered agent API for Anti-Money Laundering (AML) suspicious activity detection. "
        "Accepts natural language queries, dynamically plans tool execution paths, runs hybrid ML+rule detection, "
        "and returns structured, explainable risk assessments. "
        "Use ?detailed=true for full domain records, or /report/{query_id}/download for a full report file."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/health", response_model=HealthResponse, summary="Service Health Check", tags=["System"])
def health_check() -> HealthResponse:
    """Check application health status and database connection."""
    config = AppConfig.load()
    duckdb_connected = False
    try:
        db_client = get_duckdb_client(config)
        res = db_client.query("SELECT 1")
        duckdb_connected = len(res) > 0
        db_client.close()
    except Exception:
        duckdb_connected = False

    return HealthResponse(
        status="healthy" if duckdb_connected else "degraded",
        duckdb_connected=duckdb_connected,
        active_provider=str(config.provider),
        config_version=config.config_version,
    )


# Module-level singleton cache (lives for the API process lifetime)
_config_for_cache = AppConfig.load()
_query_cache = QueryCache(
    ttl_minutes=_config_for_cache.query_cache_ttl_minutes,
    enabled=_config_for_cache.query_cache_enabled,
    semantic_enabled=_config_for_cache.semantic_cache_enabled,
    semantic_threshold=_config_for_cache.semantic_cache_threshold,
)


@app.post(
    "/query",
    response_model=AgentResponseSummary,
    status_code=status.HTTP_200_OK,
    summary="Process Natural Language Compliance Query",
    tags=["Agent Query Execution"],
)
def process_query(
    request: QueryRequest,
    detailed: bool = Query(
        default=False,
        description="Set true to include full account_record and recent_transactions in flagged items. "
                    "Default false returns a compact response. For very large result sets, use /report/{query_id}/download instead.",
    ),
) -> AgentResponseSummary:
    """Process a natural language AML query through the LangGraph agent.

    Checks the smart semantic cache first (TF-IDF similarity). On cache miss,
    parses intent, builds an execution plan, runs the selected tool pipeline,
    caches the result, and returns a compact structured response.

    Compact mode (default): flagged items contain entity_id, score, risk_level,
    escalation_action, explanation, triggered_signals, supporting_evidence.

    Detailed mode (?detailed=true): also includes account_record and recent_transactions.
    For full reports, prefer GET /report/{query_id}/download.
    """
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty")

    # Check smart cache before executing the full agent pipeline
    cached_response = _query_cache.get(request.query)
    if cached_response is not None:
        items = cached_response.flagged_items if detailed else _strip_bulk_records(cached_response.flagged_items)
        return AgentResponseSummary(
            query_id=cached_response.query_id,
            raw_query=cached_response.raw_query,
            intent=cached_response.intent,
            execution_summary=cached_response.execution_summary,
            flagged_items=items,
            charts=cached_response.charts,
            metrics=cached_response.metrics,
            explanation=cached_response.explanation,
        )

    config = AppConfig.load()
    if request.provider:
        config.provider = request.provider
    if request.model_id:
        config.model_id = request.model_id

    db_client = get_duckdb_client(config)
    try:
        audit_repo = AuditRepo(db_client)
        registry = build_default_tool_registry()
        llm_client = LLMClient(config)

        graph = build_agent_graph(
            config=config,
            registry=registry,
            db_client=db_client,
            audit_repo=audit_repo,
            llm_client=llm_client,
        )

        initial_state = {"raw_query": request.query}
        final_state = graph.invoke(initial_state)

        agent_response = final_state.get("agent_response")
        if not agent_response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent state machine failed to produce an AgentResponse",
            )

        # Cache the full response for future repeated/similar queries
        _query_cache.put(request.query, agent_response)

        # Return compact or detailed based on request param
        items = agent_response.flagged_items if detailed else _strip_bulk_records(agent_response.flagged_items)
        return AgentResponseSummary(
            query_id=agent_response.query_id,
            raw_query=agent_response.raw_query,
            intent=agent_response.intent,
            execution_summary=agent_response.execution_summary,
            flagged_items=items,
            charts=agent_response.charts,
            metrics=agent_response.metrics,
            explanation=agent_response.explanation,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {e}",
        )
    finally:
        db_client.close()


@app.get(
    "/audit/{query_id}",
    response_model=AuditResponse,
    summary="Get Query Audit Execution Trace",
    tags=["Audit & Compliance"],
)
def get_audit_trace(
    query_id: str = FastAPIPath(..., description="Query ID string to fetch audit log for.")
) -> AuditResponse:
    """Retrieve complete audit event log for a given query_id from DuckDB audit_log."""
    config = AppConfig.load()
    db_client = get_duckdb_client(config)
    try:
        audit_repo = AuditRepo(db_client)
        events = audit_repo.get_by_query_id(query_id)
        return AuditResponse(
            query_id=query_id,
            event_count=len(events),
            events=events,
        )
    finally:
        db_client.close()


@app.get(
    "/report/{query_id}/download",
    summary="Download Full Report for a Query",
    tags=["Reports & Downloads"],
)
def download_report(
    query_id: str = FastAPIPath(..., description="Query ID to generate report for."),
    format: str = Query(default="json", description="Report format: 'json' or 'csv'."),
) -> StreamingResponse:
    """Generate and stream a downloadable full report for the given query_id.

    Returns a file containing all flagged items with complete domain records
    (account_record, recent_transactions), risk assessments, and audit events.
    Suitable for compliance teams who need the full picture in a downloadable format.

    Formats:
    - json: Full structured JSON report
    - csv: Tabular summary of flagged items (entity_id, score, risk_level, escalation, explanation)
    """
    config = AppConfig.load()
    db_client = get_duckdb_client(config)
    try:
        # Fetch risk assessments for this query
        rows = db_client.query(
            "SELECT entity_id, composite_score, risk_level, escalation_action, signals_json "
            "FROM risk_assessments WHERE query_id = ? ORDER BY composite_score DESC",
            [query_id],
        )

        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No risk assessments found for query_id='{query_id}'. Run the query first.",
            )

        audit_rows = db_client.query(
            "SELECT tool_name, step_id, started_at, ended_at, duration_ms, rows_in, rows_out, status "
            "FROM audit_log WHERE query_id = ? ORDER BY step_id",
            [query_id],
        )

        if format.lower() == "csv":
            return _generate_csv_report(query_id, rows)
        else:
            return _generate_json_report(query_id, rows, audit_rows)

    finally:
        db_client.close()


def _generate_json_report(
    query_id: str,
    assessment_rows: list[dict],
    audit_rows: list[dict],
) -> StreamingResponse:
    """Generate JSON report as a streaming file download."""
    report = {
        "query_id": query_id,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "risk_assessments": assessment_rows,
        "audit_trail": audit_rows,
        "total_flagged": len(assessment_rows),
    }
    content = json.dumps(report, indent=2, default=str)
    return StreamingResponse(
        io.StringIO(content),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="aml_report_{query_id}.json"'},
    )


def _generate_csv_report(query_id: str, assessment_rows: list[dict]) -> StreamingResponse:
    """Generate CSV report as a streaming file download."""
    output = io.StringIO()
    fieldnames = ["entity_id", "composite_score", "risk_level", "escalation_action", "signals_json"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in assessment_rows:
        writer.writerow(row)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="aml_report_{query_id}.csv"'},
    )


@app.get("/cache/stats", summary="Get Query Cache Statistics", tags=["System"])
def cache_stats() -> dict[str, Any]:
    """Return current cache hit/miss statistics and entry count."""
    return _query_cache.stats


@app.post("/cache/clear", summary="Clear Query Cache", tags=["System"])
def cache_clear() -> dict[str, str]:
    """Clear all cached query responses."""
    _query_cache.clear()
    return {"status": "cache_cleared"}
