"""FastAPI Application for AML Suspicious Activity Detector Agent.

Provides HTTP REST endpoints for query processing, service health, audit retrieval,
and full report downloads. Interactive OpenAPI/Swagger documentation at /docs.

Key design decisions:
- POST /query returns a compact AgentResponse by default (no raw account/transaction dumps)
- POST /query?detailed=true returns full domain records for reviewers
- POST /stream returns SSE events as the agent runs (use for UI streaming)
- GET /report/{query_id}/download returns a downloadable JSON or CSV report file
- Smart semantic cache (TF-IDF) is checked before running the agent pipeline
"""

import io
import csv
import json
import os
import shutil
import tempfile
import traceback
from typing import Any
from loguru import logger
from fastapi import FastAPI, HTTPException, Query, status, Path as FastAPIPath, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ProviderName, IntentType
from core.cache import QueryCache
from storage.duckdb import get_duckdb_client
from storage.repositories import AuditRepo
from tools.registry import build_default_tool_registry
from schemas.contracts import AgentResponse
from schemas.audit import AuditEvent
from agent.graph import build_agent_graph
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

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
    thread_id: str | None = Field(
        default=None,
        description="Optional thread ID for conversational memory checkpointing.",
        json_schema_extra={"example": "thread-1234"},
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
    execution_context: dict[str, Any] | None = None
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _build_agent_response_summary(agent_response: AgentResponse, detailed: bool) -> AgentResponseSummary:
    """Convert an AgentResponse to AgentResponseSummary, stripping bulk data if not detailed."""
    if detailed:
        items = [
            item.model_dump() if hasattr(item, "model_dump") else item
            for item in (agent_response.flagged_items or [])
        ]
    else:
        items = _strip_bulk_records(agent_response.flagged_items or [])

    return AgentResponseSummary(
        query_id=agent_response.query_id,
        raw_query=agent_response.raw_query,
        intent=agent_response.intent,
        execution_summary=agent_response.execution_summary or {},
        execution_context=(
            agent_response.execution_context.model_dump()
            if agent_response.execution_context
            else None
        ),
        flagged_items=items,
        charts=agent_response.charts or [],
        metrics=agent_response.metrics or {},
        explanation=agent_response.explanation or "",
    )


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

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

# Module-level singleton cache (lives for the API process lifetime)
_config_for_cache = AppConfig.load()
_query_cache = QueryCache(
    ttl_minutes=_config_for_cache.query_cache_ttl_minutes,
    enabled=_config_for_cache.query_cache_enabled,
    semantic_enabled=_config_for_cache.semantic_cache_enabled,
    semantic_threshold=_config_for_cache.semantic_cache_threshold,
)


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check service and database health",
    tags=["Ops"],
)
def health_check() -> HealthResponse:
    """Returns basic service health status and database connectivity."""
    config = AppConfig.load()
    duckdb_connected = False
    try:
        db_client = get_duckdb_client(config)
        res = db_client.query("SELECT 1")
        duckdb_connected = len(res) > 0
        db_client.close()
    except Exception:
        duckdb_connected = False

    status_val = "healthy" if duckdb_connected else "degraded"
    return HealthResponse(
        status=status_val,
        duckdb_connected=duckdb_connected,
        active_provider=config.provider,
        config_version=config.config_version,
    )


@app.get(
    "/health/tools",
    summary="Check tool health statuses",
    tags=["Ops"],
)
def tool_health_check() -> dict[str, Any]:
    """Returns the health status of all registered tools."""
    registry = build_default_tool_registry()
    tools_health = {}
    for name, spec in registry._tools.items():
        tools_health[name] = spec.metadata.health_status.value
    return {"tools": tools_health}


# ---------------------------------------------------------------------------
# Dataset ingestion
# ---------------------------------------------------------------------------

@app.post(
    "/ingest",
    summary="Ingest new datasets",
    tags=["Ops"],
)
def ingest_datasets(
    accounts: UploadFile = File(None),
    transactions: UploadFile = File(None),
    alerts: UploadFile = File(None),
) -> dict[str, Any]:
    """Upload new CSV datasets and overwrite existing tables for testing.

    Handles column mismatches (e.g. missing alert_row_id) by generating
    surrogate keys via row_number(). Drops and recreates tables to clear
    stale CHECK constraints from older schema versions.
    """
    config = AppConfig.load()
    db_client = get_duckdb_client(config)

    results: dict[str, int] = {}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            to_ingest: dict[str, str] = {}
            if accounts:
                acc_path = os.path.join(tmpdir, "accounts.csv")
                with open(acc_path, "wb") as f:
                    shutil.copyfileobj(accounts.file, f)
                to_ingest["accounts"] = acc_path
            if transactions:
                tx_path = os.path.join(tmpdir, "transactions.csv")
                with open(tx_path, "wb") as f:
                    shutil.copyfileobj(transactions.file, f)
                to_ingest["transactions"] = tx_path
            if alerts:
                al_path = os.path.join(tmpdir, "alerts.csv")
                with open(al_path, "wb") as f:
                    shutil.copyfileobj(alerts.file, f)
                to_ingest["alerts"] = al_path

            if not to_ingest:
                return {"status": "no_files", "ingested_rows": {}}

            # Drop in FK-safe order and recreate from current schema to clear stale constraints
            drop_order = []
            if "accounts" in to_ingest:
                drop_order = ["alerts", "transactions", "accounts"]
            elif "transactions" in to_ingest:
                drop_order = ["alerts", "transactions"]
            elif "alerts" in to_ingest:
                drop_order = ["alerts"]

            for table in drop_order:
                db_client.execute(f"DROP TABLE IF EXISTS {table}")

            # Recreate tables from the canonical schema definition
            db_client.init_schema()

            # Ingest in FK-safe order: accounts → transactions → alerts
            for table in ["accounts", "transactions", "alerts"]:
                if table not in to_ingest:
                    continue

                raw_path = to_ingest[table]
                # Forward slashes required for DuckDB COPY on Windows
                fwd_path = raw_path.replace("\\", "/")

                # Read CSV header to detect column mismatches
                with open(raw_path, "r", encoding="utf-8") as f:
                    csv_cols = [c.strip().lower().strip("\r") for c in f.readline().split(",")]

                # Get current table column order from DuckDB information_schema
                table_cols_raw = db_client.query(
                    "SELECT column_name FROM information_schema.columns "
                    f"WHERE table_name = '{table}' ORDER BY ordinal_position"
                )
                table_cols = [str(r[0]).lower() for r in table_cols_raw]

                missing_cols = [c for c in table_cols if c not in csv_cols]

                if missing_cols:
                    # Build SELECT in table-column order so INSERT positional mapping is correct
                    select_parts = []
                    for c in table_cols:
                        if c in csv_cols:
                            select_parts.append(f'"{c}"')
                        else:
                            # Auto-generate missing PK/serial column (e.g. alert_row_id)
                            select_parts.append(f'row_number() OVER () AS "{c}"')
                    col_list = ", ".join(f'"{c}"' for c in table_cols)
                    sel_expr = ", ".join(select_parts)
                    sql = (
                        f"INSERT INTO {table} ({col_list}) "
                        f"SELECT {sel_expr} FROM read_csv_auto('{fwd_path}', header=true, ignore_errors=true)"
                    )
                else:
                    sql = f"COPY {table} FROM '{fwd_path}' (HEADER TRUE, AUTO_DETECT TRUE)"

                logger.info("Ingesting {table}: {sql}", table=table, sql=sql[:120])
                db_client.execute(sql)
                res = db_client.query(f"SELECT COUNT(*) FROM {table}")
                results[table] = res[0][0] if res else 0

        return {"status": "success", "ingested_rows": results}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Ingest failed: {err}\n{tb}", err=str(e), tb=tb)
        raise HTTPException(status_code=500, detail=f"Failed to ingest datasets: {str(e)}")
    finally:
        db_client.close()


# ---------------------------------------------------------------------------
# Cache endpoints
# ---------------------------------------------------------------------------

@app.get("/cache/stats", summary="Get Query Cache Statistics", tags=["System"])
def cache_stats() -> dict[str, Any]:
    """Return current cache hit/miss statistics and entry count."""
    return _query_cache.stats


@app.post("/cache/clear", summary="Clear Query Cache", tags=["System"])
def cache_clear() -> dict[str, str]:
    """Clear all cached query responses."""
    _query_cache.clear()
    return {"status": "cache_cleared"}


# ---------------------------------------------------------------------------
# Query execution — synchronous /query
# ---------------------------------------------------------------------------

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
    """
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty")

    # Check smart cache before executing the full agent pipeline
    cached_response = _query_cache.get(request.query)
    if cached_response is not None:
        return _build_agent_response_summary(cached_response, detailed)

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
        thread_id = request.thread_id or "default_thread"
        invoke_config = {"configurable": {"thread_id": thread_id}}

        final_state = graph.invoke(initial_state, config=invoke_config)

        agent_response = final_state.get("agent_response")
        if not agent_response:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent state machine completed but produced no AgentResponse. Check server logs.",
            )

        # Cache the full response for future repeated/similar queries
        _query_cache.put(request.query, agent_response)
        return _build_agent_response_summary(agent_response, detailed)

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        err_msg = str(e) or type(e).__name__
        logger.error("Query execution failed: {err}\n{tb}", err=err_msg, tb=tb)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {err_msg}",
        )
    finally:
        db_client.close()


# ---------------------------------------------------------------------------
# Query execution — streaming /stream (SSE)
# ---------------------------------------------------------------------------

@app.post(
    "/stream",
    summary="Stream Agent Execution Events (SSE)",
    tags=["Agent Query Execution"],
)
def stream_query(
    request: QueryRequest,
    detailed: bool = Query(default=False),
) -> StreamingResponse:
    """Stream a natural language AML query through the LangGraph agent using SSE.

    Yields `data: {...}\\n\\n` events as each graph node executes.
    Final event has `event: complete` with the full AgentResponseSummary.
    Error events have `event: error` with `detail` and optional `traceback`.
    """
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty")

    # Snapshot config at request time so it stays consistent inside the generator
    config = AppConfig.load()
    if request.provider:
        config.provider = request.provider
    if request.model_id:
        config.model_id = request.model_id

    raw_query = request.query
    thread_id = request.thread_id or "default_thread"
    req_detailed = detailed

    def event_generator():
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

            initial_state = {"raw_query": raw_query}
            invoke_config = {"configurable": {"thread_id": thread_id}}
            final_state: dict[str, Any] = {}

            # Yield SSE node-update events as the graph executes
            for event in graph.stream(initial_state, config=invoke_config, stream_mode="updates"):
                if not isinstance(event, dict):
                    continue
                for node_name, node_state in event.items():
                    payload = {"event": "node_update", "node": node_name}
                    yield f"data: {json.dumps(payload)}\n\n"
                    # Accumulate state; guard against None node states (LangGraph may yield them)
                    if node_state and isinstance(node_state, dict):
                        final_state.update(node_state)

            # --- Post-stream result assembly ---
            if not final_state:
                yield f"data: {json.dumps({'event': 'error', 'detail': 'Agent graph produced no state updates.'})}\n\n"
                return

            agent_response: AgentResponse | None = final_state.get("agent_response")
            if not agent_response:
                yield f"data: {json.dumps({'event': 'error', 'detail': 'Agent completed but produced no AgentResponse. Check server logs.'})}\n\n"
                return

            # Cache for repeated queries
            _query_cache.put(raw_query, agent_response)

            summary = _build_agent_response_summary(agent_response, req_detailed)
            payload = {"event": "complete", "data": summary.model_dump(mode="json")}
            yield f"data: {json.dumps(payload)}\n\n"

        except Exception as e:
            tb = traceback.format_exc()
            err_detail = str(e) or type(e).__name__
            logger.error("Stream execution failed: {err}\n{tb}", err=err_detail, tb=tb)
            # Always yield the error as a proper SSE event — never let the exception escape
            safe_payload = {"event": "error", "detail": f"Execution failed: {err_detail}", "traceback": tb}
            yield f"data: {json.dumps(safe_payload)}\n\n"

        finally:
            db_client.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Audit & reports
# ---------------------------------------------------------------------------

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
    """Generate and stream a downloadable full report for the given query_id."""
    config = AppConfig.load()
    db_client = get_duckdb_client(config)
    try:
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
        return _generate_json_report(query_id, rows, audit_rows)

    finally:
        db_client.close()


def _generate_json_report(
    query_id: str,
    assessment_rows: list,
    audit_rows: list,
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


def _generate_csv_report(query_id: str, assessment_rows: list) -> StreamingResponse:
    """Generate CSV report as a streaming file download."""
    output = io.StringIO()
    fieldnames = ["entity_id", "composite_score", "risk_level", "escalation_action", "signals_json"]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in assessment_rows:
        writer.writerow(row if isinstance(row, dict) else dict(zip(fieldnames, row)))
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="aml_report_{query_id}.csv"'},
    )
