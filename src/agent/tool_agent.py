"""Tool-calling LLM agent graph — LLM dynamically selects tools via function calling.

Instead of a hardcoded Tier-1 decision table, the LLM sees four high-level AML tools
and decides which to invoke based on the natural language query.

Four LLM-callable tools (principle: LLM decides WORKFLOW, not algorithm steps):
  - detect_suspicious_activity  →  feature_engineering → detection → scoring (chained)
  - query_transactions          →  data_query (SQL aggregations and filters)
  - analyze_dataset             →  eda (exploratory data analysis and charts)
  - lookup_account              →  entity_lookup (single account metadata + alerts)

Collapsing the ML pipeline into one tool eliminates:
  - LLM parallel tool-call ordering problems (scoring before features)
  - DuckDB threading contention from parallel ToolNode execution
  - Empty Pydantic schema issues (scoring has no LLM-visible args)

Graph shape:
    parse_intent → tool_agent ⟵→ tool_node (loop) → extract_state → explain → report
                       ↑              ↓ (tool_calls present in last AI message)
                       └──────────────┘

The old Tier-1 planner is preserved in graph.py as build_legacy_agent_graph().
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from loguru import logger

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.audit import AuditEvent
from schemas.contracts import AgentResponse
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry
from agent.state import AgentState
from agent.nodes import parse_intent_node, explain_node, report_node
from llm.client import LLMClient


# ---------------------------------------------------------------------------
# LLM-facing tool input schemas
# Keep these minimal — only expose what the LLM should decide.
# Infrastructure args (config, db_client, query_id, step_id) are injected by wrappers.
# ---------------------------------------------------------------------------

class _DetectArgs(BaseModel):
    """Run full AML suspicious activity detection: compute behavioral features,
    apply rule engine + ML anomaly detection ensemble, and score results.
    Use this for: suspicious activity, pattern detection, money laundering, risk analysis."""

    pattern_type: str = Field(
        default="unknown",
        description=(
            "Target AML pattern:\n"
            "  'structuring'       — transactions split to stay below $10,000 BSA threshold\n"
            "  'smurfing'          — multiple accounts coordinating to layer funds\n"
            "  'layering'          — rapid in-out transfers to obscure fund origin\n"
            "  'rapid_cashout'     — funds withdrawn quickly after deposit\n"
            "  'velocity'          — unusually high transaction rate spike\n"
            "  'dormant'           — dormant account sudden reactivation after inactivity\n"
            "  'duplicate'         — repeated identical transfer amounts between same accounts\n"
            "  'spike'             — large sudden volume spike exceeding historical norm\n"
            "  'multi_destination' — transfers dispersing funds to numerous distinct counterparties\n"
            "  'unknown'           — general anomaly detection (no specific pattern known)"
        ),
    )
    window_days: int = Field(
        default=30, ge=1, le=365,
        description="Look-back window in days (e.g. 30 for 'last 30 days', 7 for 'last week').",
    )


class _DataQueryArgs(BaseModel):
    """Run a direct SQL query on transactions or accounts. Use for counting, filtering,
    threshold lookups, and aggregations — no ML involved.
    Examples: 'how many transactions exceed $10,000?', 'which accounts made 10+ transfers?'"""

    table: str = Field(
        default="transactions",
        description="Table to query: 'transactions' or 'accounts'.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Column filters. Examples:\n"
            "  Amount filter: {\"tx_amount\": {\"op\": \">=\", \"value\": 10000}}\n"
            "  Fraud flag: {\"is_fraud\": true}\n"
            "  Date range: {\"tx_date\": {\"op\": \">=\", \"value\": \"2024-01-01\"}}"
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description="Columns to group by for aggregation. E.g. ['sender_account_id']",
    )
    agg_func: str = Field(
        default="",
        description="Aggregation function: 'COUNT', 'SUM', 'AVG'. Leave empty for plain filtered lookup.",
    )
    having: dict[str, Any] = Field(
        default_factory=dict,
        description="HAVING clause condition. E.g. {\"op\": \">=\", \"value\": 10} for 10+ results.",
    )
    limit: int = Field(default=50, description="Max rows to return.")


class _EDAArgs(BaseModel):
    """Generate exploratory data analysis of the dataset: distributions, baseline stats, and charts.
    Use for: 'show distribution of amounts', 'visualize transaction patterns', 'overview stats'."""

    eda_type: str = Field(
        default="full_profile",
        description="Analysis depth: 'full_profile' (all stats + charts), 'distribution', 'summary'.",
    )


class _EntityLookupArgs(BaseModel):
    """Look up a single account: alerts, metadata, and cached risk assessments.
    Use when the query mentions a specific account ID or customer ID."""

    account_id: str = Field(description="The account or customer ID to look up (as a string).")


# ---------------------------------------------------------------------------
# Per-graph-instance execution context
# ---------------------------------------------------------------------------

class _RunContext:
    """Mutable per-graph-instance execution context shared across tool wrappers.

    One instance is created at graph-build time. `reset()` is called at the
    start of each query invocation to ensure counters and results don't bleed
    across concurrent or sequential requests.
    """

    __slots__ = (
        "query_id",
        "step",
        "detection_results",
        "feature_results",
        "scoring_results",
        "data_query_results",
        "eda_results",
        "entity_lookup_results",
    )

    def __init__(self) -> None:
        self.query_id = "q_unknown"
        self.step = 0
        self.detection_results: dict[str, Any] = {}
        self.feature_results: dict[str, Any] = {}
        self.scoring_results: dict[str, Any] = {}
        self.data_query_results: dict[str, Any] = {}
        self.eda_results: dict[str, Any] = {}
        self.entity_lookup_results: dict[str, Any] = {}

    def reset(self, query_id: str) -> None:
        self.query_id = query_id
        self.step = 0
        self.detection_results = {}
        self.feature_results = {}
        self.scoring_results = {}
        self.data_query_results = {}
        self.eda_results = {}
        self.entity_lookup_results = {}

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def get_domain_state(self) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        if self.feature_results:
            updates["feature_results"] = self.feature_results
        if self.detection_results:
            updates["detection_results"] = self.detection_results
        if self.scoring_results:
            updates["scoring_results"] = self.scoring_results
            if "risk_assessments" in self.scoring_results:
                updates["risk_assessments"] = self.scoring_results["risk_assessments"]
            if "flagged_items" in self.scoring_results:
                updates["flagged_items"] = self.scoring_results["flagged_items"]
        elif self.detection_results:
            if "risk_assessments" in self.detection_results:
                updates["risk_assessments"] = self.detection_results["risk_assessments"]
            if "flagged_items" in self.detection_results:
                updates["flagged_items"] = self.detection_results["flagged_items"]

        if self.data_query_results:
            updates["data_query_results"] = self.data_query_results
        if self.eda_results:
            updates["eda_results"] = self.eda_results
            if "charts" in self.eda_results:
                updates["charts"] = self.eda_results["charts"]
            if "metrics" in self.eda_results:
                updates["metrics"] = self.eda_results["metrics"]
        if self.entity_lookup_results:
            updates["entity_lookup_results"] = self.entity_lookup_results
            if "risk_assessments" in self.entity_lookup_results:
                updates["risk_assessments"] = self.entity_lookup_results["risk_assessments"]
            if "flagged_items" in self.entity_lookup_results:
                updates["flagged_items"] = self.entity_lookup_results["flagged_items"]

        return updates


# ---------------------------------------------------------------------------
# LangChain StructuredTool factory
# ---------------------------------------------------------------------------

def _make_lc_tools(
    registry: ToolRegistry,
    config: AppConfig,
    db_client: DuckDBClient,
    ctx: _RunContext,
) -> list[StructuredTool]:
    """Build LangChain StructuredTool wrappers for the four LLM-callable workflows.

    Each wrapper:
    - Routes through registry.validate_and_call → full audit logging preserved
    - Injects infrastructure args (config, db_client, query_id, step_id) from ctx
    - The detect tool chains feature_engineering → detection → scoring sequentially
    """

    # --- detect_suspicious_activity ---
    def _detect_run(pattern_type: str = "unknown", window_days: int = 30) -> str:
        """Run the full ML detection pipeline: features → detection → scoring."""
        errors: list[str] = []

        # Step 1: Feature engineering
        feat_step = ctx.next_step()
        feat_result = registry.validate_and_call(
            ToolName.FEATURE_ENGINEERING,
            {"feature_family": "all", "window_days": window_days},
            config, db_client, ctx.query_id, feat_step,
        )
        if feat_result.status == "ok":
            ctx.feature_results = feat_result.data
        else:
            errors.append(f"feature_engineering: {feat_result.error_summary}")
            logger.warning("Feature engineering failed in detect tool: {err}", err=feat_result.error_summary)

        # Step 2: Detection (runs even if features failed — uses pre-computed cache)
        det_step = ctx.next_step()
        det_result = registry.validate_and_call(
            ToolName.DETECTION,
            {"pattern_type": pattern_type, "window_days": window_days},
            config, db_client, ctx.query_id, det_step,
        )
        if det_result.status == "ok":
            ctx.detection_results = det_result.data
        else:
            errors.append(f"detection: {det_result.error_summary}")
            logger.warning("Detection failed in detect tool: {err}", err=det_result.error_summary)

        # Step 3: Scoring (uses detection_results from context)
        score_step = ctx.next_step()
        score_result = registry.validate_and_call(
            ToolName.SCORING,
            {
                "detection_results": ctx.detection_results,
                "query_id": ctx.query_id,
            },
            config, db_client, ctx.query_id, score_step,
        )
        if score_result.status == "ok":
            ctx.scoring_results = score_result.data
        else:
            errors.append(f"scoring: {score_result.error_summary}")

        # Construct token-efficient summary for LLM context window (avoids HTTP 413)
        risk_assessments = ctx.scoring_results.get("risk_assessments", [])
        top_entities = [
            {
                "entity_id": str(ra.get("entity_id")),
                "score": ra.get("composite_score"),
                "risk_level": ra.get("risk_level"),
                "signals": ra.get("triggered_signals", []),
            }
            for ra in risk_assessments[:10]
        ]

        summary = {
            "status": "ok" if not errors else ("partial" if (ctx.detection_results or ctx.scoring_results) else "error"),
            "pattern_type": pattern_type,
            "window_days": window_days,
            "total_accounts_assessed": ctx.feature_results.get("account_count", 0),
            "flagged_accounts_count": ctx.detection_results.get("flagged_count", 0),
            "high_risk_count": ctx.scoring_results.get("high_risk_count", 0),
            "top_flagged_entities": top_entities,
            "tool_chain": "feature_engineering → detection → scoring",
        }
        if errors:
            summary["warnings"] = errors
        return json.dumps(summary)

    # --- query_transactions ---
    def _query_run(
        table: str = "transactions",
        filters: dict[str, Any] | None = None,
        group_by: list[str] | None = None,
        agg_func: str = "",
        having: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> str:
        step_id = ctx.next_step()
        result = registry.validate_and_call(
            ToolName.DATA_QUERY,
            {
                "table": table,
                "filters": filters or {},
                "group_by": group_by or [],
                "agg_func": agg_func,
                "having": having or {},
                "limit": limit,
            },
            config, db_client, ctx.query_id, step_id,
        )
        if result.status == "ok":
            ctx.data_query_results = result.data
        return result.model_dump_json()

    # --- analyze_dataset ---
    def _eda_run(eda_type: str = "full_profile") -> str:
        step_id = ctx.next_step()
        result = registry.validate_and_call(
            ToolName.EDA,
            {"eda_type": eda_type},
            config, db_client, ctx.query_id, step_id,
        )
        if result.status == "ok":
            ctx.eda_results = result.data
        return result.model_dump_json()

    # --- lookup_account ---
    def _entity_run(account_id: str) -> str:
        step_id = ctx.next_step()
        result = registry.validate_and_call(
            ToolName.ENTITY_LOOKUP,
            {"account_id": account_id},
            config, db_client, ctx.query_id, step_id,
        )
        if result.status == "ok":
            ctx.entity_lookup_results = result.data
        return result.model_dump_json()

    return [
        StructuredTool.from_function(
            func=_detect_run,
            name="detect_suspicious_activity",
            description=_DetectArgs.__doc__,
            args_schema=_DetectArgs,
        ),
        StructuredTool.from_function(
            func=_query_run,
            name="query_transactions",
            description=_DataQueryArgs.__doc__,
            args_schema=_DataQueryArgs,
        ),
        StructuredTool.from_function(
            func=_eda_run,
            name="analyze_dataset",
            description=_EDAArgs.__doc__,
            args_schema=_EDAArgs,
        ),
        StructuredTool.from_function(
            func=_entity_run,
            name="lookup_account",
            description=_EntityLookupArgs.__doc__,
            args_schema=_EntityLookupArgs,
        ),
    ]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(query_spec: Any, config: AppConfig) -> str:
    """Build a dynamic system prompt grounding the LLM in query context and tool guidance."""
    intent = str(getattr(query_spec, "intent_type", "unknown")) if query_spec else "unknown"
    pattern = str(getattr(query_spec, "pattern_type", "unknown")) if query_spec else "unknown"
    filters = getattr(query_spec, "filters", {}) or {}
    entity_id = getattr(query_spec, "target_entity_id", None) if query_spec else None

    filter_lines: list[str] = []
    if filters.get("days"):
        filter_lines.append(f"  - Time window: last {filters['days']} days")
    if filters.get("min_amount") is not None:
        filter_lines.append(f"  - Minimum amount: ${float(filters['min_amount']):,.0f}")
    if filters.get("max_amount") is not None:
        filter_lines.append(f"  - Maximum amount: ${float(filters['max_amount']):,.0f}")
    if entity_id:
        filter_lines.append(f"  - Target account: {entity_id}")
    filter_str = "\n".join(filter_lines) if filter_lines else "  - None"

    return f"""You are an AML (Anti-Money Laundering) Compliance Analysis Agent.

Parsed query intent: {intent}
Detected AML pattern: {pattern}
Active filters:
{filter_str}

Available tools:
- detect_suspicious_activity  → Full ML detection pipeline. Use for suspicious activity, pattern detection, risk analysis.
- query_transactions          → Direct SQL. Use for counting, filtering, aggregations (no ML).
- analyze_dataset             → EDA and charts. Use for distributions, overviews, visualizations.
- lookup_account              → Single account lookup by ID. Use when a specific account is mentioned.

Select the ONE tool that best answers the query and call it. Call only what is needed."""


# ---------------------------------------------------------------------------
# State extraction — parse ToolMessages → domain state fields
# ---------------------------------------------------------------------------

def _extract_domain_state_from_messages(messages: list[BaseMessage]) -> dict[str, Any]:
    """Parse accumulated ToolMessages to populate AML domain state fields.

    Called after the tool-calling loop ends. Maps tool results to the
    domain state keys that explain_node and report_node expect.
    """
    updates: dict[str, Any] = {}

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        tool_name = getattr(msg, "name", "") or ""
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            continue

        # detect_suspicious_activity returns a custom dict (not ToolResult format)
        if tool_name == "detect_suspicious_activity":
            if "feature_results" in payload:
                updates["feature_results"] = payload["feature_results"]
            if "detection_results" in payload:
                det = payload["detection_results"]
                updates["detection_results"] = det
                if isinstance(det, dict):
                    if "risk_assessments" in det:
                        updates["risk_assessments"] = det["risk_assessments"]
                    if "flagged_items" in det:
                        updates["flagged_items"] = det["flagged_items"]
            if "scoring_results" in payload:
                sc = payload["scoring_results"]
                updates["scoring_results"] = sc
                if isinstance(sc, dict):
                    # Scoring supersedes detection's preliminary risk assessments
                    if "risk_assessments" in sc:
                        updates["risk_assessments"] = sc["risk_assessments"]
                    if "flagged_items" in sc:
                        updates["flagged_items"] = sc["flagged_items"]
            continue

        # Other tools return standard ToolResult JSON (status, data)
        if payload.get("status") != "ok":
            continue
        data = payload.get("data", {})

        if tool_name == "query_transactions":
            updates["data_query_results"] = data
        elif tool_name == "analyze_dataset":
            updates["eda_results"] = data
            if "charts" in data:
                updates["charts"] = data["charts"]
            if "metrics" in data:
                updates["metrics"] = data["metrics"]
        elif tool_name == "lookup_account":
            updates["entity_lookup_results"] = data
            if "risk_assessments" in data:
                updates["risk_assessments"] = data["risk_assessments"]
            if "flagged_items" in data:
                updates["flagged_items"] = data["flagged_items"]

    return updates


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _write_query_start_audit(
    audit_repo: AuditRepo,
    query_id: str,
    config_version: str,
) -> None:
    """Write a query_start audit event to DuckDB (Core Design Rule 8).

    The tool-calling graph has no plan_node, so this provides the initial
    audit row that /audit/{query_id} relies on.
    """
    try:
        event = AuditEvent(
            audit_id=f"audit_start_{uuid.uuid4().hex[:8]}",
            query_id=query_id,
            tool_name=ToolName.REPORTING,  # sentinel; actual tools logged by registry
            step_id=0,
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=datetime.now(timezone.utc).isoformat(),
            duration_ms=0.0,
            rows_in=0,
            rows_out=0,
            provider=None,
            model_name=None,
            config_version=config_version,
            status="ok",
            error_summary=None,
        )
        audit_repo.save_audit_event(event)
        logger.debug("Audit start event written for query [{qid}]", qid=query_id)
    except Exception as e:
        logger.warning("Could not write query start audit event: {err}", err=e)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_tool_calling_graph(
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
    audit_repo: AuditRepo,
    llm_client: LLMClient | None = None,
) -> Any:
    """Build a tool-calling LLM agent graph.

    The LLM chooses from four high-level AML tools based on the query.
    The ML detection pipeline (features → detection → scoring) is wrapped into
    a single tool to eliminate parallel execution ordering issues.

    Falls back gracefully to legacy graph if bind_tools is not supported.
    """
    resolved_llm = llm_client or LLMClient(config)

    # Per-graph-instance run context — shared closure across all tool wrappers
    ctx = _RunContext()

    # Build LangChain tool wrappers
    lc_tools = _make_lc_tools(registry, config, db_client, ctx)

    # Bind tools to the LLM (enables function-calling / tool use)
    # parallel_tool_calls=False: forces one tool call per turn → safe sequential execution
    try:
        model_with_tools = resolved_llm.model.bind_tools(
            lc_tools,
            parallel_tool_calls=False,  # one call per turn; prevents out-of-order execution
        )
    except TypeError:
        # Some providers don't accept parallel_tool_calls kwarg
        model_with_tools = resolved_llm.model.bind_tools(lc_tools)

    tool_node = ToolNode(lc_tools)
    workflow = StateGraph(AgentState)

    # --- parse_intent node ---
    def _parse_intent(state: AgentState) -> dict[str, Any]:
        result = parse_intent_node(state, config, resolved_llm)
        qid = result.get("query_id", state.get("query_id", "q_unknown"))
        ctx.reset(query_id=qid)
        # Write audit event so every query has at least one audit row (Core Design Rule 8)
        _write_query_start_audit(audit_repo, qid, config.config_version)
        # Seed message list with the raw query
        raw = state.get("raw_query", "")
        result["messages"] = [HumanMessage(content=raw)]
        return result

    # --- tool_agent node: LLM decides which tool to call ---
    def _tool_agent(state: AgentState) -> dict[str, Any]:
        query_spec = state.get("query_spec")
        system_prompt = _build_system_prompt(query_spec, config)
        messages = list(state.get("messages") or [])
        response = model_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + messages
        )
        n_calls = len(getattr(response, "tool_calls", []) or [])
        logger.debug("tool_agent: LLM response has {n} tool call(s)", n=n_calls)
        return {"messages": [response]}

    # --- tool_node: LangGraph executes the tool calls ---
    # (ToolNode runs synchronously per call since parallel_tool_calls=False)

    # --- extract_state: parse ToolMessages & _RunContext → domain state ---
    def _extract_state(state: AgentState) -> dict[str, Any]:
        messages = list(state.get("messages") or [])
        # Pull rich domain data directly from ctx (avoids serializing huge dicts in LLM messages)
        domain_updates = ctx.get_domain_state()
        # Fallback/merge anything from message payloads
        msg_updates = _extract_domain_state_from_messages(messages)
        for k, v in msg_updates.items():
            if k not in domain_updates or not domain_updates[k]:
                domain_updates[k] = v

        tool_names = [
            getattr(msg, "name", "unknown")
            for msg in messages
            if isinstance(msg, ToolMessage)
        ]
        logger.info("extract_state: tools called = {tools}", tools=tool_names)
        return domain_updates

    # --- explain and report ---
    def _explain(state: AgentState) -> dict[str, Any]:
        return explain_node(state, config, registry, db_client, resolved_llm)

    def _report(state: AgentState) -> dict[str, Any]:
        return report_node(state, config, registry, db_client)

    # --- fallback to fixed flow when autonomous execution is undecided/fails ---
    def _fallback_fixed_flow(state: AgentState) -> dict[str, Any]:
        """Fallback to deterministic Tier-1 decision table when autonomous execution is undecided/fails."""
        logger.info("Autonomous execution undecided or yielded no domain results. Invoking fallback_fixed_flow.")
        from agent.planner import Planner
        from agent.nodes import execute_step_node
        from agent.guardrails import audit_write_plan_start

        planner = Planner(registry=registry, llm_client=resolved_llm)
        query_spec = state.get("query_spec")
        query_id = state.get("query_id", "q_unknown")

        if not query_spec:
            logger.error("No query_spec available for fallback planner.")
            return {}

        plan = planner.create_plan(query_spec, query_id=query_id)
        # Core Design Rule 8: Write ExecutionPlan to audit_log before executing tools
        audit_write_plan_start(plan, audit_repo, config_version=config.config_version)

        state_with_plan = dict(state)
        state_with_plan["execution_plan"] = plan
        state_with_plan["current_step_index"] = 0
        state_with_plan["tool_results"] = []
        state_with_plan["replan_count"] = 0

        # Execute all steps in the plan sequentially
        while state_with_plan["current_step_index"] < len(plan.steps):
            step_res = execute_step_node(state_with_plan, config, registry, db_client)
            state_with_plan.update(step_res)

        # Map execution results to domain updates for explain/report nodes
        updates: dict[str, Any] = {
            "execution_plan": plan,
            "tool_results": state_with_plan.get("tool_results", []),
            "current_step_index": state_with_plan.get("current_step_index", 0),
        }
        for k in ("detection_results", "scoring_results", "data_query_results", "eda_results", "entity_lookup_results", "risk_assessments", "flagged_items", "charts", "metrics", "feature_results"):
            if k in state_with_plan and state_with_plan[k]:
                updates[k] = state_with_plan[k]
        return updates

    # --- conditional edge: continue tool loop or proceed to extraction ---
    def _should_continue(state: AgentState) -> Literal["tools", "extract_state"]:
        messages = list(state.get("messages") or [])
        if not messages:
            return "extract_state"
        last = messages[-1]
        if isinstance(last, AIMessage) and (getattr(last, "tool_calls", None) or []):
            return "tools"
        return "extract_state"

    # --- conditional edge after extraction: explain or fallback ---
    def _after_extract(state: AgentState) -> Literal["explain", "fallback_fixed_flow"]:
        has_results = any(
            state.get(k) is not None and bool(state.get(k))
            for k in ("detection_results", "scoring_results", "data_query_results", "eda_results", "entity_lookup_results", "risk_assessments", "flagged_items")
        )
        if not has_results:
            logger.warning("Autonomous execution yielded no actionable domain data. Routing to fallback_fixed_flow.")
            return "fallback_fixed_flow"
        return "explain"

    # --- edge after parse_intent: skip loop if already rejected ---
    def _after_parse(state: AgentState) -> Literal["tool_agent", "report"]:
        if state.get("agent_response") is not None:
            return "report"
        return "tool_agent"

    # Register nodes
    workflow.add_node("parse_intent", _parse_intent)
    workflow.add_node("tool_agent", _tool_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("extract_state", _extract_state)
    workflow.add_node("fallback_fixed_flow", _fallback_fixed_flow)
    workflow.add_node("explain", _explain)
    workflow.add_node("report", _report)

    # Wire edges
    workflow.set_entry_point("parse_intent")
    workflow.add_conditional_edges(
        "parse_intent",
        _after_parse,
        {"tool_agent": "tool_agent", "report": "report"},
    )
    workflow.add_conditional_edges(
        "tool_agent",
        _should_continue,
        {"tools": "tools", "extract_state": "extract_state"},
    )
    workflow.add_edge("tools", "tool_agent")
    workflow.add_conditional_edges(
        "extract_state",
        _after_extract,
        {"explain": "explain", "fallback_fixed_flow": "fallback_fixed_flow"},
    )
    workflow.add_edge("fallback_fixed_flow", "explain")
    workflow.add_edge("explain", "report")
    workflow.add_edge("report", END)

    app = workflow.compile()
    logger.info("Compiled tool-calling LangGraph agent graph with fallback successfully")
    return app
