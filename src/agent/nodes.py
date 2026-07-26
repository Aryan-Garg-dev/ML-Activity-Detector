"""LangGraph node execution functions for the agent state machine."""

import re
import uuid
from typing import Any
from loguru import logger

from core.config import AppConfig
from core.types import ToolName, IntentType, PatternType
from core.cache import artefact_cache
from schemas.contracts import QuerySpec, ExecutionPlan, ToolResult, AgentResponse
from schemas.audit import AuditEvent
from storage.duckdb import DuckDBClient
from storage.repositories import AuditRepo
from tools.registry import ToolRegistry
from agent.state import AgentState
from agent.planner import Planner
from agent.guardrails import validate_execution_plan, audit_write_plan_start, validate_query_relevance
from llm.client import LLMClient
from llm.prompts.intent_parser import parse_query_intent
from llm.prompts.explanation_polish import polish_explanation


def _extract_time_filter(query_lower: str) -> dict[str, Any]:
    """Extract time-related filters from query text into a filters dict."""
    filters: dict[str, Any] = {}

    # Match "last N days/weeks/months/years"
    m = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", query_lower)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit == "day":
            filters["days"] = n
        elif unit == "week":
            filters["days"] = n * 7
        elif unit == "month":
            filters["days"] = n * 30
        elif unit == "year":
            filters["days"] = n * 365
        return filters

    # Match shorthand: "yesterday", "last week", "last month", "last year"
    if "yesterday" in query_lower:
        filters["days"] = 1
    elif "last week" in query_lower:
        filters["days"] = 7
    elif "last month" in query_lower:
        filters["days"] = 30
    elif "last year" in query_lower:
        filters["days"] = 365
    elif "last 90 days" in query_lower:
        filters["days"] = 90

    # Match "between X and Y" — extract as date range hint
    between_match = re.search(r"between\s+(\w+)\s+and\s+(\w+)", query_lower)
    if between_match:
        filters["date_range_hint"] = f"{between_match.group(1)}-{between_match.group(2)}"
        # Fallback: None defers to caller to apply config.default_window_days
        filters.setdefault("days", None)

    return filters


def _extract_amount_filter(query_lower: str) -> dict[str, Any]:
    """Extract amount thresholds from query text."""
    filters: dict[str, Any] = {}
    # Match "under $10,000" or "under 10000" or "below £10,000"
    under_match = re.search(r"(?:under|below|less than)\s*[\$£€]?\s*([\d,]+)", query_lower)
    if under_match:
        filters["max_amount"] = float(under_match.group(1).replace(",", ""))

    # Match "over $50,000" or "more than 50000" or "exceeding £50,000"
    over_match = re.search(r"(?:over|above|more than|exceeding|greater than)\s*[\$£€]?\s*([\d,]+)", query_lower)
    if over_match:
        filters["min_amount"] = float(over_match.group(1).replace(",", ""))

    return filters


def _extract_count_filter(query_lower: str) -> dict[str, Any]:
    """Extract transaction count thresholds from query text."""
    agg: dict[str, Any] = {}
    # Match "10+", "more than 10", ">10", "10 or more"
    count_match = re.search(r"(?:more than|over|>|at least)\s*(\d+)", query_lower)
    if count_match:
        agg["min_count"] = int(count_match.group(1))

    plus_match = re.search(r"(\d+)\+", query_lower)
    if plus_match:
        agg["min_count"] = int(plus_match.group(1))

    return agg


def _heuristic_query_spec(raw_query: str, config: AppConfig | None = None) -> QuerySpec:
    """Comprehensive heuristic fallback for query intent parsing when LLM is unavailable.

    Covers all query categories from the hackathon test suite: entity lookup, aggregation,
    EDA, pattern search (structuring/smurfing/velocity/layering/rapid_cashout),
    risk scoring batch, and off-topic rejection.
    """
    q_lower = raw_query.lower().strip()
    q_words = set(q_lower.split())  # Word-level set for exact word matching

    # 1. Entity lookup: specific customer/account/transaction ID
    id_match = re.search(r"(?:customer|account|entity)\s+(?:id\s+)?(\d+)", q_lower)
    if id_match:
        return QuerySpec(
            intent_type=IntentType.ENTITY_LOOKUP,
            pattern_type=PatternType.UNKNOWN,
            target_entity_id=id_match.group(1),
            raw_query=raw_query,
        )

    # Match "is 4521 suspicious", "explain why 12345 was flagged", "profile 9999"
    # Flexible: verb ... optional words ... numeric ID
    is_id_match = re.search(r"(?:is|explain|profile|why is|why was|check)\s+(?:\w+\s+)*?(\d{3,})", q_lower)
    if is_id_match:
        return QuerySpec(
            intent_type=IntentType.ENTITY_LOOKUP,
            pattern_type=PatternType.UNKNOWN,
            target_entity_id=is_id_match.group(1),
            raw_query=raw_query,
        )

    # Transaction lookup: "explain transaction TXN001" — require ID-like token (contains digits)
    tx_match = re.search(r"(?:transaction|txn)\s+(?:id\s+)?(\w*\d+\w*)", q_lower)
    if tx_match:
        return QuerySpec(
            intent_type=IntentType.ENTITY_LOOKUP,
            pattern_type=PatternType.UNKNOWN,
            target_entity_id=tx_match.group(1),
            raw_query=raw_query,
        )

    # Extract common filters
    time_filters = _extract_time_filter(q_lower)
    amount_filters = _extract_amount_filter(q_lower)
    count_agg = _extract_count_filter(q_lower)

    all_filters = {**time_filters, **amount_filters}

    # 2. Pattern-specific detection queries (check BEFORE aggregation/EDA to avoid substring collisions)
    # Structuring
    if any(k in q_lower for k in ["structuring", "threshold avoidance", "just below", "below 10"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.STRUCTURING,
            filters=all_filters,
            raw_query=raw_query,
        )

    # Smurfing
    if any(k in q_lower for k in ["smurfing", "coordinated transfer", "splitting transaction"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.SMURFING,
            filters=all_filters,
            raw_query=raw_query,
        )

    # Velocity
    if any(k in q_lower for k in ["velocity", "rapid transaction burst", "burst", "within 10 minute", "high frequency"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.VELOCITY,
            filters=all_filters,
            raw_query=raw_query,
        )

    # Rapid cashout
    if any(k in q_lower for k in ["rapid cash", "cashout", "cash-out", "cash out"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.RAPID_CASHOUT,
            filters=all_filters,
            raw_query=raw_query,
        )

    # Layering / geography / cross-border
    if any(k in q_lower for k in ["layering", "cross-border", "international", "risky countries", "geography"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type=PatternType.LAYERING,
            filters=all_filters,
            raw_query=raw_query,
        )

    # Dormant reactivation
    if any(k in q_lower for k in ["dormant", "inactivity", "reactivation", "quiet account", "sudden activity after"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type="dormant",
            filters=all_filters,
            raw_query=raw_query,
        )

    # Duplicate transfer
    if any(k in q_lower for k in ["duplicate", "identical amount", "repeated transfer", "same amount"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type="duplicate",
            filters=all_filters,
            raw_query=raw_query,
        )

    # Volume spike
    if any(k in q_lower for k in ["spike", "sudden increase", "volume surge", "burst in volume"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type="spike",
            filters=all_filters,
            raw_query=raw_query,
        )

    # Multi destination
    if any(k in q_lower for k in ["multi destination", "many counterparties", "numerous receivers", "fan out", "dispersing"]):
        return QuerySpec(
            intent_type=IntentType.PATTERN_SEARCH,
            pattern_type="multi_destination",
            filters=all_filters,
            raw_query=raw_query,
        )

    # 3. Risk scoring batch / broad suspicious activity queries (before aggregation/EDA)
    risk_keywords = [
        "suspicious", "risky", "anomal", "unusual", "abnormal",
        "behavioural", "behavioral", "compliance", "aml report",
        "find suspicious", "who should be",
    ]
    if any(k in q_lower for k in risk_keywords):
        return QuerySpec(
            intent_type=IntentType.RISK_SCORING_BATCH,
            pattern_type=PatternType.UNKNOWN,
            filters=all_filters,
            raw_query=raw_query,
        )

    # 4. Aggregation queries: counting, threshold, top-N lookups
    # Use word-boundary-safe keywords (avoid bare "count" which matches inside "accounts")
    aggregation_keywords = [
        "how many", "total number", "transactions under", "transactions over",
        "10+", "5+", "20+", "top 20", "top risky", "top suspicious", "highest",
        "customers with more than", "customers exceeding", "customers with >",
        "repeated transfers", "duplicate transfers", "show repeated", "show duplicate",
    ]
    # Also check for exact word "count" (not substring inside other words)
    has_aggregation = any(k in q_lower for k in aggregation_keywords) or "count" in q_words
    if has_aggregation:
        pattern = PatternType.STRUCTURING if "under" in q_lower else PatternType.UNKNOWN
        return QuerySpec(
            intent_type=IntentType.AGGREGATION_QUERY,
            pattern_type=pattern,
            filters=all_filters,
            aggregation_spec=count_agg,
            raw_query=raw_query,
        )

    # 5. Broad EDA queries
    eda_keywords = [
        "overview", "distribution", "eda", "trend", "summary", "summarise", "summarize",
        "profiling", "profile the", "missing values", "visualise", "visualize",
        "country distribution", "outliers visually", "baseline", "explore",
    ]
    if any(k in q_lower for k in eda_keywords):
        return QuerySpec(
            intent_type=IntentType.BROAD_EDA,
            pattern_type=PatternType.UNKNOWN,
            filters=all_filters,
            raw_query=raw_query,
        )

    # 6. Secondary risk keywords (lower confidence, checked after EDA)
    secondary_risk = ["risk", "flag", "monitor", "review", "report", "escalat", "detect"]
    if any(k in q_lower for k in secondary_risk):
        return QuerySpec(
            intent_type=IntentType.RISK_SCORING_BATCH,
            pattern_type=PatternType.UNKNOWN,
            filters=all_filters,
            raw_query=raw_query,
        )

    # 7. Catch-all: default to pattern search
    # Apply default_window_days from config for date-range queries with no explicit window
    if config is not None:
        for key in ("days",):
            if all_filters.get(key) is None and "date_range_hint" in all_filters:
                all_filters[key] = config.default_window_days

    return QuerySpec(
        intent_type=IntentType.PATTERN_SEARCH,
        pattern_type=PatternType.UNKNOWN,
        filters=all_filters,
        raw_query=raw_query,
    )


def parse_intent_node(state: AgentState, config: AppConfig, llm_client: LLMClient) -> dict[str, Any]:
    """Node: Parse raw natural language query into a typed QuerySpec.

    First validates query relevance (rejects off-topic queries), then parses intent
    via LLM with heuristic fallback.
    """
    raw_query = state.get("raw_query", "")
    query_id = state.get("query_id") or f"q_{uuid.uuid4().hex[:8]}"
    logger.info("Executing parse_intent_node for query [{query_id}]: '{query}'", query_id=query_id, query=raw_query)

    # Off-topic query rejection guardrail
    is_relevant, rejection_reason = validate_query_relevance(raw_query)
    if not is_relevant:
        logger.warning("Query [{query_id}] rejected as off-topic: {reason}", query_id=query_id, reason=rejection_reason)
        rejection_response = AgentResponse(
            query_id=query_id,
            raw_query=raw_query,
            intent=IntentType.BROAD_EDA,
            execution_summary={"status": "rejected", "reason": rejection_reason, "tools_invoked": [], "tools_skipped": []},
            explanation=rejection_reason,
        )
        return {
            "query_id": query_id,
            "query_spec": QuerySpec(intent_type=IntentType.BROAD_EDA, pattern_type=PatternType.UNKNOWN, raw_query=raw_query),
            "agent_response": rejection_response,
            "current_step_index": 999,  # Signal to skip all steps
            "tool_results": [],
            "audit_events": [],
            "replan_count": 0,
            "error": rejection_reason,
        }

    try:
        query_spec = parse_query_intent(llm_client, raw_query)
    except Exception as e:
        logger.error("Failed to parse query intent via LLM ({err}). Falling back to heuristic query spec", err=e)
        query_spec = _heuristic_query_spec(raw_query, config)

    return {
        "query_id": query_id,
        "query_spec": query_spec,
        "current_step_index": 0,
        "tool_results": [],
        "audit_events": [],
        "replan_count": 0,
    }


def plan_node(
    state: AgentState,
    config: AppConfig,
    planner: Planner,
    audit_repo: AuditRepo,
) -> dict[str, Any]:
    """Node: Formulate ExecutionPlan using planner and validate with guardrails."""
    query_spec = state["query_spec"]
    query_id = state["query_id"]

    logger.info("Executing plan_node for query [{query_id}]", query_id=query_id)
    plan = planner.create_plan(query_spec, query_id=query_id)

    # Validate plan with guardrails
    is_valid, reason = validate_execution_plan(plan, planner.registry, max_steps=config.max_plan_steps)
    if not is_valid:
        logger.error("Generated plan failed guardrails ({reason})", reason=reason)
        raise ValueError(f"Plan validation failed: {reason}")

    # Write plan to audit log before execution begins (Core Design Rule 8)
    audit_write_plan_start(plan, audit_repo, config_version=config.config_version)

    return {
        "execution_plan": plan,
        "current_step_index": 0,
    }


def execute_step_node(
    state: AgentState,
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
) -> dict[str, Any]:
    """Node: Execute the current step from the ExecutionPlan."""
    plan = state["execution_plan"]
    step_index = state.get("current_step_index", 0)
    query_id = state["query_id"]
    tool_results = list(state.get("tool_results", []))

    if step_index >= len(plan.steps):
        logger.info("All plan steps completed for [{query_id}]", query_id=query_id)
        return {"current_step_index": step_index}

    step = plan.steps[step_index]
    logger.info(
        "Executing step {step_id}/{total}: {tool_name}",
        step_id=step.step_id,
        total=len(plan.steps),
        tool_name=step.tool_name,
    )

    if step.tool_name in {ToolName.EXPLANATION, ToolName.REPORTING}:
        logger.debug(
            "Skipping terminal plan placeholder step {step_id}: {tool_name}",
            step_id=step.step_id,
            tool_name=step.tool_name,
        )
        return {"current_step_index": step_index + 1}

    completed_tools = [ToolName(t["tool_name"]) for t in tool_results if "tool_name" in t] if tool_results else []

    tool_args = {
        **step.args,
        **(
            {"detection_results": state.get("detection_results", {}), "query_id": query_id}
            if step.tool_name == ToolName.SCORING
            else {}
        ),
    }

    tool_name_str = step.tool_name.value if hasattr(step.tool_name, "value") else str(step.tool_name)
    cached_data = artefact_cache.get(tool_name_str, tool_args)
    
    if cached_data is not None:
        result = ToolResult(
            tool_name=step.tool_name,
            step_id=step.step_id,
            status="ok",
            data=cached_data,
            rows_count=len(cached_data) if isinstance(cached_data, dict) else 0
        )
        logger.info("Using cached result for tool {tool_name}", tool_name=tool_name_str)
    else:
        result = registry.validate_and_call(
            name=step.tool_name,
            args=tool_args,
            config=config,
            db_client=db_client,
            query_id=query_id,
            step_id=step.step_id,
            completed_tools=completed_tools,
        )
        if result.status == "ok":
            artefact_cache.put(tool_name_str, tool_args, result.data)

    tool_results.append(result)
    updates: dict[str, Any] = {
        "tool_results": tool_results,
        "current_step_index": step_index + 1,
    }

    # Extract domain data from tool output if present
    if result.status == "ok":
        tool_str = str(result.tool_name)
        if tool_str == str(ToolName.DATA_QUERY):
            updates["data_query_results"] = result.data
        if tool_str == str(ToolName.EDA):
            updates["eda_results"] = result.data
        if tool_str == str(ToolName.FEATURE_ENGINEERING):
            updates["feature_results"] = result.data
        if tool_str == str(ToolName.DETECTION):
            updates["detection_results"] = result.data
        if tool_str == str(ToolName.SCORING):
            updates["scoring_results"] = result.data
        if tool_str == str(ToolName.EXPLANATION):
            updates["explanation_results"] = result.data
        if tool_str == str(ToolName.ENTITY_LOOKUP):
            updates["entity_lookup_results"] = result.data


        if "risk_assessments" in result.data:
            updates["risk_assessments"] = result.data["risk_assessments"]
        if "flagged_items" in result.data:
            updates["flagged_items"] = result.data["flagged_items"]

        if "charts" in result.data:
            updates["charts"] = result.data["charts"]
        if "metrics" in result.data:
            updates["metrics"] = result.data["metrics"]

    return updates


def explain_node(
    state: AgentState,
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Node: Run ExplanationTool and apply optional LLM polish with numeric drift check.

    Passes query_context from the parsed QuerySpec into the ExplanationTool so
    each flagged entity explanation references the user's original query intent
    and any active filters (amount thresholds, time windows, pattern type).
    """
    query_id = state["query_id"]
    tool_results = list(state.get("tool_results", []))
    step_id = len(tool_results) + 1
    query_spec = state.get("query_spec")

    logger.info("Executing explain_node for [{query_id}]", query_id=query_id)

    # Build query_context dict to ground explanations in the user's original query
    query_context: dict[str, Any] | None = None
    if query_spec:
        query_context = {
            "raw_query": getattr(query_spec, "raw_query", ""),
            "filters": getattr(query_spec, "filters", {}),
            "intent_type": str(getattr(query_spec, "intent_type", "")),
            "pattern_type": str(getattr(query_spec, "pattern_type", "unknown")),
            # Propagate config-driven truncation limit to template builder
            "query_display_max_chars": config.query_display_max_chars,  # fallback: 80
        }

    # Run deterministic ExplanationTool template generation
    exp_result = registry.validate_and_call(
        name=ToolName.EXPLANATION,
        args={
            "risk_assessments": state.get("risk_assessments", []),
            "detection_results": state.get("detection_results", {}),
            "feature_sets": state.get("feature_results", {}),
            "query_context": query_context,
            "enable_llm_polish": False,
        },
        config=config,
        db_client=db_client,
        query_id=query_id,
        step_id=step_id,
    )
    tool_results.append(exp_result)

    explanations_dict = exp_result.data.get("explanations", {}) if isinstance(exp_result.data, dict) else {}
    risk_assessments = list(state.get("risk_assessments", []))

    # Dynamic top-N from config — entity_lookup shows 1, pattern_search shows top 5, batch shows top 10
    # Fallbacks: entity=1 (single account), default=5 (pattern search), batch=10 (full scan)
    intent_str = str(getattr(query_spec, "intent_type", "")) if query_spec else ""
    if intent_str == IntentType.ENTITY_LOOKUP:
        top_n = config.explain_top_n_entity  # fallback: 1
    elif intent_str == IntentType.RISK_SCORING_BATCH:
        top_n = config.explain_top_n_batch   # fallback: 10
    else:
        top_n = config.explain_top_n_default  # fallback: 5

    sorted_assessments = sorted(
        risk_assessments,
        key=lambda item: float(item.get("composite_score", 0.0)) if isinstance(item, dict) else float(getattr(item, "composite_score", 0.0)),
        reverse=True,
    )[:top_n]

    summary_fragments: list[str] = []
    for assessment in sorted_assessments:
        entity_id = str(assessment.get("entity_id") if isinstance(assessment, dict) else getattr(assessment, "entity_id", "unknown"))
        score = float(assessment.get("composite_score", 0.0)) if isinstance(assessment, dict) else float(getattr(assessment, "composite_score", 0.0))
        entity_explanation = explanations_dict.get(entity_id)
        if entity_explanation:
            summary_fragments.append(f"{entity_id} ({score:.1f}): {entity_explanation}")

    if summary_fragments:
        raw_explanation = "Top flagged accounts: " + " | ".join(summary_fragments)
    else:
        raw_explanation = exp_result.data.get("explanation", "Analysis complete. No suspicious flags triggered.")

    # Apply LLM polish if enabled via config and LLM client is available
    if config.explanation_polish_enabled and llm_client:
        polished = polish_explanation(
            llm_client,
            raw_explanation,
            tolerance=config.explanation_polish_tolerance,  # fallback: 0.20
        )
    else:
        polished = raw_explanation

    return {
        "explanation": polished,
        "explanation_results": exp_result.data,
        "tool_results": tool_results,
    }


def verify_node(state: AgentState) -> dict[str, Any]:
    """Node: Verify AgentState consistency before final reporting.
    
    Checks that intermediate results (like tool_results, risk_assessments)
    are properly formatted and catches missing data early.
    """
    query_id = state.get("query_id", "q_unknown")
    logger.info("Executing verify_node for [{query_id}]", query_id=query_id)
    
    # 1. Check if tool_results is a list
    tool_results = state.get("tool_results", [])
    if not isinstance(tool_results, list):
        logger.error("tool_results is not a list in state")
        return {"error": "Invalid state: tool_results must be a list"}
        
    # 2. Check risk assessments if scoring ran
    if "scoring_results" in state and state["scoring_results"]:
        risk_assessments = state.get("risk_assessments", [])
        if not isinstance(risk_assessments, list):
            logger.error("risk_assessments is not a list in state")
            return {"error": "Invalid state: risk_assessments must be a list"}
            
    # 3. Validation passed
    return {}


def report_node(
    state: AgentState,
    config: AppConfig,
    registry: ToolRegistry,
    db_client: DuckDBClient,
) -> dict[str, Any]:
    """Node: Run ReportingTool to assemble final AgentResponse."""
    query_id = state.get("query_id", "q_unknown")
    tool_results = list(state.get("tool_results") or [])
    # execution_plan is only populated by the legacy planner graph; None in tool-calling graph
    plan = state.get("execution_plan")
    # query_spec is always set by parse_intent_node
    query_spec = state.get("query_spec")
    explanation = state.get("explanation", "Analysis complete.")
    step_id = len(tool_results) + 1

    logger.info("Executing report_node for [{query_id}]", query_id=query_id)

    # Short-circuit: rejection and clarification paths already build a complete AgentResponse
    # in parse_intent_node / clarify_node. Avoid re-running the reporting tool unnecessarily.
    existing_response = state.get("agent_response")
    if existing_response is not None:
        logger.info("report_node: agent_response already set (rejection/clarification path), skipping tool call.")
        return {"agent_response": existing_response}

    # Derive tools_invoked for execution_summary from ToolResults (legacy) or messages (tool-calling)
    if tool_results:
        tools_invoked_list = [
            tr.tool_name.value if hasattr(tr.tool_name, "value") else str(tr.tool_name)
            for tr in tool_results
        ]
    else:
        # Tool-calling graph: extract tool names from ToolMessages in state messages
        from langchain_core.messages import ToolMessage as LCToolMessage
        messages = list(state.get("messages") or [])
        tools_invoked_list = [
            getattr(msg, "name", "unknown")
            for msg in messages
            if isinstance(msg, LCToolMessage)
        ]

    rep_result = registry.validate_and_call(
        name=ToolName.REPORTING,
        args={
            "raw_query": state.get("raw_query", ""),
            "intent": query_spec.intent_type if hasattr(query_spec, "intent_type") else IntentType.PATTERN_SEARCH,
            "query_spec": query_spec.model_dump() if hasattr(query_spec, "model_dump") else query_spec,
            "execution_plan": plan.model_dump() if hasattr(plan, "model_dump") else plan,
            "eda_results": state.get("eda_results", {}),
            "data_query_results": state.get("data_query_results", {}),
            "feature_results": state.get("feature_results", {}),
            "detection_results": state.get("detection_results", {}),
            "scoring_results": state.get("scoring_results", {}),
            "explanation_results": state.get("explanation_results", {}),
            "entity_lookup_results": state.get("entity_lookup_results", {}),
            "tools_invoked": tools_invoked_list,
            "explanation": explanation,
        },
        config=config,
        db_client=db_client,
        query_id=query_id,
        step_id=step_id,
    )
    tool_results.append(rep_result)

    agent_response_dict = rep_result.data.get("agent_response", {})
    
    from schemas.contracts import ExecutionContext
    execution_context = ExecutionContext(
        run_id=query_id,
        total_duration_ms=sum(t.duration_ms for t in tool_results),
        tools_executed=tools_invoked_list,
        errors=[t.error_summary for t in tool_results if getattr(t, "error_summary", None)]
    )
    
    if isinstance(agent_response_dict, dict) and agent_response_dict:
        agent_response = AgentResponse(**agent_response_dict)
        agent_response.execution_context = execution_context
    else:
        agent_response = AgentResponse(
            query_id=query_id,
            raw_query=state.get("raw_query", ""),
            intent=query_spec.intent_type,
            explanation=explanation,
            execution_context=execution_context,
        )

    return {
        "agent_response": agent_response,
        "tool_results": tool_results,
    }


def replan_node(state: AgentState, planner: Planner) -> dict[str, Any]:
    """Node: Attempt single replan after a step execution failure."""
    replan_count = state.get("replan_count", 0)
    query_id = state["query_id"]
    query_spec = state["query_spec"]

    logger.warning("Executing replan_node (attempt {count}) for [{query_id}]", count=replan_count + 1, query_id=query_id)

    if replan_count >= 1:
        logger.error("Maximum replan attempts exceeded for [{query_id}]", query_id=query_id)
        return {"error": "Maximum replan attempts reached"}

    new_plan = planner.create_plan(query_spec, query_id=query_id)
    return {
        "execution_plan": new_plan,
        "current_step_index": 0,
        "replan_count": replan_count + 1,
    }


def clarify_node(state: AgentState, config: AppConfig) -> dict[str, Any]:
    """Node: Handle ambiguous or low-confidence queries by returning a clarification prompt."""
    query_id = state.get("query_id", f"q_{uuid.uuid4().hex[:8]}")
    query_spec = state.get("query_spec")

    if not query_spec:
        clarification_msg = "I could not understand your query. Please rephrase it with more specific AML terminology."
        intent = IntentType.PATTERN_SEARCH
    else:
        logger.info("Executing clarify_node for query [{query_id}] (confidence: {conf:.2f})", query_id=query_id, conf=query_spec.confidence_score)
        missing = ", ".join(query_spec.missing_entities) if query_spec.missing_entities else "key details"
        clarification_msg = f"I am not confident in understanding your query (confidence {query_spec.confidence_score:.2f}). Please clarify the following: {missing}."
        intent = query_spec.intent_type

    rejection_response = AgentResponse(
        query_id=query_id,
        raw_query=state.get("raw_query", ""),
        intent=intent,
        execution_summary={"status": "clarification_required", "reason": clarification_msg, "tools_invoked": [], "tools_skipped": []},
        explanation=clarification_msg,
    )
    
    return {
        "agent_response": rejection_response,
        "current_step_index": 999,  # Signal to skip all steps
        "error": clarification_msg,
    }
