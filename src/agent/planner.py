"""Planner module: Tier 1 Decision Table & Tier 2 LLM Fallback Planner."""

import uuid
from typing import Any
from loguru import logger
from core.types import IntentType, PatternType, ToolName
from schemas.contracts import QuerySpec, ExecutionPlan, PlanStep
from tools.registry import ToolRegistry
from agent.guardrails import validate_execution_plan
from llm.client import LLMClient


class Planner:
    """Agent planner constructing execution plans deterministically or via LLM fallback."""

    def __init__(self, registry: ToolRegistry, llm_client: LLMClient | None = None) -> None:
        self.registry = registry
        self.llm_client = llm_client

    def create_plan(self, query_spec: QuerySpec, query_id: str = "q_001") -> ExecutionPlan:
        """Construct an ExecutionPlan using Tier 1 decision table or Tier 2 LLM fallback."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Attempt Tier 1 Decision Table matching
        tier1_plan = self._tier1_decision_table(query_spec, query_id, plan_id)
        if tier1_plan is not None:
            logger.info(
                "Planner selected Tier 1 Decision Table for intent [{intent}] pattern [{pattern}]",
                intent=query_spec.intent_type,
                pattern=query_spec.pattern_type,
            )
            return tier1_plan

        # Tier 2 LLM Fallback Planner
        if self.llm_client:
            logger.info("Tier 1 template not matched. Invoking Tier 2 LLM Fallback Planner...")
            tier2_plan = self._tier2_llm_planner(query_spec, query_id, plan_id)
            if tier2_plan:
                is_valid, reason = validate_execution_plan(tier2_plan, self.registry)
                if is_valid:
                    return tier2_plan
                logger.warning("Tier 2 LLM plan failed guardrails ({reason}), falling back to default plan", reason=reason)

        # Fallback default plan
        return self._build_default_plan(query_spec, query_id, plan_id)

    def _tier1_decision_table(self, query_spec: QuerySpec, query_id: str, plan_id: str) -> ExecutionPlan | None:
        intent = query_spec.intent_type
        pattern = query_spec.pattern_type
        entity_id = query_spec.target_entity_id

        # 1. Pattern Search Path
        if intent == IntentType.PATTERN_SEARCH:
            family = pattern.value if pattern != PatternType.UNKNOWN else "volume"
            # Extract user-specified window from filters (e.g. "last 7 days" → days=7)
            window_days = int(query_spec.filters.get("days", 30))
            steps = [
                PlanStep(step_id=1, tool_name=ToolName.FEATURE_ENGINEERING, args={"feature_family": family, "window_days": window_days}, reason=f"Compute {family} features for {window_days}-day window"),
                PlanStep(step_id=2, tool_name=ToolName.DETECTION, args={"pattern_type": pattern.value, "window_days": window_days}, reason=f"Run detection for {pattern} pattern"),
                PlanStep(step_id=3, tool_name=ToolName.SCORING, args={}, reason="Compute composite risk score and bands"),
                PlanStep(step_id=4, tool_name=ToolName.EXPLANATION, args={}, reason="Generate natural language explanation"),
                PlanStep(step_id=5, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse"),
            ]
            skipped = [ToolName.DATA_QUERY, ToolName.EDA, ToolName.ENTITY_LOOKUP]
            skip_reasons = {str(t): "Not required for targeted pattern search" for t in skipped}
            return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=skipped, skip_reasons=skip_reasons)

        # 2. Aggregation Query Path
        if intent == IntentType.AGGREGATION_QUERY:
            table = "transactions"
            group_by = ["sender_account_id"]
            agg_func = "COUNT"

            data_filters: dict[str, Any] = {}
            has_amount_filter = False
            if "max_amount" in query_spec.filters and query_spec.filters["max_amount"] is not None:
                data_filters["tx_amount"] = {"op": "<", "value": float(query_spec.filters["max_amount"])}
                has_amount_filter = True
            elif "min_amount" in query_spec.filters and query_spec.filters["min_amount"] is not None:
                data_filters["tx_amount"] = {"op": ">=", "value": float(query_spec.filters["min_amount"])}
                has_amount_filter = True

            # Only apply HAVING (count threshold) when the user explicitly asked for a count
            # e.g. "10+ transactions" sets min_count. Amount-only queries use filtered_lookup.
            has_count_filter = "min_count" in query_spec.aggregation_spec and query_spec.aggregation_spec["min_count"] is not None

            if has_amount_filter and not has_count_filter:
                # "transactions exceeding $1000" → filtered_lookup (no group-by aggregation needed)
                data_query_args = {
                    "table": table,
                    "filters": data_filters,
                    "limit": 100,
                }
            else:
                having: dict[str, Any] = {"op": ">=", "value": 10}
                if has_count_filter:
                    having = {"op": ">=", "value": int(query_spec.aggregation_spec["min_count"])}

                data_query_args = {
                    "table": table,
                    "group_by": group_by,
                    "agg_func": agg_func,
                    "filters": data_filters,
                    "having": having,
                    "limit": 100,
                }

            steps = [
                PlanStep(step_id=1, tool_name=ToolName.DATA_QUERY, args=data_query_args, reason="Execute DuckDB direct aggregation or filtered lookup"),
                PlanStep(step_id=2, tool_name=ToolName.EXPLANATION, args={}, reason="Generate lightweight explanation"),
                PlanStep(step_id=3, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse"),
            ]
            skipped = [ToolName.EDA, ToolName.FEATURE_ENGINEERING, ToolName.DETECTION, ToolName.SCORING, ToolName.ENTITY_LOOKUP]
            skip_reasons = {str(t): "Direct aggregation query does not require ML detection or full pipeline" for t in skipped}
            return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=skipped, skip_reasons=skip_reasons)


        # 3. Entity Lookup Path
        if intent == IntentType.ENTITY_LOOKUP and entity_id:
            steps = [
                PlanStep(step_id=1, tool_name=ToolName.ENTITY_LOOKUP, args={"account_id": entity_id}, reason=f"Lookup existing alerts and activity for account {entity_id}"),
                PlanStep(step_id=2, tool_name=ToolName.EXPLANATION, args={}, reason="Generate summary explanation"),
                PlanStep(step_id=3, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse"),
            ]
            skipped = [ToolName.EDA, ToolName.DATA_QUERY]
            skip_reasons = {str(t): "Targeted entity lookup skips broad exploratory analysis" for t in skipped}
            return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=skipped, skip_reasons=skip_reasons)

        # 4. Broad EDA Path — two sub-paths:
        #    4a. EDA-only: visualization/distribution queries without detection keywords → skip ML
        #    4b. Full EDA: exploration with detection (original behavior)
        if intent == IntentType.BROAD_EDA:
            window_days = int(query_spec.filters.get("days", 30))
            q_lower = query_spec.raw_query.lower()
            eda_only_keywords = ["show", "visualize", "visualise", "plot", "chart", "distribution", "histogram"]
            detect_keywords = ["suspicious", "anomal", "detect", "flag", "risk", "laundering"]
            is_eda_only = (
                any(k in q_lower for k in eda_only_keywords)
                and not any(k in q_lower for k in detect_keywords)
            )

            if is_eda_only:
                # EDA-only: just profile + charts, no ML
                steps = [
                    PlanStep(step_id=1, tool_name=ToolName.EDA, args={"eda_type": "full_profile"}, reason="Generate EDA profile, baselines, and charts"),
                    PlanStep(step_id=2, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse with charts"),
                ]
                skipped = [ToolName.DATA_QUERY, ToolName.ENTITY_LOOKUP, ToolName.FEATURE_ENGINEERING, ToolName.DETECTION, ToolName.SCORING, ToolName.EXPLANATION]
                skip_reasons = {str(t): "Visualization-only query does not require ML detection" for t in skipped}
            else:
                # Full EDA: explore then detect
                steps = [
                    PlanStep(step_id=1, tool_name=ToolName.EDA, args={"eda_type": "full_profile"}, reason="Generate EDA dataset profile, baselines, and charts"),
                    PlanStep(step_id=2, tool_name=ToolName.FEATURE_ENGINEERING, args={"feature_family": "volume", "window_days": window_days}, reason="Compute dataset baseline volume features"),
                    PlanStep(step_id=3, tool_name=ToolName.DETECTION, args={"pattern_type": "structuring", "window_days": window_days}, reason="Run baseline anomaly detection"),
                    PlanStep(step_id=4, tool_name=ToolName.SCORING, args={}, reason="Compute initial risk scores"),
                    PlanStep(step_id=5, tool_name=ToolName.EXPLANATION, args={}, reason="Generate dataset insights explanation"),
                    PlanStep(step_id=6, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse"),
                ]
                skipped = [ToolName.DATA_QUERY, ToolName.ENTITY_LOOKUP]
                skip_reasons = {str(t): "Broad EDA does not use targeted query tools" for t in skipped}

            return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=skipped, skip_reasons=skip_reasons)

        # 5. Risk Scoring Batch Path
        if intent == IntentType.RISK_SCORING_BATCH:
            window_days = int(query_spec.filters.get("days", 30))
            steps = [
                PlanStep(step_id=1, tool_name=ToolName.FEATURE_ENGINEERING, args={"feature_family": "velocity", "window_days": window_days}, reason="Compute velocity and amount deviation features"),
                PlanStep(step_id=2, tool_name=ToolName.DETECTION, args={"pattern_type": "velocity", "window_days": window_days}, reason="Run PyOD and rule detection ensemble"),
                PlanStep(step_id=3, tool_name=ToolName.SCORING, args={}, reason="Compute risk scores and assign risk bands"),
                PlanStep(step_id=4, tool_name=ToolName.EXPLANATION, args={}, reason="Generate batch risk explanations"),
                PlanStep(step_id=5, tool_name=ToolName.REPORTING, args={}, reason="Assemble final AgentResponse"),
            ]
            skipped = [ToolName.EDA, ToolName.DATA_QUERY, ToolName.ENTITY_LOOKUP]
            skip_reasons = {str(t): "Batch risk scoring uses full pipeline without interactive EDA" for t in skipped}
            return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=skipped, skip_reasons=skip_reasons)

        return None

    def _tier2_llm_planner(self, query_spec: QuerySpec, query_id: str, plan_id: str) -> ExecutionPlan | None:
        if not self.llm_client:
            return None

        prompt = f"""Construct a minimal, focused ExecutionPlan for the following query specification:
{query_spec.model_dump_json(indent=2)}

Available Tools:
- data_query: Direct DuckDB aggregations and filters (no ML). Use for: counting transactions, threshold lookups, top-N queries.
- eda: Exploratory data analysis profiles and visual charts. Use for: distribution queries, overview, baseline stats.
- feature_engineering: Compute features (families: volume, threshold, network, velocity). Required before detection.
- detection: Rule + ML anomaly detection ensemble (patterns: structuring, smurfing, layering, rapid_cashout, velocity). Requires feature_engineering first.
- scoring: Calculate risk scores, confidence, and escalation action. Requires detection first.
- explanation: Generate compliance explanation narrative. Requires scoring first.
- reporting: Formulate final AgentResponse. Always the last step.
- entity_lookup: Check single account ID cache / alert records. Use for single-entity queries.

Workflow Decision Guide (select the MINIMUM path needed):
1. If intent=aggregation_query or query has counting/threshold lookup → use: data_query → reporting
2. If intent=broad_eda and query has 'visualize/chart/distribution' without 'suspicious/detect' → use: eda → reporting  
3. If intent=broad_eda with detection needed → use: eda → feature_engineering → detection → scoring → explanation → reporting
4. If intent=entity_lookup and has target_entity_id → use: entity_lookup → explanation → reporting
5. If intent=pattern_search → use: feature_engineering → detection → scoring → explanation → reporting
6. If intent=risk_scoring_batch → use: feature_engineering → detection → scoring → explanation → reporting

Rules:
- Step IDs must be sequential integers 1, 2, 3...
- Maximum steps: 8
- NEVER add tools that are not needed for the given intent
- Always end with reporting
- Provide a clear 'reason' for each step explaining what it computes
"""
        try:
            plan = self.llm_client.structured_output(schema=ExecutionPlan, prompt=prompt)
            plan.plan_id = plan_id
            plan.query_id = query_id
            return plan
        except Exception as e:
            logger.error("Tier 2 LLM planner failed: {err}", err=e)
            return None

    def _build_default_plan(self, query_spec: QuerySpec, query_id: str, plan_id: str) -> ExecutionPlan:
        steps = [
            PlanStep(step_id=1, tool_name=ToolName.FEATURE_ENGINEERING, args={"feature_family": "volume"}, reason="Compute baseline features"),
            PlanStep(step_id=2, tool_name=ToolName.DETECTION, args={"pattern_type": "structuring"}, reason="Run anomaly detection"),
            PlanStep(step_id=3, tool_name=ToolName.SCORING, args={}, reason="Compute risk score"),
            PlanStep(step_id=4, tool_name=ToolName.EXPLANATION, args={}, reason="Explain flags"),
            PlanStep(step_id=5, tool_name=ToolName.REPORTING, args={}, reason="Report response"),
        ]
        return ExecutionPlan(plan_id=plan_id, query_id=query_id, steps=steps, tools_skipped=[ToolName.EDA, ToolName.DATA_QUERY], skip_reasons={"eda": "Default fallback plan"})
