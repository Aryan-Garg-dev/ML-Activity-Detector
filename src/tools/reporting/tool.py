"""ReportingTool: Final AgentResponse structured output assembly.

Assembles execution summaries, flagged item assessments (FlaggedItem), supporting charts,
key metrics, and executive narrative into the canonical AgentResponse contract.
"""

from typing import Any
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName, IntentType, RiskLevel, EscalationAction
from schemas.contracts import ToolResult, AgentResponse
from schemas.risk import FlaggedItem
from storage.duckdb import DuckDBClient


class ReportingInput(BaseModel):
    """Input schema for ReportingTool.

    Examples:
        Assemble final AgentResponse from completed tool outputs:
            {
                "raw_query": "Find structuring patterns in last 30 days",
                "intent": "pattern_search",
                "query_id": "q_001",
                "scoring_results": {...},
                "explanation_results": {...}
            }
    """

    raw_query: str = Field(default="Analyze dataset for suspicious activity")
    intent: IntentType | str = Field(default=IntentType.PATTERN_SEARCH)
    query_id: str = Field(default="q_default")
    explanation: str | None = Field(default=None)
    execution_plan: dict[str, Any] | None = Field(default=None)
    eda_results: dict[str, Any] | None = Field(default=None)
    data_query_results: dict[str, Any] | None = Field(default=None)
    feature_results: dict[str, Any] | None = Field(default=None)
    detection_results: dict[str, Any] | None = Field(default=None)
    scoring_results: dict[str, Any] | None = Field(default=None)
    explanation_results: dict[str, Any] | None = Field(default=None)
    entity_lookup_results: dict[str, Any] | None = Field(default=None)
    tools_invoked: list[str] | None = Field(default=None, description="Explicit list of tools invoked in plan.")


def execute_reporting(
    input: ReportingInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute final AgentResponse reporting assembly."""
    try:
        query_id_str = query_id or input.query_id
        intent_enum = IntentType(input.intent) if isinstance(input.intent, str) else input.intent

        # Extract explanations dictionary
        explanations_dict: dict[str, str] = {}
        if input.explanation_results:
            explanations_dict = input.explanation_results.get("explanations", {})

        # Extract risk assessments from scoring or entity lookup
        assessments_list: list[dict[str, Any]] = []
        if input.scoring_results:
            assessments_list = input.scoring_results.get("risk_assessments", [])
        elif input.entity_lookup_results:
            assessments_list = input.entity_lookup_results.get("risk_assessments", [])


        # Extract feature sets for evidence
        features_dict: dict[str, Any] = {}
        if input.feature_results:
            features_dict = input.feature_results.get("feature_sets", {})
            features_dict = {str(k): v for k, v in features_dict.items()}

        # Build FlaggedItem objects
        flagged_items: list[dict[str, Any]] = []
        high_risk_count = 0
        med_risk_count = 0
        low_risk_count = 0

        if input.entity_lookup_results and input.entity_lookup_results.get("flagged_items"):
            raw_items = input.entity_lookup_results.get("flagged_items", [])
            for item in raw_items:
                r_lvl = str(item.get("risk_level", "medium")).lower()
                esc_act = str(item.get("escalation_action", "review")).lower()

                if r_lvl == RiskLevel.HIGH:
                    high_risk_count += 1
                elif r_lvl == RiskLevel.MEDIUM:
                    med_risk_count += 1
                else:
                    low_risk_count += 1

                f_item = FlaggedItem(
                    entity_id=str(item.get("entity_id", "unknown")),
                    score=float(item.get("score", 0.0)),
                    risk_level=RiskLevel(r_lvl),
                    escalation_action=EscalationAction(esc_act),
                    explanation=str(item.get("explanation", "")),
                    triggered_signals=item.get("triggered_signals", []),
                    supporting_evidence=item.get("supporting_evidence", {}),
                    account_record=item.get("account_record"),
                    recent_transactions=item.get("recent_transactions", []),
                )
                flagged_items.append(f_item.model_dump())

        elif input.data_query_results and input.data_query_results.get("rows"):
            rows = input.data_query_results.get("rows", [])
            for row in rows:
                entity_id = str(row.get("sender_account_id") or row.get("account_id") or row.get("customer_id") or "unknown")
                cnt = row.get("agg_value", 0)
                exp_text = f"Account {entity_id} performed {cnt} transactions matching aggregation query threshold."

                account_data = None
                tx_data = []
                try:
                    if entity_id.isdigit() and db_client:
                        from storage.repositories import AccountRepo, TransactionRepo
                        acc_repo = AccountRepo(db_client)
                        tx_repo = TransactionRepo(db_client)

                        acc_rec = acc_repo.get_by_id(int(entity_id))
                        if acc_rec:
                            account_data = acc_rec.model_dump()

                        tx_recs = tx_repo.get_by_account(int(entity_id), limit=10)
                        tx_data = [t.model_dump() for t in tx_recs]
                except Exception as ex:
                    logger.debug("Could not fetch domain records for {eid}: {err}", eid=entity_id, err=ex)

                f_item = FlaggedItem(
                    entity_id=entity_id,
                    score=min(100.0, float(cnt) * 2.0),
                    risk_level=RiskLevel.MEDIUM if cnt < 50 else RiskLevel.HIGH,
                    escalation_action=EscalationAction.REVIEW if cnt < 50 else EscalationAction.REPORT,
                    explanation=exp_text,
                    triggered_signals=[f"aggregation_count_{cnt}"],
                    supporting_evidence={"transaction_count": cnt},
                    account_record=account_data,
                    recent_transactions=tx_data,
                )

                flagged_items.append(f_item.model_dump())
                if f_item.risk_level == RiskLevel.HIGH:
                    high_risk_count += 1
                elif f_item.risk_level == RiskLevel.MEDIUM:
                    med_risk_count += 1
                else:
                    low_risk_count += 1

        else:

            for assess in assessments_list:
                entity_id = str(assess.get("entity_id", "unknown"))
                score = float(assess.get("composite_score", 0.0))
                r_level = str(assess.get("risk_level", "medium")).lower()
                esc_action = str(assess.get("escalation_action", "review")).lower()
                signals = assess.get("triggered_signals", [])

                # Count risk levels
                if r_level == RiskLevel.HIGH:
                    high_risk_count += 1
                elif r_level == RiskLevel.MEDIUM:
                    med_risk_count += 1
                else:
                    low_risk_count += 1

                # Get explanation string for entity
                exp_text = explanations_dict.get(
                    entity_id,
                    f"Account {entity_id} flagged with composite risk score {score:.1f}/100.",
                )

                # Get supporting feature evidence
                evidence = features_dict.get(entity_id, {})

                # Fetch structured AccountRecord and TransactionRecords if available
                account_data = None
                tx_data = []
                try:
                    if entity_id.isdigit():
                        from storage.repositories import AccountRepo, TransactionRepo
                        acc_repo = AccountRepo(db_client)
                        tx_repo = TransactionRepo(db_client)

                        acc_rec = acc_repo.get_by_id(int(entity_id))
                        if acc_rec:
                            account_data = acc_rec.model_dump()

                        tx_recs = tx_repo.get_by_account(int(entity_id), limit=10)
                        tx_data = [t.model_dump() for t in tx_recs]
                except Exception as ex:
                    logger.debug("Could not fetch structured domain records for {eid}: {err}", eid=entity_id, err=ex)

                flagged_item = FlaggedItem(
                    entity_id=entity_id,
                    score=score,
                    risk_level=RiskLevel(r_level),
                    escalation_action=EscalationAction(esc_action),
                    explanation=exp_text,
                    triggered_signals=signals,
                    supporting_evidence=evidence,
                    account_record=account_data,
                    recent_transactions=tx_data,
                )
                flagged_items.append(flagged_item.model_dump())



        # Collect chart file paths from EDA results if present
        chart_paths: list[str] = []
        if input.eda_results:
            chart_paths = input.eda_results.get("chart_paths", [])

        # Build execution summary using explicit tools_invoked or non-None result checks
        if input.tools_invoked is not None:
            tools_invoked = list(input.tools_invoked)
        else:
            tools_invoked = []
            if input.eda_results is not None: tools_invoked.append(str(ToolName.EDA))
            if input.data_query_results is not None: tools_invoked.append(str(ToolName.DATA_QUERY))
            if input.feature_results is not None: tools_invoked.append(str(ToolName.FEATURE_ENGINEERING))
            if input.detection_results is not None: tools_invoked.append(str(ToolName.DETECTION))
            if input.scoring_results is not None: tools_invoked.append(str(ToolName.SCORING))
            if input.explanation_results is not None: tools_invoked.append(str(ToolName.EXPLANATION))
            if str(ToolName.REPORTING) not in tools_invoked: tools_invoked.append(str(ToolName.REPORTING))

        all_known_tools = [str(t) for t in ToolName]
        tools_skipped = [t for t in all_known_tools if t not in tools_invoked]

        execution_summary = {
            "query_id": query_id_str,
            "tools_invoked": tools_invoked,
            "tools_skipped": tools_skipped,
            "steps_executed": len(tools_invoked),
            "status": "ok",
        }

        total_assessed = len(assessments_list) if assessments_list else len(flagged_items)

        # Key metrics dictionary
        metrics = {
            "total_entities_assessed": total_assessed,
            "flagged_count": len(flagged_items),
            "high_risk_count": high_risk_count,
            "medium_risk_count": med_risk_count,
            "low_risk_count": low_risk_count,
        }

        # High-level executive narrative explanation
        exec_narrative = (
            f"Analysis completed for query '{input.raw_query}'. Evaluated {total_assessed} entities "
            f"across requested detection paths. Identified {high_risk_count} high-risk, {med_risk_count} medium-risk, "
            f"and {low_risk_count} low-risk entities requiring compliance attention."
        )
        final_explanation = input.explanation or exec_narrative


        response = AgentResponse(
            query_id=query_id_str,
            raw_query=input.raw_query,
            intent=intent_enum,
            execution_summary=execution_summary,
            flagged_items=flagged_items,
            charts=chart_paths,
            metrics=metrics,
            explanation=final_explanation,
        )

        logger.info(
            "Reporting tool completed: assembled AgentResponse query_id={qid}, flagged={cnt}",
            qid=query_id_str, cnt=len(flagged_items),
        )

        return ToolResult(
            tool_name=ToolName.REPORTING,
            step_id=step_id,
            status="ok",
            data={
                "agent_response": response.model_dump(),
                "flagged_count": len(flagged_items),
                "rows_in": len(assessments_list),
            },
            rows_count=len(flagged_items),
        )

    except Exception as e:
        logger.exception("Reporting tool failed")
        return ToolResult(
            tool_name=ToolName.REPORTING,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )
