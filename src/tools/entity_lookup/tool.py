"""EntityLookupTool: Single entity lookup for account metadata, alerts, and risk assessments.

Handles queries like "Is customer ID 4521 suspicious?" by retrieving account metadata,
historical alerts, and cached or on-demand risk assessments for a single account.
"""

from typing import Any
import json
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName, RiskLevel, EscalationAction
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from storage.repositories import AccountRepo, TransactionRepo, AlertRepo


class EntityLookupInput(BaseModel):
    """Input schema for EntityLookupTool.

    Examples:
        {"account_id": "4521"}
        {"account_id": 4521}
    """

    account_id: str = Field(description="Account ID string or integer to look up.")


def execute_entity_lookup(
    input: EntityLookupInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute single-entity lookup for account metadata, transactions, and existing risk assessments."""
    try:
        acc_id_str = str(input.account_id).strip()
        acc_repo = AccountRepo(db_client)
        tx_repo = TransactionRepo(db_client)
        alert_repo = AlertRepo(db_client)

        account_record = None
        recent_txs = []
        alerts = []

        if acc_id_str.isdigit():
            acc_int = int(acc_id_str)
            acc_rec = acc_repo.get_by_id(acc_int)
            if acc_rec:
                account_record = acc_rec.model_dump()

            tx_recs = tx_repo.get_by_account(acc_int, limit=20)
            recent_txs = [t.model_dump() for t in tx_recs]

        # Check existing alerts in database
        if acc_id_str.isdigit():
            alert_recs = alert_repo.get_by_id(int(acc_id_str))
            alerts = [a.model_dump() for a in alert_recs]

        # Check existing risk assessment in DuckDB risk_assessments table
        cached_assessments = db_client.query(
            "SELECT entity_id, composite_score, confidence, risk_level, escalation_action, signals_json FROM risk_assessments WHERE entity_id = ?",
            [acc_id_str],
        )


        risk_assessments = []
        flagged_items = []

        if cached_assessments:
            r = cached_assessments[0]
            risk_assessments.append({
                "entity_id": str(r[0]),
                "composite_score": float(r[1]),
                "confidence": float(r[2]),
                "risk_level": str(r[3]),
                "escalation_action": str(r[4]),
                "triggered_signals": json.loads(r[5]) if isinstance(r[5], str) and r[5] else [],
            })
            flagged_items.append({
                "entity_id": str(r[0]),
                "score": float(r[1]),
                "risk_level": str(r[3]),
                "escalation_action": str(r[4]),
                "explanation": f"Account {r[0]} retrieved from cache with risk score {r[1]:.1f}/100.",
                "triggered_signals": json.loads(r[5]) if isinstance(r[5], str) and r[5] else [],
                "supporting_evidence": {"account_record": account_record, "alerts_count": len(alerts)},
                "account_record": account_record,
                "recent_transactions": recent_txs,
            })
        else:
            # Baseline fallback assessment if not previously calculated
            score = 75.0 if (account_record and account_record.get("is_fraud")) else 10.0
            r_level = RiskLevel.HIGH if score >= 75.0 else RiskLevel.LOW
            esc_action = EscalationAction.REPORT if score >= 75.0 else EscalationAction.MONITOR

            risk_assessments.append({
                "entity_id": acc_id_str,
                "composite_score": score,
                "confidence": 0.9,
                "risk_level": r_level,
                "escalation_action": esc_action,
                "triggered_signals": ["KNOWN_FRAUD_FLAG"] if score >= 75.0 else [],
            })
            flagged_items.append({
                "entity_id": acc_id_str,
                "score": score,
                "risk_level": r_level,
                "escalation_action": esc_action,
                "explanation": f"Single entity lookup for account {acc_id_str}.",
                "triggered_signals": ["KNOWN_FRAUD_FLAG"] if score >= 75.0 else [],
                "supporting_evidence": {"account_record": account_record},
                "account_record": account_record,
                "recent_transactions": recent_txs,
            })

        return ToolResult(
            tool_name=ToolName.ENTITY_LOOKUP,
            step_id=step_id,
            status="ok",
            data={
                "entity_id": acc_id_str,
                "account_record": account_record,
                "recent_transactions": recent_txs,
                "alerts": alerts,
                "risk_assessments": risk_assessments,
                "flagged_items": flagged_items,
                "rows_in": 1,
            },
            rows_count=1 if account_record else 0,
        )

    except Exception as e:
        logger.exception("EntityLookupTool execution failed for account {acc}", acc=input.account_id)
        return ToolResult(
            tool_name=ToolName.ENTITY_LOOKUP,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )
