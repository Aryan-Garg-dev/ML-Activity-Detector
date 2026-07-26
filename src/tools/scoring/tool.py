"""ScoringTool: Deterministic composite risk score classification.

Converts detection signals and ML anomaly percentile ranks into composite risk scores
(0-100), maps them to RiskLevel bands (LOW, MEDIUM, HIGH) and EscalationAction,
and persists RiskAssessment records into DuckDB.
"""

import json
import uuid
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName, RiskLevel, EscalationAction
from schemas.contracts import ToolResult
from schemas.risk import RiskAssessment
from storage.duckdb import DuckDBClient


class ScoringInput(BaseModel):
    """Input schema for ScoringTool.

    Examples:
        Score detection results from DetectionTool:
            {"detection_results": {...}, "query_id": "q_001"}
    """

    detection_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Detection summary dictionary produced by DetectionTool.",
    )
    query_id: str = Field(default="q_default", description="Active query identifier.")


def execute_scoring(
    input: ScoringInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute risk scoring and classification pipeline.

    Matches signature expected by ToolRegistry:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        raw_results = input.detection_results.get("detection_results", input.detection_results)

        if not raw_results:
            return ToolResult(
                tool_name=ToolName.SCORING,
                step_id=step_id,
                status="ok",
                data={"risk_assessments": [], "assessed_count": 0, "message": "No detection results to score"},
                rows_count=0,
            )

        # Extract weights from config
        w_rule = config.ensemble_weights.get("w_rule", 0.50)
        w_ml = config.ensemble_weights.get("w_ml", 0.35)
        w_context = config.ensemble_weights.get("w_context", 0.15)

        assessments: list[RiskAssessment] = []
        assessment_dicts: list[dict[str, Any]] = []

        for acc_key, det_info in raw_results.items():
            account_id = str(acc_key)
            triggered_signals = det_info.get("triggered_signals", [])
            rule_flags = det_info.get("rule_flags", [])
            ml_score = float(det_info.get("ml_anomaly_score", 0.0))
            confidence = float(det_info.get("confidence", 0.0))

            # Rule score component (ratio of rule flags triggered, normalized to 1.0)
            rule_ratio = min(1.0, len(rule_flags) / 2.0) if rule_flags else 0.0

            # Context multiplier (placeholder context score 0.0 - 1.0)
            context_mult = 0.5 if len(triggered_signals) > 1 else 0.0

            # Raw composite formula: clip(w_rule*rule_ratio + w_ml*ml_score + w_context*context_mult, 0, 1)
            raw_score = (w_rule * rule_ratio) + (w_ml * ml_score) + (w_context * context_mult)
            clipped_score = max(0.0, min(1.0, raw_score))
            composite_score = float(round(clipped_score * 100.0, 2))

            # Risk level & escalation action mapping
            risk_level, escalation = _classify_risk(composite_score, config)

            feature_contributions = det_info.get("feature_contributions", {})

            assessment_id = f"risk_{uuid.uuid4().hex[:8]}"
            assessment = RiskAssessment(
                assessment_id=assessment_id,
                query_id=query_id or input.query_id,
                entity_id=account_id,
                composite_score=composite_score,
                confidence=confidence,
                risk_level=risk_level,
                escalation_action=escalation,
                triggered_signals=triggered_signals,
                feature_contributions=feature_contributions,
            )
            assessments.append(assessment)

            # Persist to DuckDB risk_assessments table
            _persist_risk_assessment(db_client, assessment)

            assessment_dicts.append(assessment.model_dump())

        logger.info(
            "Scoring completed: count={cnt}, high_risk={high}",
            cnt=len(assessments),
            high=sum(1 for a in assessments if a.risk_level == RiskLevel.HIGH),
        )

        return ToolResult(
            tool_name=ToolName.SCORING,
            step_id=step_id,
            status="ok",
            data={
                "assessed_count": len(assessments),
                "risk_assessments": assessment_dicts,
                "rows_in": len(assessments),
            },
            rows_count=len(assessments),
        )

    except Exception as e:
        logger.exception("Scoring tool failed")
        return ToolResult(
            tool_name=ToolName.SCORING,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )


def _classify_risk(score: float, config: AppConfig) -> tuple[RiskLevel, EscalationAction]:
    """Map composite score (0-100) to RiskLevel and EscalationAction based on thresholds config."""
    low_max = config.risk_bands.get("low", [0, 39])[1]
    med_max = config.risk_bands.get("medium", [40, 74])[1]

    if score <= low_max:
        return RiskLevel.LOW, EscalationAction.MONITOR
    elif score <= med_max:
        return RiskLevel.MEDIUM, EscalationAction.REVIEW
    else:
        return RiskLevel.HIGH, EscalationAction.REPORT


def _persist_risk_assessment(db_client: DuckDBClient, assessment: RiskAssessment) -> None:
    """Save risk assessment record to DuckDB risk_assessments table."""
    sql = """
    INSERT INTO risk_assessments (assessment_id, query_id, entity_id, composite_score, confidence, risk_level, escalation_action, signals_json)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT (assessment_id) DO UPDATE SET
        composite_score = EXCLUDED.composite_score,
        confidence = EXCLUDED.confidence,
        risk_level = EXCLUDED.risk_level,
        escalation_action = EXCLUDED.escalation_action
    """
    db_client.execute(sql, [
        assessment.assessment_id,
        assessment.query_id,
        assessment.entity_id,
        assessment.composite_score,
        assessment.confidence,
        str(assessment.risk_level),
        str(assessment.escalation_action),
        json.dumps(assessment.triggered_signals),
    ])
