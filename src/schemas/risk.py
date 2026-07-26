from typing import Any
from pydantic import BaseModel, Field
from core.types import RiskLevel, EscalationAction

# Per-entity composite risk score assessment record
class RiskAssessment(BaseModel):
    assessment_id: str | None = None
    query_id: str
    entity_id: str
    composite_score: float
    confidence: float
    risk_level: RiskLevel
    escalation_action: EscalationAction
    triggered_signals: list[str] = Field(default_factory=list)
    feature_contributions: dict[str, float] = Field(default_factory=dict)

# Flagged entity summary item included in agent response
class FlaggedItem(BaseModel):
    entity_id: str
    score: float
    risk_level: RiskLevel
    escalation_action: EscalationAction
    explanation: str
    triggered_signals: list[str] = Field(default_factory=list)
    supporting_evidence: dict[str, Any] = Field(default_factory=dict)
    account_record: dict[str, Any] | None = Field(default=None, description="Typed AccountRecord domain model for flagged account")
    recent_transactions: list[dict[str, Any]] = Field(default_factory=list, description="Typed TransactionRecord list for flagged activity")

