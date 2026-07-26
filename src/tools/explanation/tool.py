"""ExplanationTool: Evidence-grounded flag explanation generator.

Fills pattern-specific templates with real computed values and signals.
Enforces numeric preservation guardrails to reject any numeric drift or hallucinated facts.
Accepts query_context to produce query-aware explanations that reference the user's
original query intent and applied filters.
"""

from typing import Any
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.explanation.templates import (
    format_grounded_explanation,
    verify_numeric_preservation,
)


class ExplanationInput(BaseModel):
    """Input schema for ExplanationTool.

    Examples:
        Generate query-aware grounded explanations from scoring and detection outputs:
            {
                "risk_assessments": [...],
                "detection_results": {...},
                "feature_sets": {...},
                "query_context": {"raw_query": "...", "filters": {}, "pattern_type": "..."},
                "enable_llm_polish": false
            }
    """

    risk_assessments: list[dict[str, Any]] | dict[str, Any] = Field(
        default_factory=list,
        description="List of risk assessment records or dictionary of assessments.",
    )
    detection_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional detection results dictionary.",
    )
    feature_sets: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional feature sets dictionary keyed by account ID.",
    )
    query_context: dict[str, Any] | None = Field(
        default=None,
        description="Query context dict from QuerySpec (raw_query, filters, pattern_type) to ground explanations.",
    )
    enable_llm_polish: bool = Field(
        default=False,
        description="Whether to attempt LLM phrasing polish with numeric drift guardrails.",
    )


def execute_explanation(
    input: ExplanationInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute explanation generation workflow.

    Matches signature expected by ToolRegistry:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        raw_assessments = input.risk_assessments
        if isinstance(raw_assessments, dict):
            raw_assessments = raw_assessments.get("risk_assessments", [raw_assessments])

        if not raw_assessments:
            return ToolResult(
                tool_name=ToolName.EXPLANATION,
                step_id=step_id,
                status="ok",
                data={"explanations": {}, "count": 0, "message": "No risk assessments to explain"},
                rows_count=0,
            )

        features_dict = input.feature_sets.get("feature_sets", input.feature_sets) if isinstance(input.feature_sets, dict) else {}
        features_dict = {str(k): v for k, v in features_dict.items()}

        explanations: dict[str, str] = {}

        for assess in raw_assessments:
            entity_id = str(assess.get("entity_id", "unknown"))
            feat_info = features_dict.get(entity_id, {})

            # Generate grounded explanation — now query-context aware
            grounded_text = format_grounded_explanation(
                entity_id=entity_id,
                risk_info=assess,
                feature_info=feat_info,
                query_context=input.query_context,
            )

            final_text = grounded_text

            # Optional LLM polish with numeric drift check
            if input.enable_llm_polish:
                logger.debug("LLM polish requested for entity {entity}", entity=entity_id)
                polished_candidate = _apply_phrasing_polish(grounded_text)
                if verify_numeric_preservation(grounded_text, polished_candidate):
                    final_text = polished_candidate
                else:
                    logger.warning(
                        "Numeric drift detected in LLM polish for {entity}. Falling back to grounded text.",
                        entity=entity_id,
                    )
                    final_text = grounded_text

            explanations[entity_id] = final_text

        logger.info("Explanation tool completed: generated {cnt} explanations", cnt=len(explanations))

        return ToolResult(
            tool_name=ToolName.EXPLANATION,
            step_id=step_id,
            status="ok",
            data={
                "count": len(explanations),
                "explanations": explanations,
                "rows_in": len(raw_assessments),
            },
            rows_count=len(explanations),
        )

    except Exception as e:
        logger.exception("Explanation tool failed")
        return ToolResult(
            tool_name=ToolName.EXPLANATION,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )


def _apply_phrasing_polish(text: str) -> str:
    """Apply minimal stylistic normalization while preserving exact numeric facts.

    The LLM polish step is intentionally lightweight — it normalizes spacing
    and sentence breaks but does not change numeric values.
    """
    polished = text.replace(".  ", ". ").strip()
    return polished
