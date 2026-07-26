"""DetectionTool: Hybrid rule engine + PyOD ML scorer + OR-gate ensemble.

Executes pattern-specific AML rules and unsupervised ML anomaly detection (IForest + LOF),
combines signals using an OR-gate ensemble, and outputs flagged entities with confidence.
"""

from typing import Any
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName, PatternType
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.features.tool import FeatureInput, execute_feature_engineering
from tools.detection.rules import evaluate_account_rules
from tools.detection.ml_scorer import fit_predict_ml_anomalies
from tools.detection.ensemble import get_ensemble_strategy


class DetectionInput(BaseModel):
    """Input schema for DetectionTool.

    Examples:
        Detect structuring patterns across all accounts:
            {"pattern_type": "structuring", "window_days": 30}

        Detect any suspicious pattern for a single customer account:
            {"pattern_type": "unknown", "account_ids": [101]}
    """

    pattern_type: PatternType | str = Field(
        default=PatternType.UNKNOWN,
        description="Target AML pattern: 'structuring', 'smurfing', 'layering', 'rapid_cashout', 'velocity', or 'unknown'",
    )
    account_ids: list[int] | None = Field(
        default=None,
        description="Optional list of target account IDs. None detects across all active accounts.",
    )
    window_days: int = Field(default=30, ge=1, le=365)
    feature_family: str = Field(default="all", description="Feature family to use for detection.")


def execute_detection(
    input: DetectionInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute hybrid rule + ML anomaly detection workflow.

    Matches signature expected by ToolRegistry:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        # Step 1: Execute feature engineering to acquire feature matrix
        feat_input = FeatureInput(
            feature_family=input.feature_family,
            account_ids=input.account_ids,
            window_days=input.window_days,
            use_cache=True,
        )
        feat_res = execute_feature_engineering(feat_input, config, db_client, query_id, step_id)

        if feat_res.status != "ok":
            return ToolResult(
                tool_name=ToolName.DETECTION,
                step_id=step_id,
                status="error",
                error_summary=f"Feature engineering failed: {feat_res.error_summary}",
            )

        if not feat_res.data.get("feature_sets"):
            return ToolResult(
                tool_name=ToolName.DETECTION,
                step_id=step_id,
                status="ok",
                data={"detection_results": {}, "flagged_count": 0, "message": "No features available for detection"},
                rows_count=0,
            )

        features_dict: dict[int, dict[str, Any]] = feat_res.data["feature_sets"]
        # Ensure integer keys if returned as string from JSON
        features_dict = {int(k): v for k, v in features_dict.items()}

        # Step 2: Evaluate rule engine per account
        rule_results: dict[int, list[str]] = {}
        for acc_id, fdict in features_dict.items():
            rule_results[acc_id] = evaluate_account_rules(
                features=fdict,
                pattern=input.pattern_type,
                min_near_threshold_count=config.structuring_min_txn_count,
            )

        # Step 3: Evaluate PyOD ML scorers (IForest + LOF)
        bg_features_list: list[dict[str, Any]] | None = None
        if len(features_dict) < 20:
            try:
                target_ids_str = ", ".join(str(k) for k in features_dict.keys())
                bg_rows = db_client.query(
                    f"SELECT account_id FROM accounts WHERE is_fraud = false AND account_id NOT IN ({target_ids_str}) LIMIT 50"
                )
                bg_ids = [int(r[0]) for r in bg_rows]
                if bg_ids:
                    bg_input = FeatureInput(
                        feature_family=input.feature_family,
                        account_ids=bg_ids,
                        window_days=input.window_days,
                        use_cache=True,
                    )
                    bg_res = execute_feature_engineering(bg_input, config, db_client, query_id, step_id)
                    if bg_res.status == "ok" and bg_res.data.get("feature_sets"):
                        bg_features_list = list(bg_res.data["feature_sets"].values())
            except Exception as e:
                logger.debug("Could not fetch background features from DuckDB for small batch: {err}", err=e)

        ml_results = fit_predict_ml_anomalies(features_dict, config, background_features=bg_features_list)

        # Step 4: Combine rule flags + ML scores via configured ensemble strategy
        strategy = get_ensemble_strategy(config.ensemble_strategy)
        ensemble_results = strategy.combine_signals(rule_results, ml_results, config, pattern_type=str(input.pattern_type))

        flagged_results = {acc: info for acc, info in ensemble_results.items() if info["is_flagged"]}
        flagged_count = len(flagged_results)

        logger.info(
            "Detection completed: pattern={pattern}, total_accounts={total}, flagged={flagged}",
            pattern=input.pattern_type, total=len(ensemble_results), flagged=flagged_count,
        )

        return ToolResult(
            tool_name=ToolName.DETECTION,
            step_id=step_id,
            status="ok",
            data={
                "pattern_type": str(input.pattern_type),
                "total_accounts": len(ensemble_results),
                "flagged_count": flagged_count,
                "detection_results": ensemble_results,
                "flagged_results": flagged_results,
                "rows_in": len(ensemble_results),
            },
            rows_count=flagged_count,
        )

    except Exception as e:
        logger.exception("Detection tool failed")
        return ToolResult(
            tool_name=ToolName.DETECTION,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )
