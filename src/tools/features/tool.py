"""FeatureTool: Feature engineering dispatcher for AML detection pipeline.

Computes model-ready and rule-ready features on demand per pattern family:
volume, threshold, network, and velocity. Operates on DuckDB tables and uses
FeatureRepo for caching computed feature vectors in the feature_cache table.
"""

from typing import Any, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from schemas.domain import FeatureSet
from storage.duckdb import DuckDBClient
from storage.repositories import FeatureRepo
from tools.features.volume import compute_volume_features
from tools.features.threshold import compute_threshold_features
from tools.features.network import compute_network_features
from tools.features.velocity import compute_velocity_features


class FeatureInput(BaseModel):
    """Input schema for the FeatureTool.

    Examples:
        Compute volume features for all active accounts:
            {"feature_family": "volume", "window_days": 30}

        Compute structuring features for specific accounts:
            {"feature_family": "threshold", "account_ids": [101, 102], "use_cache": false}

        Compute all feature families:
            {"feature_family": "all", "window_days": 30}
    """

    feature_family: str = Field(
        default="volume",
        description="Feature family to compute: 'volume', 'threshold', 'network', 'velocity', or 'all'",
    )
    account_ids: list[int] | None = Field(
        default=None,
        description="Optional list of target account IDs. None computes for all active accounts.",
    )
    window_days: int = Field(default=30, ge=1, le=365)
    use_cache: bool = Field(default=True, description="Whether to check and update FeatureRepo cache.")


VALID_FAMILIES = {"volume", "threshold", "network", "velocity", "all"}


def execute_feature_engineering(
    input: FeatureInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute feature engineering dispatcher.

    Matches signature expected by ToolRegistry:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        family = input.feature_family.lower()
        if family not in VALID_FAMILIES:
            return ToolResult(
                tool_name=ToolName.FEATURE_ENGINEERING,
                step_id=step_id,
                status="error",
                error_summary=f"Invalid feature_family '{input.feature_family}'. Valid: {VALID_FAMILIES}",
            )

        feat_repo = FeatureRepo(db_client)

        # Families to execute
        families_to_run = (
            ["volume", "threshold", "network", "velocity"]
            if family == "all"
            else [family]
        )

        # Check cache if enabled
        cached_results: dict[int, dict[str, Any]] = {}
        missing_account_ids = input.account_ids

        if input.use_cache and input.account_ids is not None:
            uncached_accs = []
            for acc in input.account_ids:
                acc_features = {}
                all_found = True
                for fam in families_to_run:
                    cached_fset = feat_repo.get_features(acc, fam, input.window_days)
                    if cached_fset is not None:
                        acc_features.update(cached_fset.features)
                    else:
                        all_found = False
                        break
                if all_found:
                    cached_results[acc] = acc_features
                else:
                    uncached_accs.append(acc)
            missing_account_ids = uncached_accs

            if not missing_account_ids and cached_results:
                logger.info("Retrieved all {count} account features from cache", count=len(cached_results))
                return ToolResult(
                    tool_name=ToolName.FEATURE_ENGINEERING,
                    step_id=step_id,
                    status="ok",
                    data={
                        "feature_family": family,
                        "window_days": input.window_days,
                        "account_count": len(cached_results),
                        "feature_sets": cached_results,
                        "from_cache": True,
                        "rows_in": len(cached_results),
                    },
                    rows_count=len(cached_results),
                )

        # Determine dataset max timestamp and cutoff for SQL filter pushdown
        max_ts_res = db_client.query("SELECT COALESCE(MAX(timestamp), 0) FROM transactions")
        max_ts = max_ts_res[0][0] if max_ts_res else 0
        cutoff_ts = max_ts - input.window_days

        # Query DuckDB with SQL filter pushdown
        tx_params: list[Any] = [cutoff_ts]
        tx_sql = "SELECT * FROM transactions WHERE timestamp >= ?"

        if missing_account_ids:
            placeholders = ", ".join("?" for _ in missing_account_ids)
            tx_sql += f" AND (sender_account_id IN ({placeholders}) OR receiver_account_id IN ({placeholders}))"
            tx_params.extend(missing_account_ids)
            tx_params.extend(missing_account_ids)

            acc_placeholders = ", ".join("?" for _ in missing_account_ids)
            acc_sql = f"SELECT * FROM accounts WHERE account_id IN ({acc_placeholders})"
            acc_df = db_client.query_df(acc_sql, missing_account_ids)
        else:
            acc_df = db_client.query_df("SELECT * FROM accounts")

        tx_df = db_client.query_df(tx_sql, tx_params)

        if tx_df.empty:
            combined_fsets = dict(cached_results)
            target_ids = missing_account_ids if missing_account_ids is not None else [int(id_val) for id_val in acc_df["account_id"].tolist()]
            for acc in target_ids:
                if acc not in combined_fsets:
                    combined_fsets[acc] = {
                        "txn_count_7d": 0, "txn_count_30d": 0, "txn_sum_7d": 0.0, "txn_sum_30d": 0.0,
                        "avg_amount": 0.0, "amount_std": 0.0, "count_near_threshold_30d": 0,
                        "round_number_bias": 0.0, "distinct_counterparties_30d": 0,
                        "fan_in_degree": 0, "fan_out_degree": 0, "velocity_zscore": 0.0,
                        "dwell_time_avg_hours": 0.0, "in_out_ratio_30d": 0.0, "distinct_countries_30d": 1,
                    }
            return ToolResult(
                tool_name=ToolName.FEATURE_ENGINEERING,
                step_id=step_id,
                status="ok",
                data={
                    "feature_family": family,
                    "window_days": input.window_days,
                    "account_count": len(combined_fsets),
                    "feature_sets": combined_fsets,
                    "from_cache": bool(cached_results),
                    "rows_in": len(combined_fsets),
                },
                rows_count=len(combined_fsets),
            )

        computed_features: dict[int, dict[str, Any]] = {}

        # Compute feature families — parallel computation across pandas DataFrames
        # (save to DuckDB sequentially on main thread to avoid connection lock conflicts)
        family_results: dict[str, dict[int, dict[str, Any]]] = {}

        if len(families_to_run) > 1:
            max_workers = min(len(families_to_run), getattr(config, "feature_parallel_workers", 4))
            logger.debug(
                "Computing {n} feature families in parallel ({w} workers)",
                n=len(families_to_run), w=max_workers,
            )

            def _compute_family(fam: str) -> tuple[str, dict[int, dict[str, Any]]]:
                if fam == "volume":
                    return fam, compute_volume_features(tx_df, missing_account_ids, input.window_days)
                elif fam == "threshold":
                    return fam, compute_threshold_features(tx_df, missing_account_ids, config.reporting_threshold, input.window_days)
                elif fam == "network":
                    return fam, compute_network_features(tx_df, missing_account_ids, input.window_days)
                elif fam == "velocity":
                    return fam, compute_velocity_features(tx_df, acc_df, missing_account_ids, input.window_days)
                return fam, {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_fam = {executor.submit(_compute_family, fam): fam for fam in families_to_run}
                for future in as_completed(future_to_fam):
                    try:
                        fam_name, fam_data = future.result()
                        family_results[fam_name] = fam_data
                    except Exception as fam_err:
                        fam = future_to_fam[future]
                        logger.warning("Feature family '{fam}' failed in parallel execution: {err}", fam=fam, err=fam_err)
        else:
            fam = families_to_run[0]
            if fam == "volume":
                family_results[fam] = compute_volume_features(tx_df, missing_account_ids, input.window_days)
            elif fam == "threshold":
                family_results[fam] = compute_threshold_features(tx_df, missing_account_ids, config.reporting_threshold, input.window_days)
            elif fam == "network":
                family_results[fam] = compute_network_features(tx_df, missing_account_ids, input.window_days)
            elif fam == "velocity":
                family_results[fam] = compute_velocity_features(tx_df, acc_df, missing_account_ids, input.window_days)

        # Main thread: merge results and save to feature_cache safely
        for fam, fam_data in family_results.items():
            for acc, fdict in fam_data.items():
                computed_features.setdefault(acc, {}).update(fdict)
                if input.use_cache:
                    try:
                        feat_repo.save_features(FeatureSet(
                            account_id=acc, feature_family=fam,
                            window_days=input.window_days, features=fdict,
                        ))
                    except Exception as cache_err:
                        logger.debug("Feature cache save skipped: {err}", err=cache_err)

        # Merge with any cached results
        final_results = {**cached_results, **computed_features}

        logger.info(
            "Feature engineering completed: family={fam}, accounts={cnt}",
            fam=family, cnt=len(final_results),
        )

        return ToolResult(
            tool_name=ToolName.FEATURE_ENGINEERING,
            step_id=step_id,
            status="ok",
            data={
                "feature_family": family,
                "window_days": input.window_days,
                "account_count": len(final_results),
                "feature_sets": final_results,
                "from_cache": False,
                "rows_in": len(tx_df),
            },
            rows_count=len(final_results),
        )

    except Exception as e:
        logger.exception("Feature engineering failed")
        return ToolResult(
            tool_name=ToolName.FEATURE_ENGINEERING,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )
