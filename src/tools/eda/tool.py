"""EDATool: Exploratory data analysis for AML transaction data.

Provides profiling summary, transaction volume trends, amount distributions,
fraud baseline statistics, and optional chart generation. Designed for the
"Analyse this dataset for suspicious activity" class of queries where broad
exploration is needed before targeted detection.

The tool operates on DuckDB-backed tables and supports optional filters to
scope the analysis to a specific time window or account subset.
"""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.data_query.aggregator import VALID_TABLES, build_where_clause
from tools.eda.charts import (
    plot_amount_distribution,
    plot_fraud_vs_normal_comparison,
    plot_top_senders,
    plot_transaction_volume_timeline,
)


class EDAInput(BaseModel):
    """Input schema for the EDATool.

    Examples:
        Full EDA with charts:
            {"table": "transactions", "include_charts": true}

        Scoped EDA on recent transactions:
            {
                "table": "transactions",
                "filters": {"timestamp": {"op": ">=", "value": 170}},
                "include_charts": true
            }

        Lightweight profiling without charts:
            {"table": "transactions", "include_charts": false}
    """

    table: str = "transactions"
    filters: dict[str, Any] = Field(default_factory=dict)
    include_charts: bool = True
    chart_output_dir: str = "data/processed/charts"


def execute_eda(
    input: EDAInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute exploratory data analysis on the specified table.

    Computes:
        - Profiling summary: row count, column count, null rates, type breakdown
        - Amount statistics: min, max, mean, median, std, percentiles
        - Volume trends: transaction count per timestamp period
        - Fraud baseline: fraud rate, fraud vs normal counts and amounts
        - Top senders: accounts with highest outgoing volume
        - Charts: optional PNG visualizations saved to disk

    The function signature matches the ToolRegistry callable contract:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        if input.table not in VALID_TABLES:
            return ToolResult(
                tool_name=ToolName.EDA, step_id=step_id, status="error",
                error_summary=f"Unknown table '{input.table}'. Valid: {VALID_TABLES}",
            )

        # Load data with optional filters
        where_sql, params = build_where_clause(input.filters, input.table)
        sql = f"SELECT * FROM {input.table}{where_sql}"
        df = db_client.query_df(sql, params if params else None)

        if df.empty:
            return ToolResult(
                tool_name=ToolName.EDA, step_id=step_id, status="ok",
                data={"profiling": {"row_count": 0}, "message": "No data matching filters"},
                rows_count=0,
            )

        total_rows = len(df)
        result_data: dict[str, Any] = {"rows_in": total_rows}

        # Profiling summary
        null_rates = {col: float(df[col].isna().mean()) for col in df.columns}
        type_breakdown = {col: str(df[col].dtype) for col in df.columns}
        result_data["profiling"] = {
            "row_count": total_rows,
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "null_rates": null_rates,
            "type_breakdown": type_breakdown,
        }

        # Amount statistics (transactions and alerts have tx_amount)
        if "tx_amount" in df.columns:
            amounts = df["tx_amount"].dropna()
            result_data["amount_statistics"] = {
                "min": float(amounts.min()),
                "max": float(amounts.max()),
                "mean": float(amounts.mean()),
                "median": float(amounts.median()),
                "std": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "p25": float(np.percentile(amounts, 25)),
                "p75": float(np.percentile(amounts, 75)),
                "p95": float(np.percentile(amounts, 95)),
                "p99": float(np.percentile(amounts, 99)),
                "total_volume": float(amounts.sum()),
            }

        # Volume trends (transactions have timestamp)
        if "timestamp" in df.columns:
            volume_by_time = df.groupby("timestamp").size().reset_index(name="count")
            volume_by_time = volume_by_time.sort_values("timestamp")
            result_data["volume_trends"] = {
                "periods": volume_by_time["timestamp"].tolist(),
                "counts": volume_by_time["count"].tolist(),
                "total_periods": len(volume_by_time),
                "avg_per_period": float(volume_by_time["count"].mean()),
                "max_period": int(volume_by_time["count"].max()),
                "min_period": int(volume_by_time["count"].min()),
            }

        # Fraud baseline (transactions and alerts have is_fraud)
        if "is_fraud" in df.columns:
            fraud_count = int(df["is_fraud"].sum())
            normal_count = total_rows - fraud_count
            fraud_rate = fraud_count / total_rows if total_rows > 0 else 0.0
            fraud_stats: dict[str, Any] = {
                "fraud_count": fraud_count,
                "normal_count": normal_count,
                "fraud_rate": round(fraud_rate, 6),
                "total_transactions": total_rows,
            }

            if "tx_amount" in df.columns:
                fraud_amount = float(df[df["is_fraud"] == True]["tx_amount"].sum())
                normal_amount = float(df[df["is_fraud"] == False]["tx_amount"].sum())
                total_amount = fraud_amount + normal_amount
                fraud_stats["fraud_amount_total"] = fraud_amount
                fraud_stats["normal_amount_total"] = normal_amount
                fraud_stats["fraud_amount_share"] = (
                    round(fraud_amount / total_amount, 6) if total_amount > 0 else 0.0
                )

            result_data["fraud_baseline"] = fraud_stats

        # Top senders by transaction count
        if "sender_account_id" in df.columns:
            top_senders = (
                df["sender_account_id"]
                .value_counts()
                .head(20)
                .to_dict()
            )
            # Convert numpy int keys to Python int for JSON serialization
            result_data["top_senders"] = {
                int(k): int(v) for k, v in top_senders.items()
            }

        # Account-level breakdown (if accounts table or account IDs present)
        if "account_type" in df.columns:
            type_breakdown_stats = (
                df.groupby("account_type")
                .agg(count=("account_type", "size"))
                .reset_index()
                .to_dict(orient="records")
            )
            result_data["account_type_breakdown"] = type_breakdown_stats

        if "country" in df.columns:
            country_breakdown = (
                df.groupby("country")
                .agg(count=("country", "size"))
                .reset_index()
                .to_dict(orient="records")
            )
            result_data["country_breakdown"] = country_breakdown

        # Chart generation
        chart_paths: list[str] = []
        if input.include_charts and "tx_amount" in df.columns:
            chart_dir = Path(input.chart_output_dir) / query_id
            chart_dir.mkdir(parents=True, exist_ok=True)

            chart_paths.append(
                plot_transaction_volume_timeline(df, chart_dir / "volume_timeline.png")
            )
            chart_paths.append(
                plot_amount_distribution(df, chart_dir / "amount_distribution.png")
            )

            if "is_fraud" in df.columns:
                chart_paths.append(
                    plot_fraud_vs_normal_comparison(df, chart_dir / "fraud_comparison.png")
                )

            if "sender_account_id" in df.columns:
                chart_paths.append(
                    plot_top_senders(df, chart_dir / "top_senders.png")
                )

            result_data["chart_paths"] = chart_paths

        logger.info(
            "EDA completed: {rows} rows, {charts} charts generated",
            rows=total_rows, charts=len(chart_paths),
        )

        return ToolResult(
            tool_name=ToolName.EDA,
            step_id=step_id,
            status="ok",
            data=result_data,
            rows_count=total_rows,
        )

    except Exception as e:
        logger.exception("EDA execution failed")
        return ToolResult(
            tool_name=ToolName.EDA,
            step_id=step_id,
            status="error",
            error_summary=f"{type(e).__name__}: {e}",
        )
