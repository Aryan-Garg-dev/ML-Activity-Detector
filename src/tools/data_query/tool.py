"""DataQueryTool: DuckDB-backed filtered lookups and direct aggregations.

Handles queries like "Which customers made 10+ transactions under $10,000?"
or "Show all fraud transactions in the last 30 days" — no ML involved.
Supports two modes: filtered lookup (SELECT with WHERE) and aggregation
(GROUP BY with HAVING).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.data_query.aggregator import (
    VALID_AGG_FUNCTIONS,
    VALID_TABLES,
    run_aggregation,
    run_filtered_query,
)


class DataQueryInput(BaseModel):
    """Input schema for the DataQueryTool.

    Supports two query modes:
        1. Filtered lookup — set table, columns, filters; leave agg_func empty.
        2. Aggregation — set table, group_by, agg_func; optionally agg_column, having.

    Examples:
        Filtered lookup:
            {"table": "transactions", "filters": {"is_fraud": true}, "limit": 50}

        Aggregation:
            {
                "table": "transactions",
                "group_by": ["sender_account_id"],
                "agg_func": "COUNT",
                "having": {"op": ">=", "value": 10}
            }

        Aggregation with filter:
            {
                "table": "transactions",
                "group_by": ["sender_account_id"],
                "agg_func": "COUNT",
                "filters": {"tx_amount": {"op": "<", "value": 10000}},
                "having": {"op": ">=", "value": 10}
            }
    """

    table: str = "transactions"
    columns: list[str] | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    group_by: list[str] | None = None
    agg_func: str | None = None
    agg_column: str | None = None
    having: dict[str, Any] | None = None
    order_by: str | None = None
    limit: int = Field(default=100, ge=1, le=10000)


def execute_data_query(
    input: DataQueryInput,
    config: AppConfig,
    db_client: DuckDBClient,
    query_id: str,
    step_id: int,
) -> ToolResult:
    """Execute a data query, routing between filtered lookup and aggregation modes.

    The function is designed to be passed directly to `ToolRegistry.register()` as
    the tool callable, matching the expected signature:
        (input, config, db_client, query_id, step_id) -> ToolResult
    """
    try:
        is_aggregation = input.group_by is not None and input.agg_func is not None

        if is_aggregation:
            result = run_aggregation(
                client=db_client,
                table=input.table,
                group_by=input.group_by,
                agg_func=input.agg_func,
                agg_column=input.agg_column,
                filters=input.filters if input.filters else None,
                having=input.having,
                order_by=input.order_by,
                limit=input.limit,
            )
        else:
            result = run_filtered_query(
                client=db_client,
                table=input.table,
                columns=input.columns,
                filters=input.filters if input.filters else None,
                order_by=input.order_by,
                limit=input.limit,
            )

        return ToolResult(
            tool_name=ToolName.DATA_QUERY,
            step_id=step_id,
            status="ok",
            data={
                "mode": "aggregation" if is_aggregation else "filtered_lookup",
                "table": input.table,
                "columns": result["columns"],
                "rows": result["rows"],
                "row_count": result["row_count"],
                "rows_in": result["row_count"],
            },
            rows_count=result["row_count"],
        )

    except ValueError as e:
        return ToolResult(
            tool_name=ToolName.DATA_QUERY,
            step_id=step_id,
            status="error",
            error_summary=str(e),
        )
