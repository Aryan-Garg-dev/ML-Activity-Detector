"""Parameterized SQL aggregation engine for DuckDB.

Builds safe, parameterized GROUP BY queries with support for standard aggregate
functions, column-level filters, and HAVING clauses. All column names are validated
against a strict whitelist to prevent SQL injection — only known schema columns
from accounts, transactions, and alerts tables are accepted.

Design note: filter values are always parameterized (?-placeholders) and never
interpolated. Column names cannot be parameterized in SQL, so they are validated
against the whitelist instead.
"""

from typing import Any

from loguru import logger

from storage.duckdb import DuckDBClient


# Columns known to each table — the only column names accepted in queries
COLUMN_WHITELIST: dict[str, set[str]] = {
    "accounts": {
        "account_id", "customer_id", "init_balance", "country",
        "account_type", "is_fraud", "tx_behavior_id",
    },
    "transactions": {
        "tx_id", "sender_account_id", "receiver_account_id", "tx_type",
        "tx_amount", "timestamp", "is_fraud", "alert_id",
    },
    "alerts": {
        "alert_row_id", "alert_id", "alert_type", "is_fraud", "tx_id",
        "sender_account_id", "receiver_account_id", "tx_type",
        "tx_amount", "timestamp",
    },
}

VALID_TABLES = set(COLUMN_WHITELIST.keys())

VALID_AGG_FUNCTIONS = {"COUNT", "SUM", "AVG", "MIN", "MAX"}

# Operators allowed in filter expressions
VALID_OPERATORS = {"=", ">", "<", ">=", "<=", "!=", "BETWEEN", "IN"}


def validate_columns(columns: list[str], table: str) -> None:
    """Validate that all columns belong to the table's whitelist."""
    if table not in VALID_TABLES:
        raise ValueError(f"Unknown table '{table}'. Valid tables: {VALID_TABLES}")
    allowed = COLUMN_WHITELIST[table]
    invalid = [c for c in columns if c not in allowed]
    if invalid:
        raise ValueError(
            f"Invalid columns for table '{table}': {invalid}. "
            f"Allowed: {sorted(allowed)}"
        )


def build_where_clause(
    filters: dict[str, Any], table: str,
) -> tuple[str, list[Any]]:
    """Build a parameterized WHERE clause from a filter dict.

    Filter formats:
        Simple equality:  {"column": value}
        Operator:         {"column": {"op": ">", "value": 100}}
        BETWEEN:          {"column": {"op": "BETWEEN", "value": [low, high]}}
        IN:               {"column": {"op": "IN", "value": [v1, v2, v3]}}

    Returns:
        Tuple of (where_sql, params) where where_sql starts with " WHERE ..."
        or is empty string if no filters.
    """
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

    for column, condition in filters.items():
        validate_columns([column], table)

        if isinstance(condition, dict):
            op = condition.get("op", "=").upper()
            value = condition.get("value")
            if op not in VALID_OPERATORS:
                raise ValueError(f"Invalid operator '{op}'. Valid: {VALID_OPERATORS}")

            if op == "BETWEEN":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError("BETWEEN requires a [low, high] value list")
                clauses.append(f"{column} BETWEEN ? AND ?")
                params.extend(value)
            elif op == "IN":
                if not isinstance(value, (list, tuple)) or len(value) == 0:
                    raise ValueError("IN requires a non-empty value list")
                placeholders = ", ".join("?" for _ in value)
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(value)
            elif op == "!=":
                clauses.append(f"{column} != ?")
                params.append(value)
            else:
                clauses.append(f"{column} {op} ?")
                params.append(value)
        else:
            # Simple equality
            clauses.append(f"{column} = ?")
            params.append(condition)

    where_sql = " WHERE " + " AND ".join(clauses)
    return where_sql, params


def run_aggregation(
    client: DuckDBClient,
    table: str,
    group_by: list[str],
    agg_func: str,
    agg_column: str | None = None,
    filters: dict[str, Any] | None = None,
    having: dict[str, Any] | None = None,
    order_by: str | None = None,
    limit: int = 1000,
) -> dict[str, Any]:
    """Execute a parameterized aggregation query.

    Args:
        client: DuckDB client instance.
        table: Table name (must be in VALID_TABLES).
        group_by: Columns to group by.
        agg_func: Aggregation function (COUNT, SUM, AVG, MIN, MAX).
        agg_column: Column to aggregate. None defaults to * for COUNT.
        filters: Optional WHERE clause filters.
        having: Optional HAVING clause, e.g. {"op": ">=", "value": 10}.
        order_by: Column to ORDER BY. Use "agg_value" for the aggregate result.
        limit: Maximum rows to return.

    Returns:
        Dict with keys: columns, rows, row_count.

    Example:
        # "Which accounts made 10+ transactions?"
        result = run_aggregation(
            client, table="transactions",
            group_by=["sender_account_id"], agg_func="COUNT",
            having={"op": ">=", "value": 10},
        )
    """
    if table not in VALID_TABLES:
        raise ValueError(f"Unknown table '{table}'. Valid: {VALID_TABLES}")

    agg_func_upper = agg_func.upper()
    if agg_func_upper not in VALID_AGG_FUNCTIONS:
        raise ValueError(f"Invalid aggregation function '{agg_func}'. Valid: {VALID_AGG_FUNCTIONS}")

    group_by_cols = group_by or []
    if group_by_cols:
        validate_columns(group_by_cols, table)

    # Build the aggregate expression
    if agg_column:
        validate_columns([agg_column], table)
        agg_expr = f"{agg_func_upper}({agg_column})"
    else:
        agg_expr = f"{agg_func_upper}(*)"

    if group_by_cols:
        group_by_sql = ", ".join(group_by_cols)
        select_cols = f"{group_by_sql}, {agg_expr} AS agg_value"
        group_clause = f" GROUP BY {group_by_sql}"
    else:
        select_cols = f"{agg_expr} AS agg_value"
        group_clause = ""

    # Build WHERE clause
    where_sql, params = build_where_clause(filters or {}, table)

    sql = f"SELECT {select_cols} FROM {table}{where_sql}{group_clause}"

    # HAVING clause — applies to the aggregate value
    if having:
        having_op = having.get("op", ">=").upper()
        having_value = having.get("value")
        if having_op not in VALID_OPERATORS:
            raise ValueError(f"Invalid HAVING operator '{having_op}'")
        sql += f" HAVING agg_value {having_op} ?"
        params.append(having_value)

    # ORDER BY
    if order_by:
        if order_by == "agg_value":
            sql += " ORDER BY agg_value DESC"
        else:
            validate_columns([order_by], table)
            sql += f" ORDER BY {order_by}"

    sql += " LIMIT ?"
    params.append(limit)

    logger.debug("Aggregation query: {sql} params={params}", sql=sql, params=params)
    rows = client.query(sql, params)

    result_columns = group_by_cols + ["agg_value"]
    result_rows = [dict(zip(result_columns, row)) for row in rows]

    return {
        "columns": result_columns,
        "rows": result_rows,
        "row_count": len(result_rows),
    }


def run_filtered_query(
    client: DuckDBClient,
    table: str,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    order_by: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """Execute a filtered SELECT query returning matching rows.

    Args:
        client: DuckDB client instance.
        table: Table name (must be in VALID_TABLES).
        columns: Columns to select. None selects all columns.
        filters: Optional WHERE clause filters.
        order_by: Column to ORDER BY.
        limit: Maximum rows to return.

    Returns:
        Dict with keys: columns, rows, row_count.

    Example:
        # "Show fraud transactions over $5000"
        result = run_filtered_query(
            client, table="transactions",
            columns=["tx_id", "sender_account_id", "tx_amount"],
            filters={"is_fraud": True, "tx_amount": {"op": ">", "value": 5000}},
            limit=50,
        )
    """
    if table not in VALID_TABLES:
        raise ValueError(f"Unknown table '{table}'. Valid: {VALID_TABLES}")

    if columns:
        validate_columns(columns, table)
        select_cols = ", ".join(columns)
    else:
        select_cols = "*"
        columns = sorted(COLUMN_WHITELIST[table])

    where_sql, params = build_where_clause(filters or {}, table)

    sql = f"SELECT {select_cols} FROM {table}{where_sql}"

    if order_by:
        validate_columns([order_by], table)
        sql += f" ORDER BY {order_by}"

    sql += " LIMIT ?"
    params.append(limit)

    logger.debug("Filtered query: {sql} params={params}", sql=sql, params=params)
    rows = client.query(sql, params)

    # Convert tuples to dicts when specific columns are selected
    if select_cols != "*":
        result_rows = [dict(zip(columns, row)) for row in rows]
    else:
        # For SELECT *, get column names from table metadata
        col_info = client.query(f"SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [table])
        col_names = [r[0] for r in col_info]
        result_rows = [dict(zip(col_names, row)) for row in rows]
        columns = col_names

    return {
        "columns": columns,
        "rows": result_rows,
        "row_count": len(result_rows),
    }
