"""Tests for DataQueryTool: filtered lookups, aggregations, column whitelist, and edge cases."""

import pytest
from pathlib import Path

from core.config import AppConfig
from core.types import ToolName
from storage.duckdb import DuckDBClient
from tools.data_query.tool import DataQueryInput, execute_data_query
from tools.data_query.aggregator import (
    validate_columns,
    build_where_clause,
    run_aggregation,
    run_filtered_query,
    VALID_TABLES,
)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    """Creates a test DuckDB with schema and sample data."""
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")

    # Insert test accounts
    for i in range(5):
        client.execute(
            "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)",
            [i, f"C_{i}", 1000.0 + i * 100, "US", "I", i < 2, 1],
        )

    # Insert test transactions with varied amounts and timestamps
    txns = [
        (1, 0, 1, "TRANSFER", 500.0, 10, False, None),
        (2, 0, 2, "TRANSFER", 9500.0, 10, False, None),
        (3, 0, 3, "TRANSFER", 9800.0, 15, True, None),
        (4, 1, 0, "TRANSFER", 200.0, 15, False, None),
        (5, 1, 2, "TRANSFER", 15000.0, 20, True, None),
        (6, 2, 3, "TRANSFER", 3000.0, 20, False, None),
        (7, 2, 4, "TRANSFER", 7500.0, 25, False, None),
        (8, 3, 0, "TRANSFER", 100.0, 25, False, None),
        (9, 3, 1, "TRANSFER", 9999.0, 30, True, None),
        (10, 4, 0, "TRANSFER", 50.0, 30, False, None),
    ]
    for tx in txns:
        client.execute(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", list(tx),
        )

    # Insert test alerts
    client.execute(
        "INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [1, 100, "cycle", True, 3, 0, 3, "TRANSFER", 9800.0, 15],
    )

    return client


# --- Column Whitelist Validation ---

def test_validate_columns_valid():
    validate_columns(["tx_id", "tx_amount", "sender_account_id"], "transactions")


def test_validate_columns_invalid():
    with pytest.raises(ValueError, match="Invalid columns"):
        validate_columns(["tx_id", "nonexistent_col"], "transactions")


def test_validate_columns_invalid_table():
    with pytest.raises(ValueError, match="Unknown table"):
        validate_columns(["id"], "unknown_table")


# --- WHERE Clause Building ---

def test_build_where_empty():
    sql, params = build_where_clause({}, "transactions")
    assert sql == ""
    assert params == []


def test_build_where_equality():
    sql, params = build_where_clause({"is_fraud": True}, "transactions")
    assert "is_fraud = ?" in sql
    assert params == [True]


def test_build_where_operator():
    sql, params = build_where_clause(
        {"tx_amount": {"op": ">", "value": 5000}}, "transactions",
    )
    assert "tx_amount > ?" in sql
    assert params == [5000]


def test_build_where_between():
    sql, params = build_where_clause(
        {"timestamp": {"op": "BETWEEN", "value": [10, 20]}}, "transactions",
    )
    assert "timestamp BETWEEN ? AND ?" in sql
    assert params == [10, 20]


def test_build_where_in():
    sql, params = build_where_clause(
        {"sender_account_id": {"op": "IN", "value": [0, 1, 2]}}, "transactions",
    )
    assert "sender_account_id IN (?, ?, ?)" in sql
    assert params == [0, 1, 2]


def test_build_where_not_equal():
    sql, params = build_where_clause(
        {"tx_type": {"op": "!=", "value": "TRANSFER"}}, "transactions",
    )
    assert "tx_type != ?" in sql
    assert params == ["TRANSFER"]


def test_build_where_invalid_operator():
    with pytest.raises(ValueError, match="Invalid operator"):
        build_where_clause(
            {"tx_amount": {"op": "DROP", "value": 0}}, "transactions",
        )


def test_build_where_between_invalid_value():
    with pytest.raises(ValueError, match="BETWEEN requires"):
        build_where_clause(
            {"timestamp": {"op": "BETWEEN", "value": [10]}}, "transactions",
        )


def test_build_where_in_empty_list():
    with pytest.raises(ValueError, match="non-empty"):
        build_where_clause(
            {"sender_account_id": {"op": "IN", "value": []}}, "transactions",
        )


def test_build_where_multiple_filters():
    sql, params = build_where_clause(
        {"is_fraud": True, "tx_amount": {"op": ">=", "value": 1000}},
        "transactions",
    )
    assert "is_fraud = ?" in sql
    assert "tx_amount >= ?" in sql
    assert len(params) == 2


# --- Filtered Query ---

def test_filtered_query_basic(db_client: DuckDBClient):
    result = run_filtered_query(db_client, "transactions", limit=5)
    assert result["row_count"] <= 5
    assert len(result["rows"]) == result["row_count"]
    assert "columns" in result


def test_filtered_query_with_filter(db_client: DuckDBClient):
    result = run_filtered_query(
        db_client, "transactions",
        columns=["tx_id", "tx_amount"],
        filters={"is_fraud": True},
    )
    assert result["row_count"] == 3
    for row in result["rows"]:
        assert "tx_id" in row
        assert "tx_amount" in row


def test_filtered_query_with_operator_filter(db_client: DuckDBClient):
    result = run_filtered_query(
        db_client, "transactions",
        columns=["tx_id", "tx_amount"],
        filters={"tx_amount": {"op": ">", "value": 5000}},
    )
    for row in result["rows"]:
        assert row["tx_amount"] > 5000


def test_filtered_query_empty_result(db_client: DuckDBClient):
    result = run_filtered_query(
        db_client, "transactions",
        filters={"tx_amount": {"op": ">", "value": 999999}},
    )
    assert result["row_count"] == 0
    assert result["rows"] == []


def test_filtered_query_accounts_table(db_client: DuckDBClient):
    result = run_filtered_query(
        db_client, "accounts",
        columns=["account_id", "customer_id"],
        filters={"is_fraud": True},
    )
    assert result["row_count"] == 2


def test_filtered_query_alerts_table(db_client: DuckDBClient):
    result = run_filtered_query(db_client, "alerts")
    assert result["row_count"] == 1
    assert result["rows"][0]["alert_type"] == "cycle"


def test_filtered_query_order_by(db_client: DuckDBClient):
    result = run_filtered_query(
        db_client, "transactions",
        columns=["tx_id", "tx_amount"],
        order_by="tx_amount",
        limit=3,
    )
    amounts = [r["tx_amount"] for r in result["rows"]]
    assert amounts == sorted(amounts)


def test_filtered_query_invalid_column(db_client: DuckDBClient):
    with pytest.raises(ValueError, match="Invalid columns"):
        run_filtered_query(
            db_client, "transactions", columns=["sql_injection_attempt"],
        )


def test_filtered_query_invalid_table(db_client: DuckDBClient):
    with pytest.raises(ValueError, match="Unknown table"):
        run_filtered_query(db_client, "nonexistent_table")


# --- Aggregation Query ---

def test_aggregation_count(db_client: DuckDBClient):
    """Count transactions per sender account."""
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
    )
    assert result["row_count"] > 0
    assert "agg_value" in result["columns"]


def test_aggregation_sum(db_client: DuckDBClient):
    """Sum transaction amounts per sender."""
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="SUM",
        agg_column="tx_amount",
    )
    assert result["row_count"] > 0
    for row in result["rows"]:
        assert row["agg_value"] > 0


def test_aggregation_avg(db_client: DuckDBClient):
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="AVG",
        agg_column="tx_amount",
    )
    assert result["row_count"] > 0


def test_aggregation_min_max(db_client: DuckDBClient):
    result_min = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="MIN",
        agg_column="tx_amount",
    )
    result_max = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="MAX",
        agg_column="tx_amount",
    )
    assert result_min["row_count"] > 0
    assert result_max["row_count"] > 0


def test_aggregation_with_having(db_client: DuckDBClient):
    """Accounts with 3+ transactions (accounts 0, 1, 2, 3 each have 2-3)."""
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
        having={"op": ">=", "value": 3},
    )
    for row in result["rows"]:
        assert row["agg_value"] >= 3


def test_aggregation_with_filter(db_client: DuckDBClient):
    """Count only transactions under $10,000 per sender."""
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
        filters={"tx_amount": {"op": "<", "value": 10000}},
    )
    assert result["row_count"] > 0


def test_aggregation_with_filter_and_having(db_client: DuckDBClient):
    """Accounts with 2+ transactions under $10,000 — the core AML structuring query."""
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
        filters={"tx_amount": {"op": "<", "value": 10000}},
        having={"op": ">=", "value": 2},
    )
    for row in result["rows"]:
        assert row["agg_value"] >= 2


def test_aggregation_order_by_agg_value(db_client: DuckDBClient):
    result = run_aggregation(
        db_client, "transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
        order_by="agg_value",
    )
    if result["row_count"] > 1:
        values = [r["agg_value"] for r in result["rows"]]
        assert values == sorted(values, reverse=True)


def test_aggregation_invalid_function(db_client: DuckDBClient):
    with pytest.raises(ValueError, match="Invalid aggregation"):
        run_aggregation(
            db_client, "transactions",
            group_by=["sender_account_id"],
            agg_func="DROP",
        )


def test_aggregation_invalid_column(db_client: DuckDBClient):
    with pytest.raises(ValueError, match="Invalid columns"):
        run_aggregation(
            db_client, "transactions",
            group_by=["nonexistent_column"],
            agg_func="COUNT",
        )


# --- DataQueryTool execute() ---

def test_execute_filtered_lookup(db_client: DuckDBClient, config: AppConfig):
    input = DataQueryInput(
        table="transactions",
        columns=["tx_id", "tx_amount"],
        filters={"is_fraud": True},
        limit=50,
    )
    result = execute_data_query(input, config, db_client, "q_001", 1)
    assert result.status == "ok"
    assert result.tool_name == ToolName.DATA_QUERY
    assert result.data["mode"] == "filtered_lookup"
    assert result.data["row_count"] == 3


def test_execute_aggregation(db_client: DuckDBClient, config: AppConfig):
    input = DataQueryInput(
        table="transactions",
        group_by=["sender_account_id"],
        agg_func="COUNT",
        having={"op": ">=", "value": 2},
    )
    result = execute_data_query(input, config, db_client, "q_002", 2)
    assert result.status == "ok"
    assert result.data["mode"] == "aggregation"
    assert result.data["row_count"] > 0


def test_execute_invalid_table(db_client: DuckDBClient, config: AppConfig):
    input = DataQueryInput(table="bad_table")
    result = execute_data_query(input, config, db_client, "q_003", 1)
    assert result.status == "error"
    assert "Unknown table" in result.error_summary


def test_execute_invalid_column_returns_error(db_client: DuckDBClient, config: AppConfig):
    input = DataQueryInput(
        table="transactions",
        columns=["sql_injection"],
    )
    result = execute_data_query(input, config, db_client, "q_004", 1)
    assert result.status == "error"
    assert "Invalid columns" in result.error_summary


    def test_empty_group_by_aggregation(self, db_client: DuckDBClient):
        """Test aggregation without group_by columns (global table aggregation)."""
        res = run_aggregation(
            client=db_client,
            table="transactions",
            group_by=[],
            agg_func="COUNT",
            filters={"tx_amount": {"op": "<", "value": 500}},
        )
        assert res["row_count"] == 1
        assert "agg_value" in res["rows"][0]
        assert res["rows"][0]["agg_value"] == 3


# --- Integration: real activity.duckdb ---

@pytest.mark.skipif(
    not Path("activity.duckdb").exists(),
    reason="Integration test requires real activity.duckdb",
)
class TestDataQueryIntegration:
    """Integration tests against the real IBM AMLSim dataset."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = DuckDBClient(db_path="activity.duckdb")
        self.config = AppConfig()
        yield
        self.client.close()

    def test_count_fraud_transactions(self):
        input = DataQueryInput(
            table="transactions",
            group_by=["is_fraud"],
            agg_func="COUNT",
        )
        result = execute_data_query(input, self.config, self.client, "q_int_001", 1)
        assert result.status == "ok"
        assert result.data["row_count"] == 2

    def test_top_senders_above_threshold(self):
        """Which accounts sent 10+ transactions under $10,000?"""
        input = DataQueryInput(
            table="transactions",
            group_by=["sender_account_id"],
            agg_func="COUNT",
            filters={"tx_amount": {"op": "<", "value": 10000}},
            having={"op": ">=", "value": 10},
            order_by="agg_value",
            limit=20,
        )
        result = execute_data_query(input, self.config, self.client, "q_int_002", 1)
        assert result.status == "ok"
        for row in result.data["rows"]:
            assert row["agg_value"] >= 10

    def test_filtered_fraud_lookup(self):
        input = DataQueryInput(
            table="transactions",
            columns=["tx_id", "sender_account_id", "tx_amount"],
            filters={"is_fraud": True},
            limit=10,
        )
        result = execute_data_query(input, self.config, self.client, "q_int_003", 1)
        assert result.status == "ok"
        assert result.data["row_count"] <= 10
