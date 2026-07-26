"""Tests for EDATool: profiling, statistics, volume trends, fraud baseline, charts, and edge cases."""

import pytest
from pathlib import Path

from core.config import AppConfig
from core.types import ToolName
from storage.duckdb import DuckDBClient
from tools.eda.tool import EDAInput, execute_eda
from tools.eda.charts import (
    plot_transaction_volume_timeline,
    plot_amount_distribution,
    plot_fraud_vs_normal_comparison,
    plot_top_senders,
)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    """Creates a test DuckDB with schema and sample data for EDA."""
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")

    # Insert test accounts
    for i in range(5):
        client.execute(
            "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)",
            [i, f"C_{i}", 1000.0 + i * 100, "US", "I", i < 2, 1],
        )

    # Insert test transactions across multiple timestamps
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

    return client


# --- EDA Profiling ---

def test_eda_profiling(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_001", 1)
    assert result.status == "ok"
    profiling = result.data["profiling"]
    assert profiling["row_count"] == 10
    assert profiling["column_count"] == 8
    assert "tx_amount" in profiling["columns"]
    assert isinstance(profiling["null_rates"], dict)
    assert isinstance(profiling["type_breakdown"], dict)


def test_eda_profiling_accounts(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="accounts", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_002", 1)
    assert result.status == "ok"
    assert result.data["profiling"]["row_count"] == 5


# --- Amount Statistics ---

def test_eda_amount_statistics(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_003", 1)
    stats = result.data["amount_statistics"]
    assert stats["min"] == 50.0
    assert stats["max"] == 15000.0
    assert stats["mean"] > 0
    assert stats["median"] > 0
    assert stats["std"] > 0
    assert stats["p25"] <= stats["median"] <= stats["p75"]
    assert stats["p95"] <= stats["p99"]
    assert stats["total_volume"] > 0


# --- Volume Trends ---

def test_eda_volume_trends(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_004", 1)
    trends = result.data["volume_trends"]
    assert trends["total_periods"] == 5  # timestamps: 10, 15, 20, 25, 30
    assert len(trends["periods"]) == 5
    assert len(trends["counts"]) == 5
    assert sum(trends["counts"]) == 10
    assert trends["avg_per_period"] == 2.0
    assert trends["max_period"] == 2
    assert trends["min_period"] == 2


# --- Fraud Baseline ---

def test_eda_fraud_baseline(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_005", 1)
    fraud = result.data["fraud_baseline"]
    assert fraud["fraud_count"] == 3
    assert fraud["normal_count"] == 7
    assert fraud["total_transactions"] == 10
    assert 0 < fraud["fraud_rate"] < 1
    assert fraud["fraud_amount_total"] > 0
    assert fraud["normal_amount_total"] > 0
    assert 0 < fraud["fraud_amount_share"] < 1


# --- Top Senders ---

def test_eda_top_senders(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_006", 1)
    top = result.data["top_senders"]
    assert isinstance(top, dict)
    assert len(top) > 0
    # Account 0 has 3 outgoing transactions
    assert top.get(0) == 3


# --- Filter Application ---

def test_eda_with_filters(db_client: DuckDBClient, config: AppConfig):
    """Filters narrow the analysis scope correctly."""
    input = EDAInput(
        table="transactions",
        filters={"timestamp": {"op": ">=", "value": 20}},
        include_charts=False,
    )
    result = execute_eda(input, config, db_client, "q_eda_007", 1)
    assert result.status == "ok"
    # Timestamps >= 20: rows at 20, 25, 30 = 6 transactions
    assert result.data["profiling"]["row_count"] == 6


def test_eda_with_fraud_filter(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(
        table="transactions",
        filters={"is_fraud": True},
        include_charts=False,
    )
    result = execute_eda(input, config, db_client, "q_eda_008", 1)
    assert result.data["profiling"]["row_count"] == 3
    assert result.data["fraud_baseline"]["fraud_count"] == 3


# --- Empty Dataset ---

def test_eda_empty_result(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(
        table="transactions",
        filters={"tx_amount": {"op": ">", "value": 999999}},
        include_charts=False,
    )
    result = execute_eda(input, config, db_client, "q_eda_009", 1)
    assert result.status == "ok"
    assert result.data["profiling"]["row_count"] == 0


# --- Invalid Table ---

def test_eda_invalid_table(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="nonexistent")
    result = execute_eda(input, config, db_client, "q_eda_010", 1)
    assert result.status == "error"
    assert "Unknown table" in result.error_summary


# --- Chart Generation ---

def test_eda_chart_generation(db_client: DuckDBClient, config: AppConfig, tmp_path: Path):
    chart_dir = str(tmp_path / "charts")
    input = EDAInput(
        table="transactions",
        include_charts=True,
        chart_output_dir=chart_dir,
    )
    result = execute_eda(input, config, db_client, "q_eda_011", 1)
    assert result.status == "ok"
    chart_paths = result.data.get("chart_paths", [])
    assert len(chart_paths) == 4  # timeline, distribution, fraud comparison, top senders

    for path in chart_paths:
        assert Path(path).exists()
        assert Path(path).suffix == ".png"
        assert Path(path).stat().st_size > 0


def test_eda_no_charts_when_disabled(db_client: DuckDBClient, config: AppConfig):
    input = EDAInput(table="transactions", include_charts=False)
    result = execute_eda(input, config, db_client, "q_eda_012", 1)
    assert "chart_paths" not in result.data


# --- Individual Chart Functions ---

def test_chart_volume_timeline(db_client: DuckDBClient, tmp_path: Path):
    df = db_client.query_df("SELECT * FROM transactions")
    path = plot_transaction_volume_timeline(df, tmp_path / "vol.png")
    assert Path(path).exists()


def test_chart_amount_distribution(db_client: DuckDBClient, tmp_path: Path):
    df = db_client.query_df("SELECT * FROM transactions")
    path = plot_amount_distribution(df, tmp_path / "dist.png")
    assert Path(path).exists()


def test_chart_fraud_comparison(db_client: DuckDBClient, tmp_path: Path):
    df = db_client.query_df("SELECT * FROM transactions")
    path = plot_fraud_vs_normal_comparison(df, tmp_path / "fraud.png")
    assert Path(path).exists()


def test_chart_top_senders(db_client: DuckDBClient, tmp_path: Path):
    df = db_client.query_df("SELECT * FROM transactions")
    path = plot_top_senders(df, tmp_path / "senders.png")
    assert Path(path).exists()


def test_chart_empty_dataframe(tmp_path: Path):
    """Chart functions handle empty DataFrames without crashing."""
    import pandas as pd
    empty_df = pd.DataFrame(columns=["tx_amount", "timestamp", "is_fraud", "sender_account_id"])
    path = plot_transaction_volume_timeline(empty_df, tmp_path / "empty_vol.png")
    assert Path(path).exists()


# --- Integration: real activity.duckdb ---

@pytest.mark.skipif(
    not Path("activity.duckdb").exists(),
    reason="Integration test requires real activity.duckdb",
)
class TestEDAIntegration:
    """Integration tests against the real IBM AMLSim dataset."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        self.client = DuckDBClient(db_path="activity.duckdb")
        self.config = AppConfig()
        self.chart_dir = str(tmp_path / "integration_charts")
        yield
        self.client.close()

    def test_full_eda_with_charts(self):
        input = EDAInput(
            table="transactions",
            include_charts=True,
            chart_output_dir=self.chart_dir,
        )
        result = execute_eda(input, self.config, self.client, "q_int_eda_001", 1)
        assert result.status == "ok"
        assert result.data["profiling"]["row_count"] > 1_000_000
        assert result.data["fraud_baseline"]["fraud_count"] == 1719
        assert len(result.data.get("chart_paths", [])) == 4

    def test_eda_scoped_to_recent_timestamps(self):
        input = EDAInput(
            table="transactions",
            filters={"timestamp": {"op": ">=", "value": 190}},
            include_charts=False,
        )
        result = execute_eda(input, self.config, self.client, "q_int_eda_002", 1)
        assert result.status == "ok"
        assert result.data["profiling"]["row_count"] < 1_000_000
