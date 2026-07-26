"""Tests for Feature Engineering Tool: volume, threshold, network, velocity, caching, and LangChain integration."""

import pytest
from pathlib import Path
import pandas as pd

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.registry import ToolRegistry, ToolSpec
from tools.features.tool import FeatureInput, execute_feature_engineering
from tools.features.volume import compute_volume_features
from tools.features.threshold import compute_threshold_features
from tools.features.network import compute_network_features
from tools.features.velocity import compute_velocity_features


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    """Creates a test DuckDB with schema and transactions/accounts sample data."""
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")

    # Insert test accounts
    accounts = [
        (101, "C_101", 5000.0, "US", "I", False, 1),
        (102, "C_102", 10000.0, "US", "I", True, 1),
        (103, "C_103", 2000.0, "MX", "I", False, 1),
    ]
    for acc in accounts:
        client.execute("INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?, ?)", list(acc))

    # Insert test transactions
    # Structuring/smurfing pattern: 101 sends multiple ~9000 amounts to 102
    txns = [
        (1, 101, 102, "TRANSFER", 9500.0, 10, True, None),
        (2, 101, 102, "TRANSFER", 9800.0, 12, True, None),
        (3, 101, 102, "TRANSFER", 9000.0, 15, True, None),
        (4, 102, 103, "TRANSFER", 25000.0, 20, True, None),
        (5, 103, 101, "TRANSFER", 500.0, 25, False, None),
    ]
    for tx in txns:
        client.execute("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", list(tx))

    return client


# --- Individual Feature Module Tests ---

def test_volume_features():
    df = pd.DataFrame([
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 9500.0, "timestamp": 10},
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 9800.0, "timestamp": 12},
    ])
    res = compute_volume_features(df, account_ids=[101], window_days=30)
    assert 101 in res
    assert res[101]["txn_count_30d"] == 2
    assert res[101]["txn_sum_30d"] == 19300.0
    assert res[101]["avg_amount"] == 9650.0


def test_threshold_features():
    df = pd.DataFrame([
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 9543.21, "timestamp": 10}, # near threshold, non-round
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 100.0, "timestamp": 12},   # round number
    ])
    res = compute_threshold_features(df, account_ids=[101], reporting_threshold=10000.0, window_days=30)
    assert 101 in res
    assert res[101]["count_near_threshold_30d"] == 1
    assert res[101]["round_number_bias"] == 0.5


def test_network_features():
    df = pd.DataFrame([
        {"sender_account_id": 101, "receiver_account_id": 102, "timestamp": 10},
        {"sender_account_id": 103, "receiver_account_id": 102, "timestamp": 12},
    ])
    res = compute_network_features(df, account_ids=[102], window_days=30)
    assert 102 in res
    assert res[102]["fan_in_degree"] == 2
    assert res[102]["fan_out_degree"] == 0
    assert res[102]["distinct_counterparties_30d"] == 2


def test_velocity_features():
    df = pd.DataFrame([
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 9500.0, "timestamp": 10},
        {"sender_account_id": 101, "receiver_account_id": 102, "tx_amount": 9800.0, "timestamp": 12},
    ])
    acc_df = pd.DataFrame([
        {"account_id": 101, "country": "US"},
        {"account_id": 102, "country": "US"},
    ])
    res = compute_velocity_features(df, acc_df, account_ids=[101], window_days=30)
    assert 101 in res
    assert "velocity_zscore" in res[101]
    assert "dwell_time_avg_hours" in res[101]


# --- FeatureTool Dispatcher & Cache Tests ---

def test_execute_feature_engineering_volume(db_client: DuckDBClient, config: AppConfig):
    input = FeatureInput(feature_family="volume", account_ids=[101], use_cache=False)
    result = execute_feature_engineering(input, config, db_client, "q_feat_001", 1)
    assert result.status == "ok"
    assert result.tool_name == ToolName.FEATURE_ENGINEERING
    assert result.data["account_count"] == 1
    fsets = result.data["feature_sets"]
    assert 101 in fsets or "101" in fsets
    feats = fsets.get(101) or fsets.get("101")
    assert "txn_count_30d" in feats


def test_execute_feature_engineering_all(db_client: DuckDBClient, config: AppConfig):
    input = FeatureInput(feature_family="all", account_ids=[101, 102], use_cache=False)
    result = execute_feature_engineering(input, config, db_client, "q_feat_002", 1)
    assert result.status == "ok"
    fsets = result.data["feature_sets"]
    feats_101 = fsets.get(101) or fsets.get("101")
    # All families merged
    assert "txn_count_30d" in feats_101
    assert "count_near_threshold_30d" in feats_101
    assert "fan_in_degree" in feats_101
    assert "velocity_zscore" in feats_101


def test_execute_feature_engineering_caching(db_client: DuckDBClient, config: AppConfig):
    input_first = FeatureInput(feature_family="volume", account_ids=[101], use_cache=True)
    res1 = execute_feature_engineering(input_first, config, db_client, "q_feat_003", 1)
    assert res1.status == "ok"
    assert res1.data["from_cache"] is False

    # Second run should read from cache
    res2 = execute_feature_engineering(input_first, config, db_client, "q_feat_004", 1)
    assert res2.status == "ok"
    assert res2.data["from_cache"] is True


def test_execute_feature_engineering_partial_cache_empty_slice(db_client: DuckDBClient, config: AppConfig):
    """Partial cache hit preserves cached features when uncached accounts have zero transactions."""
    # Seed cache for account 101
    input_seed = FeatureInput(feature_family="volume", account_ids=[101], use_cache=True)
    execute_feature_engineering(input_seed, config, db_client, "q_seed", 1)

    # Request cached account 101 + uncached nonexistent account 999999
    input_mixed = FeatureInput(feature_family="volume", account_ids=[101, 999999], use_cache=True)
    res_mixed = execute_feature_engineering(input_mixed, config, db_client, "q_mixed", 1)

    assert res_mixed.status == "ok"
    fsets = res_mixed.data["feature_sets"]
    # Account 101 MUST be preserved from cache even though account 999999 had 0 transactions!
    assert 101 in fsets or "101" in fsets
    assert 999999 in fsets or "999999" in fsets


def test_execute_invalid_feature_family(db_client: DuckDBClient, config: AppConfig):
    input = FeatureInput(feature_family="invalid_family")
    result = execute_feature_engineering(input, config, db_client, "q_feat_005", 1)
    assert result.status == "error"
    assert "Invalid feature_family" in result.error_summary


# --- LangChain Integration Export Test ---

def test_langchain_structured_tool_export(db_client: DuckDBClient, config: AppConfig):
    spec = ToolSpec(
        name=ToolName.FEATURE_ENGINEERING,
        description="Compute AML features for accounts",
        input_schema=FeatureInput,
        callable=execute_feature_engineering,
    )
    lc_tool = spec.to_langchain_tool(config, db_client, "q_lc_001")
    assert lc_tool.name == "feature_engineering"
    assert lc_tool.description == "Compute AML features for accounts"
    assert lc_tool.args_schema == FeatureInput

    registry = ToolRegistry()
    registry.register(spec)
    lc_tools = registry.get_langchain_tools(config, db_client)
    assert len(lc_tools) == 1
    assert lc_tools[0].name == "feature_engineering"


# --- Integration Test on Real activity.duckdb ---

@pytest.mark.skipif(
    not Path("activity.duckdb").exists(),
    reason="Integration test requires real activity.duckdb",
)
class TestFeatureIntegration:
    """Integration tests against real IBM AMLSim dataset."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.client = DuckDBClient(db_path="activity.duckdb")
        self.config = AppConfig()
        yield
        self.client.close()

    def test_real_dataset_volume_features(self):
        input = FeatureInput(feature_family="volume", account_ids=[0, 1], use_cache=False)
        result = execute_feature_engineering(input, self.config, self.client, "q_int_feat_001", 1)
        assert result.status == "ok"
        assert result.data["account_count"] == 2

    def test_real_dataset_all_families(self):
        input = FeatureInput(feature_family="all", account_ids=[0], use_cache=False)
        result = execute_feature_engineering(input, self.config, self.client, "q_int_feat_002", 1)
        assert result.status == "ok"
        fsets = result.data["feature_sets"]
        assert 0 in fsets or "0" in fsets
