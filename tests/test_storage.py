from pathlib import Path
from storage.duckdb import DuckDBClient
from storage.repositories import AccountRepo, TransactionRepo, AlertRepo, AuditRepo, FeatureRepo
from schemas.domain import AccountRecord, FeatureSet
from schemas.audit import AuditEvent
from core.types import ToolName
# pyrefly: ignore [missing-import]
from scripts.load_data import load_dataset
from core.config import AppConfig

# Test repository query operations on DuckDB database
def test_storage_repositories(tmp_path: Path):
    db_path = tmp_path / "test.duckdb"
    client = DuckDBClient(db_path=db_path)
    client.init_schema("sql/duckdb_schema.sql")

    # Test AccountRepo
    acc_repo = AccountRepo(client)
    client.execute(
        "INSERT INTO accounts (account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [101, "C_101", 500.0, "US", "I", False, 1]
    )
    acc = acc_repo.get_by_id(101)
    assert acc is not None
    assert acc.customer_id == "C_101"
    all_accs = acc_repo.get_all(limit=10)
    assert len(all_accs) == 1

    # Test TransactionRepo with sentinel alert_id = -1
    tx_repo = TransactionRepo(client)
    client.execute(
        "INSERT INTO transactions (tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [1, 101, 101, "TRANSFER", 150.0, 1000, False, None]
    )
    tx = tx_repo.get_by_id(1)
    assert tx is not None
    assert tx.alert_id is None

    # Test AlertRepo
    alert_repo = AlertRepo(client)
    client.execute(
        "INSERT INTO alerts (alert_row_id, alert_id, alert_type, is_fraud, tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [1, 500, "structuring", True, 1, 101, 101, "TRANSFER", 150.0, 1000]
    )
    alerts = alert_repo.get_by_id(500)
    assert len(alerts) == 1
    assert alerts[0].alert_type == "structuring"

    # Test AuditRepo
    audit_repo = AuditRepo(client)
    event = AuditEvent(
        query_id="q_test", step_id=1, tool_name=ToolName.DATA_QUERY,
        started_at="2026-07-25T10:00:00Z", ended_at="2026-07-25T10:00:01Z",
        duration_ms=500.0, rows_in=5, rows_out=5, config_version="v1.0.0", status="ok"
    )
    audit_repo.save_audit_event(event)
    logs = audit_repo.get_by_query_id("q_test")
    assert len(logs) == 1
    assert logs[0].tool_name == ToolName.DATA_QUERY

    # Test FeatureRepo
    feat_repo = FeatureRepo(client)
    fset = FeatureSet(account_id=101, feature_family="volume", window_days=14, features={"txn_count": 10})
    feat_repo.save_features(fset)
    loaded_fset = feat_repo.get_features(101, "volume", 14)
    assert loaded_fset is not None
    assert loaded_fset.features["txn_count"] == 10

    client.close()

# Test dataset ingestion repeatability and repository query on real DuckDB database
def test_dataset_ingestion_and_repo_query(tmp_path: Path):
    db_path = tmp_path / "activity.duckdb"
    config = AppConfig(db_path=db_path, raw_data_dir=Path("dataset"))
    load_dataset(config)

    client = DuckDBClient(db_path=db_path)
    acc_repo = AccountRepo(client)
    accounts = acc_repo.get_all(limit=5)
    assert len(accounts) == 5

    tx_repo = TransactionRepo(client)
    txs = tx_repo.get_by_account(accounts[0].account_id, limit=5)
    assert isinstance(txs, list)
    client.close()
