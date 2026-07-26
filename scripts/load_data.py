import sys
from pathlib import Path

# Add project root and src directory to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.config import AppConfig
from core.logging import logger
from storage.duckdb import DuckDBClient

# Ingests IBM AMLSim synthetic dataset CSV files into DuckDB database
def load_dataset(config: AppConfig | None = None) -> None:
    if config is None:
        config = AppConfig.load()

    db_path = config.db_path
    dataset_dir = config.raw_data_dir

    logger.info(f"Connecting to DuckDB at {db_path}...")
    client = DuckDBClient(db_path=db_path)
    client.init_schema("sql/duckdb_schema.sql")

    accounts_csv = dataset_dir / "accounts.csv"
    transactions_csv = dataset_dir / "transactions.csv"
    alerts_csv = dataset_dir / "alerts.csv"

    if not accounts_csv.exists() and (dataset_dir / "test" / "accounts.csv").exists():
        dataset_dir = dataset_dir / "test"
        accounts_csv = dataset_dir / "accounts.csv"
        transactions_csv = dataset_dir / "transactions.csv"
        alerts_csv = dataset_dir / "alerts.csv"

    if accounts_csv.exists() or transactions_csv.exists() or alerts_csv.exists():
        logger.info("Clearing existing tables in reverse dependency order...")
        client.execute("DELETE FROM alerts")
        client.execute("DELETE FROM transactions")
        client.execute("DELETE FROM accounts")

    if accounts_csv.exists():
        logger.info(f"Loading {accounts_csv}...")
        client.execute(f"""
            INSERT INTO accounts (account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id)
            SELECT ACCOUNT_ID, CUSTOMER_ID, INIT_BALANCE, COUNTRY, ACCOUNT_TYPE, IS_FRAUD, TX_BEHAVIOR_ID
            FROM read_csv_auto('{accounts_csv.as_posix()}')
        """)
        count = client.query("SELECT COUNT(*) FROM accounts")[0][0]
        logger.info(f"Loaded {count} accounts.")

    if transactions_csv.exists():
        logger.info(f"Loading {transactions_csv}...")
        client.execute(f"""
            INSERT INTO transactions (tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id)
            SELECT TX_ID, SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_TYPE, TX_AMOUNT, TIMESTAMP, IS_FRAUD,
                   CASE WHEN ALERT_ID = -1 THEN NULL ELSE ALERT_ID END
            FROM read_csv_auto('{transactions_csv.as_posix()}')
        """)
        count = client.query("SELECT COUNT(*) FROM transactions")[0][0]
        logger.info(f"Loaded {count} transactions.")

    if alerts_csv.exists():
        logger.info(f"Loading {alerts_csv}...")
        client.execute(f"""
            INSERT INTO alerts (alert_row_id, alert_id, alert_type, is_fraud, tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp)
            SELECT row_number() OVER () AS alert_row_id, ALERT_ID, ALERT_TYPE, IS_FRAUD, TX_ID, SENDER_ACCOUNT_ID, RECEIVER_ACCOUNT_ID, TX_TYPE, TX_AMOUNT, TIMESTAMP
            FROM read_csv_auto('{alerts_csv.as_posix()}')
        """)
        count = client.query("SELECT COUNT(*) FROM alerts")[0][0]
        logger.info(f"Loaded {count} alert rows.")

    client.close()
    logger.info("Dataset ingestion complete.")

if __name__ == "__main__":
    load_dataset()
