from pathlib import Path
from typing import Any
import time
import duckdb
from pandas import DataFrame
from core.config import AppConfig
from core.logging import logger

# Manages connection lifecycle and parameterized query execution for DuckDB
class DuckDBClient:
    def __init__(self, db_path: str | Path = "activity.duckdb", read_only: bool = False) -> None:
        self.db_path = str(db_path)
        self.read_only = read_only
        # On Windows, DuckDB uses exclusive file locks. Retry with backoff when
        # another process/test holds the write lock (common in full test suites).
        _max_retries = 3
        _retry_delay = 0.3  # seconds
        for attempt in range(_max_retries):
            try:
                self.conn = duckdb.connect(self.db_path, read_only=read_only)
                return  # connected successfully
            except Exception as e:
                is_lock_error = "used by another process" in str(e).lower() or "cannot open file" in str(e).lower()
                if not is_lock_error:
                    raise  # not a lock error — re-raise immediately
                if attempt < _max_retries - 1:
                    logger.debug("DuckDB lock contention on attempt {n}/{m} — retrying in {d}s", n=attempt + 1, m=_max_retries, d=_retry_delay)
                    time.sleep(_retry_delay)
                    _retry_delay *= 2  # exponential backoff
                else:
                    # All retries exhausted — fall back to read-only
                    if not read_only:
                        logger.warning("DuckDB read-write connection busy, falling back to read-only mode")
                        try:
                            self.conn = duckdb.connect(self.db_path, read_only=True)
                            self.read_only = True
                            return
                        except Exception:
                            pass  # fall through to re-raise original
                    raise

    # Executes schema SQL script to create tables if they do not exist
    def init_schema(self, schema_file: str | Path = "sql/duckdb_schema.sql") -> None:
        if self.read_only:
            return
        schema_path = Path(schema_file)
        if schema_path.exists():
            sql = schema_path.read_text(encoding="utf-8")
            self.conn.execute(sql)
            logger.info("DuckDB schema initialized successfully.")

    # Executes parameterized SQL query returning list of tuples
    def query(self, sql: str, params: list[Any] | dict[str, Any] | None = None) -> list[tuple]:
        if params is None:
            return self.conn.execute(sql).fetchall()
        return self.conn.execute(sql, params).fetchall()

    # Executes parameterized SQL query returning Pandas DataFrame
    def query_df(self, sql: str, params: list[Any] | dict[str, Any] | None = None) -> DataFrame:
        if params is None:
            return self.conn.execute(sql).df()
        return self.conn.execute(sql, params).df()

    # Executes parameterized SQL DML/DDL statement
    def execute(self, sql: str, params: list[Any] | dict[str, Any] | None = None) -> None:
        if self.read_only:
            logger.debug("Write operation skipped on read-only connection: {sql}", sql=sql[:50])
            return
        if params is None:
            self.conn.execute(sql)
        else:
            self.conn.execute(sql, params)

    # Closes the DuckDB connection
    def close(self) -> None:
        self.conn.close()

# Factory function creating initialized DuckDB client from AppConfig
def get_duckdb_client(config: AppConfig | None = None, read_only: bool = False) -> DuckDBClient:
    db_path = config.db_path if config else "activity.duckdb"
    client = DuckDBClient(db_path=db_path, read_only=read_only)
    client.init_schema()
    return client
