import json
import uuid
from typing import Any
from schemas.domain import AccountRecord, TransactionRecord, AlertRecord, FeatureSet
from schemas.audit import AuditEvent
from storage.duckdb import DuckDBClient

# Repository for account metadata queries
class AccountRepo:
    def __init__(self, client: DuckDBClient) -> None:
        self.client = client

    def get_by_id(self, account_id: int) -> AccountRecord | None:
        sql = "SELECT account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id FROM accounts WHERE account_id = ?"
        rows = self.client.query(sql, [account_id])
        if not rows:
            return None
        r = rows[0]
        return AccountRecord(
            account_id=r[0], customer_id=r[1], init_balance=float(r[2]),
            country=r[3], account_type=r[4], is_fraud=bool(r[5]), tx_behavior_id=r[6]
        )

    def get_all(self, limit: int = 100) -> list[AccountRecord]:
        sql = "SELECT account_id, customer_id, init_balance, country, account_type, is_fraud, tx_behavior_id FROM accounts LIMIT ?"
        rows = self.client.query(sql, [limit])
        return [
            AccountRecord(
                account_id=r[0], customer_id=r[1], init_balance=float(r[2]),
                country=r[3], account_type=r[4], is_fraud=bool(r[5]), tx_behavior_id=r[6]
            )
            for r in rows
        ]

# Repository for transaction queries
class TransactionRepo:
    def __init__(self, client: DuckDBClient) -> None:
        self.client = client

    def get_by_id(self, tx_id: int) -> TransactionRecord | None:
        sql = "SELECT tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id FROM transactions WHERE tx_id = ?"
        rows = self.client.query(sql, [tx_id])
        if not rows:
            return None
        r = rows[0]
        return TransactionRecord(
            tx_id=r[0], sender_account_id=r[1], receiver_account_id=r[2],
            tx_type=r[3], tx_amount=float(r[4]), timestamp=r[5], is_fraud=bool(r[6]),
            alert_id=r[7] if r[7] != -1 else None
        )

    def get_by_account(self, account_id: int, limit: int = 100) -> list[TransactionRecord]:
        sql = "SELECT tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp, is_fraud, alert_id FROM transactions WHERE sender_account_id = ? OR receiver_account_id = ? ORDER BY timestamp DESC LIMIT ?"
        rows = self.client.query(sql, [account_id, account_id, limit])
        return [
            TransactionRecord(
                tx_id=r[0], sender_account_id=r[1], receiver_account_id=r[2],
                tx_type=r[3], tx_amount=float(r[4]), timestamp=r[5], is_fraud=bool(r[6]),
                alert_id=r[7] if r[7] != -1 else None
            )
            for r in rows
        ]

# Repository for alert queries
class AlertRepo:
    def __init__(self, client: DuckDBClient) -> None:
        self.client = client

    def get_by_id(self, alert_id: int) -> list[AlertRecord]:
        sql = "SELECT alert_row_id, alert_id, alert_type, is_fraud, tx_id, sender_account_id, receiver_account_id, tx_type, tx_amount, timestamp FROM alerts WHERE alert_id = ?"
        rows = self.client.query(sql, [alert_id])
        return [
            AlertRecord(
                alert_row_id=r[0], alert_id=r[1], alert_type=r[2], is_fraud=bool(r[3]),
                tx_id=r[4], sender_account_id=r[5], receiver_account_id=r[6],
                tx_type=r[7], tx_amount=float(r[8]), timestamp=r[9]
            )
            for r in rows
        ]

# Repository for persistent audit log events
class AuditRepo:
    def __init__(self, client: DuckDBClient) -> None:
        self.client = client

    def save_audit_event(self, event: AuditEvent) -> str:
        audit_id = event.audit_id or f"aud_{uuid.uuid4().hex[:8]}"
        sql = """
        INSERT INTO audit_log (audit_id, query_id, step_id, tool_name, started_at, ended_at, duration_ms, rows_in, rows_out, provider, model_name, config_version, status, error_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.client.execute(sql, [
            audit_id, event.query_id, event.step_id, str(event.tool_name),
            event.started_at, event.ended_at, event.duration_ms, event.rows_in,
            event.rows_out, event.provider, event.model_name, event.config_version,
            event.status, event.error_summary
        ])
        return audit_id

    save_event = save_audit_event

    def get_by_query_id(self, query_id: str) -> list[AuditEvent]:
        sql = "SELECT audit_id, query_id, step_id, tool_name, started_at, ended_at, duration_ms, rows_in, rows_out, provider, model_name, config_version, status, error_summary FROM audit_log WHERE query_id = ? ORDER BY step_id ASC"
        rows = self.client.query(sql, [query_id])
        return [
            AuditEvent(
                audit_id=r[0], query_id=r[1], step_id=r[2], tool_name=r[3],
                started_at=r[4], ended_at=r[5], duration_ms=float(r[6]),
                rows_in=r[7], rows_out=r[8], provider=r[9], model_name=r[10],
                config_version=r[11], status=r[12], error_summary=r[13]
            )
            for r in rows
        ]

# Repository for feature cache persistence
class FeatureRepo:
    def __init__(self, client: DuckDBClient) -> None:
        self.client = client

    def save_features(self, feature_set: FeatureSet) -> None:
        sql = """
        INSERT INTO feature_cache (account_id, feature_family, window_days, features_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (account_id, feature_family, window_days) DO UPDATE SET features_json = EXCLUDED.features_json
        """
        self.client.execute(sql, [
            feature_set.account_id, feature_set.feature_family,
            feature_set.window_days, json.dumps(feature_set.features)
        ])

    def get_features(self, account_id: int, feature_family: str, window_days: int) -> FeatureSet | None:
        sql = "SELECT account_id, feature_family, window_days, features_json, computed_at FROM feature_cache WHERE account_id = ? AND feature_family = ? AND window_days = ?"
        rows = self.client.query(sql, [account_id, feature_family, window_days])
        if not rows:
            return None
        r = rows[0]
        return FeatureSet(
            account_id=r[0], feature_family=r[1], window_days=r[2],
            features=json.loads(r[3]), computed_at=str(r[4])
        )
