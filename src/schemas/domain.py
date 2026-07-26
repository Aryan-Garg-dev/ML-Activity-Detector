from pydantic import BaseModel, Field

# Account record schema representing accounts table rows
class AccountRecord(BaseModel):
    account_id: int
    customer_id: str
    init_balance: float
    country: str
    account_type: str
    is_fraud: bool
    tx_behavior_id: int

# Transaction record schema representing transactions table rows
class TransactionRecord(BaseModel):
    tx_id: int
    sender_account_id: int
    receiver_account_id: int
    tx_type: str
    tx_amount: float
    timestamp: int
    is_fraud: bool
    alert_id: int | None = None

# Alert record schema representing alerts table rows
class AlertRecord(BaseModel):
    alert_row_id: int | None = None
    alert_id: int
    alert_type: str
    is_fraud: bool
    tx_id: int
    sender_account_id: int
    receiver_account_id: int
    tx_type: str
    tx_amount: float
    timestamp: int

# Computed feature set per account and window
class FeatureSet(BaseModel):
    account_id: int
    feature_family: str
    window_days: int
    features: dict[str, float | int | str | bool] = Field(default_factory=dict)
    computed_at: str | None = None
