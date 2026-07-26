CREATE TABLE IF NOT EXISTS accounts (
    account_id BIGINT PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    init_balance DECIMAL(18, 2) NOT NULL CHECK (init_balance >= 0),
    country VARCHAR NOT NULL,
    account_type VARCHAR NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    tx_behavior_id INTEGER NOT NULL CHECK (tx_behavior_id >= 0)
);

CREATE TABLE IF NOT EXISTS transactions (
    tx_id BIGINT PRIMARY KEY,
    sender_account_id BIGINT NOT NULL REFERENCES accounts(account_id),
    receiver_account_id BIGINT NOT NULL REFERENCES accounts(account_id),
    tx_type VARCHAR NOT NULL,
    tx_amount DECIMAL(18, 2) NOT NULL CHECK (tx_amount >= 0),
    timestamp BIGINT NOT NULL CHECK (timestamp >= 0),
    is_fraud BOOLEAN NOT NULL,
    alert_id BIGINT
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_row_id BIGINT PRIMARY KEY,
    alert_id BIGINT NOT NULL,
    alert_type VARCHAR NOT NULL,
    is_fraud BOOLEAN NOT NULL,
    tx_id BIGINT NOT NULL REFERENCES transactions(tx_id),
    sender_account_id BIGINT NOT NULL REFERENCES accounts(account_id),
    receiver_account_id BIGINT NOT NULL REFERENCES accounts(account_id),
    tx_type VARCHAR NOT NULL,
    tx_amount DECIMAL(18, 2) NOT NULL CHECK (tx_amount >= 0),
    timestamp BIGINT NOT NULL CHECK (timestamp >= 0)
);

CREATE TABLE IF NOT EXISTS feature_cache (
    account_id BIGINT NOT NULL,
    feature_family VARCHAR NOT NULL,
    window_days INTEGER NOT NULL,
    features_json JSON NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (account_id, feature_family, window_days)
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id VARCHAR PRIMARY KEY,
    query_id VARCHAR NOT NULL,
    step_id INTEGER NOT NULL,
    tool_name VARCHAR NOT NULL,
    started_at VARCHAR NOT NULL,
    ended_at VARCHAR NOT NULL,
    duration_ms DOUBLE NOT NULL,
    rows_in INTEGER NOT NULL,
    rows_out INTEGER NOT NULL,
    provider VARCHAR,
    model_name VARCHAR,
    config_version VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    error_summary VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id VARCHAR PRIMARY KEY,
    query_id VARCHAR NOT NULL,
    entity_id VARCHAR NOT NULL,
    composite_score DOUBLE NOT NULL,
    confidence DOUBLE NOT NULL,
    risk_level VARCHAR NOT NULL,
    escalation_action VARCHAR NOT NULL,
    signals_json JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
