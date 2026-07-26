import os
import json
from pathlib import Path
from core.config import AppConfig
from core.types import IntentType, PatternType, RiskLevel, EscalationAction, ToolName, ProviderName
from core.logging import setup_logging, log_tool_execution

# Test loading application configuration from custom YAML files
def test_app_config_load_custom_yaml(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("PROVIDER", raising=False)
    monkeypatch.delenv("MAX_PLAN_STEPS", raising=False)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    app_yaml = config_dir / "app.yaml"
    app_yaml.write_text("provider: 'ollama'\nmax_plan_steps: 12\n", encoding="utf-8")

    config = AppConfig.load(config_dir=config_dir, env_file=tmp_path / ".env_empty")
    assert config.provider == ProviderName.OLLAMA
    assert config.max_plan_steps == 12


# Test environment variable overrides over YAML config settings
def test_app_config_env_overrides(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "app.yaml").write_text("provider: 'groq'\nmodel_id: 'default_model'\n", encoding="utf-8")

    monkeypatch.setenv("PROVIDER", "openrouter")
    monkeypatch.setenv("MODEL_ID", "custom_model_id")
    monkeypatch.setenv("LOG_TO_FILE", "false")

    # Use a non-existent .env so load_dotenv doesn't override monkeypatched values
    config = AppConfig.load(config_dir=config_dir, env_file=tmp_path / ".env.nonexistent")
    assert config.provider == ProviderName.OPENROUTER
    assert config.model_id == "custom_model_id"
    assert config.log_to_file is False

# Test Enum string representations and validity
def test_enums():
    assert IntentType.PATTERN_SEARCH == "pattern_search"
    assert PatternType.STRUCTURING == "structuring"
    assert RiskLevel.HIGH == "high"
    assert EscalationAction.REPORT == "report"
    assert ToolName.DETECTION == "detection"
    assert ProviderName.GROQ == "groq"

# Test setup_logging and structured jsonl tool execution logs
def test_logging_structured_jsonl(tmp_path: Path):
    logs_dir = tmp_path / "logs"
    config = AppConfig(log_dir=logs_dir, log_to_file=True, log_to_console=False)
    setup_logging(config)

    log_tool_execution(
        query_id="q_001",
        tool_name="detection",
        step_id=1,
        started_at="2026-07-25T10:00:00Z",
        ended_at="2026-07-25T10:00:01Z",
        duration_ms=1000.0,
        rows_in=10,
        rows_out=2,
        provider="groq",
        model_name="llama-3.3-70b-versatile",
        config_version="v1.0.0",
        status="ok"
    )

    jsonl_file = logs_dir / "tool_execution.jsonl"
    assert jsonl_file.exists()
    content = jsonl_file.read_text(encoding="utf-8").strip()
    data = json.loads(content)
    assert data["record"]["extra"]["query_id"] == "q_001"
    assert data["record"]["extra"]["tool_name"] == "detection"

# Test console logging fallback when file logging is disabled
def test_logging_console_fallback(tmp_path: Path):
    config = AppConfig(log_dir=tmp_path / "logs", log_to_file=False, log_to_console=False)
    setup_logging(config)
    # console handler is enabled as fallback even if log_to_console is False
