"""Tests for the ToolRegistry: registration, lookup, validation, dispatch, and audit logging."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from pydantic import BaseModel, Field

from core.config import AppConfig
from core.types import ToolName
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient
from tools.registry import ToolRegistry, ToolSpec, ToolNotFoundError, ToolValidationError


# Minimal input/output schemas for test tools
class _MockInput(BaseModel):
    value: int
    label: str = "default"


class _MockInputRequired(BaseModel):
    required_field: str


def _mock_tool_callable(input: _MockInput, config, db_client, query_id, step_id) -> ToolResult:
    """A test tool that doubles the input value."""
    return ToolResult(
        tool_name=ToolName.DATA_QUERY,
        step_id=step_id,
        status="ok",
        data={"result": input.value * 2, "label": input.label},
        rows_count=1,
    )


def _mock_tool_raises(input: _MockInput, config, db_client, query_id, step_id) -> ToolResult:
    """A test tool that always raises an exception."""
    raise RuntimeError("Intentional test failure")


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def db_client(tmp_path: Path) -> DuckDBClient:
    client = DuckDBClient(db_path=tmp_path / "test.duckdb")
    client.init_schema("sql/duckdb_schema.sql")
    return client


@pytest.fixture
def sample_spec() -> ToolSpec:
    return ToolSpec(
        name=ToolName.DATA_QUERY,
        description="Test data query tool",
        input_schema=_MockInput,
        callable=_mock_tool_callable,
    )


# --- Registration and Lookup ---

def test_register_and_get(registry: ToolRegistry, sample_spec: ToolSpec):
    registry.register(sample_spec)
    retrieved = registry.get(ToolName.DATA_QUERY)
    assert retrieved.name == ToolName.DATA_QUERY
    assert retrieved.description == "Test data query tool"


def test_register_duplicate_raises(registry: ToolRegistry, sample_spec: ToolSpec):
    registry.register(sample_spec)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(sample_spec)


def test_get_unknown_tool_raises(registry: ToolRegistry):
    with pytest.raises(ToolNotFoundError, match="not registered"):
        registry.get(ToolName.EDA)


def test_has_tool(registry: ToolRegistry, sample_spec: ToolSpec):
    assert not registry.has_tool(ToolName.DATA_QUERY)
    registry.register(sample_spec)
    assert registry.has_tool(ToolName.DATA_QUERY)


def test_list_tools_empty(registry: ToolRegistry):
    assert registry.list_tools() == []


def test_list_tools_returns_all_registered(registry: ToolRegistry):
    spec1 = ToolSpec(
        name=ToolName.DATA_QUERY,
        description="Tool 1",
        input_schema=_MockInput,
        callable=_mock_tool_callable,
    )
    spec2 = ToolSpec(
        name=ToolName.EDA,
        description="Tool 2",
        input_schema=_MockInput,
        callable=_mock_tool_callable,
    )
    registry.register(spec1)
    registry.register(spec2)
    tools = registry.list_tools()
    assert len(tools) == 2
    names = {t.name for t in tools}
    assert names == {ToolName.DATA_QUERY, ToolName.EDA}


# --- validate_and_call: Success Cases ---

def test_validate_and_call_success(registry: ToolRegistry, sample_spec: ToolSpec, config: AppConfig, db_client: DuckDBClient):
    registry.register(sample_spec)
    result = registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": 5, "label": "test"},
        config=config,
        db_client=db_client,
        query_id="q_test_001",
        step_id=1,
    )
    assert result.status == "ok"
    assert result.data["result"] == 10
    assert result.data["label"] == "test"
    assert result.rows_count == 1
    assert result.duration_ms >= 0


def test_validate_and_call_with_defaults(registry: ToolRegistry, sample_spec: ToolSpec, config: AppConfig, db_client: DuckDBClient):
    """Input schema defaults are applied when not provided."""
    registry.register(sample_spec)
    result = registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": 3},
        config=config,
        db_client=db_client,
        query_id="q_test_002",
        step_id=2,
    )
    assert result.status == "ok"
    assert result.data["label"] == "default"


# --- validate_and_call: Error Cases ---

def test_validate_and_call_unknown_tool(registry: ToolRegistry, config: AppConfig, db_client: DuckDBClient):
    result = registry.validate_and_call(
        name=ToolName.EDA,
        args={},
        config=config,
        db_client=db_client,
        query_id="q_test_003",
        step_id=1,
    )
    assert result.status == "error"
    assert "not found" in result.error_summary


def test_validate_and_call_invalid_args(registry: ToolRegistry, config: AppConfig, db_client: DuckDBClient):
    """Missing required field triggers validation error."""
    spec = ToolSpec(
        name=ToolName.DATA_QUERY,
        description="Test",
        input_schema=_MockInputRequired,
        callable=_mock_tool_callable,
    )
    registry.register(spec)
    result = registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={},
        config=config,
        db_client=db_client,
        query_id="q_test_004",
        step_id=1,
    )
    assert result.status == "error"
    assert "validation failed" in result.error_summary.lower() or "required" in result.error_summary.lower()


def test_validate_and_call_wrong_type(registry: ToolRegistry, sample_spec: ToolSpec, config: AppConfig, db_client: DuckDBClient):
    """Wrong type for a field triggers validation error."""
    registry.register(sample_spec)
    result = registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": "not_an_int"},
        config=config,
        db_client=db_client,
        query_id="q_test_005",
        step_id=1,
    )
    assert result.status == "error"
    assert "validation failed" in result.error_summary.lower() or "int" in result.error_summary.lower()


def test_validate_and_call_tool_exception(registry: ToolRegistry, config: AppConfig, db_client: DuckDBClient):
    """Tool that raises an exception returns error ToolResult instead of propagating."""
    spec = ToolSpec(
        name=ToolName.DATA_QUERY,
        description="Failing tool",
        input_schema=_MockInput,
        callable=_mock_tool_raises,
    )
    registry.register(spec)
    result = registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": 1},
        config=config,
        db_client=db_client,
        query_id="q_test_006",
        step_id=1,
    )
    assert result.status == "error"
    assert "Intentional test failure" in result.error_summary
    assert result.duration_ms >= 0


# --- Structured Logging ---

@patch("tools.registry.log_tool_execution")
def test_validate_and_call_emits_structured_log(mock_log, registry: ToolRegistry, sample_spec: ToolSpec, config: AppConfig, db_client: DuckDBClient):
    registry.register(sample_spec)
    registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": 7},
        config=config,
        db_client=db_client,
        query_id="q_log_test",
        step_id=3,
    )
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args
    # Positional args passed to log_tool_execution
    assert call_kwargs[1]["query_id"] == "q_log_test" or call_kwargs[0][0] == "q_log_test"


@patch("tools.registry.log_tool_execution")
def test_validate_and_call_logs_on_error(mock_log, registry: ToolRegistry, config: AppConfig, db_client: DuckDBClient):
    """Structured log is emitted even when tool execution fails."""
    spec = ToolSpec(
        name=ToolName.DATA_QUERY,
        description="Failing tool",
        input_schema=_MockInput,
        callable=_mock_tool_raises,
    )
    registry.register(spec)
    registry.validate_and_call(
        name=ToolName.DATA_QUERY,
        args={"value": 1},
        config=config,
        db_client=db_client,
        query_id="q_error_log",
        step_id=1,
    )
    mock_log.assert_called_once()
