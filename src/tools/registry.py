"""Tool registry: maps ToolName → ToolSpec, validates inputs, dispatches calls with audit logging.

The registry is the single boundary where all tool execution is validated, timed, and audited.
Every tool call flows through `validate_and_call`, which:
  1. Validates the tool name exists in the registry whitelist
  2. Validates input args against the tool's declared Pydantic input schema
  3. Executes the tool callable
  4. Captures timing, row counts, and status
  5. Emits a structured loguru log entry and returns the typed ToolResult
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger
from pydantic import BaseModel, ValidationError

from core.config import AppConfig
from core.logging import log_tool_execution
from core.types import ToolName
from schemas.audit import AuditEvent
from schemas.contracts import ToolResult
from storage.duckdb import DuckDBClient


class ToolNotFoundError(Exception):
    """Raised when a requested tool name is not registered."""

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' is not registered")


class ToolValidationError(Exception):
    """Raised when tool input arguments fail schema validation."""

    def __init__(self, tool_name: str, details: str) -> None:
        self.tool_name = tool_name
        self.details = details
        super().__init__(f"Validation failed for tool '{tool_name}': {details}")


class ToolSpec(BaseModel):
    """Specification for a registered tool.

    Attributes:
        name: Unique tool identifier matching a ToolName enum value.
        description: One-sentence summary used for LLM context in Phase 3.
        input_schema: Pydantic model class defining the tool's expected input shape.
        output_schema: Pydantic model class defining the tool's output shape (ToolResult by default).
        callable: The function or method that executes the tool logic.
            Signature: (input, config, db_client, query_id, step_id) -> ToolResult
    """

    name: ToolName
    description: str
    input_schema: Any  # type[BaseModel] — stored as Any to avoid Pydantic serialization issues
    output_schema: Any = ToolResult
    callable: Any  # Callable[..., ToolResult]

    model_config = {"arbitrary_types_allowed": True}

    def to_langchain_tool(
        self,
        config: AppConfig,
        db_client: DuckDBClient,
        query_id: str = "langchain_run",
        step_id: int = 1,
    ) -> Any:
        """Export tool spec as a LangChain StructuredTool for LangGraph tool-calling."""
        from langchain_core.tools import StructuredTool

        def _tool_run(**kwargs) -> str:
            validated_input = self.input_schema(**kwargs)
            result: ToolResult = self.callable(
                validated_input, config, db_client, query_id, step_id
            )
            return result.model_dump_json()

        return StructuredTool.from_function(
            func=_tool_run,
            name=str(self.name),
            description=self.description,
            args_schema=self.input_schema,
        )


class ToolRegistry:
    """Central dispatch for all tool execution.

    Usage:
        registry = ToolRegistry()
        registry.register(tool_spec)
        result = registry.validate_and_call("data_query", args_dict, config, db_client, query_id, step_id)
    """

    def __init__(self) -> None:
        self._tools: dict[ToolName, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool. Rejects duplicate names."""
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec
        logger.debug("Registered tool: {name}", name=spec.name)

    def get(self, name: ToolName) -> ToolSpec:
        """Look up a tool by name. Raises ToolNotFoundError if not registered."""
        if name not in self._tools:
            raise ToolNotFoundError(str(name))
        return self._tools[name]

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tools (useful for LLM context injection in Phase 3)."""
        return list(self._tools.values())

    def get_langchain_tools(
        self,
        config: AppConfig,
        db_client: DuckDBClient,
        query_id: str = "langchain_run",
    ) -> list[Any]:
        """Export all registered tools as LangChain StructuredTool objects."""
        return [
            spec.to_langchain_tool(config, db_client, query_id)
            for spec in self._tools.values()
        ]

    def has_tool(self, name: ToolName) -> bool:
        return name in self._tools

    def validate_and_call(
        self,
        name: ToolName,
        args: dict[str, Any],
        config: AppConfig,
        db_client: DuckDBClient,
        query_id: str,
        step_id: int,
    ) -> ToolResult:
        """Validate input args, execute the tool, capture timing and audit.

        This is the single execution path for all tools. It ensures:
        - The tool exists (whitelist check)
        - Input args match the declared schema
        - Execution is timed
        - Success/failure is logged via structured loguru output
        - A ToolResult is always returned (even on error)

        Example:
            result = registry.validate_and_call(
                name=ToolName.DATA_QUERY,
                args={"table": "transactions", "filters": {"is_fraud": True}},
                config=config,
                db_client=client,
                query_id="q_001",
                step_id=1,
            )
        """
        started_at = datetime.now(timezone.utc)
        started_at_iso = started_at.isoformat()

        # Validate tool exists
        try:
            spec = self.get(name)
        except ToolNotFoundError:
            ended_at_iso = datetime.now(timezone.utc).isoformat()
            _log_execution(
                query_id, str(name), step_id, started_at_iso, ended_at_iso,
                0.0, 0, 0, config.config_version, "error",
                error_summary=f"Tool '{name}' not found in registry",
            )
            return ToolResult(
                tool_name=name, step_id=step_id, status="error",
                error_summary=f"Tool '{name}' not found in registry",
            )

        # Validate input args against declared schema
        try:
            validated_input = spec.input_schema(**args)
        except (ValidationError, TypeError) as e:
            ended_at_iso = datetime.now(timezone.utc).isoformat()
            error_msg = f"Input validation failed: {e}"
            _log_execution(
                query_id, str(name), step_id, started_at_iso, ended_at_iso,
                0.0, 0, 0, config.config_version, "error",
                error_summary=error_msg,
            )
            return ToolResult(
                tool_name=name, step_id=step_id, status="error",
                error_summary=error_msg,
            )

        # Execute the tool
        try:
            result: ToolResult = spec.callable(
                validated_input, config, db_client, query_id, step_id,
            )
            ended_at = datetime.now(timezone.utc)
            duration_ms = (ended_at - started_at).total_seconds() * 1000
            result.duration_ms = duration_ms

            _log_execution(
                query_id, str(name), step_id, started_at_iso, ended_at.isoformat(),
                duration_ms, result.data.get("rows_in", 0), result.rows_count,
                config.config_version, result.status,
            )
            return result

        except Exception as e:
            ended_at = datetime.now(timezone.utc)
            duration_ms = (ended_at - started_at).total_seconds() * 1000
            error_msg = f"{type(e).__name__}: {e}"
            logger.exception("Tool execution failed: {name}", name=name)

            _log_execution(
                query_id, str(name), step_id, started_at_iso, ended_at.isoformat(),
                duration_ms, 0, 0, config.config_version, "error",
                error_summary=error_msg,
            )
            return ToolResult(
                tool_name=name, step_id=step_id, status="error",
                duration_ms=duration_ms, error_summary=error_msg,
            )


def _log_execution(
    query_id: str,
    tool_name: str,
    step_id: int,
    started_at: str,
    ended_at: str,
    duration_ms: float,
    rows_in: int,
    rows_out: int,
    config_version: str,
    status: str,
    error_summary: str | None = None,
) -> None:
    """Emit structured tool execution log entry via loguru."""
    log_tool_execution(
        query_id=query_id,
        tool_name=tool_name,
        step_id=step_id,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=duration_ms,
        rows_in=rows_in,
        rows_out=rows_out,
        provider=None,
        model_name=None,
        config_version=config_version,
        status=status,
        error_summary=error_summary,
    )


def build_default_tool_registry() -> ToolRegistry:
    """Construct and return a ToolRegistry populated with all default tools."""
    from tools.data_query.tool import execute_data_query, DataQueryInput
    from tools.eda.tool import execute_eda, EDAInput
    from tools.features.tool import execute_feature_engineering, FeatureInput
    from tools.detection.tool import execute_detection, DetectionInput
    from tools.scoring.tool import execute_scoring, ScoringInput
    from tools.explanation.tool import execute_explanation, ExplanationInput
    from tools.reporting.tool import execute_reporting, ReportingInput
    from tools.entity_lookup.tool import execute_entity_lookup, EntityLookupInput

    registry = ToolRegistry()
    registry.register(ToolSpec(name=ToolName.DATA_QUERY, description="DuckDB-backed direct aggregations and filtered lookups", input_schema=DataQueryInput, callable=execute_data_query))
    registry.register(ToolSpec(name=ToolName.EDA, description="Exploratory data analysis, profiles, and visual charts", input_schema=EDAInput, callable=execute_eda))
    registry.register(ToolSpec(name=ToolName.FEATURE_ENGINEERING, description="Computes AML feature families", input_schema=FeatureInput, callable=execute_feature_engineering))
    registry.register(ToolSpec(name=ToolName.DETECTION, description="Rule engine + ML anomaly detection ensemble", input_schema=DetectionInput, callable=execute_detection))
    registry.register(ToolSpec(name=ToolName.SCORING, description="Composite risk scoring and risk band classification", input_schema=ScoringInput, callable=execute_scoring))
    registry.register(ToolSpec(name=ToolName.EXPLANATION, description="Grounded natural language compliance explanations", input_schema=ExplanationInput, callable=execute_explanation))
    registry.register(ToolSpec(name=ToolName.REPORTING, description="Final AgentResponse payload assembly", input_schema=ReportingInput, callable=execute_reporting))
    registry.register(ToolSpec(name=ToolName.ENTITY_LOOKUP, description="Single entity lookup for account metadata, alerts, and risk assessments", input_schema=EntityLookupInput, callable=execute_entity_lookup))

    return registry

