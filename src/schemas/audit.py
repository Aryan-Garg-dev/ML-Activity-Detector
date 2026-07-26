from typing import Any
from pydantic import BaseModel, Field
from core.types import ToolName

# Structured log and database event for tool execution audit
class AuditEvent(BaseModel):
    audit_id: str | None = None
    query_id: str
    step_id: int
    tool_name: ToolName
    started_at: str
    ended_at: str
    duration_ms: float
    rows_in: int
    rows_out: int
    provider: str | None = None
    model_name: str | None = None
    config_version: str
    status: str
    error_summary: str | None = None

# Individual log record for tool calls
class ToolCallLog(BaseModel):
    call_id: str
    query_id: str
    tool_name: str
    input_args: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    success: bool

# Overall run metadata for an end-to-end query execution
class RunMetadata(BaseModel):
    query_id: str
    raw_query: str
    total_duration_ms: float
    config_version: str
    status: str
