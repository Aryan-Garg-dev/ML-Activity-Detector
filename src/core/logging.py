import sys
from pathlib import Path
from loguru import logger
from core.config import AppConfig

# Configures Loguru logger with console fallback toggle and log file specificity settings
def setup_logging(config: AppConfig) -> None:
    logger.remove()

    # Enable console logging if log_to_console is True OR if log_to_file is False (safety toggle fallback)
    should_log_to_console = config.log_to_console or not config.log_to_file
    if should_log_to_console:
        logger.add(
            sys.stdout,
            level=config.console_log_level,
            format=config.log_format,
            colorize=config.dev_mode,
        )

    # Enable file logging with specificity level if log_to_file is True
    if config.log_to_file:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = config.log_dir / "app.log"
        logger.add(
            log_file,
            level=config.file_log_level,
            format=config.log_format,
            rotation="10 MB",
            retention="30 days",
        )
        jsonl_file = config.log_dir / "tool_execution.jsonl"
        logger.add(
            jsonl_file,
            level=config.file_log_level,
            serialize=True,
            filter=lambda record: "tool_name" in record["extra"],
        )

# Emits structured tool execution log binding payload keys into Loguru record context
def log_tool_execution(
    query_id: str,
    tool_name: str,
    step_id: int,
    started_at: str,
    ended_at: str,
    duration_ms: float,
    rows_in: int,
    rows_out: int,
    provider: str | None,
    model_name: str | None,
    config_version: str,
    status: str,
    error_summary: str | None = None,
) -> None:
    payload = {
        "query_id": query_id,
        "tool_name": tool_name,
        "step_id": step_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "provider": provider,
        "model_name": model_name,
        "config_version": config_version,
        "status": status,
        "error_summary": error_summary,
    }
    logger.bind(**payload).info("Tool execution: {tool_name} step={step_id} status={status}", tool_name=tool_name, step_id=step_id, status=status)
