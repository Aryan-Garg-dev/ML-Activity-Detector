import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from core.types import ProviderName

# Configuration container holding app, provider, logging, and threshold settings
class AppConfig(BaseModel):
    # Provider & LLM settings
    provider: ProviderName = ProviderName.GROQ
    model_id: str = "llama-3.3-70b-versatile"
    fallback_model_id: str = "llama-3.1-8b-instant"
    base_url: str | None = None
    api_key_env_var: str = "GROQ_API_KEY"
    api_key: str | None = None
    temperature: float = 0.0
    max_retries: int = 3
    timeout_seconds: float = 30.0

    def get_api_key(self) -> str | None:
        """Dynamically resolve API key from explicit setting, provider defaults, or designated env var."""
        if self.api_key:
            return self.api_key

        provider_str = str(self.provider).lower()
        if provider_str == ProviderName.LMSTUDIO:
            return os.getenv("LMSTUDIO_API_KEY") or "lm-studio"
        elif provider_str == ProviderName.OLLAMA:
            return "ollama"
        elif provider_str == ProviderName.GROQ:
            return os.getenv("GROQ_API_KEY")
        elif provider_str == ProviderName.OPENROUTER:
            return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif provider_str == ProviderName.OPENAI:
            return os.getenv("OPENAI_API_KEY")

        if self.api_key_env_var:
            env_val = os.getenv(self.api_key_env_var)
            if env_val:
                return env_val

        return None


    def get_base_url(self) -> str | None:
        """Dynamically resolve base URL for custom / local provider endpoints."""
        if self.base_url:
            return self.base_url
        
        provider_str = str(self.provider).lower()
        if provider_str == ProviderName.OPENROUTER:
            return os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        elif provider_str == ProviderName.LMSTUDIO:
            return os.getenv("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1"
        elif provider_str == ProviderName.OLLAMA:
            return os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        elif provider_str == ProviderName.OPENAI:
            return os.getenv("OPENAI_BASE_URL")
        return None


    # Data & storage settings
    db_path: Path = Path("activity.duckdb")
    raw_data_dir: Path = Path("dataset")
    processed_data_dir: Path = Path("data/processed")
    feature_cache_ttl_hours: int = 24

    # Runtime settings
    max_plan_steps: int = 8
    ml_random_seed: int = 42
    config_version: str = "v1.0.0"
    feature_parallel_workers: int = 4

    # Query response cache settings
    query_cache_enabled: bool = True
    query_cache_ttl_minutes: int = 30
    semantic_cache_enabled: bool = True
    # Lowered from 0.92 — catches near-paraphrases like "suspicious activity in records" ≈ "suspicious activity"
    # Fallback: 0.78 provides good recall without matching unrelated queries
    semantic_cache_threshold: float = 0.78

    # Logging settings
    log_level: str = "INFO"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    log_to_console: bool = True
    log_to_file: bool = True
    log_dir: Path = Path("logs")
    dev_mode: bool = True
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

    # Explanation output shaping — top-N flagged accounts shown per intent type
    # Fallbacks: entity=1 (single account), default=5 (pattern search), batch=10 (full scan)
    explain_top_n_entity: int = 1
    explain_top_n_default: int = 5
    explain_top_n_batch: int = 10

    # Explanation polish — LLM narrative refinement settings
    explanation_polish_enabled: bool = True
    # Fraction of grounded numbers allowed to be reformatted/omitted without triggering fallback
    # Fallback: 0.20 = up to 20% of numbers may be benignly reformatted (e.g. 81.1 -> 81.1/100)
    explanation_polish_tolerance: float = 0.20

    # Agent graph selection
    # True (default): tool-calling LLM agent — LLM dynamically selects tools
    # False: legacy Tier-1 decision table + Tier-2 LLM fallback planner
    use_tool_calling_agent: bool = True
    # Max tool-calling loop iterations before breaking (guards against infinite loops)
    # Fallback: 10 is sufficient for any AML pipeline (max real depth is 3-4 tools)
    tool_call_max_iterations: int = 10

    # Time window used when query says "between X and Y" with no explicit days
    # Fallback: 90 days is a standard AML look-back window
    default_window_days: int = 90

    # Max characters to display from raw_query in explanation context sentences
    # Fallback: 80 chars keeps single-line readability in compliance reports
    query_display_max_chars: int = 80

    # Thresholds & risk scoring settings
    reporting_threshold: float = 10000.0
    structuring_window_days: int = 14
    structuring_min_txn_count: int = 3
    risk_bands: dict[str, list[int]] = Field(default_factory=lambda: {"low": [0, 39], "medium": [40, 74], "high": [75, 100]})
    ensemble_weights: dict[str, float] = Field(default_factory=lambda: {"w_rule": 0.50, "w_ml": 0.35, "w_context": 0.15})
    ml_contamination: float = 0.05
    ml_percentile_cutoff: float = 95.0

    # Load configuration applying precedence: env vars > YAML files > default model values
    @classmethod
    def load(cls, config_dir: str | Path = "configs", env_file: str | Path = ".env") -> "AppConfig":
        project_root = _find_project_root()

        # Resolve .env and configs/ relative to project root, not CWD
        env_path = Path(env_file)
        if not env_path.is_absolute():
            env_path = project_root / env_path
        config_path = Path(config_dir)
        if not config_path.is_absolute():
            config_path = project_root / config_path

        load_dotenv(dotenv_path=env_path, override=True)
        merged_data: dict[str, Any] = {}

        for yaml_name in ["app.yaml", "logging.yaml", "thresholds.yaml"]:
            file_path = config_path / yaml_name
            if file_path.exists():
                with open(file_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)
                    if content and isinstance(content, dict):
                        merged_data.update(content)

        # Apply environment variable overrides for matching fields
        for field_name, field_info in cls.model_fields.items():
            env_val = os.getenv(field_name.upper()) or os.getenv(field_name)
            if env_val is not None:
                target_type = field_info.annotation
                merged_data[field_name] = _parse_env_value(env_val, target_type)

        return cls(**merged_data)


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (directory containing .env or pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / ".env").exists() or (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # Fallback: CWD
    return Path.cwd()


# Helper function to convert raw string env variables to model field types
def _parse_env_value(val: str, target_type: Any) -> Any:
    from enum import EnumMeta
    if target_type is bool:
        return val.lower() in ("true", "1", "yes")
    if target_type is int:
        return int(val)
    if target_type is float:
        return float(val)
    if target_type is Path:
        return Path(val)
    # Handle StrEnum / Enum types (e.g. ProviderName)
    if isinstance(target_type, EnumMeta):
        try:
            return target_type(val.strip().lower())
        except (ValueError, KeyError):
            return val
    return val

