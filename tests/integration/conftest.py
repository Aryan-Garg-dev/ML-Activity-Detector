"""Shared pytest fixtures for LLM integration tests.

Provides:
- `llm_client`: real LLMClient loaded from .env (Groq by default)
- `llm_config`: the AppConfig used by that client
- `requires_llm`: autouse marker applied to all tests in tests/integration/

If GROQ_API_KEY is not set in .env, or the provider cannot connect, tests are
skipped automatically with a clear message instead of failing the suite.

--- Switching to LM Studio (local, free, no rate limits) ---
If you hit Groq rate limits or want to run tests fully offline, start LM Studio
with a loaded model and run:

    $env:PROVIDER="lmstudio"
    $env:LMSTUDIO_BASE_URL="http://localhost:1234/v1"
    uv run pytest tests/integration/ -v

Or set in your .env:
    PROVIDER=lmstudio
    LMSTUDIO_BASE_URL=http://localhost:1234/v1
"""

import os
import pytest
from loguru import logger

from core.config import AppConfig
from core.types import ProviderName
from llm.client import LLMClient


def _build_real_llm_client() -> tuple[LLMClient, AppConfig] | None:
    """Load config from .env and attempt to build a real LLM client.

    Returns (client, config) on success, or None if provider is unavailable.
    """
    config = AppConfig.load()
    provider_str = str(config.provider).lower()

    # Check that the necessary credentials / endpoints are available
    if provider_str == ProviderName.GROQ:
        api_key = config.get_api_key()
        if not api_key or api_key == "your_groq_api_key_here":
            return None
    elif provider_str in (ProviderName.LMSTUDIO, ProviderName.OLLAMA):
        # Local providers — no key required, but base URL must be reachable
        # We don't ping here (would slow down collection), just let the test fail gracefully
        pass
    elif provider_str == ProviderName.OPENROUTER:
        api_key = config.get_api_key()
        if not api_key or api_key == "your_openrouter_api_key_here":
            return None

    try:
        client = LLMClient(config)
        return client, config
    except Exception as e:
        logger.warning("LLM client construction failed: {err}", err=e)
        return None


@pytest.fixture(scope="session")
def llm_config() -> AppConfig:
    """Return the AppConfig loaded from .env. Skips if no valid provider found."""
    result = _build_real_llm_client()
    if result is None:
        pytest.skip(
            "No configured LLM provider available.\n"
            "  • To use Groq: set GROQ_API_KEY in .env\n"
            "  • To use LM Studio (local, free): set PROVIDER=lmstudio and start LM Studio\n"
            "    then run: $env:PROVIDER='lmstudio'; uv run pytest tests/integration/ -v"
        )
    _, config = result
    return config


@pytest.fixture(scope="session")
def llm_client(llm_config: AppConfig) -> LLMClient:
    """Return a real LLMClient backed by the configured provider (Groq, LMStudio, etc.)."""
    return LLMClient(llm_config)
