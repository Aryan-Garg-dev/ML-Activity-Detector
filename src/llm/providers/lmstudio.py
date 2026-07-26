"""LM Studio provider adapter using ChatOpenAI for local OpenAI-compatible endpoints."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import AppConfig
from llm.providers.base import BaseProviderAdapter


class LMStudioProviderAdapter(BaseProviderAdapter):
    """Adapter for local LM Studio / OpenAI-compatible models."""

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        selected_model = model_id or config.model_id
        api_key = config.get_api_key() or "lm-studio"
        base_url = config.get_base_url() or "http://localhost:1234/v1"

        return ChatOpenAI(
            model=selected_model,
            api_key=api_key,
            base_url=base_url,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=config.temperature,
            max_retries=config.max_retries,
            request_timeout=config.timeout_seconds,
        )

