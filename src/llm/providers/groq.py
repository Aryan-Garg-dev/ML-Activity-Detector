"""Groq provider adapter using ChatGroq."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from core.config import AppConfig
from llm.providers.base import BaseProviderAdapter


class GroqProviderAdapter(BaseProviderAdapter):
    """Adapter for Groq models via langchain-groq."""

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        selected_model = model_id or config.model_id
        api_key = config.get_api_key()

        return ChatGroq(
            model=selected_model,
            groq_api_key=api_key,
            temperature=config.temperature,
            max_retries=config.max_retries,
            timeout=config.timeout_seconds,
        )
