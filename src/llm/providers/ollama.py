"""Ollama provider adapter using ChatOllama."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from core.config import AppConfig
from llm.providers.base import BaseProviderAdapter


class OllamaProviderAdapter(BaseProviderAdapter):
    """Adapter for local Ollama models via langchain-ollama."""

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        selected_model = model_id or config.model_id
        base_url = config.get_base_url() or "http://localhost:11434"

        return ChatOllama(
            model=selected_model,
            base_url=base_url,
            temperature=config.temperature,
        )
