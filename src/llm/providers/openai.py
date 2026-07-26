"""OpenAI provider adapter using ChatOpenAI."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import AppConfig
from llm.providers.base import BaseProviderAdapter


class OpenAIProviderAdapter(BaseProviderAdapter):
    """Adapter for official OpenAI models."""

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        selected_model = model_id or config.model_id
        api_key = config.get_api_key()
        base_url = config.get_base_url()

        kwargs = {
            "model": selected_model,
            "openai_api_key": api_key,
            "temperature": config.temperature,
            "max_retries": config.max_retries,
            "request_timeout": config.timeout_seconds,
        }
        if base_url:
            kwargs["openai_api_base"] = base_url

        return ChatOpenAI(**kwargs)
