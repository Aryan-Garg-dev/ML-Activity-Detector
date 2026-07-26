"""OpenRouter provider adapter using ChatOpenAI."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config import AppConfig
from llm.providers.base import BaseProviderAdapter


class OpenRouterProviderAdapter(BaseProviderAdapter):
    """Adapter for OpenRouter using ChatOpenAI integration."""

    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        selected_model = model_id or config.model_id
        api_key = config.get_api_key() or "openrouter-api-key"
        base_url = config.get_base_url() or "https://openrouter.ai/api/v1"

        return ChatOpenAI(
            model=selected_model,
            api_key=api_key,
            base_url=base_url,
            openai_api_key=api_key,
            openai_api_base=base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/Aryan-Garg-dev/ML-Activity-Detector",
                "X-Title": "ML Activity Detector",
            },
            temperature=config.temperature,
            max_retries=config.max_retries,
            request_timeout=config.timeout_seconds,
        )

