"""Abstract base class for LLM provider adapters."""

from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from core.config import AppConfig


class BaseProviderAdapter(ABC):
    """Abstract interface that all LLM provider adapters must implement."""

    @abstractmethod
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        """Instantiate and return a LangChain BaseChatModel configured according to AppConfig.

        Args:
            config: Application configuration instance.
            model_id: Optional explicit model ID override. If None, config.model_id is used.

        Returns:
            Configured BaseChatModel ready for invocation or structured output binding.
        """
        pass
