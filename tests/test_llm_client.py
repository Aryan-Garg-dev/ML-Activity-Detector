"""Unit tests for LLM client and provider adapters."""

import pytest
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage

from core.config import AppConfig
from core.types import ProviderName
from llm.client import LLMClient
from llm.providers.base import BaseProviderAdapter


class DummyOutputSchema(BaseModel):
    summary: str
    risk_score: float


from langchain_core.outputs import ChatResult, ChatGeneration


class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = '{"summary": "Test AML summary", "risk_score": 85.5}'
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])


    @property
    def _llm_type(self) -> str:
        return "mock"


class MockAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return MockChatModel()


def test_llm_client_mock_invocation():
    config = AppConfig()
    client = LLMClient(config=config, adapter_override=MockAdapter())

    res = client.invoke("Test query")
    assert "summary" in res

    structured = client.structured_output(DummyOutputSchema, "Test prompt")
    assert isinstance(structured, DummyOutputSchema)
    assert structured.summary == "Test AML summary"
    assert structured.risk_score == 85.5


def test_provider_adapter_resolution():
    config = AppConfig(provider=ProviderName.LMSTUDIO, base_url="http://localhost:1234/v1")
    client = LLMClient(config=config)
    assert client.adapter.__class__.__name__ == "LMStudioProviderAdapter"
