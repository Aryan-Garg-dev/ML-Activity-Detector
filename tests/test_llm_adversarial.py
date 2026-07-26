"""Adversarial stress tests for LLM integration, prompt parsers, and guardrails."""

import pytest
import json
from pydantic import BaseModel, ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

from core.config import AppConfig
from core.types import ProviderName, IntentType, PatternType, ToolName
from schemas.contracts import QuerySpec, PlanStep, ExecutionPlan
from llm.client import LLMClient, extract_json_substring
from llm.providers.base import BaseProviderAdapter
from llm.prompts.explanation_polish import polish_explanation
from agent.guardrails import validate_execution_plan, GuardrailValidationError
from tools.registry import ToolRegistry, ToolSpec


# --- Test 1: Adversarial JSON Extraction ---

def test_extract_json_substring_preamble_and_fences():
    raw_response_1 = "Here is your requested QuerySpec object:\n```json\n{\n  \"intent_type\": \"pattern_search\"\n}\n```\nLet me know if you need anything else."
    clean_1 = extract_json_substring(raw_response_1)
    assert clean_1 == '{\n  "intent_type": "pattern_search"\n}'
    assert json.loads(clean_1)["intent_type"] == "pattern_search"

    raw_response_2 = "Sure thing! JSON response: {\"intent_type\": \"broad_eda\", \"raw_query\": \"test\"} End of JSON."
    clean_2 = extract_json_substring(raw_response_2)
    assert json.loads(clean_2)["intent_type"] == "broad_eda"

    raw_response_3 = "```\n[{\"step_id\": 1, \"tool_name\": \"data_query\"}]\n```"
    clean_3 = extract_json_substring(raw_response_3)
    assert json.loads(clean_3)[0]["tool_name"] == "data_query"


# --- Test 2: Flexible Enum Normalization in QuerySpec & PlanStep ---

def test_query_spec_enum_normalization():
    # Uppercase and mixed case inputs
    qs = QuerySpec(
        intent_type="PATTERN_SEARCH",  # type: ignore
        pattern_type="Structuring",     # type: ignore
        raw_query="Test query",
    )
    assert qs.intent_type == IntentType.PATTERN_SEARCH
    assert qs.pattern_type == PatternType.STRUCTURING

    # PlanStep tool_name normalization
    step = PlanStep(
        step_id=1,
        tool_name="DATA_QUERY",  # type: ignore
        reason="Test",
    )
    assert step.tool_name == ToolName.DATA_QUERY


# --- Test 3: LLM Failure & Malformed JSON Fallbacks ---

class RaisingChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("API Connection Error: Provider down")

    @property
    def _llm_type(self) -> str:
        return "raising"


class RaisingAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return RaisingChatModel()


class GarbageJsonChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        content = "Sorry, I cannot process your request. I am a helpful AI assistant."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "garbage"


class GarbageJsonAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return GarbageJsonChatModel()


def test_llm_client_raising_model_error():
    config = AppConfig()
    client = LLMClient(config=config, adapter_override=RaisingAdapter())

    with pytest.raises(RuntimeError, match="Provider down"):
        client.invoke("Test prompt")


def test_llm_client_garbage_json_fallback_raises_json_decode():
    class TestSchema(BaseModel):
        val: int

    config = AppConfig()
    client = LLMClient(config=config, adapter_override=GarbageJsonAdapter())

    with pytest.raises(json.JSONDecodeError):
        client.structured_output(TestSchema, "Test prompt")


# --- Test 4: Numeric Drift Detection in Explanation Polish ---

class AlteredNumericChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Altered 10000 to 5000 and 85.5 to 90.0 (hallucinated numbers)
        content = "Account A101 triggered structuring flags with 5 transactions near $5,000 threshold and score 90.0."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "altered"


class AlteredNumericAdapter(BaseProviderAdapter):
    def get_chat_model(self, config: AppConfig, model_id: str | None = None) -> BaseChatModel:
        return AlteredNumericChatModel()


def test_explanation_polish_rejects_numeric_drift():
    raw_explanation = "Flagged as HIGH risk (composite score: 85.5/100). Account A101 exhibited structuring indicators with 10 transactions near the $10,000 threshold."
    config = AppConfig()
    client = LLMClient(config=config, adapter_override=AlteredNumericAdapter())

    # Should detect numeric drift ($10,000 -> $5,000, 85.5 -> 90.0) and revert to raw_explanation
    polished = polish_explanation(client, raw_explanation)
    assert polished == raw_explanation


# --- Test 5: Guardrails Adversarial Execution Plan Validation ---

class DummyArgs(BaseModel):
    limit: int


@pytest.fixture
def test_registry():
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name=ToolName.DATA_QUERY,
            description="Data query tool",
            input_schema=DummyArgs,
            callable=lambda *args: None,
        )
    )
    return registry


def test_guardrail_rejects_empty_plan(test_registry):
    plan = ExecutionPlan(plan_id="p1", query_id="q1", steps=[])
    is_valid, reason = validate_execution_plan(plan, test_registry)
    assert is_valid is False
    assert "no steps" in reason


def test_guardrail_rejects_bad_args_type(test_registry):
    plan = ExecutionPlan(
        plan_id="p1",
        query_id="q1",
        steps=[PlanStep(step_id=1, tool_name=ToolName.DATA_QUERY, args={"limit": "not_an_int"}, reason="Bad arg")],
    )
    is_valid, reason = validate_execution_plan(plan, test_registry)
    assert is_valid is False
    assert "failed validation" in reason
