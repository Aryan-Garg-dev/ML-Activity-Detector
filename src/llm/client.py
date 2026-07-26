"""LLMClient wrapper for provider resolution and structured generation."""

import json
from typing import TypeVar, Type
from loguru import logger
from pydantic import BaseModel, ValidationError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from core.config import AppConfig
from core.types import ProviderName
from llm.providers.base import BaseProviderAdapter
from llm.providers.groq import GroqProviderAdapter
from llm.providers.ollama import OllamaProviderAdapter
from llm.providers.openrouter import OpenRouterProviderAdapter
from llm.providers.lmstudio import LMStudioProviderAdapter
from llm.providers.openai import OpenAIProviderAdapter

import re
from typing import TypeVar, Type

T = TypeVar("T", bound=BaseModel)



def extract_json_substring(text: str) -> str:
    """Extract first valid JSON object or array from text, stripping preambles and codeblocks."""
    # Find ```json ... ``` codeblock contents anywhere in text
    codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if codeblock_match:
        return codeblock_match.group(1).strip()

    # Find standalone JSON object {...} or array [...]
    json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if json_match:
        return json_match.group(1).strip()

    return text.strip()


class LLMClient:
    """Unified LLM client dispatching to the configured provider adapter."""

    def __init__(self, config: AppConfig, adapter_override: BaseProviderAdapter | None = None) -> None:
        self.config = config
        if adapter_override:
            self.adapter = adapter_override
        else:
            self.adapter = self._resolve_adapter(config.provider)
        
        self.model: BaseChatModel = self.adapter.get_chat_model(config)

    def _resolve_adapter(self, provider: ProviderName) -> BaseProviderAdapter:
        provider_str = str(provider).lower()
        if provider_str == ProviderName.GROQ:
            return GroqProviderAdapter()
        elif provider_str == ProviderName.OLLAMA:
            return OllamaProviderAdapter()
        elif provider_str == ProviderName.OPENROUTER:
            return OpenRouterProviderAdapter()
        elif provider_str == ProviderName.LMSTUDIO:
            return LMStudioProviderAdapter()
        elif provider_str == ProviderName.OPENAI:
            return OpenAIProviderAdapter()
        else:
            logger.warning("Unrecognized provider '{provider}', falling back to Groq", provider=provider)
            return GroqProviderAdapter()

    def invoke(self, prompt: str, system_prompt: str | None = None) -> str:
        """Invoke the chat model with optional system prompt and return the text response."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        logger.debug("Invoking LLM [{provider}:{model}]", provider=self.config.provider, model=self.config.model_id)
        response = self.model.invoke(messages)
        return str(response.content).strip()

    def structured_output(self, schema: Type[T], prompt: str, system_prompt: str | None = None) -> T:
        """Generate structured output validated against a Pydantic schema."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        logger.debug(
            "Invoking LLM for structured output [{schema}] via [{provider}]",
            schema=schema.__name__,
            provider=self.config.provider,
        )

        # For local endpoints (LMStudio, Ollama), bypass native function calling to avoid 401/400 unsupported tool errors
        provider_str = str(self.config.provider).lower()
        if provider_str in (ProviderName.LMSTUDIO, ProviderName.OLLAMA):
            example_dict = {}
            if hasattr(schema, "model_config") and isinstance(schema.model_config, dict):
                json_extra = schema.model_config.get("json_schema_extra", {})
                if isinstance(json_extra, dict) and "examples" in json_extra and json_extra["examples"]:
                    example_dict = json_extra["examples"][0]

            json_instruction = (
                f"\n\nRespond strictly with a single JSON object conforming to the `{schema.__name__}` structure.\n"
                f"Required fields: {list(schema.model_fields.keys())}.\n"
                f"Example JSON response:\n{json.dumps(example_dict if example_dict else {'intent_type': 'pattern_search', 'pattern_type': 'structuring', 'raw_query': prompt}, indent=2)}\n"
                f"Return ONLY valid JSON. Do not include markdown codeblocks or schema definitions."
            )
            fallback_prompt = prompt + json_instruction
            fallback_messages = []
            if system_prompt:
                fallback_messages.append(SystemMessage(content=system_prompt))
            fallback_messages.append(HumanMessage(content=fallback_prompt))

            raw_response = self.model.invoke(fallback_messages)
            raw_text = str(raw_response.content).strip()
            clean_json = extract_json_substring(raw_text)
            data = json.loads(clean_json)

            # Handle edge case where LLM echoes schema with $defs/properties
            if isinstance(data, dict) and ("$defs" in data or "properties" in data):
                if "properties" in data and isinstance(data["properties"], dict):
                    data = {k: v.get("default") for k, v in data["properties"].items() if isinstance(v, dict)}

            return schema(**data)

        # Attempt native LangChain structured output binding for cloud providers
        try:
            structured_llm = self.model.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema(**result)
        except Exception as e:
            logger.warning("Native structured output failed ({err}), falling back to JSON parsing", err=e)

        # Fallback: prompt LLM to produce JSON matching schema
        json_instruction = (
            f"\n\nRespond strictly with valid JSON conforming to `{schema.__name__}` fields: {list(schema.model_fields.keys())}.\n"
            f"Do not include markdown codeblocks or extra text outside JSON."
        )
        fallback_prompt = prompt + json_instruction
        fallback_messages = []
        if system_prompt:
            fallback_messages.append(SystemMessage(content=system_prompt))
        fallback_messages.append(HumanMessage(content=fallback_prompt))

        raw_response = self.model.invoke(fallback_messages)
        raw_text = str(raw_response.content).strip()
        clean_json = extract_json_substring(raw_text)

        data = json.loads(clean_json)
        if isinstance(data, dict) and ("$defs" in data or "properties" in data):
            if "properties" in data and isinstance(data["properties"], dict):
                data = {k: v.get("default") for k, v in data["properties"].items() if isinstance(v, dict)}

        return schema(**data)



