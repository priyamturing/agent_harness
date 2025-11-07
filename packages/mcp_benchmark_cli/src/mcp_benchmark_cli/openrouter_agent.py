"""OpenRouter agent implementation for the CLI layer.

Provides a thin wrapper around the OpenAI-compatible OpenRouter API so we can
use any tool-capable model from https://openrouter.ai while staying entirely in
the CLI (SDK remains untouched).

References:
- Quickstart guide (headers, base URL): https://openrouter.ai/docs/quickstart
- Reasoning tokens guide: https://openrouter.ai/docs/use-cases/reasoning-tokens
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from mcp_benchmark_sdk import Agent, AgentResponse
from mcp_benchmark_sdk.parsers import OpenAIResponseParser, ResponseParser
from mcp_benchmark_sdk.utils import retry_with_backoff


class _OpenRouterModelCatalog:
    """Minimal cache for the OpenRouter `/models` endpoint."""

    _cache: dict[str, dict[str, Any]] = {}
    _last_refresh: float = 0.0
    _ttl_seconds: int = 600

    @classmethod
    def resolve(cls, slug: str) -> dict[str, Any]:
        """Return model metadata, refreshing the cache if needed."""
        if not slug:
            raise ValueError("OpenRouter model name cannot be empty")

        normalized = slug.lower()
        now = time.time()
        if cls._needs_refresh(now):
            cls._refresh(now)

        entry = cls._cache.get(normalized)
        if entry:
            return entry

        # One more refresh in case the catalog was updated out-of-band
        cls._refresh(time.time())
        entry = cls._cache.get(normalized)
        if not entry:
            raise ValueError(
                f"OpenRouter model '{slug}' not found. "
                "List available models with `curl https://openrouter.ai/api/v1/models`."
            )
        return entry

    @classmethod
    def _needs_refresh(cls, now: float) -> bool:
        return not cls._cache or (now - cls._last_refresh) > cls._ttl_seconds

    @classmethod
    def _refresh(cls, now: float) -> None:
        endpoint = os.environ.get(
            "OPENROUTER_MODELS_ENDPOINT",
            "https://openrouter.ai/api/v1/models",
        )
        try:
            response = httpx.get(endpoint, timeout=30.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Failed to load OpenRouter models from {endpoint}: {exc}") from exc

        payload = response.json()
        models = payload.get("data", [])
        cache: dict[str, dict[str, Any]] = {}

        for model in models:
            identifiers = {
                model.get("id"),
                model.get("canonical_slug"),
            }
            for identifier in identifiers:
                if identifier:
                    cache[identifier.lower()] = model

        if not cache:
            raise RuntimeError("OpenRouter models endpoint returned no entries")

        cls._cache = cache
        cls._last_refresh = now


class OpenRouterAgent(Agent):
    """Agent that talks to OpenRouter's OpenAI-compatible chat endpoint."""

    _REASONING_KEYS = ("reasoning", "include_reasoning")

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        tool_call_limit: int = 1000,
        system_prompt: Optional[str] = None,
        reasoning: Optional[dict[str, Any]] = None,
        include_reasoning: Optional[bool] = None,
        http_referer: Optional[str] = None,
        app_title: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the OpenRouter agent."""
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.extra_kwargs = kwargs

        self.base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Create an API key at https://openrouter.ai/keys and export it."
            )

        self.model = self._normalize_model(model)
        self.model_info = _OpenRouterModelCatalog.resolve(self.model)

        supported_parameters = set(self.model_info.get("supported_parameters") or [])
        if "tools" not in supported_parameters:
            raise ValueError(
                f"OpenRouter model '{self.model}' does not expose tool-calling capabilities."
            )

        # Reasoning is optional: only wire it up when model + config allow it
        self._reasoning_key = self._select_reasoning_key(supported_parameters)
        self.reasoning_config = self._resolve_reasoning_config(reasoning, include_reasoning)
        self.reasoning_payload = self._build_reasoning_payload()

        # Attribution headers (optional per Quickstart guide)
        headers = kwargs.pop("default_headers", None) or {}
        referer = http_referer or os.environ.get("OPENROUTER_HTTP_REFERER")
        title = app_title or os.environ.get("OPENROUTER_APP_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        self.default_headers = headers or None

    def _normalize_model(self, raw: str) -> str:
        if not raw:
            raise ValueError("Model name is required for OpenRouter agent")
        candidate = raw.strip()
        lowered = candidate.lower()
        if lowered.startswith("openrouter:"):
            return candidate.split(":", 1)[1].strip()
        if lowered.startswith("openrouter/"):
            return candidate.split("/", 1)[1].strip()
        return candidate

    def _select_reasoning_key(self, supported: set[str]) -> Optional[str]:
        for key in self._REASONING_KEYS:
            if key in supported:
                return key
        return None

    def _resolve_reasoning_config(
        self,
        explicit_config: Optional[dict[str, Any]],
        include_reasoning: Optional[bool],
    ) -> Optional[dict[str, Any]]:
        if explicit_config is not None:
            return explicit_config

        env_reasoning = os.environ.get("OPENROUTER_REASONING")
        if env_reasoning:
            try:
                return json.loads(env_reasoning)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "OPENROUTER_REASONING must be valid JSON (example: '{\"effort\":\"high\"}')"
                ) from exc

        env_include = os.environ.get("OPENROUTER_INCLUDE_REASONING")
        if env_include is not None:
            include_reasoning = self._to_bool(env_include)

        if include_reasoning is not None:
            return {} if include_reasoning else {"exclude": True}

        # Default to enabling reasoning when supported (per docs)
        if self._reasoning_key:
            return {}
        return None

    def _build_reasoning_payload(self) -> Optional[dict[str, Any]]:
        if self.reasoning_config is None or not self._reasoning_key:
            return None

        if self._reasoning_key == "reasoning":
            return {"reasoning": self.reasoning_config}

        include_flag = not bool(self.reasoning_config.get("exclude"))
        return {"include_reasoning": include_flag}

    @staticmethod
    def _to_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _build_llm(self) -> BaseChatModel:
        """Instantiate ChatOpenAI with OpenRouter settings."""
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": 3,
            "api_key": self.api_key,
            "base_url": self.base_url,
        }

        if self.max_output_tokens is not None:
            config["max_tokens"] = self.max_output_tokens

        if self.default_headers:
            config["default_headers"] = self.default_headers

        config.update(self.extra_kwargs)

        extra_body = dict(config.get("extra_body") or {})
        if self.reasoning_payload:
            extra_body.update(self.reasoning_payload)
        if extra_body:
            config["extra_body"] = extra_body

        llm = ChatOpenAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        async def _invoke():
            return await self._llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
            timeout_seconds=600.0,
            on_retry=lambda attempt, exc, delay: None,
        )

        # Normalize OpenRouter-style reasoning payloads for the SDK parser
        if hasattr(ai_message, "additional_kwargs"):
            extras = ai_message.additional_kwargs
            if isinstance(extras, dict):
                reasoning_details = extras.get("reasoning_details")
                if reasoning_details and "reasoning" not in extras:
                    extras["reasoning"] = reasoning_details

        parser = self.get_response_parser()
        parsed = parser.parse(ai_message)

        agent_response = AgentResponse(
            content=parsed.content,
            tool_calls=parsed.tool_calls,
            reasoning="\n".join(parsed.reasoning) if parsed.reasoning else None,
            done=not bool(parsed.tool_calls),
            info={"raw_reasoning": parsed.raw_reasoning},
        )

        return agent_response, ai_message

    def get_response_parser(self) -> ResponseParser:
        return OpenAIResponseParser()


__all__ = ["OpenRouterAgent"]
