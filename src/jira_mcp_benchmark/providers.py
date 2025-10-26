"""LLM provider selection utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, Optional

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from langchain_openai import ChatOpenAI
except ImportError as exc:  # pragma: no cover - handled by dependency management
    raise ImportError(
        "langchain-openai is required. Install with `pip install langchain-openai`."
    ) from exc

try:
    from langchain_anthropic import ChatAnthropic
except ImportError as exc:  # pragma: no cover - handled by dependency management
    raise ImportError(
        "langchain-anthropic is required. Install with `pip install langchain-anthropic`."
    ) from exc

try:
    from langchain_xai import ChatXAI
except ImportError as exc:  # pragma: no cover - handled by dependency management
    raise ImportError(
        "langchain-xai is required. Install with `pip install langchain-xai`."
    ) from exc


ModelFactory = Callable[[str, float, Optional[int], Dict[str, Any]], BaseChatModel]


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration metadata for a chat model provider."""

    name: str
    default_model: str
    api_key_env: str
    factory: ModelFactory


def _build_openai(
    model: str, temperature: float, max_output_tokens: Optional[int], extra: Dict[str, Any]
) -> BaseChatModel:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": None,
        "max_retries": 3,
    }
    if max_output_tokens is not None:
        kwargs["max_completion_tokens"] = max_output_tokens
    kwargs.update(extra)
    return ChatOpenAI(**kwargs)


def _build_anthropic(
    model: str, temperature: float, max_output_tokens: Optional[int], extra: Dict[str, Any]
) -> BaseChatModel:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": None,
        "max_retries": 3,
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(extra)
    return ChatAnthropic(**kwargs)


def _build_xai(
    model: str, temperature: float, max_output_tokens: Optional[int], extra: Dict[str, Any]
) -> BaseChatModel:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": None,
        "max_retries": 3,
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(extra)
    return ChatXAI(**kwargs)


PROVIDERS: Final[Dict[str, ProviderConfig]] = {
    "openai": ProviderConfig(
        name="openai",
        default_model="gpt-5-high",
        api_key_env="OPENAI_API_KEY",
        factory=_build_openai,
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        default_model="claude-4.5-sonnet-reasoning",
        api_key_env="ANTHROPIC_API_KEY",
        factory=_build_anthropic,
    ),
    "xai": ProviderConfig(
        name="xai",
        default_model="grok-4",
        api_key_env="XAI_API_KEY",
        factory=_build_xai,
    ),
}


def create_chat_model(
    provider_name: str,
    *,
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Instantiate a chat model for the requested provider."""

    normalized = provider_name.lower()
    if normalized not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")

    config = PROVIDERS[normalized]
    env_val = os.environ.get(config.api_key_env)
    if not env_val:
        raise EnvironmentError(
            f"{config.api_key_env} is not set. Export the API key before running."
        )

    model_name = model or config.default_model
    extra_kwargs: Dict[str, Any] = {}

    if config.name == "openai":
        normalized_model = model_name.lower()
        if normalized_model == "gpt-5-high":
            model_name = "gpt-5"
            extra_kwargs["reasoning_effort"] = "high"
        include = set(extra_kwargs.get("include") or [])
        include.add("reasoning.encrypted_content")
        extra_kwargs["include"] = sorted(include)
        extra_kwargs["output_version"] = "responses/v1"
        extra_kwargs["use_responses_api"] = True
    elif config.name == "xai":
        normalized_model = model_name.lower()
        if normalized_model == "grok-4":
            extra_kwargs["output_version"] = "responses/v1"
            include = set(extra_kwargs.get("include") or [])
            include.add("reasoning.encrypted_content")
            extra_kwargs["include"] = sorted(include)
            extra_kwargs["use_responses_api"] = True
            body = dict(extra_kwargs.get("extra_body") or {})
            body.setdefault("use_encrypted_content", True)
            extra_kwargs["extra_body"] = body
    elif config.name == "anthropic":
        normalized_model = model_name.lower()
        if normalized_model == "claude-4.5-sonnet-reasoning":
            extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 2000}

    return config.factory(model_name, temperature, max_output_tokens, extra_kwargs)


__all__ = ["PROVIDERS", "ProviderConfig", "create_chat_model"]
