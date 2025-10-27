"""LLM provider selection utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, Optional, Tuple

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
    aliases: Tuple[str, ...] = ()

    def matches_model(self, model_name: str) -> bool:
        """Return True if the model name belongs to this provider."""

        normalized = model_name.lower()
        if normalized == self.default_model.lower():
            return True
        return normalized in {alias.lower() for alias in self.aliases}


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


_OPENAI_REASONING_MODELS: Final[Tuple[str, ...]] = (
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "o4-mini",
)


PROVIDERS: Final[Dict[str, ProviderConfig]] = {
    "openai": ProviderConfig(
        name="openai",
        default_model="gpt-5-high",
        api_key_env="OPENAI_API_KEY",
        factory=_build_openai,
        aliases=("gpt-5", "o4-mini", "gpt-5-high", "gpt-4o", "gpt4o"),
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        default_model="claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
        factory=_build_anthropic,
        aliases=(
            "claude-sonnet-4-5",
            "claude-4.5-sonnet",
            "claude-4.5-sonnet-reasoning",
            "claude-sonnet-4.5",
            "claude-sonnet-4.5-reasoning",
        ),
    ),
    "xai": ProviderConfig(
        name="xai",
        default_model="grok-4",
        api_key_env="XAI_API_KEY",
        factory=_build_xai,
        aliases=("grok-4", "grok4", "grok-3-mini", "grok-2"),
    ),
}

_MODEL_PROVIDER_MAP: Dict[str, str] = {}
for provider_name, config in PROVIDERS.items():
    _MODEL_PROVIDER_MAP[config.default_model.lower()] = provider_name
    for alias in config.aliases:
        _MODEL_PROVIDER_MAP[alias.lower()] = provider_name


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
        reasoning_capable = normalized_model in _OPENAI_REASONING_MODELS

        if normalized_model == "gpt-5-high":
            model_name = "gpt-5"
            extra_kwargs["reasoning_effort"] = "high"
            reasoning_capable = True
        elif normalized_model not in _OPENAI_REASONING_MODELS:
            # If the alias maps to a reasoning-capable model (e.g. "gpt-5"),
            # check the resolved target once more.
            reasoning_capable = model_name.lower() in _OPENAI_REASONING_MODELS

        if reasoning_capable:
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
        if normalized_model in {
            "claude-4.5-sonnet-reasoning",
            "claude-4.5-sonnet",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
        }:
            model_name = "claude-sonnet-4-5"
            extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 42000}
            if temperature != 1.0:
                temperature = 1.0

    return config.factory(model_name, temperature, max_output_tokens, extra_kwargs)


def resolve_provider_for_model(
    model: Optional[str],
    *,
    provider_hint: Optional[str] = None,
) -> str:
    """Infer the provider key for a given model string."""

    if provider_hint:
        normalized_hint = provider_hint.lower()
        if normalized_hint not in PROVIDERS:
            raise ValueError(
                f"Unknown provider hint '{provider_hint}'. "
                f"Known providers: {', '.join(sorted(PROVIDERS))}."
            )
        return normalized_hint

    if model is None:
        return "openai"

    normalized = model.lower()
    if normalized in _MODEL_PROVIDER_MAP:
        return _MODEL_PROVIDER_MAP[normalized]

    if normalized.startswith("gpt") or normalized.startswith("o"):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    if normalized.startswith("grok"):
        return "xai"

    raise ValueError(
        f"Unable to infer provider for model '{model}'. "
        "Specify it explicitly with the form '<provider>:<model>'."
    )


__all__ = ["PROVIDERS", "ProviderConfig", "create_chat_model", "resolve_provider_for_model"]
