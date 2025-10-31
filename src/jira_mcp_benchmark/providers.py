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

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as exc:  # pragma: no cover - handled by dependency management
    raise ImportError(
        "langchain-google-genai is required. Install with `pip install langchain-google-genai`."
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


def _build_google(
    model: str, temperature: float, max_output_tokens: Optional[int], extra: Dict[str, Any]
) -> BaseChatModel:
    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": None,
        "max_retries": 3,
        "include_thoughts": True,
        "thinking_budget": 24576,  # Max allowed by Gemini API
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    kwargs.update(extra)
    return ChatGoogleGenerativeAI(**kwargs)


def _build_openrouter(
    model: str, temperature: float, max_output_tokens: Optional[int], extra: Dict[str, Any]
) -> BaseChatModel:
    """Build a ChatOpenAI client configured for OpenRouter."""

    kwargs: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": None,
        "max_retries": 3,
    }
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    kwargs.update(extra)

    base_url = kwargs.get("base_url") or os.environ.get("OPENROUTER_BASE_URL")
    kwargs["base_url"] = base_url or "https://openrouter.ai/api/v1"

    api_key = kwargs.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Export the API key before running."
        )
    kwargs["api_key"] = api_key

    # Merge optional headers recommended by OpenRouter (used for rankings/analytics).
    default_headers = dict(kwargs.get("default_headers") or {})
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    if referer and "HTTP-Referer" not in default_headers:
        default_headers["HTTP-Referer"] = referer
    title = os.environ.get("OPENROUTER_APP_TITLE")
    if title and "X-Title" not in default_headers:
        default_headers["X-Title"] = title
    if default_headers:
        kwargs["default_headers"] = default_headers

    # Ensure we only route through providers that support our required capabilities.
    extra_body = dict(kwargs.get("extra_body") or {})
    provider_preferences = dict(extra_body.get("provider") or {})

    allowed_providers_env = os.environ.get("OPENROUTER_PROVIDER_ONLY")
    if allowed_providers_env:
        allowed_providers = [
            slug.strip() for slug in allowed_providers_env.split(",") if slug.strip()
        ]
    else:
        allowed_providers = ["fireworks", "groq"]
    if allowed_providers and "only" not in provider_preferences:
        provider_preferences["only"] = allowed_providers

    allow_fallbacks_env = os.environ.get("OPENROUTER_ALLOW_FALLBACKS")
    if (
        allow_fallbacks_env is not None
        and "allow_fallbacks" not in provider_preferences
    ):
        allow_fallbacks = allow_fallbacks_env.strip().lower() in {"1", "true", "yes"}
        provider_preferences["allow_fallbacks"] = allow_fallbacks
    if "allow_fallbacks" not in provider_preferences:
        provider_preferences["allow_fallbacks"] = True

    if "require_parameters" not in provider_preferences:
        provider_preferences["require_parameters"] = True

    if provider_preferences:
        extra_body["provider"] = provider_preferences
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


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
    "google": ProviderConfig(
        name="google",
        default_model="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        factory=_build_google,
        aliases=(
            "gemini-2.5-pro",
            "models/gemini-2.5-pro",
            "gemini-2.5-pro-exp",
            "models/gemini-2.5-pro-exp",
            "gemini-2.5-flash",
            "models/gemini-2.5-flash",
            "gemini-2.5-flash-exp",
            "models/gemini-2.5-flash-exp",
        ),
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        default_model="meta-llama/llama-4-maverick",
        api_key_env="OPENROUTER_API_KEY",
        factory=_build_openrouter,
        aliases=(
            "llama-4-maverick",
            "meta-llama/llama-4-maverick:nitro",
            "meta-llama/llama-4-maverick:floor",
        ),
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
    elif config.name == "google":
        include_thoughts = extra_kwargs.get("include_thoughts", True)
        extra_kwargs["include_thoughts"] = include_thoughts
        extra_kwargs.setdefault("thinking_budget", 24576)  # Max allowed by Gemini API
    elif config.name == "openrouter":
        use_responses_env = os.environ.get("OPENROUTER_USE_RESPONSES_API")
        if use_responses_env is not None:
            use_responses = use_responses_env.strip().lower() in {"1", "true", "yes"}
        else:
            use_responses = False
        if use_responses:
            extra_kwargs.setdefault("use_responses_api", True)
            extra_kwargs.setdefault("output_version", "responses/v1")

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
    if normalized.startswith("gemini") or normalized.startswith("models/gemini"):
        return "google"
    if normalized.startswith("meta-llama/") or normalized.startswith("llama-"):
        return "openrouter"

    raise ValueError(
        f"Unable to infer provider for model '{model}'. "
        "Specify it explicitly with the form '<provider>:<model>'."
    )


__all__ = ["PROVIDERS", "ProviderConfig", "create_chat_model", "resolve_provider_for_model"]
