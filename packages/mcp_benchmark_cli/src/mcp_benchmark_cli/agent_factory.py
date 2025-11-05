"""Agent factory for CLI - maps model strings to SDK agent classes."""

from typing import Any, Optional

from mcp_benchmark_sdk import Agent, ClaudeAgent, GPTAgent, GeminiAgent, GrokAgent
from .qwen_agent import QwenAgent
from .prompts import PROJECT_MANAGEMENT_SYSTEM_PROMPT


def create_agent_from_string(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    tool_call_limit: int = 1000,
    **kwargs: Any,
) -> Agent:
    """Create agent from model string (CLI convenience).

    Supports:
    - Auto-detection from model name
    - Explicit provider prefix (provider:model)

    Args:
        model: Model name or provider:model
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        tool_call_limit: Maximum tool calls per run
        **kwargs: Additional arguments passed to agent

    Returns:
        Agent instance

    Raises:
        ValueError: If model/provider cannot be determined

    Examples:
        >>> agent = create_agent_from_string("gpt-5-high")
        >>> agent = create_agent_from_string("anthropic:claude-sonnet-4-5")
        >>> agent = create_agent_from_string("gemini-2.5-pro", temperature=0.5, tool_call_limit=500)
    """
    # Parse provider prefix if present
    provider_hint: Optional[str] = None
    model_name = model

    if ":" in model:
        provider_hint, model_name = model.split(":", 1)
        provider_hint = provider_hint.lower()

    # Normalize model name
    model_lower = model_name.lower()

    # Explicit provider
    if provider_hint:
        if provider_hint == "openai":
            return GPTAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
                **kwargs,
            )
        elif provider_hint == "anthropic":
            return ClaudeAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
                **kwargs,
            )
        elif provider_hint == "google":
            return GeminiAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
                **kwargs,
            )
        elif provider_hint == "xai":
            return GrokAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown provider: {provider_hint}")

    # Auto-detect from model name
    if model_lower.startswith("claude"):
        return ClaudeAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
            **kwargs,
        )
    elif model_lower.startswith("gpt") or model_lower.startswith("o"):
        return GPTAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
            **kwargs,
        )
    elif model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return GeminiAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
            **kwargs,
        )
    elif model_lower.startswith("grok"):
        return GrokAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
            **kwargs,
        )
    elif model_lower.startswith("qwen"):
        return QwenAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=PROJECT_MANAGEMENT_SYSTEM_PROMPT,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unable to determine provider for model '{model}'. "
            "Use explicit format 'provider:model' (e.g., 'openai:gpt-4o')"
        )


__all__ = ["create_agent_from_string"]

