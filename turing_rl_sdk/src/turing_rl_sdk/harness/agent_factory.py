"""Agent factory for creating agents from model specifications."""

from __future__ import annotations

from typing import Any, Optional, Union

from ..agents import Agent, ClaudeAgent, GeminiAgent, GPTAgent, GrokAgent, TracingAgent, with_tracing


def create_agent(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    tool_call_limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any,
) -> Agent:
    """Create agent from model string.

    Supports:
    - Auto-detection from model name
    - Explicit provider prefix (provider:model)

    Args:
        model: Model name or provider:model
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        tool_call_limit: Maximum tool calls per run (None uses DEFAULT_TOOL_CALL_LIMIT)
        system_prompt: Optional system prompt (defaults to None - no system message)
        timeout: Timeout in seconds for LLM API calls (None uses DEFAULT_LLM_TIMEOUT_SECONDS)
        max_retries: Maximum retries for failed LLM calls (None uses DEFAULT_LLM_MAX_RETRIES)
        **kwargs: Additional arguments passed to agent

    Returns:
        Agent instance

    Raises:
        ValueError: If model/provider cannot be determined

    Examples:
        >>> agent = create_agent("gpt-4o")
        >>> agent = create_agent("anthropic:claude-sonnet-4-5")
        >>> agent = create_agent("gemini-2.5-pro", temperature=0.5, tool_call_limit=500)
        >>> agent = create_agent("gpt-4o", system_prompt="You are a helpful assistant")
        >>> agent = create_agent("gpt-4o", timeout=300, max_retries=5)
    """
    # System prompt defaults to None (no system message)

    provider_hint: Optional[str] = None
    model_name = model

    if ":" in model:
        provider_hint, model_name = model.split(":", 1)
        provider_hint = provider_hint.lower()

    model_lower = model_name.lower()

    if provider_hint:
        if provider_hint == "openai":
            return GPTAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        elif provider_hint == "anthropic":
            return ClaudeAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        elif provider_hint == "google":
            return GeminiAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        elif provider_hint == "xai":
            return GrokAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown provider: {provider_hint}")

    if model_lower.startswith("claude"):
        return ClaudeAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    elif model_lower.startswith("gpt") or model_lower.startswith("o"):
        return GPTAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    elif model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return GeminiAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    elif model_lower.startswith("grok"):
        return GrokAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unable to determine provider for model '{model}'. "
            "Use explicit format 'provider:model' (e.g., 'openai:gpt-4o')"
        )


def create_traced_agent(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    tool_call_limit: Optional[int] = None,
    system_prompt: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    **kwargs: Any,
) -> Union[Agent, TracingAgent]:
    """Create agent with automatic LangSmith tracing if enabled.
    
    Convenience wrapper around create_agent() + with_tracing().
    
    Args:
        model: Model name or provider:model
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        tool_call_limit: Maximum tool calls per run (None uses DEFAULT_TOOL_CALL_LIMIT)
        system_prompt: Optional system prompt
        timeout: Timeout in seconds for LLM API calls
        max_retries: Maximum retries for failed LLM calls
        **kwargs: Additional arguments passed to agent
        
    Returns:
        Agent instance (wrapped with TracingAgent if LANGCHAIN_TRACING_V2=true)
        
    Example:
        >>> # If LANGCHAIN_TRACING_V2=true, automatically wraps with tracing
        >>> agent = create_traced_agent("gpt-4o")
        >>> result = await agent.run(task)  # Automatically traced!
    """
    agent = create_agent(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_call_limit=tool_call_limit,
        system_prompt=system_prompt,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )
    return with_tracing(agent)
