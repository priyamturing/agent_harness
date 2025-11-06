"""Agent factory for creating agents from model specifications."""

from __future__ import annotations

from typing import Any, Optional

from ..agents import Agent, ClaudeAgent, GeminiAgent, GPTAgent, GrokAgent


DEFAULT_SYSTEM_PROMPT = """
You are an autonomous project-management agent operating inside an MCP server.
You receive: (a) a user request and (b) a set of tool definitions (schemas, params, return types).
Your goal is to complete the tasks assigned to you by using these tools effectively and efficiently.
Your only way to read or modify project state is via these tools.

Operating constraints
- Non-interactive: The user will not answer follow-ups. Do not ask questions. Do not halt for clarification.
- Obligation: Treat the user request as correct and feasible with the provided context. Execute to completion with best effort.
- Tools-first: Treat tool definitions as the single source of truth. Do not fabricate data, IDs, or results. Never assume hidden state.

Core behavior
- Objective-first: Extract the core objective succinctly and decompose into the minimal set of steps to achieve it.
- Read-before-write: When safe and efficient, fetch current state to avoid duplicates, race conditions, or destructive updates.
- Preconditions: Check the parameters before calling tools.

Tool usage policy
- Error Handling: Incase a tool call results in an error, retry the tool calling adjusting the parameters based on the error message, retrying with same parameters will only result in the same error.
""".strip()


def create_agent(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    tool_call_limit: int = 1000,
    system_prompt: Optional[str] = None,
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
        tool_call_limit: Maximum tool calls per run
        system_prompt: Optional system prompt (defaults to DEFAULT_SYSTEM_PROMPT)
        **kwargs: Additional arguments passed to agent

    Returns:
        Agent instance

    Raises:
        ValueError: If model/provider cannot be determined

    Examples:
        >>> agent = create_agent("gpt-4o")
        >>> agent = create_agent("anthropic:claude-sonnet-4-5")
        >>> agent = create_agent("gemini-2.5-pro", temperature=0.5, tool_call_limit=500)
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

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
                **kwargs,
            )
        elif provider_hint == "anthropic":
            return ClaudeAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif provider_hint == "google":
            return GeminiAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
                **kwargs,
            )
        elif provider_hint == "xai":
            return GrokAgent(
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
                system_prompt=system_prompt,
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
            **kwargs,
        )
    elif model_lower.startswith("gpt") or model_lower.startswith("o"):
        return GPTAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            **kwargs,
        )
    elif model_lower.startswith("gemini") or model_lower.startswith("models/gemini"):
        return GeminiAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            **kwargs,
        )
    elif model_lower.startswith("grok"):
        return GrokAgent(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unable to determine provider for model '{model}'. "
            "Use explicit format 'provider:model' (e.g., 'openai:gpt-4o')"
        )
