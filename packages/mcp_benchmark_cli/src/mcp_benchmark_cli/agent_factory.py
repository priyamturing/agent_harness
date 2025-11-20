"""Agent factory for CLI - extends SDK agent factory with CLI-specific agents."""

from typing import Any, Optional

from mcp_benchmark_sdk.agents.core import Agent
from mcp_benchmark_sdk.agents.telemetry import with_tracing
from mcp_benchmark_sdk.harness.agent_factory import create_agent as sdk_create_agent

from .openrouter_agent import OpenRouterAgent
from .qwen_agent import QwenAgent
from .prompts import PROJECT_MANAGEMENT_SYSTEM_PROMPT


def create_agent_from_string(
    model: str,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    tool_call_limit: int = 1000,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> Agent:
    """Create agent from model string (CLI convenience with Qwen support).

    Extends SDK's create_agent with CLI-specific agents (Qwen).
    
    Args:
        model: Model name or provider:model
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        tool_call_limit: Maximum tool calls per run
        system_prompt: Optional system prompt (defaults to PROJECT_MANAGEMENT_SYSTEM_PROMPT)
        **kwargs: Additional arguments passed to agent

    Returns:
        Agent instance

    Raises:
        ValueError: If model/provider cannot be determined
    """
    if system_prompt is None:
        system_prompt = PROJECT_MANAGEMENT_SYSTEM_PROMPT

    model_lower = model.lower()
    
    if model_lower.startswith("qwen"):
        agent = QwenAgent(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            **kwargs,
        )
        return with_tracing(agent)

    if model_lower.startswith("openrouter"):
        agent = OpenRouterAgent(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            tool_call_limit=tool_call_limit,
            system_prompt=system_prompt,
            **kwargs,
        )
        return with_tracing(agent)
    
    agent = sdk_create_agent(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_call_limit=tool_call_limit,
        system_prompt=system_prompt,
        **kwargs,
    )
    
    return with_tracing(agent)


__all__ = ["create_agent_from_string"]
