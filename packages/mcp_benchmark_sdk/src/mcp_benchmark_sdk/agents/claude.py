"""Anthropic Claude agent implementation."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from ..parsers import AnthropicResponseParser, ResponseParser
from ..tasks import AgentResponse
from ..utils import retry_with_backoff
from .base import Agent


class ClaudeAgent(Agent):
    """Agent implementation for Anthropic Claude models.

    Features:
    - Extended thinking support
    - Temperature override for reasoning
    - Automatic budget token configuration
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        thinking_budget_tokens: int = 42000,
        system_prompt: Optional[str] = None,
        tool_call_limit: int = 1000,
        **kwargs,
    ):
        """Initialize Claude agent.

        Args:
            model: Model name (claude-sonnet-4-5, etc.)
            temperature: Sampling temperature (overridden to 1.0 for thinking models)
            max_output_tokens: Maximum output tokens
            enable_thinking: Enable extended thinking
            thinking_budget_tokens: Token budget for thinking
            system_prompt: Optional system prompt
            tool_call_limit: Maximum tool calls
            **kwargs: Additional arguments for ChatAnthropic
        """
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.extra_kwargs = kwargs

    def _build_llm(self) -> BaseChatModel:
        """Build Claude model with configuration."""
        # Check API key
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Export the API key before running."
            )

        # Normalize model name
        model_name = self.model
        normalized_model = model_name.lower()

        if normalized_model in {
            "claude-4.5-sonnet-reasoning",
            "claude-4.5-sonnet",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
        }:
            model_name = "claude-sonnet-4-5"

        # Build configuration
        config: dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": 3,
        }

        # Enable thinking for supported models
        if self.enable_thinking and normalized_model in {
            "claude-4.5-sonnet-reasoning",
            "claude-4.5-sonnet",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
        }:
            config["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            # Claude thinking requires temperature = 1.0
            if config["temperature"] != 1.0:
                config["temperature"] = 1.0
            
            # max_tokens must be > thinking_budget_tokens
            if self.max_output_tokens is not None:
                config["max_tokens"] = max(self.max_output_tokens, self.thinking_budget_tokens + 1000)
            else:
                config["max_tokens"] = self.thinking_budget_tokens + 8192  # Default to budget + 8k
        elif self.max_output_tokens is not None:
            config["max_tokens"] = self.max_output_tokens

        config.update(self.extra_kwargs)

        llm = ChatAnthropic(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Claude model response with retry logic."""
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        async def _invoke():
            return await self._llm.ainvoke(messages)

        # Retry with backoff
        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
            timeout_seconds=600.0,  # 10 minutes
            on_retry=lambda attempt, exc, delay: None,  # Could log via RunContext
        )

        # Parse response
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
        """Get Anthropic response parser."""
        return AnthropicResponseParser()

