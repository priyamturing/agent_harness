"""Anthropic Claude agent implementation."""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from ..constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_TOOL_CALL_LIMIT,
    RETRY_DEFAULT_MAX_ATTEMPTS,
    THINKING_DEFAULT_BUDGET_TOKENS,
    THINKING_DEFAULT_OUTPUT_TOKENS,
    THINKING_SAFETY_MARGIN_TOKENS,
)
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
        temperature: float = 1.0,
        max_output_tokens: Optional[int] = None,
        enable_thinking: bool = True,
        thinking_budget_tokens: int = THINKING_DEFAULT_BUDGET_TOKENS,
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = DEFAULT_TOOL_CALL_LIMIT,
        **kwargs,
    ):
        """Initialize Claude agent with extended thinking support.

        Args:
            model (str): Model name identifier. Accepts variations like 
                "claude-sonnet-4-5", "claude-4.5-sonnet", etc. Defaults to 
                "claude-sonnet-4-5".
            temperature (float): Sampling temperature (0.0-1.0). Note: automatically 
                overridden to 1.0 when thinking mode is enabled, as required by 
                Claude's extended thinking feature.
            max_output_tokens (Optional[int]): Maximum output tokens for generation. 
                If None, defaults are used. Must be positive if provided. For 
                thinking mode, automatically adjusted to accommodate thinking budget.
            enable_thinking (bool): Enable Claude's extended thinking feature. When 
                True, model will use explicit reasoning process before responding.
            thinking_budget_tokens (int): Token budget allocated for thinking process. 
                Only used when enable_thinking=True. Must be positive.
            system_prompt (Optional[str]): Optional system prompt for the agent. 
                If None, no system message is included.
            tool_call_limit (Optional[int]): Maximum number of tool calls allowed 
                before stopping. If None, no limit is enforced.
            **kwargs: Additional keyword arguments passed directly to ChatAnthropic 
                constructor (e.g., api_key, timeout, etc.).
            
        Raises:
            EnvironmentError: If ANTHROPIC_API_KEY environment variable is not set.
            ValueError: If max_output_tokens or thinking_budget_tokens is non-positive.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Export the API key before running."
            )
        
        # Validate token parameters
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {max_output_tokens}"
            )
        
        if thinking_budget_tokens <= 0:
            raise ValueError(
                f"thinking_budget_tokens must be positive, got {thinking_budget_tokens}"
            )
        
        # Warn about temperature override for thinking mode
        if enable_thinking and temperature != 1.0:
            warnings.warn(
                f"Claude thinking mode requires temperature=1.0. "
                f"Your temperature={temperature} will be overridden. "
                f"Set temperature=1.0 or enable_thinking=False to suppress this warning.",
                UserWarning,
                stacklevel=2
            )
        
        # Warn if max_output_tokens is too small for thinking mode
        if (
            enable_thinking 
            and max_output_tokens is not None 
            and max_output_tokens < thinking_budget_tokens + THINKING_SAFETY_MARGIN_TOKENS
        ):
            adjusted_tokens = thinking_budget_tokens + THINKING_SAFETY_MARGIN_TOKENS
            warnings.warn(
                f"max_output_tokens={max_output_tokens} is less than required for thinking mode "
                f"(thinking_budget_tokens={thinking_budget_tokens} + safety_margin={THINKING_SAFETY_MARGIN_TOKENS}). "
                f"It will be automatically increased to {adjusted_tokens}.",
                UserWarning,
                stacklevel=2
            )
        
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget_tokens = thinking_budget_tokens
        self.extra_kwargs = kwargs

    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build Claude LLM instance with extended thinking configuration.
        
        Returns:
            Union[BaseChatModel, Runnable]: Configured ChatAnthropic instance, 
                optionally with tools bound. Automatically handles model name 
                normalization, thinking mode setup, and token budget adjustments.
        """
        model_name = self.model
        normalized_model = model_name.lower()

        if normalized_model in {
            "claude-4.5-sonnet-reasoning",
            "claude-4.5-sonnet",
            "claude-sonnet-4.5",
            "claude-sonnet-4-5",
        }:
            model_name = "claude-sonnet-4-5"

        config: dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }

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
                config["max_tokens"] = max(
                    self.max_output_tokens,
                    self.thinking_budget_tokens + THINKING_SAFETY_MARGIN_TOKENS
                )
            else:
                config["max_tokens"] = self.thinking_budget_tokens + THINKING_DEFAULT_OUTPUT_TOKENS
        elif self.max_output_tokens is not None:
            config["max_tokens"] = self.max_output_tokens

        config.update(self.extra_kwargs)
        self._apply_llm_tracing_callbacks(config)

        llm = ChatAnthropic(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Claude model response with automatic retry and backoff logic.
        
        Args:
            messages (list[BaseMessage]): Complete conversation history to send 
                to Claude, including system, user, assistant, and tool messages.
                
        Returns:
            tuple[AgentResponse, AIMessage]: A tuple containing:
                - AgentResponse with content, tool_calls, reasoning (extracted 
                  from thinking blocks), and completion status
                - Raw AIMessage for conversation history maintenance
                
        Raises:
            RuntimeError: If LLM is not initialized (initialize() not called).
            Exception: Any unrecoverable errors from Claude API after retries.
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        llm = self._llm  # Capture for type narrowing

        async def _invoke():
            return await llm.ainvoke(messages)

        # Retry with backoff
        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=RETRY_DEFAULT_MAX_ATTEMPTS,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            on_retry=lambda attempt, exc, delay: None,
        )

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
        """Get Anthropic-specific response parser.
        
        Returns:
            ResponseParser: AnthropicResponseParser instance that can extract 
                thinking blocks, tool calls, and completion signals from Claude's 
                response format.
        """
        return AnthropicResponseParser()
