"""OpenAI GPT agent implementation."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from ..constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_TOOL_CALL_LIMIT,
    RETRY_DEFAULT_MAX_ATTEMPTS,
)
from ..parsers import OpenAIResponseParser, ResponseParser
from ..tasks import AgentResponse
from ..utils import retry_with_backoff
from .base import Agent


_REASONING_MODELS = frozenset({
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5.1",
    "gpt-5.1-codex",
    "o4-mini",
})


class GPTAgent(Agent):
    """Agent implementation for OpenAI GPT models.

    Features:
    - Reasoning model support (GPT-5, o-series)
    - Encrypted reasoning content
    - Configurable reasoning effort
    """

    def __init__(
        self,
        model: str = "gpt-5",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: str = "high",
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        """Initialize GPT agent with reasoning model support.

        Args:
            model (str): Exact model name as supported by OpenAI API 
                (e.g., "gpt-5", "gpt-4o", "o4-mini"). Defaults to "gpt-5".
            temperature (float): Sampling temperature (0.0-2.0) controlling 
                randomness in generation. Lower values are more deterministic.
            max_output_tokens (Optional[int]): Maximum completion tokens for 
                generation. If None, model defaults are used. Must be positive 
                if provided.
            reasoning_effort (str): Reasoning effort level for reasoning-capable 
                models. Must be one of "low", "medium", or "high". Only applies 
                to reasoning models.
            system_prompt (Optional[str]): Optional system prompt for the agent. 
                If None, no system message is included.
            tool_call_limit (Optional[int]): Maximum number of tool calls allowed 
                before stopping. If None, uses DEFAULT_TOOL_CALL_LIMIT.
            timeout (Optional[float]): Timeout in seconds for LLM API calls.
                If None, uses DEFAULT_LLM_TIMEOUT_SECONDS.
            max_retries (Optional[int]): Maximum number of retries for failed LLM calls.
                If None, uses DEFAULT_LLM_MAX_RETRIES.
            **kwargs: Additional keyword arguments passed directly to ChatOpenAI 
                constructor (e.g., api_key, base_url, etc.).
            
        Raises:
            EnvironmentError: If OPENAI_API_KEY environment variable is not set.
            ValueError: If max_output_tokens is non-positive.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export the API key before running."
            )
        
        # Validate token parameters
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {max_output_tokens}"
            )
        
        super().__init__(
            system_prompt=system_prompt,
            tool_call_limit=tool_call_limit,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_kwargs = kwargs

    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build OpenAI LLM instance with reasoning and encryption support.
        
        Returns:
            Union[BaseChatModel, Runnable]: Configured ChatOpenAI instance, 
                optionally with tools bound. Automatically handles:
                - Reasoning effort configuration for reasoning models
                - Encrypted reasoning content inclusion
                - Responses API v1 format
        """
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": self.max_retries,
        }

        if self.max_output_tokens is not None:
            config["max_completion_tokens"] = self.max_output_tokens

        reasoning_capable = self.model.lower() in _REASONING_MODELS

        if reasoning_capable:
            config["reasoning_effort"] = self.reasoning_effort
            include = set(config.get("include") or [])
            include.add("reasoning.encrypted_content")
            config["include"] = sorted(include)

        config["output_version"] = "responses/v1"
        config["use_responses_api"] = True

        config.update(self.extra_kwargs)
        config = self._get_llm_config_with_callbacks(config)

        llm = ChatOpenAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get GPT model response with automatic retry and backoff logic.
        
        Args:
            messages (list[BaseMessage]): Complete conversation history to send 
                to GPT, including system, user, assistant, and tool messages.
                
        Returns:
            tuple[AgentResponse, AIMessage]: A tuple containing:
                - AgentResponse with content, tool_calls, reasoning (extracted 
                  from encrypted reasoning content for reasoning models), and 
                  completion status
                - Raw AIMessage for conversation history maintenance
                
        Raises:
            RuntimeError: If LLM is not initialized (initialize() not called).
            Exception: Any unrecoverable errors from OpenAI API after retries.
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        llm = self._llm  # Capture for type narrowing

        async def _invoke():
            return await llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=RETRY_DEFAULT_MAX_ATTEMPTS,
            timeout_seconds=self.timeout,
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
        """Get OpenAI-specific response parser.
        
        Returns:
            ResponseParser: OpenAIResponseParser instance that can extract 
                encrypted reasoning content, tool calls, and completion signals 
                from OpenAI's response format (including GPT-5 reasoning models).
        """
        return OpenAIResponseParser()
