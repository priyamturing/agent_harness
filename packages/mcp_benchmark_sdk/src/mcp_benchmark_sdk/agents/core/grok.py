"""xAI Grok agent implementation."""

from __future__ import annotations

import os
from typing import Any, Optional, Union

from langchain_xai import ChatXAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from ..constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_TOOL_CALL_LIMIT,
    RETRY_DEFAULT_MAX_ATTEMPTS,
)
from ..parsers import XAIResponseParser, ResponseParser
from ..tasks import AgentResponse
from ..utils import retry_with_backoff
from .base import Agent


class GrokAgent(Agent):
    """Agent implementation for xAI Grok models.

    Features:
    - Reasoning support for Grok-4
    - Encrypted reasoning content
    """

    def __init__(
        self,
        model: str = "grok-4",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = DEFAULT_TOOL_CALL_LIMIT,
        **kwargs,
    ):
        """Initialize Grok agent with reasoning support for Grok-4.

        Args:
            model (str): Model name identifier like "grok-4", "grok-3-mini", 
                "grok-2", etc. Defaults to "grok-4". Grok-4 gets special 
                encrypted reasoning support.
            temperature (float): Sampling temperature (0.0-2.0) controlling 
                randomness in generation. Lower values are more deterministic.
            max_output_tokens (Optional[int]): Maximum output tokens for 
                generation. If None, model defaults are used.
            system_prompt (Optional[str]): Optional system prompt for the agent. 
                If None, no system message is included.
            tool_call_limit (Optional[int]): Maximum number of tool calls allowed 
                before stopping. If None, no limit is enforced.
            **kwargs: Additional keyword arguments passed directly to ChatXAI 
                constructor (e.g., api_key, timeout, base_url, etc.).
            
        Raises:
            EnvironmentError: If XAI_API_KEY environment variable is not set.
        """
        if not os.environ.get("XAI_API_KEY"):
            raise EnvironmentError(
                "XAI_API_KEY is not set. Export the API key before running."
            )
        
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.extra_kwargs = kwargs

    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build Grok LLM instance with encrypted reasoning support for Grok-4.
        
        Returns:
            Union[BaseChatModel, Runnable]: Configured ChatXAI instance, 
                optionally with tools bound. For Grok-4, automatically configures:
                - Responses API v1 format
                - Encrypted reasoning content inclusion
                - use_encrypted_content in extra_body
        """
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }

        if self.max_output_tokens is not None:
            config["max_tokens"] = self.max_output_tokens

        normalized_model = self.model.lower()
        if normalized_model in {"grok-4", "grok4"}:
            config["output_version"] = "responses/v1"
            include = set(config.get("include") or [])
            include.add("reasoning.encrypted_content")
            config["include"] = sorted(include)
            config["use_responses_api"] = True

            body = dict(config.get("extra_body") or {})
            body.setdefault("use_encrypted_content", True)
            config["extra_body"] = body

        config.update(self.extra_kwargs)
        config = self._get_llm_config_with_callbacks(config)

        llm = ChatXAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Grok model response with automatic retry and backoff logic.
        
        Args:
            messages (list[BaseMessage]): Complete conversation history to send 
                to Grok, including system, user, assistant, and tool messages.
                
        Returns:
            tuple[AgentResponse, AIMessage]: A tuple containing:
                - AgentResponse with content, tool_calls, reasoning (extracted 
                  from encrypted reasoning content for Grok-4), and completion status
                - Raw AIMessage for conversation history maintenance
                
        Raises:
            RuntimeError: If LLM is not initialized (initialize() not called).
            Exception: Any unrecoverable errors from xAI API after retries.
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        llm = self._llm  # Capture for type narrowing

        async def _invoke():
            return await llm.ainvoke(messages)

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
        """Get xAI-specific response parser.
        
        Returns:
            ResponseParser: XAIResponseParser instance that can extract encrypted 
                reasoning content, tool calls, and completion signals from Grok's 
                response format (especially Grok-4 reasoning).
        """
        return XAIResponseParser()
