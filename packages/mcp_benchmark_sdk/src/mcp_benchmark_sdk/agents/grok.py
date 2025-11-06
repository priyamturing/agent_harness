"""xAI Grok agent implementation."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_xai import ChatXAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from ..constants import (
    DEFAULT_LLM_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_TOOL_CALL_LIMIT,
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
        """Initialize Grok agent.

        Args:
            model: Model name (grok-4, grok-3-mini, grok-2)
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            system_prompt: Optional system prompt
            tool_call_limit: Maximum tool calls (None = no limit)
            **kwargs: Additional arguments for ChatXAI
            
        Raises:
            EnvironmentError: If XAI_API_KEY is not set
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

    def _build_llm(self) -> BaseChatModel:
        """Build Grok model with configuration."""
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }

        if self.max_output_tokens is not None:
            config["max_tokens"] = self.max_output_tokens

        # Enable reasoning for Grok-4
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

        llm = ChatXAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Grok model response with retry logic."""
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        async def _invoke():
            return await self._llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            on_retry=lambda attempt, exc, delay: None,
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
        """Get xAI response parser."""
        return XAIResponseParser()

