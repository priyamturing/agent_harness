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
        model: str = "gpt-5-high",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: str = "high",
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = DEFAULT_TOOL_CALL_LIMIT,
        **kwargs,
    ):
        """Initialize GPT agent.

        Args:
            model: Model name (gpt-5-high, gpt-4o, etc.)
            temperature: Sampling temperature
            max_output_tokens: Maximum completion tokens
            reasoning_effort: Reasoning effort (low, medium, high)
            system_prompt: Optional system prompt
            tool_call_limit: Maximum tool calls (None = no limit)
            **kwargs: Additional arguments for ChatOpenAI
            
        Raises:
            EnvironmentError: If OPENAI_API_KEY is not set
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Export the API key before running."
            )
        
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_kwargs = kwargs

    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build OpenAI model with configuration."""
        model_name = self.model
        normalized_model = model_name.lower()

        config: dict[str, Any] = {
            "model": model_name,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": DEFAULT_LLM_MAX_RETRIES,
        }

        if self.max_output_tokens is not None:
            config["max_completion_tokens"] = self.max_output_tokens

        reasoning_capable = normalized_model in _REASONING_MODELS

        if normalized_model == "gpt-5-high":
            model_name = "gpt-5"
            config["model"] = model_name
            config["reasoning_effort"] = self.reasoning_effort
            reasoning_capable = True

        if reasoning_capable:
            include = set(config.get("include") or [])
            include.add("reasoning.encrypted_content")
            config["include"] = sorted(include)

        config["output_version"] = "responses/v1"
        config["use_responses_api"] = True

        config.update(self.extra_kwargs)

        llm = ChatOpenAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get GPT model response with retry logic."""
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        llm = self._llm  # Capture for type narrowing

        async def _invoke():
            return await llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
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
        """Get OpenAI response parser."""
        return OpenAIResponseParser()

