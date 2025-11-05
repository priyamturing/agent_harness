"""Google Gemini agent implementation."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage

from ..parsers import GoogleResponseParser, ResponseParser
from ..tasks import AgentResponse
from ..utils import retry_with_backoff
from .base import Agent, _DEFAULT_LLM_TIMEOUT_SECONDS


class GeminiAgent(Agent):
    """Agent implementation for Google Gemini models.

    Features:
    - Thinking budget support
    - Thought inclusion
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = True,
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = 1000,
        **kwargs,
    ):
        """Initialize Gemini agent.

        Args:
            model: Model name (gemini-2.5-pro, etc.)
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            thinking_budget: Optional thinking budget (tokens)
            include_thoughts: Include thoughts in response
            system_prompt: Optional system prompt
            tool_call_limit: Maximum tool calls (None = no limit)
            **kwargs: Additional arguments for ChatGoogleGenerativeAI
            
        Raises:
            EnvironmentError: If GOOGLE_API_KEY is not set
        """
        if not os.environ.get("GOOGLE_API_KEY"):
            raise EnvironmentError(
                "GOOGLE_API_KEY is not set. Export the API key before running."
            )
        
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.include_thoughts = include_thoughts
        self.extra_kwargs = kwargs

    def _build_llm(self) -> BaseChatModel:
        """Build Gemini model with configuration."""
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_output_tokens": self.max_output_tokens,
        }

        # Check environment for thinking budget
        env_budget = os.environ.get("GOOGLE_THINKING_BUDGET")
        if env_budget and "thinking_budget" not in config:
            try:
                config["thinking_budget"] = int(env_budget)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GOOGLE_THINKING_BUDGET value '{env_budget}'. Must be an integer."
                ) from exc

        # Override with explicit thinking_budget if provided
        if self.thinking_budget is not None:
            config["thinking_budget"] = self.thinking_budget

        config["include_thoughts"] = self.include_thoughts

        config.update(self.extra_kwargs)

        llm = ChatGoogleGenerativeAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Gemini model response with retry logic."""
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        async def _invoke():
            return await self._llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
            timeout_seconds=_DEFAULT_LLM_TIMEOUT_SECONDS,
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
        """Get Google response parser."""
        return GoogleResponseParser()

