"""Google Gemini agent implementation."""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from io import StringIO
from typing import Any, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable

from ..constants import DEFAULT_LLM_TIMEOUT_SECONDS, DEFAULT_TOOL_CALL_LIMIT, RETRY_DEFAULT_MAX_ATTEMPTS
from ..parsers import GoogleResponseParser, ResponseParser
from ..tasks import AgentResponse
from ..utils import retry_with_backoff
from .base import Agent

# Suppress noisy schema warnings emitted by the Google GenAI SDK when tools include
# additionalProperties (a common pattern for MCP schema definitions).
warnings.filterwarnings(
    "ignore",
    message=r"Key 'additionalProperties' is not supported in schema, ignoring",
)
logging.getLogger(
    "google.ai.generativelanguage_v1beta.services.generative_service"
).setLevel(logging.ERROR)


@contextmanager
def _suppress_schema_warnings():
    """Suppress Google GenAI schema warnings during tool binding.
    
    Context manager that suppresses both Python warnings and stdout/stderr 
    output from the Google GenAI library when binding tools. This is necessary 
    because the library emits warnings about additionalProperties in tool schemas.
    
    Yields:
        None: Context where warnings are suppressed.
    """
    # Suppress Python warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*additionalProperties.*")
        
        # Suppress stdout/stderr (for print statements in the library)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


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
        thinking_level: Optional[str] = None,
        include_thoughts: bool = True,
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = DEFAULT_TOOL_CALL_LIMIT,
        **kwargs,
    ):
        """Initialize Gemini agent with thinking budget support.

        Args:
            model (str): Model name identifier like "gemini-2.5-pro", 
                "gemini-2.0-flash", etc. Defaults to "gemini-2.5-pro".
            temperature (float): Sampling temperature (0.0-2.0) controlling 
                randomness in generation. Lower values are more deterministic.
            max_output_tokens (Optional[int]): Maximum output tokens for 
                generation. If None, model defaults are used.
            thinking_budget (Optional[int]): Optional token budget allocated 
                for the thinking process. Can also be set via GOOGLE_THINKING_BUDGET 
                environment variable. If None, no explicit budget is set.
            thinking_level (Optional[str]): Optional thinking level ("low", "high").
                Controls the depth of reasoning. If None, model defaults are used.
            include_thoughts (bool): Whether to include thought process in the 
                response. When True, intermediate reasoning is visible.
            system_prompt (Optional[str]): Optional system prompt for the agent. 
                If None, no system message is included.
            tool_call_limit (Optional[int]): Maximum number of tool calls allowed 
                before stopping. If None, no limit is enforced.
            **kwargs: Additional keyword arguments passed directly to 
                ChatGoogleGenerativeAI constructor (e.g., api_key, timeout, etc.).
            
        Raises:
            EnvironmentError: If GOOGLE_API_KEY environment variable is not set.
            ValueError: If GOOGLE_THINKING_BUDGET env var is set but not a valid integer.
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
        self.thinking_level = thinking_level
        self.include_thoughts = include_thoughts
        self.extra_kwargs = kwargs

    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build Gemini LLM instance with thinking budget configuration.
        
        Returns:
            Union[BaseChatModel, Runnable]: Configured ChatGoogleGenerativeAI 
                instance, optionally with tools bound. Automatically handles:
                - Thinking budget from instance setting or GOOGLE_THINKING_BUDGET env var
                - Thinking level configuration
                - Thought inclusion configuration
                - Schema warning suppression during tool binding
        """
        config: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "timeout": None,
            "max_output_tokens": self.max_output_tokens,
        }

        env_budget = os.environ.get("GOOGLE_THINKING_BUDGET")
        if env_budget and "thinking_budget" not in config:
            try:
                config["thinking_budget"] = int(env_budget)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GOOGLE_THINKING_BUDGET value {env_budget!r}: {exc!r}"
                ) from exc

        if self.thinking_budget is not None:
            config["thinking_budget"] = self.thinking_budget

        if self.thinking_level is not None:
            config["thinking_level"] = self.thinking_level

        config["include_thoughts"] = self.include_thoughts

        config.update(self.extra_kwargs)
        config = self._get_llm_config_with_callbacks(config)

        with _suppress_schema_warnings():
            llm = ChatGoogleGenerativeAI(**config)
            if self._tools:
                llm = llm.bind_tools(self._tools)
        return llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Gemini model response with automatic retry and backoff logic.
        
        Args:
            messages (list[BaseMessage]): Complete conversation history to send 
                to Gemini, including system, user, assistant, and tool messages.
                
        Returns:
            tuple[AgentResponse, AIMessage]: A tuple containing:
                - AgentResponse with content, tool_calls, reasoning (extracted 
                  from thought blocks if include_thoughts=True), and completion status
                - Raw AIMessage for conversation history maintenance
                
        Raises:
            RuntimeError: If LLM is not initialized (initialize() not called).
            Exception: Any unrecoverable errors from Google API after retries.
        """
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")
        
        llm = self._llm  # Capture for type narrowing

        async def _invoke():
            with _suppress_schema_warnings():
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
        """Get Google-specific response parser.
        
        Returns:
            ResponseParser: GoogleResponseParser instance that can extract 
                thought blocks, tool calls, and completion signals from Gemini's 
                response format.
        """
        return GoogleResponseParser()
