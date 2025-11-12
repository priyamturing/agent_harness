"""Qwen agent implementation for Alibaba Cloud Model Studio.

Uses OpenAI-compatible API without modifying the SDK.
Documentation: https://www.alibabacloud.com/help/en/model-studio/first-api-call-to-qwen
"""

import os
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from mcp_benchmark_sdk.agents.core import Agent
from mcp_benchmark_sdk.agents.tasks import AgentResponse
from mcp_benchmark_sdk.agents.parsers import OpenAIResponseParser, ResponseParser
from mcp_benchmark_sdk.agents.utils import retry_with_backoff


class QwenAgent(Agent):
    """Agent implementation for Alibaba Cloud Qwen models.

    Uses OpenAI-compatible API from DashScope.
    Requires DASHSCOPE_API_KEY environment variable.

    Supported models:
    - qwen-plus (default)
    - qwen3-14b
    - qwen-turbo
    - qwen-max
    - qwen2.5-72b-instruct
    - qwen-14b (maps to qwen-plus)
    """

    # Models that support reasoning/thinking
    _REASONING_MODELS = frozenset({
        "qwen-plus",
        "qwen-max",
        "qwen2.5-72b-instruct",
    })

    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        max_output_tokens: Optional[int] = None,
        tool_call_limit: int = 1000,
        system_prompt: Optional[str] = None,
        enable_thinking: bool = True,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Qwen agent.

        Args:
            model: Qwen model name
            temperature: Sampling temperature
            max_output_tokens: Maximum output tokens
            tool_call_limit: Maximum tool calls per run
            system_prompt: Optional system prompt for the agent
            enable_thinking: Enable reasoning for supported models
            base_url: Optional custom base URL (default: Singapore endpoint)
            **kwargs: Additional arguments for ChatOpenAI (not Agent base class)

        Raises:
            EnvironmentError: If DASHSCOPE_API_KEY is not set
        """
        # Pass system_prompt and tool_call_limit to parent Agent class
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking
        
        # Map common model names to actual DashScope model IDs
        model_map = {
            "qwen-14b": "qwen3-14b",
            "qwen3-14b": "qwen3-14b",
            "qwen-plus": "qwen-plus",
            "qwen-turbo": "qwen-turbo",
            "qwen-max": "qwen-max",
            "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
        }
        
        self.actual_model = model_map.get(model.lower(), model)
        
        # Get base URL from environment or use default (Singapore)
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        # Get API key
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "DASHSCOPE_API_KEY is not set. Export the API key before running Qwen models.\n"
                "Get your API key from: https://modelstudio.console.alibabacloud.com/?tab=model#/api-key\n"
                "Set it with: export DASHSCOPE_API_KEY='your-key-here'"
            )
        
        self.extra_kwargs = kwargs

    def _build_llm(self) -> BaseChatModel:
        """Build Qwen model using OpenAI-compatible interface."""
        config: dict[str, Any] = {
            "model": self.actual_model,
            "temperature": self.temperature,
            "timeout": None,
            "max_retries": 3,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }

        if self.max_output_tokens is not None:
            config["max_completion_tokens"] = self.max_output_tokens

        # Pass through any additional kwargs - let the API fail if they're unsupported
        config.update(self.extra_kwargs)
        
        # Note: Qwen's thinking/reasoning requires streaming mode which we don't use
        # For non-streaming mode, explicitly disable thinking in the request body
        # Use extra_body to pass additional parameters to the API
        extra_body = config.get("extra_body", {})
        extra_body["enable_thinking"] = False
        config["extra_body"] = extra_body

        llm = ChatOpenAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm

    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get Qwen model response with retry logic."""
        if not self._llm:
            raise RuntimeError("LLM not initialized. Call initialize() first.")

        async def _invoke():
            return await self._llm.ainvoke(messages)

        ai_message = await retry_with_backoff(
            _invoke,
            max_retries=2,
            timeout_seconds=600.0,
            on_retry=lambda attempt, exc, delay: None,
        )

        # Parse response (use OpenAI parser since it's compatible)
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
        """Get OpenAI-compatible response parser."""
        return OpenAIResponseParser()

