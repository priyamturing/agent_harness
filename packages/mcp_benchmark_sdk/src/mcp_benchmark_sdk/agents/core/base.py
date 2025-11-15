"""Base agent class with multi-level API."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Union
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from ..constants import (
    DEFAULT_MAX_STEPS,
    DEFAULT_TOOL_CALL_LIMIT,
    TOOL_CALL_ID_HEX_LENGTH,
)
from ..mcp import MCPClientManager 
from ..parsers import  ResponseParser
from ..runtime import RunContext
from ..tasks import AgentResponse, Result, Task, ToolCall, ToolResult
from ..telemetry import get_langfuse_trace_url, maybe_attach_langfuse_callback


class Agent(ABC):
    """Base agent class with multi-level API.

    Supports three levels of usage:
    1. High-level: agent.run(task) - complete automation
    2. Mid-level: Override get_response(), get_model_config() - custom LLM
    3. Low-level: Manual loop control via primitives
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        tool_call_limit: Optional[int] = DEFAULT_TOOL_CALL_LIMIT,
    ):
        """Initialize agent.

        Args:
            system_prompt (Optional[str]): Optional system prompt for the agent. 
                If None, no system message will be included in the conversation.
            tool_call_limit (Optional[int]): Maximum number of tool calls allowed 
                before stopping execution. If None, no limit is enforced.
        """
        self.system_prompt = system_prompt
        self.tool_call_limit = tool_call_limit

        self._task: Optional[Task] = None
        self._run_context: Optional[RunContext] = None
        self._mcp_manager: Optional[MCPClientManager] = None
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._llm: Union[BaseChatModel, Runnable, None] = None
        
        self._owns_run_context: bool = False

    def _apply_llm_tracing_callbacks(self, config: dict[str, Any]) -> None:
        """Allow subclasses to inject telemetry callbacks before building LLMs."""
        maybe_attach_langfuse_callback(self, config)

    def _get_tool_invoke_config(self) -> dict[str, Any]:
        """Get configuration dict with callbacks for tool invocations."""
        config: dict[str, Any] = {}
        maybe_attach_langfuse_callback(self, config)
        return config

    def _attach_langfuse_trace_url(self, result: Result) -> None:
        if result.langfuse_url:
            return

        url = get_langfuse_trace_url(self)
        if url:
            result.langfuse_url = url

    # ============ High-Level API ============

    async def run(
        self,
        task: Task,
        max_steps: int = DEFAULT_MAX_STEPS,
        *,
        run_context: Optional[RunContext] = None,
    ) -> Result:
        """Complete task execution with automatic agent loop.

        Args:
            task (Task): The task to execute, containing prompt, MCP configuration,
                and optional database ID.
            max_steps (int): Maximum number of agent turns/iterations before 
                stopping. Defaults to DEFAULT_MAX_STEPS.
            run_context (Optional[RunContext]): Runtime context for telemetry,
                status updates, and database communication. If None, a new 
                RunContext will be created and managed by the agent.

        Returns:
            Result: Execution result containing:
                - success (bool): Whether task completed successfully
                - messages (list[BaseMessage]): Complete conversation history
                - metadata (dict): Execution metadata (steps, errors, etc.)
                - reasoning_traces (list[str]): All reasoning steps
                - error (Optional[str]): Error message if failed
            
        Note:
            If run_context is not provided, the agent creates and owns it,
            and will clean it up automatically. If you provide your own
            run_context, you're responsible for cleaning it up (e.g., using
            'async with RunContext() as ctx').
            
            For LangSmith/Langfuse tracing, wrap the agent with `with_tracing()`:
                agent = with_tracing(ClaudeAgent())
        """
        if run_context is None:
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)
            owns_context = True
        else:
            owns_context = False
        
        self._run_context = run_context
        self._owns_run_context = owns_context

        try:
            await self.initialize(task, run_context)
            result = await self._execute_loop(max_steps, run_context)
            self._attach_langfuse_trace_url(result)
            return result
        finally:
            await self.cleanup()

    # ============ Mid-Level Overridable ============

    @abstractmethod
    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get model response (override for custom LLM).

        Args:
            messages (list[BaseMessage]): Complete conversation history including 
                system messages, user messages, assistant messages, and tool messages.

        Returns:
            tuple[AgentResponse, AIMessage]: A tuple containing:
                - AgentResponse: Parsed response with content, tool_calls, reasoning, and done flag
                - AIMessage: Raw LangChain AIMessage for maintaining conversation history
                
        Raises:
            Exception: Any LLM-specific errors (network, API, rate limits, etc.)
        """
        ...

    @abstractmethod
    def get_response_parser(self) -> ResponseParser:
        """Get response parser (override for custom parsing).

        Returns:
            ResponseParser: Parser instance specific to the LLM provider 
                (e.g., AnthropicParser, OpenAIParser, GoogleParser) that 
                knows how to extract tool calls, reasoning, and completion 
                signals from the provider's response format.
        """
        ...

    def get_model_config(self) -> dict[str, Any]:
        """Get model-specific configuration (override for custom config).

        Returns:
            dict[str, Any]: Configuration dictionary for LLM instantiation.
                May include API keys, model names, temperature, timeout settings,
                and other provider-specific parameters. Empty dict means use defaults.
        """
        return {}

    # ============ Low-Level Primitives ============

    async def initialize(self, task: Task, run_context: RunContext) -> None:
        """Initialize agent with task, connect MCP servers, and load tools.

        Args:
            task (Task): Task to execute, containing the prompt, MCP server 
                configuration, and optional database ID.
            run_context (RunContext): Runtime context for telemetry, status 
                updates, and database communication.
            
        Note:
            When using the high-level run() API, self._run_context is already set
            before this is called. For low-level API usage, set self._run_context
            before calling this method. This method connects to all MCP servers,
            loads available tools, and builds the LLM instance.
        """
        self._task = task

        self._mcp_manager = MCPClientManager()
        await self._mcp_manager.connect(task.mcp, run_context.database_id)

        self._tools = self._mcp_manager.get_all_tools()
        self._tool_map = {tool.name: tool for tool in self._tools}

        self._llm = self._build_llm()

        await run_context.notify_status(
            f"Initialized with {len(self._tools)} tools from MCP server"
        )

    async def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute MCP tool calls and return results.

        Args:
            tool_calls (list[ToolCall]): List of tool calls to execute. Each 
                ToolCall contains name, arguments, and an optional ID. If ID 
                is missing, one will be generated automatically.

        Returns:
            list[ToolResult]: List of tool execution results, one per tool call.
                Each ToolResult contains the output content, tool_call_id, error 
                status, and optionally structured content. Results are returned 
                in the same order as input tool_calls.
                
        Raises:
            TypeError: If a tool returns non-JSON-serializable data (MCP protocol violation)
        
        Note:
            - Unknown tool names result in error ToolResults, not exceptions
            - Tool execution errors are captured in ToolResult.is_error
            - All results are notified to run_context for telemetry
        """
        results: list[ToolResult] = []

        for tc in tool_calls:
            if tc.id is None:
                tc.id = f"{tc.name}_{uuid4().hex[:TOOL_CALL_ID_HEX_LENGTH]}"
                if self._run_context:
                    await self._run_context.notify_status(
                        f"⚠️  Tool call '{tc.name}' missing ID - generated: {tc.id}",
                        "warning"
                    )
            
            if not tc.name or not tc.name.strip():
                error_msg = (
                    "Invalid tool call: missing or empty tool name. "
                    "This indicates a malformed LLM response."
                )
                results.append(
                    ToolResult(
                        content=error_msg,
                        tool_call_id=tc.id,
                        is_error=True,
                    )
                )
                if self._run_context:
                    await self._run_context.notify_tool_call(
                        "<empty_name>", tc.arguments, error_msg, is_error=True
                    )
                continue
            
            tool = self._tool_map.get(tc.name)

            if tool is None:
                error_msg = f"Unknown tool '{tc.name}'"
                results.append(
                    ToolResult(
                        content=error_msg,
                        tool_call_id=tc.id,
                        is_error=True,
                    )
                )
                if self._run_context:
                    await self._run_context.notify_tool_call(
                        tc.name, tc.arguments, error_msg, is_error=True
                    )
                continue

            try:
                # Get config with tracing callbacks (Langfuse, etc.)
                tool_config = self._get_tool_invoke_config()
                
                # Invoke tool - check for async vs sync
                # Some LangChain tools are async (ainvoke), others are sync (invoke)
                # For sync tools, run in thread pool to avoid blocking the event loop
                if hasattr(tool, "ainvoke"):
                    output = await tool.ainvoke(tc.arguments, config=tool_config)
                else:
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(None, lambda: tool.invoke(tc.arguments, config=tool_config))

                serialized = self._normalize_tool_output(output)
                
                # MCP protocol requires JSON-serializable responses
                # If serialization fails, it's a protocol violation by the MCP server
                try:
                    content_str = json.dumps(serialized, ensure_ascii=False)
                except TypeError as e:
                    raise TypeError(
                        f"Tool '{tc.name}' returned non-JSON-serializable data. "
                        f"MCP servers must return JSON-compatible types. "
                        f"Received type: {type(serialized).__name__}. "
                        f"Original error: {e!r}"
                    ) from e

                results.append(
                    ToolResult(
                        content=content_str,
                        tool_call_id=tc.id,
                        is_error=False,
                        structured_content=serialized if isinstance(serialized, dict) else None,
                    )
                )

                if self._run_context:
                    await self._run_context.notify_tool_call(
                        tc.name, tc.arguments, serialized, is_error=False
                    )

            except Exception as exc:
                error_msg = f"Tool '{tc.name}' failed: {exc!r}"
                results.append(
                    ToolResult(
                        content=error_msg,
                        tool_call_id=tc.id,
                        is_error=True,
                    )
                )

                if self._run_context:
                    await self._run_context.notify_tool_call(
                        tc.name, tc.arguments, error_msg, is_error=True
                    )

        return results

    def get_available_tools(self) -> list[BaseTool]:
        """Get all available tools loaded from MCP servers.

        Returns:
            list[BaseTool]: List of LangChain BaseTool instances loaded from 
                all connected MCP servers. These tools can be bound to the LLM 
                for function calling.
        """
        return self._tools

    def get_initial_messages(self) -> list[BaseMessage]:
        """Build initial conversation messages (system prompt + task prompt).

        Returns:
            list[BaseMessage]: List of initial messages to start the conversation.
                Includes SystemMessage (if system_prompt is set) followed by 
                HumanMessage (containing the task prompt).
        """
        messages: list[BaseMessage] = []
        
        if self.system_prompt is not None:
            messages.append(SystemMessage(content=self.system_prompt))

        if self._task:
            messages.append(HumanMessage(content=self._task.prompt))

        return messages

    def format_tool_results(
        self, tool_calls: list[ToolCall], results: list[ToolResult]
    ) -> list[BaseMessage]:
        """Format tool results into model messages.

        Args:
            tool_calls: Tool calls that were made
            results: Results from tool execution

        Returns:
            List of ToolMessage objects
            
        Raises:
            RuntimeError: If tool_calls and results length mismatch
        """
        if len(tool_calls) != len(results):
            error_msg = (
                f"Tool calls/results mismatch: {len(tool_calls)} calls "
                f"but {len(results)} results. This indicates a bug in call_tools() "
                f"or its override."
            )
            raise RuntimeError(error_msg)
        
        messages: list[BaseMessage] = []

        for tc, result in zip(tool_calls, results):
            messages.append(
                ToolMessage(
                    content=result.content,
                    tool_call_id=result.tool_call_id,
                    name=tc.name,
                )
            )

        return messages

    async def cleanup(self) -> None:
        """Cleanup resources (MCP connections, LLM clients, RunContext, etc.).
        
        Performs defensive cleanup of all agent resources:
        - MCP client connections
        - LangChain LLM clients (if they support cleanup)
        - RunContext (HTTP client) if agent owns it
        
        This method is safe to call multiple times and will attempt
        cleanup even if resources don't explicitly support it.
        """
        if self._mcp_manager:
            await self._mcp_manager.cleanup()
        
        # Clean RunContext if we own it (created internally in run())
        # If user provided their own RunContext, they're responsible for cleanup
        if self._owns_run_context and self._run_context:
            await self._run_context.cleanup()
        
        # Defensive cleanup of LLM resources
        # Different LangChain providers may or may not have cleanup methods
        if self._llm:
            if hasattr(self._llm, 'aclose'):
                try:
                    await self._llm.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass  
            
            elif hasattr(self._llm, 'close'):
                try:
                    self._llm.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            elif hasattr(self._llm, '__aexit__'):
                try:
                    await self._llm.__aexit__(None, None, None)  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            elif hasattr(self._llm, 'async_client') and hasattr(self._llm.async_client, 'aclose'):  # type: ignore[attr-defined]
                try:
                    await self._llm.async_client.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            elif hasattr(self._llm, 'client') and hasattr(self._llm.client, 'close'):  # type: ignore[attr-defined]
                try:
                    self._llm.client.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            


    # ============ Internal (overridable for full control) ============

    async def _execute_loop(self, max_steps: int, run_context: RunContext) -> Result:
        """Execute the main agent loop - override for custom execution flow.

        Args:
            max_steps (int): Maximum number of agent turns/iterations before 
                stopping. Each turn includes getting a response and optionally 
                executing tool calls.
            run_context (RunContext): Runtime context for telemetry and status 
                updates throughout the execution loop.

        Returns:
            Result: Final execution result containing:
                - success: True if agent signaled completion, False if stopped by limits
                - messages: Complete conversation history
                - metadata: Execution details (step count, stop reason, etc.)
                - reasoning_traces: All reasoning steps collected during execution
                - error: Error message if failed, None if successful
        """
        messages = self.get_initial_messages()
        remaining_tool_calls = self.tool_call_limit
        all_reasoning: list[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = self._extract_text_content(msg.content)
                await run_context.notify_message("system", content)
            elif isinstance(msg, HumanMessage):
                content = self._extract_text_content(msg.content)
                await run_context.notify_message("user", content)

        for step in range(max_steps):
            await run_context.notify_status(f"Step {step + 1}/{max_steps}", "info")

            try:
                response, ai_message = await self.get_response(messages)
            except Exception as exc:
                await run_context.notify_status(f"Model error: {exc!r}", "error")
                return Result(
                    success=False,
                    messages=messages,
                    metadata={"step": step, "error": str(exc)},
                    database_id=run_context.database_id,
                    error=str(exc),
                )

            # Log assistant message
            await run_context.notify_message(
                "assistant",
                response.content,
                {"reasoning": response.reasoning} if response.reasoning else None,
            )

            if response.reasoning:
                all_reasoning.append(response.reasoning)

            # Add AI message to history BEFORE checking limit
            # This ensures the conversation log shows what the AI attempted
            messages.append(ai_message)

            # Check tool call limit AFTER adding to history
            # If limit is exceeded, the message is included but tool calls are blocked
            if (
                response.tool_calls
                and remaining_tool_calls is not None
                and remaining_tool_calls < len(response.tool_calls)
            ):
                tool_call_names = [tc.name for tc in response.tool_calls]
                error_msg = (
                    f"Tool call limit reached. Agent attempted to make {len(response.tool_calls)} "
                    f"tool call(s) {tool_call_names} but only {remaining_tool_calls} call(s) remaining. "
                    f"These tool calls were NOT executed."
                )
                await run_context.notify_status(error_msg, "warning")
                return Result(
                    success=False,
                    messages=messages,
                    metadata={
                        "steps": step + 1,
                        "reason": "tool_call_limit_reached",
                        "attempted_tool_calls": len(response.tool_calls),
                        "remaining_tool_calls": remaining_tool_calls,
                    },
                    database_id=run_context.database_id,
                    reasoning_traces=all_reasoning,
                    error=error_msg,
                )

            if response.done and not response.tool_calls:
                await run_context.notify_status("Agent completed", "info")

                return Result(
                    success=True,
                    messages=messages,
                    metadata={"steps": step + 1},
                    database_id=run_context.database_id,
                    reasoning_traces=all_reasoning,
                    error=None,
                )

            if response.tool_calls:
                if remaining_tool_calls is not None:
                    remaining_tool_calls -= len(response.tool_calls)

                tool_results = await self.call_tools(response.tool_calls)
                tool_messages = self.format_tool_results(response.tool_calls, tool_results)
                messages.extend(tool_messages)

        await run_context.notify_status("Max steps reached", "warning")

        return Result(
            success=False,
            messages=messages,
            metadata={"steps": max_steps, "reason": "max_steps_reached"},
            database_id=run_context.database_id,
            reasoning_traces=all_reasoning,
            error="Maximum steps reached",
        )

    # ============ Internal Helpers ============

    @abstractmethod
    def _build_llm(self) -> Union[BaseChatModel, Runnable]:
        """Build the LLM instance (provider-specific).

        Returns:
            Union[BaseChatModel, Runnable]: Configured LLM instance. Can be a 
                raw BaseChatModel or a Runnable chain with tools bound, depending 
                on provider requirements. This method is called during initialize().
        """
        ...

    def _normalize_tool_output(self, result: object) -> object:
        """Normalize tool output to consistent format for JSON serialization.
        
        Args:
            result (object): Raw tool output from MCP server, which can be a 
                string, number, boolean, None, dict, list, or other object.
                
        Returns:
            object: Normalized result. If input is a JSON string, attempts to 
                parse it. Otherwise returns input as-is for JSON serialization.
        """
        if isinstance(result, str):
            stripped = result.strip()
            if stripped:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return result
            return result

        if isinstance(result, (int, float, bool)) or result is None:
            return result

        return result

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract text from message content (handles multimodal content).
        
        Args:
            content: Message content (str, list, or other)
            
        Returns:
            Text representation of content
        """
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        text_parts.append(repr(block))
                else:
                    text_parts.append(str(block))
            return "\n".join(text_parts)
        
        return str(content)
