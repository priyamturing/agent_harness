"""Base agent class with multi-level API."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.tools import ToolException, _convert_call_tool_result
from mcp.types import CallToolResult, Tool as MCPSpecTool, TextContent
from mcp_agent.agents.agent import Agent as MCPAgent
from mcp_agent.config import MCPSettings, MCPServerSettings, Settings
from mcp_agent.core.context import Context
from mcp_agent.executor.executor import AsyncioExecutor
from mcp_agent.mcp.mcp_server_registry import ServerRegistry

from ..constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_TOOL_CALL_LIMIT,
    TOOL_CALL_ID_HEX_LENGTH,
)
from ..mcp import MCPConfig
from ..parsers import ParsedResponse, ResponseParser
from ..runtime import RunContext
from ..tasks import AgentResponse, Result, Task, ToolCall, ToolResult
from ..utils import retry_with_backoff


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
            system_prompt: Optional system prompt for the agent (None = no system message)
            tool_call_limit: Maximum tool calls before stopping (None = no limit)
        """
        self.system_prompt = system_prompt
        self.tool_call_limit = tool_call_limit

        # Runtime state (set during initialize)
        self._task: Optional[Task] = None
        self._run_context: Optional[RunContext] = None
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._llm: Optional[BaseChatModel] = None
        self._mcp_agent_core: Optional[MCPAgent] = None
        self._mcp_context: Optional[Context] = None
        self._mcp_executor: Optional[AsyncioExecutor] = None
        self._server_registry: Optional[ServerRegistry] = None
        
        # Ownership tracking for resource cleanup
        # If agent creates RunContext internally, it's responsible for cleaning it up
        self._owns_run_context: bool = False

    # ============ High-Level API ============

    async def run(
        self,
        task: Task,
        max_steps: int = DEFAULT_MAX_STEPS,
        *,
        run_context: Optional[RunContext] = None,
    ) -> Result:
        """Complete task execution with auto-loop.

        Args:
            task: Task to execute
            max_steps: Maximum number of agent turns
            run_context: Optional runtime context (created if not provided)

        Returns:
            Result with final status and messages
            
        Note:
            If run_context is not provided, the agent creates and owns it,
            and will clean it up automatically. If you provide your own
            run_context, you're responsible for cleaning it up (e.g., using
            'async with RunContext() as ctx').
        """
        # Track RunContext ownership for proper cleanup
        if run_context is None:
            # Agent creates and owns the RunContext
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)
            self._run_context = run_context
            self._owns_run_context = True
        else:
            # User provided RunContext - they're responsible for cleanup
            self._run_context = run_context
            self._owns_run_context = False

        try:
            await self.initialize(task, run_context)
            result = await self._execute_loop(max_steps, run_context)
            return result
        finally:
            await self.cleanup()

    # ============ Mid-Level Overridable ============

    @abstractmethod
    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get model response (override for custom LLM).

        Args:
            messages: Conversation history

        Returns:
            Tuple of (AgentResponse, raw AIMessage for conversation history)
        """
        ...

    @abstractmethod
    def get_response_parser(self) -> ResponseParser:
        """Get response parser (override for custom parsing).

        Returns:
            ResponseParser instance
        """
        ...

    def get_model_config(self) -> dict[str, Any]:
        """Get model-specific configuration (override for custom config).

        Returns:
            Configuration dict for model instantiation
        """
        return {}

    # ============ Low-Level Primitives ============

    async def initialize(self, task: Task, run_context: RunContext) -> None:
        """Initialize with task (connect MCPs, load tools, setup hooks).

        Args:
            task: Task to execute
            run_context: Runtime context
            
        Note:
            When using the high-level run() API, self._run_context is already set
            before this is called. For low-level API usage, set self._run_context
            before calling this method.
        """
        self._task = task

        server_settings = self._build_server_settings(task.mcps, run_context.database_id)
        settings = Settings(mcp=MCPSettings(servers=server_settings))

        context = Context(config=settings)
        server_registry = ServerRegistry(config=settings)
        context.server_registry = server_registry

        executor = AsyncioExecutor()
        executor._context = context  # Bind executor to context for callbacks
        context.executor = executor

        agent_name = task.metadata.get("agent_name") if task.metadata else None
        if not agent_name:
            agent_name = f"benchmark-agent-{uuid4().hex[:8]}"

        self._mcp_context = context
        self._mcp_executor = executor
        self._server_registry = server_registry

        self._mcp_agent_core = MCPAgent(
            name=agent_name,
            instruction=self.system_prompt,
            server_names=list(server_settings.keys()),
            context=context,
            connection_persistence=False,
        )

        await self._mcp_agent_core.initialize()

        tools_result = await self._mcp_agent_core.list_tools()
        self._tools = self._build_structured_tools(tools_result.tools)
        self._tool_map = {tool.name: tool for tool in self._tools}

        self._llm = self._build_llm()

        await run_context.notify_status(
            f"Initialized with {len(self._tools)} tools from {len(server_settings)} MCP(s)"
        )

    async def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute MCP tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool results
        """
        if not self._mcp_agent_core:
            raise RuntimeError("MCP agent not initialized. Call initialize() first.")

        results: list[ToolResult] = []

        for tc in tool_calls:
            if tc.id is None:
                tc.id = f"{tc.name}_{uuid4().hex[:TOOL_CALL_ID_HEX_LENGTH]}"
                if self._run_context:
                    await self._run_context.notify_status(
                        f"⚠️  Tool call '{tc.name}' missing ID - generated: {tc.id}",
                        "warning",
                    )

            if not tc.name or not tc.name.strip():
                error_msg = (
                    "Invalid tool call: missing or empty tool name. "
                    "This indicates a malformed LLM response."
                )
                tool_result = ToolResult(
                    content=error_msg,
                    tool_call_id=tc.id,
                    is_error=True,
                )
                results.append(tool_result)
                if self._run_context:
                    await self._run_context.notify_tool_call(
                        "<empty_name>", tc.arguments, error_msg, is_error=True
                    )
                continue

            if tc.name not in self._tool_map:
                error_msg = f"Unknown tool '{tc.name}'"
                tool_result = ToolResult(
                    content=error_msg,
                    tool_call_id=tc.id,
                    is_error=True,
                )
                results.append(tool_result)
                if self._run_context:
                    await self._run_context.notify_tool_call(
                        tc.name, tc.arguments, error_msg, is_error=True
                    )
                continue

            try:
                call_result = await self._mcp_agent_core.call_tool(
                    tc.name, arguments=tc.arguments
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Tool '{tc.name}' failed: {exc!r}"
                tool_result = ToolResult(
                    content=error_msg,
                    tool_call_id=tc.id,
                    is_error=True,
                )
                results.append(tool_result)
                if self._run_context:
                    await self._run_context.notify_tool_call(
                        tc.name, tc.arguments, error_msg, is_error=True
                    )
                continue

            tool_result = self._convert_call_result(tc, call_result)
            results.append(tool_result)

            if self._run_context:
                await self._run_context.notify_tool_call(
                    tc.name,
                    tc.arguments,
                    tool_result.structured_content or tool_result.content,
                    is_error=tool_result.is_error,
                )

        return results

    def get_available_tools(self) -> list[BaseTool]:
        """Get filtered tools (excluding lifecycle tools).

        Returns:
            List of available tools
        """
        return self._tools

    def get_initial_messages(self) -> list[BaseMessage]:
        """Build system + task messages.

        Returns:
            List of initial messages
        """
        messages: list[BaseMessage] = []
        
        # Only add system message if prompt exists (not None)
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
        # Clean MCP connections
        if self._mcp_agent_core:
            try:
                await self._mcp_agent_core.shutdown()
            except Exception:
                pass
            finally:
                self._mcp_agent_core = None

        if self._server_registry and getattr(self._server_registry, "connection_manager", None):
            try:
                await self._server_registry.connection_manager.close()  # type: ignore[call-arg]
            except Exception:
                pass
            finally:
                self._server_registry = None
                self._mcp_context = None
                self._mcp_executor = None
        else:
            self._server_registry = None
            self._mcp_context = None
            self._mcp_executor = None
        
        # Clean RunContext if we own it (created internally in run())
        # If user provided their own RunContext, they're responsible for cleanup
        if self._owns_run_context and self._run_context:
            await self._run_context.cleanup()
        
        # Defensive cleanup of LLM resources
        # Different LangChain providers may or may not have cleanup methods
        if self._llm:
            # Strategy 1: Check for async close method (most common)
            if hasattr(self._llm, 'aclose'):
                try:
                    await self._llm.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass  
            
            # Strategy 2: Check for sync close method
            elif hasattr(self._llm, 'close'):
                try:
                    self._llm.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Strategy 3: Check for async context manager exit
            elif hasattr(self._llm, '__aexit__'):
                try:
                    await self._llm.__aexit__(None, None, None)  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Strategy 4: Direct access to internal httpx async client (ChatOpenAI, etc.)
            elif hasattr(self._llm, 'async_client') and hasattr(self._llm.async_client, 'aclose'):  # type: ignore[attr-defined]
                try:
                    await self._llm.async_client.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # Strategy 5: Direct access to sync client
            elif hasattr(self._llm, 'client') and hasattr(self._llm.client, 'close'):  # type: ignore[attr-defined]
                try:
                    self._llm.client.close()  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # If none of the above work, rely on garbage collection
            # This is fine for most modern LangChain versions

    # ============ Internal (overridable for full control) ============

    async def _execute_loop(self, max_steps: int, run_context: RunContext) -> Result:
        """Agent execution loop - override for custom flow.

        Args:
            max_steps: Maximum number of turns
            run_context: Runtime context

        Returns:
            Result with final state
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
                await run_context.notify_status(f"Model error: {exc}", "error")
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

            # Collect reasoning traces
            if response.reasoning:
                all_reasoning.append(response.reasoning)

            # Check tool call limit BEFORE adding to history to maintain consistency
            if response.tool_calls:
                if remaining_tool_calls is not None:
                    if remaining_tool_calls < len(response.tool_calls):
                        await run_context.notify_status("Tool call limit reached", "warning")
                        return Result(
                            success=False,
                            messages=messages,
                            verifier_results=[],
                            metadata={"steps": step + 1, "reason": "tool_call_limit_reached"},
                            database_id=run_context.database_id,
                            reasoning_traces=all_reasoning,
                            error="Tool call limit reached",
                        )

            messages.append(ai_message)

            # Check if done
            if response.done and not response.tool_calls:
                await run_context.notify_status("Agent completed", "info")

                return Result(
                    success=True,
                    messages=messages,
                    verifier_results=[],
                    metadata={"steps": step + 1},
                    database_id=run_context.database_id,
                    reasoning_traces=all_reasoning,
                    error=None,
                )

            # Execute tool calls
            if response.tool_calls:
                if remaining_tool_calls is not None:
                    remaining_tool_calls -= len(response.tool_calls)

                tool_results = await self.call_tools(response.tool_calls)
                tool_messages = self.format_tool_results(response.tool_calls, tool_results)
                messages.extend(tool_messages)

        # Max steps reached
        await run_context.notify_status("Max steps reached", "warning")

        return Result(
            success=False,
            messages=messages,
            verifier_results=[],
            metadata={"steps": max_steps, "reason": "max_steps_reached"},
            database_id=run_context.database_id,
            reasoning_traces=all_reasoning,
            error="Maximum steps reached",
        )

    # ============ Internal Helpers ============

    @abstractmethod
    def _build_llm(self) -> BaseChatModel:
        """Build the LLM instance (provider-specific).

        Returns:
            Configured BaseChatModel
        """
        ...

    def _build_server_settings(
        self, configs: list[MCPConfig], database_id: Optional[str]
    ) -> dict[str, MCPServerSettings]:
        servers: dict[str, MCPServerSettings] = {}
        for config in configs:
            headers = dict(config.headers) if config.headers else {}
            if database_id and "x-database-id" not in headers:
                headers["x-database-id"] = database_id

            servers[config.name] = MCPServerSettings(
                name=config.name,
                transport=config.transport,
                command=config.command,
                args=list(config.args) if config.args else [],
                url=config.url,
                headers=headers or None,
            )

        return servers

    def _build_structured_tools(self, tools: list[MCPSpecTool]) -> list[StructuredTool]:
        structured: list[StructuredTool] = []
        for tool in sorted(tools, key=lambda t: t.name):
            structured.append(self._create_structured_tool(tool))
        return structured

    def _create_structured_tool(self, tool: MCPSpecTool) -> StructuredTool:
        tool_name = tool.name

        async def call_tool(**arguments: Any) -> Any:
            if not self._mcp_agent_core:
                raise RuntimeError("MCP agent not initialized. Call initialize() first.")

            call_result = await self._mcp_agent_core.call_tool(tool_name, arguments=arguments)
            try:
                text_content, non_text_content = _convert_call_tool_result(call_result)
            except ToolException as exc:
                raise ToolException(str(exc) or self._stringify_call_tool_result(call_result)) from exc

            return (text_content, non_text_content) if non_text_content else text_content

        return StructuredTool(
            name=tool.name,
            description=tool.description or "",
            args_schema=tool.inputSchema,
            coroutine=call_tool,
            response_format="content_and_artifact",
        )

    @staticmethod
    def _stringify_call_tool_result(call_result: CallToolResult) -> str:
        parts: list[str] = []
        for content_block in call_result.content:
            try:
                if isinstance(content_block, TextContent):
                    if content_block.text:
                        parts.append(content_block.text)
                else:
                    parts.append(
                        json.dumps(content_block.model_dump(mode="json"), ensure_ascii=False)
                    )
            except Exception:
                parts.append(str(content_block))

        if not parts:
            serialized = call_result.model_dump(mode="json")
            content = serialized.get("content")
            try:
                return json.dumps(content, ensure_ascii=False)
            except Exception:
                return str(content)

        return "\n".join(parts)

    def _convert_call_result(self, tool_call: ToolCall, call_result: CallToolResult) -> ToolResult:
        content_str = self._stringify_call_tool_result(call_result)
        structured = call_result.model_dump(mode="json")
        return ToolResult(
            content=content_str,
            tool_call_id=tool_call.id,
            is_error=bool(call_result.isError),
            structured_content=structured,
        )

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
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Handle {"type": "text", "text": "..."}
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        # For non-text blocks, use repr
                        text_parts.append(repr(block))
                else:
                    text_parts.append(str(block))
            return "\n".join(text_parts)
        
        # Fallback: convert to string
        return str(content)

