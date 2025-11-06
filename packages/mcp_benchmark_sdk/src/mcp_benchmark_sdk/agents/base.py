"""Base agent class with multi-level API."""

from __future__ import annotations

import asyncio
import json
import os
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langsmith import traceable

from ..constants import (
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_MAX_STEPS,
    DEFAULT_TOOL_CALL_LIMIT,
    TOOL_CALL_ID_HEX_LENGTH,
)
from ..mcp import MCPClientManager, MCPConfig
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
        self._mcp_manager: Optional[MCPClientManager] = None
        self._tools: list[BaseTool] = []
        self._tool_map: dict[str, BaseTool] = {}
        self._llm: Optional[BaseChatModel] = None
        
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
            
        LangSmith Tracing:
            Creates a parent trace that groups all LLM calls and tool executions.
            Each run gets its own trace with metadata for easy identification.
        """
        # Setup run context first to get database_id for trace metadata
        if run_context is None:
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)
            owns_context = True
        else:
            owns_context = False
        
        # Wrap execution in traceable context if tracing is enabled
        if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
            # Create descriptive name for this trace
            model_name = self.model if hasattr(self, "model") else "unknown"
            
            # Extract scenario info from task metadata
            scenario_name = "task"
            run_number = None
            if task and hasattr(task, "metadata") and task.metadata:
                scenario_name = task.metadata.get("scenario_name", scenario_name)
                run_number = task.metadata.get("run_number")
            
            # Build readable trace name: "grok-4: scenario_name (run 2)"
            trace_name = f"{model_name}: {scenario_name}"
            if run_number is not None:
                trace_name = f"{trace_name} (run {run_number})"
            
            # Get database_id from run_context (more reliable than task.database_id)
            db_id = run_context.database_id if run_context else (task.database_id if task else None)
            
            # Get session_id and thread_id for grouping (LangSmith Threads feature)
            # session_id: Groups ALL runs from one mcp-benchmark invocation
            # thread_id: Unique ID for this specific run (model+scenario+run_number)
            session_id = None
            thread_id = None
            if task and task.metadata:
                session_id = task.metadata.get("session_id")
                thread_id = task.metadata.get("thread_id")
                
                # Generate thread_id if not provided (unique per model+scenario+run)
                if not thread_id and session_id:
                    thread_id = f"{session_id}_{model_name}_{scenario_name}_run{run_number or 1}"
            
            # Create metadata for filtering/search
            trace_metadata = {
                "model": model_name,
                "scenario": scenario_name,
                "database_id": db_id,
                "prompt_preview": task.prompt[:100] if task and task.prompt else "N/A",
            }
            if run_number is not None:
                trace_metadata["run_number"] = run_number
            
            # Add session_id for grouping all runs from same benchmark session
            # This enables LangSmith's Threads view: https://docs.langchain.com/langsmith/threads
            if session_id:
                trace_metadata["session_id"] = session_id
            if thread_id:
                trace_metadata["thread_id"] = thread_id
            
            # Use traceable wrapper with descriptive name
            @traceable(name=trace_name, run_type="chain", metadata=trace_metadata)
            async def _run_with_trace():
                return await self._run_impl(task, max_steps, run_context, owns_context)
            
            return await _run_with_trace()
        else:
            # No tracing - run directly
            return await self._run_impl(task, max_steps, run_context, owns_context)
    
    async def _run_impl(
        self,
        task: Task,
        max_steps: int,
        run_context: RunContext,
        owns_context: bool,
    ) -> Result:
        """Internal implementation of run (separated for tracing wrapper).
        
        Args:
            task: Task to execute
            max_steps: Maximum agent turns
            run_context: Already initialized RunContext
            owns_context: Whether this agent owns the context (for cleanup)
        """
        # Set context ownership
        self._run_context = run_context
        self._owns_run_context = owns_context

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

        # Connect to MCP servers
        self._mcp_manager = MCPClientManager()
        await self._mcp_manager.connect(task.mcps, run_context.database_id)

        # Load tools
        self._tools = self._mcp_manager.get_all_tools()
        self._tool_map = {tool.name: tool for tool in self._tools}

        # Build LLM with tools
        self._llm = self._build_llm()

        await run_context.notify_status(
            f"Initialized with {len(self._tools)} tools from {len(task.mcps)} MCP(s)"
        )

    async def call_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """Execute MCP tool calls.

        Args:
            tool_calls: List of tool calls to execute

        Returns:
            List of tool results
        """
        results: list[ToolResult] = []

        for tc in tool_calls:
            # Generate unique ID if LLM didn't provide one
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
                # Invoke tool
                if hasattr(tool, "ainvoke"):
                    output = await tool.ainvoke(tc.arguments)
                else:
                    loop = asyncio.get_running_loop()
                    output = await loop.run_in_executor(None, tool.invoke, tc.arguments)

                # Serialize output
                serialized = self._serialize_tool_output(output)
                
                # MCP protocol requires JSON-serializable responses
                # If serialization fails, it's a protocol violation by the MCP server
                try:
                    content_str = json.dumps(serialized, ensure_ascii=False)
                except TypeError as e:
                    raise TypeError(
                        f"Tool '{tc.name}' returned non-JSON-serializable data. "
                        f"MCP servers must return JSON-compatible types. "
                        f"Received type: {type(serialized).__name__}. "
                        f"Original error: {e}"
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
        if self._mcp_manager:
            await self._mcp_manager.cleanup()
        
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
                    langsmith_url=self._get_langsmith_trace_url(messages),
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
                            langsmith_url=self._get_langsmith_trace_url(messages),
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
                    langsmith_url=self._get_langsmith_trace_url(messages),
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
            langsmith_url=self._get_langsmith_trace_url(messages),
        )

    def _get_langsmith_trace_url(self, messages: list[BaseMessage]) -> Optional[str]:
        """Get LangSmith trace URL for the parent run (Agent.run).
        
        Since Agent.run() is decorated with @traceable, it creates a parent trace
        that groups all LLM calls and tool executions. This method fetches that
        parent trace's shareable URL.
        
        Returns:
            LangSmith trace URL (shareable link) or None if tracing is disabled
        """
        if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() != "true":
            return None
        
        # Get the current run tree (this is the Agent.run parent trace)
        try:
            from langsmith import get_current_run_tree, Client
            
            run_tree = get_current_run_tree()
            if not run_tree or not run_tree.id:
                return None
            
            # Fetch the run from LangSmith to get the correct shareable URL
            try:
                client = Client()
                run = client.read_run(str(run_tree.id))
                
                # The run.url property gives us the correct shareable URL
                if run and hasattr(run, "url") and run.url:
                    return run.url
            except Exception:
                pass
            
            # Fallback: use the trace_url or share_url from run_tree
            if hasattr(run_tree, "trace_url") and run_tree.trace_url:
                return run_tree.trace_url
            
            # Last resort: construct URL manually using the format from LangSmith docs
            # The actual URL format varies, so we'll just return None if we can't get it properly
            return None
            
        except Exception:
            return None

    # ============ Internal Helpers ============

    @abstractmethod
    def _build_llm(self) -> BaseChatModel:
        """Build the LLM instance (provider-specific).

        Returns:
            Configured BaseChatModel
        """
        ...

    def _serialize_tool_output(self, result: object) -> object:
        """Serialize tool output to consistent format."""
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

