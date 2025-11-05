"""Base agent class with multi-level API."""

from __future__ import annotations

import asyncio
import json
import textwrap
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from ..mcp import MCPClientManager, MCPConfig
from ..parsers import ParsedResponse, ResponseParser
from ..runtime import RunContext
from ..tasks import AgentResponse, Result, Task, ToolCall, ToolResult
from ..utils import retry_with_backoff
from ..verifiers import VerifierResult, execute_verifiers


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
        tool_call_limit: int = 1000,
    ):
        """Initialize agent.

        Args:
            system_prompt: Optional system prompt for the agent (None = no system message)
            tool_call_limit: Maximum tool calls before stopping
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

    # ============ High-Level API ============

    async def run(
        self,
        task: Task,
        max_steps: int = 1000,
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
        """
        if run_context is None:
            # Only pass database_id if provided; otherwise let RunContext auto-generate UUID
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)

        await self.initialize(task, run_context)
        try:
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
        """
        self._task = task
        self._run_context = run_context

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
            tool = self._tool_map.get(tc.name)

            if tool is None:
                error_msg = f"Unknown tool '{tc.name}'"
                results.append(
                    ToolResult(
                        content=error_msg,
                        tool_call_id=tc.id or tc.name,
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
                content_str = (
                    json.dumps(serialized, ensure_ascii=False)
                    if isinstance(serialized, (dict, list))
                    else str(serialized)
                )

                results.append(
                    ToolResult(
                        content=content_str,
                        tool_call_id=tc.id or tc.name,
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
                        tool_call_id=tc.id or tc.name,
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

    async def run_verifiers(self) -> list[VerifierResult]:
        """Execute verifiers on current state.

        Returns:
            List of verifier results
        """
        if not self._task or not self._task.verifiers:
            return []

        if not self._run_context or not self._run_context.sql_runner_url:
            return []

        results = await execute_verifiers(
            self._task.verifiers,
            self._run_context.sql_runner_url,
            self._run_context.database_id,
            self._run_context.http_client,
        )

        if self._run_context:
            await self._run_context.notify_verifier_update(results)

        return results

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
        """
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
        """Cleanup resources (MCP connections, etc.)."""
        if self._mcp_manager:
            await self._mcp_manager.cleanup()

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

        # Notify initial messages
        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = self._extract_text_content(msg.content)
                await run_context.notify_message("system", content)
            elif isinstance(msg, HumanMessage):
                content = self._extract_text_content(msg.content)
                await run_context.notify_message("user", content)

        for step in range(max_steps):
            await run_context.notify_status(f"Step {step + 1}/{max_steps}", "info")

            # Get model response
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

            # Add AIMessage to conversation history (CRITICAL for tool call tracking)
            messages.append(ai_message)

            # Check if done
            if response.done and not response.tool_calls:
                await run_context.notify_status("Agent completed", "info")

                # Run final verifiers
                verifier_results = await self.run_verifiers()

                # Check verifier success
                all_passed = all(v.success for v in verifier_results) if verifier_results else True
                error_msg = None
                if verifier_results and not all_passed:
                    failed_verifiers = [v.name for v in verifier_results if not v.success]
                    error_msg = f"Verifier(s) failed: {', '.join(failed_verifiers)}"

                return Result(
                    success=all_passed,
                    messages=messages,
                    verifier_results=verifier_results,
                    metadata={"steps": step + 1},
                    database_id=run_context.database_id,
                    reasoning_traces=all_reasoning,
                    error=error_msg,
                )

            # Execute tool calls
            if response.tool_calls:
                if remaining_tool_calls is not None:
                    if remaining_tool_calls < len(response.tool_calls):
                        await run_context.notify_status("Tool call limit reached", "warning")
                        break
                    remaining_tool_calls -= len(response.tool_calls)

                tool_results = await self.call_tools(response.tool_calls)
                tool_messages = self.format_tool_results(response.tool_calls, tool_results)
                messages.extend(tool_messages)

                # Run verifiers after tool calls
                await self.run_verifiers()

        # Max steps reached
        await run_context.notify_status("Max steps reached", "warning")
        verifier_results = await self.run_verifiers()

        return Result(
            success=False,
            messages=messages,
            verifier_results=verifier_results,
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

