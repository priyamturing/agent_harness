"""Result and response data structures for agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the agent.
    
    Attributes:
        name (str): Name of the tool to call, matching a tool from MCP server.
        arguments (dict[str, Any]): Arguments to pass to the tool as key-value pairs.
        id (Optional[str]): Unique identifier for this tool call. Generated 
            automatically if None.
    """

    name: str
    arguments: dict[str, Any]
    id: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool call.
    
    Attributes:
        content (str): String representation of the tool output, typically JSON.
        tool_call_id (str): ID matching the ToolCall that was executed.
        is_error (bool): Whether the tool execution resulted in an error.
        structured_content (Optional[dict[str, Any]]): Parsed structured content 
            if the output is a dict, None otherwise.
    """

    content: str
    tool_call_id: str
    is_error: bool = False
    structured_content: Optional[dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Single agent turn/iteration response from the LLM.
    
    Attributes:
        content (str): Text content of the response from the agent.
        tool_calls (list[ToolCall]): List of tool calls requested by agent in this turn.
        reasoning (Optional[str]): Extracted reasoning/thinking from the agent 
            (e.g., Claude thinking blocks, GPT-5 reasoning).
        done (bool): Whether the agent has finished and requires no further turns.
        info (dict[str, Any]): Additional metadata about this response, such as 
            raw_reasoning content before parsing.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Final execution result from running a task.
    
    Represents the agent's execution outcome, including conversation history,
    success status, and any errors. This is the agent layer result only;
    verifier results are managed separately by the harness layer.
    
    Attributes:
        success (bool): Whether the task completed successfully (agent signaled done).
        messages (list[Any]): Complete LangChain message history from the execution.
        metadata (dict[str, Any]): Execution metadata including step count, stop reason, etc.
        database_id (Optional[str]): Database ID if multi-tenant scenario was used.
        reasoning_traces (list[str]): All reasoning/thinking traces collected during execution.
        error (Optional[str]): Error message if execution failed, None if successful.
        langsmith_url (Optional[str]): LangSmith trace URL if tracing was enabled.
        langfuse_url (Optional[str]): Langfuse trace URL if tracing was enabled.
    """

    success: bool
    messages: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    reasoning_traces: list[str] = field(default_factory=list)
    error: Optional[str] = None
    langsmith_url: Optional[str] = None  # LangSmith trace URL if tracing is enabled
    langfuse_url: Optional[str] = None   # Langfuse trace URL if tracing is enabled

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Convert LangChain messages to structured conversation history.
        
        Returns:
            list[dict[str, Any]]: List of conversation entries with type, role, 
                and content. Each entry is one of:
                - message: {"type": "message", "role": str, "content": str, "reasoning": list[str]?}
                - tool_call: {"type": "tool_call", "tool": str, "args": dict}
                - tool_result: {"type": "tool_result", "tool": str, "output": Any}
        """
        if not self.messages:
            return []

        conversation: list[dict[str, Any]] = []
        for msg in self.messages:
            msg_type = getattr(msg, "type", None)

            if msg_type == "ai":
                conversation.extend(self._format_ai_message(msg))
            elif msg_type == "tool":
                entry = self._format_tool_result(msg)
                if entry:
                    conversation.append(entry)
            elif msg_type == "human":
                entry = self._format_role_message(msg, "user")
                if entry:
                    conversation.append(entry)
            elif msg_type == "system":
                entry = self._format_role_message(msg, "system")
                if entry:
                    conversation.append(entry)

        return conversation

    def _format_ai_message(self, msg: Any) -> list[dict[str, Any]]:
        content_text, reasoning_blocks = self._extract_ai_content(msg)
        entries: list[dict[str, Any]] = []

        if content_text or reasoning_blocks:
            message_entry: dict[str, Any] = {
                "type": "message",
                "role": "assistant",
            }
            if content_text:
                message_entry["content"] = content_text
            if reasoning_blocks:
                message_entry["reasoning"] = reasoning_blocks
            entries.append(message_entry)

        entries.extend(self._collect_tool_calls(msg))
        return entries

    def _extract_ai_content(self, msg: Any) -> tuple[str, list[str]]:
        content = getattr(msg, "content", None)

        if isinstance(content, list):
            text = ""
            reasoning: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "thinking":
                    reasoning.append(block.get("thinking", ""))
                elif block_type == "text":
                    text = block.get("text", "")
            return text, reasoning

        if isinstance(content, str):
            return content, []

        return "", []

    def _collect_tool_calls(self, msg: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []

        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    entries.append(self._tool_call_entry(block.get("name"), block.get("input", {})))

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    entries.append(self._tool_call_entry(tc.get("name"), tc.get("args", {})))
                else:
                    entries.append(
                        self._tool_call_entry(
                            getattr(tc, "name", None),
                            getattr(tc, "arguments", {}),
                        )
                    )

        return entries

    def _tool_call_entry(self, name: Any, args: Any) -> dict[str, Any]:
        formatted_args = args if isinstance(args, dict) else args
        return {
            "type": "tool_call",
            "tool": name or "unknown",
            "args": formatted_args,
        }

    def _format_tool_result(self, msg: Any) -> Optional[dict[str, Any]]:
        tool_name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", "")

        output = self._safe_json_load(content)
        return {
            "type": "tool_result",
            "tool": tool_name or "unknown",
            "output": output,
        }

    def _format_role_message(self, msg: Any, role: str) -> Optional[dict[str, Any]]:
        content = getattr(msg, "content", None)
        if not content:
            return None

        return {
            "type": "message",
            "role": role,
            "content": content,
        }

    def _safe_json_load(self, content: Any) -> Any:
        if isinstance(content, str):
            try:
                return json.loads(content)
            except (json.JSONDecodeError, TypeError):
                return content
        return content
