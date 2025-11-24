"""Result and response data structures for agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ..runtime import MessageRole


class MessageType(str, Enum):
    """LangChain message type identifiers."""
    
    AI = "ai"
    TOOL = "tool"
    HUMAN = "human"
    SYSTEM = "system"


class ConversationEntryType(str, Enum):
    """Conversation entry type identifiers for formatted output."""
    
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ASSISTANT = "assistant"
    TOOL_RESULTS = "tool_results"


class ContentBlockType(str, Enum):
    """Content block type identifiers within messages."""
    
    THINKING = "thinking"
    TEXT = "text"
    TOOL_USE = "tool_use"


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
                - message: {"type": ConversationEntryType.MESSAGE, "role": MessageRole, "content": str, "reasoning": list[str]?}
                - tool_call: {"type": ConversationEntryType.TOOL_CALL, "tool": str, "args": dict}
                - tool_result: {"type": ConversationEntryType.TOOL_RESULT, "tool": str, "output": Any}
        """
        if not self.messages:
            return []

        conversation: list[dict[str, Any]] = []
        for msg in self.messages:
            msg_type = getattr(msg, "type", None)

            if msg_type == MessageType.AI:
                conversation.extend(self._format_ai_message(msg))
            elif msg_type == MessageType.TOOL:
                entry = self._format_tool_result(msg)
                if entry:
                    conversation.append(entry)
            elif msg_type == MessageType.HUMAN:
                entry = self._format_role_message(msg, MessageRole.USER)
                if entry:
                    conversation.append(entry)
            elif msg_type == MessageType.SYSTEM:
                entry = self._format_role_message(msg, MessageRole.SYSTEM)
                if entry:
                    conversation.append(entry)

        return conversation

    def _format_ai_message(self, msg: Any) -> list[dict[str, Any]]:
        content_text, reasoning_blocks = self._extract_ai_content(msg)
        entries: list[dict[str, Any]] = []

        if content_text or reasoning_blocks:
            message_entry: dict[str, Any] = {
                "type": ConversationEntryType.MESSAGE,
                "role": MessageRole.ASSISTANT,
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
            text_parts: list[str] = []
            reasoning: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == ContentBlockType.THINKING:
                    reasoning.append(block.get("thinking", ""))
                elif block_type == ContentBlockType.TEXT:
                    text_value = block.get("text", "")
                    if text_value:
                        text_parts.append(text_value)
            text = "".join(text_parts)
            return text, reasoning

        if isinstance(content, str):
            return content, []

        return "", []

    def _collect_tool_calls(self, msg: Any) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _append_call(name: Any, args: Any, call_id: Optional[str]) -> None:
            if not name and not call_id:
                return
            normalized_args = args if isinstance(args, dict) else args
            key = call_id or f"{name}:{json.dumps(normalized_args, sort_keys=True)}"
            if key in seen:
                return
            seen.add(key)
            entries.append(self._tool_call_entry(name, normalized_args, call_id))

        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == ContentBlockType.TOOL_USE:
                    _append_call(block.get("name"), block.get("input", {}), block.get("id"))

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    _append_call(tc.get("name"), tc.get("args", {}), tc.get("id"))
                else:
                    _append_call(
                        getattr(tc, "name", None),
                        getattr(tc, "arguments", {}),
                        getattr(tc, "id", None),
                    )

        return entries

    def _tool_call_entry(self, name: Any, args: Any, call_id: Optional[str] = None) -> dict[str, Any]:
        formatted_args = args if isinstance(args, dict) else args
        entry: dict[str, Any] = {
            "type": ConversationEntryType.TOOL_CALL,
            "tool": name or "unknown",
            "args": formatted_args,
        }
        if call_id:
            entry["id"] = call_id
        return entry

    def _format_tool_result(self, msg: Any) -> Optional[dict[str, Any]]:
        tool_name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", "")

        output = self._safe_json_load(content)
        entry = {
            "type": ConversationEntryType.TOOL_RESULT,
            "tool": tool_name or "unknown",
            "output": output,
        }
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        return entry

    def build_benchmark_response(self) -> list[dict[str, Any]]:
        """Create assistant/tool entries formatted for benchmark exports."""
        if not self.messages:
            return []

        entries: list[dict[str, Any]] = []
        iteration = 0
        idx = 0
        total = len(self.messages)

        while idx < total:
            msg = self.messages[idx]
            msg_type = getattr(msg, "type", None)

            if msg_type == MessageType.AI:
                iteration += 1
                content, reasoning_blocks = self._extract_ai_content(msg)
                assistant_entry: dict[str, Any] = {
                    "type": ConversationEntryType.ASSISTANT,
                    "iteration": iteration,
                }
                if content:
                    assistant_entry["content"] = content
                if reasoning_blocks:
                    assistant_entry["reasoning"] = reasoning_blocks

                tool_calls = self._collect_tool_calls(msg)
                if tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "type": ConversationEntryType.TOOL_CALL,
                            "name": tc.get("tool"),
                            "args": tc.get("args", {}),
                            "id": tc.get("id"),
                        }
                        for tc in tool_calls
                    ]

                entries.append(assistant_entry)

                tool_messages = []
                lookahead = idx + 1
                while lookahead < total:
                    next_msg = self.messages[lookahead]
                    if getattr(next_msg, "type", None) != MessageType.TOOL:
                        break
                    tool_messages.append(next_msg)
                    lookahead += 1

                if tool_messages:
                    entries.append(
                        {
                            "type": ConversationEntryType.TOOL_RESULTS,
                            "iteration": iteration,
                            "results": [
                                self._format_benchmark_tool_result(tool_msg)
                                for tool_msg in tool_messages
                            ],
                        }
                    )

                idx = lookahead
                continue

            idx += 1

        return entries

    def _format_benchmark_tool_result(self, tool_msg: Any) -> dict[str, Any]:
        content = getattr(tool_msg, "content", "")
        if isinstance(content, str):
            output = content
        else:
            output = json.dumps(content, ensure_ascii=False)

        additional_metadata = getattr(tool_msg, "additional_kwargs", {}) or {}
        is_error = additional_metadata.get("is_error")
        if isinstance(is_error, bool):
            success = not is_error
        else:
            success = self._is_serialized_json(output)

        return {
            "tool_name": getattr(tool_msg, "name", "unknown"),
            "tool_call_id": getattr(tool_msg, "tool_call_id", None),
            "success": success,
            "output": output,
        }

    def _is_serialized_json(self, value: str) -> bool:
        try:
            json.loads(value)
            return True
        except Exception:
            return False

    def _format_role_message(self, msg: Any, role: MessageRole) -> Optional[dict[str, Any]]:
        content = getattr(msg, "content", None)
        if not content:
            return None

        return {
            "type": ConversationEntryType.MESSAGE,
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
