"""Result and response data structures for agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    """Represents a tool invocation requested by the agent."""

    name: str
    arguments: dict[str, Any]
    id: Optional[str] = None


@dataclass
class ToolResult:
    """Result from executing a tool call."""

    content: str
    tool_call_id: str
    is_error: bool = False
    structured_content: Optional[dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Single agent turn response."""

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Final execution result from running a task.
    
    This represents the agent's execution result only.
    Verifier results are managed separately by the harness layer.
    """

    success: bool
    messages: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    reasoning_traces: list[str] = field(default_factory=list)
    error: Optional[str] = None
    langsmith_url: Optional[str] = None  # LangSmith trace URL if tracing is enabled

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Return conversation entries derived from LangChain messages."""
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

