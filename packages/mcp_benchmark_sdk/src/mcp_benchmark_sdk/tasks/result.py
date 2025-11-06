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
    """Final execution result from running a task."""

    success: bool
    messages: list[Any]
    verifier_results: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    reasoning_traces: list[str] = field(default_factory=list)
    error: Optional[str] = None
    langsmith_url: Optional[str] = None  # LangSmith trace URL if tracing is enabled

    def get_conversation_history(self) -> list[dict[str, Any]]:
        """Extract conversation history in user-friendly format.
        
        Converts LangChain messages to a clean format:
        - Messages: {"type": "message", "role": "user|assistant|system", "content": "...", "reasoning": [...]}
        - Tool calls: {"type": "tool_call", "tool": "tool_name", "args": {...}}
        - Tool results: {"type": "tool_result", "tool": "tool_name", "output": {...}}
        
        Returns:
            List of conversation entries in clean format
        """
        if not self.messages:
            return []
        
        conversation = []
        for msg in self.messages:
            msg_type = msg.type if hasattr(msg, "type") else "unknown"
            
            # Handle AI messages (may have complex content structure)
            if msg_type == "ai":
                content_text = ""
                reasoning_blocks = []
                tool_uses = []
                
                # Parse complex content structure if it's a list
                if hasattr(msg, "content") and isinstance(msg.content, list):
                    for block in msg.content:
                        if isinstance(block, dict):
                            if block.get("type") == "thinking":
                                reasoning_blocks.append(block.get("thinking", ""))
                            elif block.get("type") == "text":
                                content_text = block.get("text", "")
                            elif block.get("type") == "tool_use":
                                tool_uses.append({
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "input": block.get("input", {}),
                                })
                elif hasattr(msg, "content") and isinstance(msg.content, str):
                    content_text = msg.content
                
                # Add AI message if it has content or reasoning
                if content_text or reasoning_blocks:
                    entry = {
                        "type": "message",
                        "role": "assistant",
                    }
                    if content_text:
                        entry["content"] = content_text
                    if reasoning_blocks:
                        entry["reasoning"] = reasoning_blocks
                    conversation.append(entry)
                
                # Add tool calls from content blocks or tool_calls attribute
                tool_calls_to_add = tool_uses if tool_uses else []
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_to_add.append({
                            "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                            "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "arguments", {}),
                        })
                
                for tc in tool_calls_to_add:
                    conversation.append({
                        "type": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "args": tc.get("input") if "input" in tc else tc.get("args", {}),
                    })
            
            # Handle tool result messages
            elif msg_type == "tool":
                tool_name = msg.name if hasattr(msg, "name") else "unknown"
                content = msg.content if hasattr(msg, "content") else ""
                
                # Try to parse JSON output
                try:
                    output = json.loads(content) if isinstance(content, str) else content
                except (json.JSONDecodeError, TypeError):
                    output = content
                
                conversation.append({
                    "type": "tool_result",
                    "tool": tool_name,
                    "output": output,
                })
            
            # Handle human messages
            elif msg_type == "human":
                if hasattr(msg, "content") and msg.content:
                    conversation.append({
                        "type": "message",
                        "role": "user",
                        "content": msg.content,
                    })
            
            # Handle system messages
            elif msg_type == "system":
                if hasattr(msg, "content") and msg.content:
                    conversation.append({
                        "type": "message",
                        "role": "system",
                        "content": msg.content,
                    })
        
        return conversation

