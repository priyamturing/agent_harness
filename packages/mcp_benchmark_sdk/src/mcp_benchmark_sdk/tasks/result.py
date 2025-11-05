"""Result and response data structures for agent execution."""

from __future__ import annotations

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

