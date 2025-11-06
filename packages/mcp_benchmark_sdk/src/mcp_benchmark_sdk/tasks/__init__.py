"""Task and result data structures."""

from .task import Task
from .result import Result, AgentResponse, ToolCall, ToolResult

__all__ = [
    "Task",
    "Result",
    "AgentResponse",
    "ToolCall",
    "ToolResult",
]

