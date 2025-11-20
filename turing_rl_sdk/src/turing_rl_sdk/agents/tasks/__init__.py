"""Task and result data structures."""

from .task import Task
from .result import (
    Result,
    AgentResponse,
    ToolCall,
    ToolResult,
    MessageType,
    ConversationEntryType,
    ContentBlockType,
)

__all__ = [
    "Task",
    "Result",
    "AgentResponse",
    "ToolCall",
    "ToolResult",
    "MessageType",
    "ConversationEntryType",
    "ContentBlockType",
]

