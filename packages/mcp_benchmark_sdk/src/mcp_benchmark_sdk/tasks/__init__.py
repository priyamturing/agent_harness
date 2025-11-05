"""Task and result data structures."""

from .task import Task
from .scenario import Scenario, ScenarioPrompt
from .result import Result, AgentResponse, ToolCall, ToolResult

__all__ = [
    "Task",
    "Scenario",
    "ScenarioPrompt",
    "Result",
    "AgentResponse",
    "ToolCall",
    "ToolResult",
]

