"""Observer pattern for agent execution events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional




class RunObserver(ABC):
    """Abstract base class for observing agent execution events.
    
    Implement this interface to receive notifications about agent execution
    events such as messages, tool calls, and status updates. Useful for:
    - Custom logging implementations
    - Telemetry and monitoring
    - Real-time UI updates
    - Debugging and development tools
    """

    @abstractmethod
    async def on_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Called when a message is added to the conversation.

        Args:
            role (str): Message role - one of "system", "user", "assistant", "tool".
            content (str): The message content/text.
            metadata (Optional[dict[str, Any]]): Additional metadata such as 
                reasoning traces, timestamps, or custom fields.
        """

    @abstractmethod
    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Called when a tool is invoked by the agent.

        Args:
            tool_name (str): Name of the tool that was called.
            arguments (dict[str, Any]): Arguments passed to the tool as key-value pairs.
            result (Any): The result returned by the tool execution. Could be 
                a string, dict, list, or error message.
            is_error (bool): Whether the tool call resulted in an error.
        """

    @abstractmethod
    async def on_status(self, message: str, level: str = "info") -> None:
        """Called for status updates during execution.

        Args:
            message (str): Human-readable status message.
            level (str): Log level indicating severity - "info", "warning", or "error".
        """


class NoOpObserver(RunObserver):
    """No-op observer implementation that ignores all events.
    
    Useful as a default observer or for testing when you want to disable
    event notifications without changing code structure.
    """

    async def on_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Ignore message events."""
        pass

    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Ignore tool call events."""
        pass

    async def on_status(self, message: str, level: str = "info") -> None:
        """Ignore status update events."""
        pass

