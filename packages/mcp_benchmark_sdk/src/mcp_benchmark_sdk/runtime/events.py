"""Observer pattern for agent execution events."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..tasks import AgentResponse, ToolCall, ToolResult


class RunObserver(ABC):
    """Observer interface for agent execution events."""

    @abstractmethod
    async def on_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Called when a message is added to the conversation.

        Args:
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Additional metadata (reasoning traces, etc.)
        """

    @abstractmethod
    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Called when a tool is invoked.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
            is_error: Whether the tool call resulted in an error
        """

    @abstractmethod
    async def on_verifier_update(self, verifier_results: list[Any]) -> None:
        """Called when verifiers are executed.

        Args:
            verifier_results: List of verifier results
        """

    @abstractmethod
    async def on_status(self, message: str, level: str = "info") -> None:
        """Called for status updates.

        Args:
            message: Status message
            level: Log level (info, warning, error)
        """


class NoOpObserver(RunObserver):
    """No-op observer that does nothing."""

    async def on_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        pass

    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        pass

    async def on_verifier_update(self, verifier_results: list[Any]) -> None:
        pass

    async def on_status(self, message: str, level: str = "info") -> None:
        pass

