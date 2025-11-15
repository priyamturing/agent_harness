"""Runtime context for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from .events import RunObserver


@dataclass
class RunContext:
    """Runtime context for agent execution with telemetry and state management.

    Provides centralized management of:
    - Unique database ID for multi-tenant scenarios
    - Event observers for logging, telemetry, and monitoring
    - Metadata for execution context
    
    Can be used as an async context manager for automatic cleanup:
    ```python
    async with RunContext() as ctx:
        # Resources are automatically cleaned up on exit
        await agent.run(task, run_context=ctx)
    ```
    
    Attributes:
        database_id (str): Unique identifier for this execution run, generated 
            automatically as UUID if not provided.
        observers (list[RunObserver]): List of observers that will be notified 
            of execution events (messages, tool calls, status updates).
        metadata (dict[str, Any]): Additional metadata for this execution context.
    """

    database_id: str = field(default_factory=lambda: str(uuid4()))
    observers: list[RunObserver] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observer(self, observer: RunObserver) -> None:
        """Add an event observer to receive execution notifications.
        
        Args:
            observer (RunObserver): Observer instance to add to the list.
        """
        self.observers.append(observer)

    def remove_observer(self, observer: RunObserver) -> None:
        """Remove an event observer from the notification list.
        
        Args:
            observer (RunObserver): Observer instance to remove.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    async def notify_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Notify all observers of a new message in the conversation.
        
        Args:
            role (str): Message role - "system", "user", "assistant", or "tool".
            content (str): The message content/text.
            metadata (Optional[dict[str, Any]]): Additional metadata like 
                reasoning traces, timestamps, etc.
        """
        for observer in self.observers:
            await observer.on_message(role, content, metadata)

    async def notify_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Notify all observers of a tool invocation.
        
        Args:
            tool_name (str): Name of the tool that was called.
            arguments (dict[str, Any]): Arguments passed to the tool.
            result (Any): The result returned by the tool execution.
            is_error (bool): Whether the tool call resulted in an error.
        """
        for observer in self.observers:
            await observer.on_tool_call(tool_name, arguments, result, is_error)

    async def notify_status(self, message: str, level: str = "info") -> None:
        """Notify all observers of a status update.
        
        Args:
            message (str): Status message describing the update.
            level (str): Log level - "info", "warning", or "error".
        """
        for observer in self.observers:
            await observer.on_status(message, level)

    async def cleanup(self) -> None:
        """Cleanup resources (no-op placeholder for future use).
        
        Currently a no-op, but can be extended to cleanup HTTP clients,
        database connections, or other resources in the future.
        """
        return None
    
    async def __aenter__(self) -> "RunContext":
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, ensuring cleanup."""
        await self.cleanup()

