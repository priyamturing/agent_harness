"""Runtime context for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from .events import RunObserver


@dataclass
class RunContext:
    """Runtime context for agent execution.

    Centralizes:
    - Unique database ID per run
    - SQL runner URL for verifiers
    - Event observers for logging/telemetry
    
    Can be used as an async context manager for automatic cleanup:
    ```python
    async with RunContext(sql_runner_url=url) as ctx:
        # Resources are automatically cleaned up on exit
        await agent.run(task, run_context=ctx)
    ```
    """

    database_id: str = field(default_factory=lambda: str(uuid4()))
    sql_runner_url: Optional[str] = None
    observers: list[RunObserver] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_observer(self, observer: RunObserver) -> None:
        """Add an event observer."""
        self.observers.append(observer)

    def remove_observer(self, observer: RunObserver) -> None:
        """Remove an event observer."""
        if observer in self.observers:
            self.observers.remove(observer)

    async def notify_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Notify observers of a message."""
        for observer in self.observers:
            await observer.on_message(role, content, metadata)

    async def notify_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Notify observers of a tool call."""
        for observer in self.observers:
            await observer.on_tool_call(tool_name, arguments, result, is_error)

    async def notify_status(self, message: str, level: str = "info") -> None:
        """Notify observers of status updates."""
        for observer in self.observers:
            await observer.on_status(message, level)

    async def cleanup(self) -> None:
        """Cleanup resources (no-op placeholder)."""
        return None
    
    async def __aenter__(self) -> "RunContext":
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, ensuring cleanup."""
        await self.cleanup()

