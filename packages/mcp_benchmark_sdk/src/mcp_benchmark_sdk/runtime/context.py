"""Runtime context for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import httpx

from .events import NoOpObserver, RunObserver

# HTTP client timeout configuration (used for database verifiers)
_DEFAULT_HTTP_TIMEOUT = 30.0  # Overall timeout in seconds
_DEFAULT_HTTP_CONNECT_TIMEOUT = 10.0  # Connection timeout in seconds


@dataclass
class RunContext:
    """Runtime context for agent execution.

    Centralizes:
    - Unique database ID per run
    - SQL runner URL for verifiers
    - Shared HTTP client
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
    http_client: Optional[httpx.AsyncClient] = None
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

    async def notify_verifier_update(self, verifier_results: list[Any]) -> None:
        """Notify observers of verifier results."""
        for observer in self.observers:
            await observer.on_verifier_update(verifier_results)

    async def notify_status(self, message: str, level: str = "info") -> None:
        """Notify observers of status updates."""
        for observer in self.observers:
            await observer.on_status(message, level)

    def get_http_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.
        
        Returns:
            Active httpx.AsyncClient instance
            
        Note:
            If the client was closed externally without being set to None,
            this method will detect it and create a new client.
        """
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(_DEFAULT_HTTP_TIMEOUT, connect=_DEFAULT_HTTP_CONNECT_TIMEOUT)
            )
        elif self.http_client.is_closed:
            # Client was closed externally without being set to None
            # Recreate it instead of returning closed client
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(_DEFAULT_HTTP_TIMEOUT, connect=_DEFAULT_HTTP_CONNECT_TIMEOUT)
            )
        return self.http_client

    async def cleanup(self) -> None:
        """Cleanup resources (HTTP client, etc.)."""
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None
    
    async def __aenter__(self) -> "RunContext":
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, ensuring cleanup."""
        await self.cleanup()

