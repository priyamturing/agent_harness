"""MCP client management for connecting to a single server."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MCPConfig
from .tool_fixer import fix_tool_schemas


class MCPClientManager:
    """Manager for MCP client connection.

    Handles:
    - Connecting to an MCP server
    - Tool retrieval from server
    - Adding x-database-id header when needed
    """

    def __init__(self):
        self._client: Optional[MultiServerMCPClient] = None
        self._config: Optional[MCPConfig] = None
        self._tools: list[BaseTool] = []

    async def connect(
        self,
        config: MCPConfig,
        database_id: Optional[str] = None,
    ) -> None:
        """Connect to MCP server.

        Args:
            config: MCP configuration
            database_id: Optional database ID to inject into headers
        """
        if not config:
            raise ValueError("MCP configuration required")

        # Build server configuration
        connection: dict[str, Any] = {"transport": config.transport}

        if config.url:
            connection["url"] = config.url
        if config.command:
            connection["command"] = config.command
        if config.args:
            connection["args"] = config.args

        # Merge headers with database_id
        headers = dict(config.headers) if config.headers else {}
        if database_id and "x-database-id" not in headers:
            headers["x-database-id"] = database_id

        if headers:
            connection["headers"] = headers

        server_configs: dict[str, Mapping[str, Any]] = {config.name: connection}
        self._config = config

        # Create and connect client
        try:
            self._client = MultiServerMCPClient(server_configs)  # type: ignore[arg-type]
            raw_tools = await self._client.get_tools()
            
            # Fix tools with reserved keyword parameters
            self._tools = fix_tool_schemas(raw_tools)
        except Exception as exc:
            # Extract root cause from exception chain/groups
            root_cause = exc
            
            # Check for ExceptionGroup (Python 3.11+)
            if hasattr(exc, 'exceptions'):
                if exc.exceptions:  # type: ignore[attr-defined]
                    root_cause = exc.exceptions[0]  # type: ignore[attr-defined]
            
            while root_cause.__cause__ is not None:
                root_cause = root_cause.__cause__
            
            error_parts = ["Failed to connect to MCP server:"]
            if config.url:
                error_parts.append(f"  URL: {config.url}")
            if config.command:
                error_parts.append(f"  Command: {config.command}")
            error_parts.append(f"  Cause: {type(root_cause).__name__}: {root_cause}")
            error_parts.append("")
            error_parts.append("  Make sure the MCP server is running and accessible.")
            
            raise ConnectionError("\n".join(error_parts)) from exc

    def get_all_tools(self) -> list[BaseTool]:
        """Get all tools from all connected servers.
        
        Returns:
            List of LangChain tools from all MCP servers
            
        Raises:
            RuntimeError: If client not connected
        """
        if not self._client:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        return self._tools

    async def cleanup(self) -> None:
        """Cleanup MCP connection.

        Note: MultiServerMCPClient doesn't provide cleanup mechanism,
        so this is a no-op for now. Concurrency controlled via semaphore.
        """
        # MultiServerMCPClient doesn't have cleanup() or __aexit__
        # Connection will be cleaned up when object is garbage collected
        pass

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._client is not None

    @property
    def server_name(self) -> Optional[str]:
        """Get connected server name."""
        return self._config.name if self._config else None

