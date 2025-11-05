"""MCP client management for connecting to multiple servers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MCPConfig
from .tool_fixer import fix_tool_schemas


class MCPClientManager:
    """Manager for MCP client connections.

    Handles:
    - Connecting to multiple MCP servers
    - Tool retrieval from servers
    - Adding x-database-id header when needed
    """

    def __init__(self):
        self._client: Optional[MultiServerMCPClient] = None
        self._configs: dict[str, MCPConfig] = {}
        self._tools: list[BaseTool] = []

    async def connect(
        self,
        configs: list[MCPConfig],
        database_id: Optional[str] = None,
    ) -> None:
        """Connect to multiple MCP servers.

        Args:
            configs: List of MCP configurations
            database_id: Optional database ID to inject into headers
        """
        if not configs:
            raise ValueError("At least one MCP configuration required")

        # Build server configurations
        server_configs: dict[str, Mapping[str, Any]] = {}

        for config in configs:
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

            server_configs[config.name] = connection
            self._configs[config.name] = config

        # Create and connect client
        # Type ignore: server_configs is structurally compatible with Connection type
        self._client = MultiServerMCPClient(server_configs)  # type: ignore[arg-type]
        raw_tools = await self._client.get_tools()
        
        # Fix tools with reserved keyword parameters
        self._tools = fix_tool_schemas(raw_tools)

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
        """Cleanup MCP connections.

        Note: MultiServerMCPClient doesn't provide cleanup mechanism,
        so this is a no-op for now. Concurrency controlled via semaphore.
        """
        # MultiServerMCPClient doesn't have cleanup() or __aexit__
        # Connections will be cleaned up when object is garbage collected
        pass

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._client is not None

    @property
    def server_names(self) -> list[str]:
        """Get list of connected server names."""
        return list(self._configs.keys())

