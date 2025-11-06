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
        try:
            # Type ignore: server_configs is structurally compatible with Connection type
            self._client = MultiServerMCPClient(server_configs)  # type: ignore[arg-type]
            raw_tools = await self._client.get_tools()
            
            # Fix tools with reserved keyword parameters
            self._tools = fix_tool_schemas(raw_tools)
        except Exception as exc:
            # Extract root cause from exception chain/groups
            root_cause = exc
            
            # Check for ExceptionGroup (Python 3.11+)
            if hasattr(exc, 'exceptions'):
                # Get first exception from group
                if exc.exceptions:  # type: ignore[attr-defined]
                    root_cause = exc.exceptions[0]  # type: ignore[attr-defined]
            
            # Walk the exception chain to find root cause
            while root_cause.__cause__ is not None:
                root_cause = root_cause.__cause__
            
            # Provide clear error message for connection failures
            mcp_urls = [cfg.url for cfg in configs if cfg.url]
            mcp_commands = [cfg.command for cfg in configs if cfg.command]
            
            error_parts = ["Failed to connect to MCP server(s):"]
            if mcp_urls:
                error_parts.append(f"  URLs: {', '.join(mcp_urls)}")
            if mcp_commands:
                error_parts.append(f"  Commands: {', '.join(mcp_commands)}")
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

