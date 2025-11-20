"""MCP client management for connecting to a single server."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..constants import RETRY_DEFAULT_MAX_ATTEMPTS
from ..utils import retry_with_backoff
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
        """Connect to MCP server and load all available tools.

        Args:
            config (MCPConfig): MCP server configuration specifying transport 
                type (SSE/stdio), connection details (URL/command), and headers.
            database_id (Optional[str]): Optional database ID to inject as 
                'x-database-id' header. Used for multi-tenant scenarios where 
                the MCP server needs to know which database to operate on.
                
        Raises:
            ValueError: If config is None or empty.
            ConnectionError: If connection to MCP server fails. Includes detailed 
                error message with server details and root cause.
        """

        connection: dict[str, Any] = {"transport": config.transport}

        if config.url:
            connection["url"] = config.url
        if config.command:
            connection["command"] = config.command
        if config.args:
            connection["args"] = config.args

        headers = dict(config.headers) if config.headers else {}
        if database_id and "x-database-id" not in headers:
            headers["x-database-id"] = database_id

        if headers:
            connection["headers"] = headers

        server_configs: dict[str, Mapping[str, Any]] = {config.name: connection}
        self._config = config

        async def _attempt_connect() -> None:
            self._client = MultiServerMCPClient(server_configs)  # type: ignore[arg-type]
            raw_tools = await self._client.get_tools()
            self._tools = fix_tool_schemas(raw_tools)

        try:
            await retry_with_backoff(
                _attempt_connect,
                max_retries=RETRY_DEFAULT_MAX_ATTEMPTS,
            )
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
        """Cleanup MCP connection resources.

        Note: MultiServerMCPClient from langchain-mcp-adapters doesn't currently 
        provide a cleanup() or __aexit__() method, so this is a no-op. The 
        connection will be cleaned up when the object is garbage collected. 
        Concurrency is controlled via semaphore elsewhere in the SDK.
        """
        # MultiServerMCPClient doesn't have cleanup() or __aexit__
        # Connection will be cleaned up when object is garbage collected
        pass

    @property
    def is_connected(self) -> bool:
        """Check if MCP client is connected.
        
        Returns:
            bool: True if connect() has been called successfully and client 
                is initialized, False otherwise.
        """
        return self._client is not None

    @property
    def server_name(self) -> Optional[str]:
        """Get the name of the connected MCP server.
        
        Returns:
            Optional[str]: Server name from MCPConfig if connected, None otherwise.
        """
        return self._config.name if self._config else None
