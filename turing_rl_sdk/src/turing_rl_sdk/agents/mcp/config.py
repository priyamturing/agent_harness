"""MCP configuration data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCPConfig:
    """Configuration for connecting to an MCP server.
    
    Specifies connection details for either HTTP/SSE or stdio transport.
    Must provide either 'url' for HTTP transport or 'command' for stdio transport.
    
    Attributes:
        name (str): Unique server name identifier. Required, cannot be empty.
        transport (str): Transport protocol type. Typically "streamable_http" 
            for HTTP/SSE or "stdio" for command-line MCP servers.
        url (Optional[str]): Server URL for HTTP/SSE transport, e.g., 
            "http://localhost:8015/mcp". Required if using HTTP transport.
        command (Optional[str]): Command to launch stdio MCP server, e.g., 
            "npx" or "python". Required if using stdio transport.
        args (Optional[list[str]]): Arguments for stdio command, e.g., 
            ["@modelcontextprotocol/server-filesystem", "/path/to/dir"].
        headers (dict[str, str]): HTTP headers for HTTP/SSE transport. Can 
            include auth tokens or database IDs.
    """

    name: str
    transport: str = "streamable_http"
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    headers: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration after dataclass initialization.
        
        Raises:
            ValueError: If neither url nor command is provided, or if name is empty.
        """
        if not self.url and not self.command:
            raise ValueError(
                f"MCPConfig for '{self.name}' requires either 'url' or 'command' to be set. "
                "Provide at least one connection method:\n"
                "  - url: for HTTP/SSE transport (e.g., 'http://localhost:8015/mcp')\n"
                "  - command: for stdio transport (e.g., 'npx', with args=['@modelcontextprotocol/server-filesystem'])"
            )
        
        if not self.name or not self.name.strip():
            raise ValueError(
                "MCPConfig.name cannot be empty. Provide a server name."
            )

