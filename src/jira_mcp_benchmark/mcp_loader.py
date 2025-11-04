"""Helpers for loading MCP tools via langchain-mcp-adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from langchain_core.tools import BaseTool

from langchain_mcp_adapters.client import MultiServerMCPClient


@dataclass(slots=True)
class MCPConfig:
    """Runtime configuration for connecting to an MCP server."""

    name: str
    transport: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    headers: Optional[Mapping[str, str]] = None


async def load_tools_from_mcp(config: MCPConfig) -> list[BaseTool]:
    """Load LangChain-compatible tools from the configured MCP server.
    
    Note: The MCP client is not returned because it cannot be properly cleaned up
    (MultiServerMCPClient.cleanup() doesn't exist, __aexit__ raises NotImplementedError).
    Concurrency is controlled via semaphore in cli.py to prevent file descriptor exhaustion.
    """

    connection: MutableMapping[str, Any] = {"transport": config.transport}
    if config.url:
        connection["url"] = config.url
    if config.command:
        connection["command"] = config.command
    if config.args:
        connection["args"] = config.args
    if config.headers:
        connection["headers"] = dict(config.headers)

    client = MultiServerMCPClient({config.name: connection})
    return await client.get_tools()
