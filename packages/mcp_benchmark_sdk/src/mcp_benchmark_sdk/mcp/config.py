"""MCP configuration data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCPConfig:
    """Configuration for connecting to an MCP server."""

    name: str
    transport: str = "streamable_http"
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[list[str]] = None
    headers: dict[str, str] = field(default_factory=dict)

