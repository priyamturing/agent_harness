"""Task definition for SDK users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..mcp import MCPConfig


@dataclass
class Task:
    """User-facing task definition.
    
    Task defines the execution configuration for an agent.
    """

    prompt: str
    mcps: list["MCPConfig"]
    max_steps: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    conversation_mode: bool = False

