"""Task definition for SDK users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..mcp import MCPConfig
    from ..verifiers import Verifier


@dataclass
class Task:
    """User-facing task definition (HUD-style)."""

    prompt: str
    mcps: list["MCPConfig"]
    verifiers: list["Verifier"] = field(default_factory=list)
    max_steps: int = 1000
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    conversation_mode: bool = False

