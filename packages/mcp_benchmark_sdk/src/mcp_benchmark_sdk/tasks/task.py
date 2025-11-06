"""Task definition for SDK users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ..constants import DEFAULT_MAX_STEPS

if TYPE_CHECKING:
    from ..mcp import MCPConfig


@dataclass
class Task:
    """User-facing task definition.
    
    Task defines the execution configuration for an agent.
    """

    prompt: str
    mcp: "MCPConfig"
    max_steps: int = DEFAULT_MAX_STEPS
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    conversation_mode: bool = False

    def __post_init__(self) -> None:
        """Validate task fields after initialization."""
        if not self.prompt or not self.prompt.strip():
            raise ValueError(
                "Task.prompt cannot be empty. Provide a non-empty prompt string."
            )
        
        if not self.mcp:
            raise ValueError(
                "Task.mcp cannot be None. Provide an MCPConfig instance."
            )
        
        if self.max_steps <= 0:
            raise ValueError(
                f"Task.max_steps must be positive, got {self.max_steps}. "
                "Provide a positive integer for max_steps."
            )

