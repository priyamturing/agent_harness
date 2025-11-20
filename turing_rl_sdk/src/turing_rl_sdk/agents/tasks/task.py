"""Task definition for SDK users."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ..constants import DEFAULT_MAX_STEPS

if TYPE_CHECKING:
    from ..mcp import MCPConfig


@dataclass
class Task:
    """User-facing task definition specifying what an agent should execute.
    
    Combines a prompt with MCP server configuration and execution parameters.
    Used as input to `agent.run(task)`.
    
    Attributes:
        prompt (str): The user prompt/instruction for the agent. Must be non-empty.
        mcp (MCPConfig): MCP server configuration specifying how to connect 
            to the MCP server and what tools are available.
        max_steps (int): Maximum number of agent turns/iterations before 
            stopping. Defaults to DEFAULT_MAX_STEPS.
        metadata (dict[str, Any]): Additional metadata for this task, such as 
            scenario_name, run_number, session_id for tracing and logging.
        database_id (Optional[str]): Optional database ID for multi-tenant 
            scenarios. Injected as 'x-database-id' header to MCP server.
        conversation_mode (bool): Whether this task uses conversation mode. 
            Currently unused as multi-turn is not fully supported.
    """

    prompt: str
    mcp: "MCPConfig"
    max_steps: int = DEFAULT_MAX_STEPS
    metadata: dict[str, Any] = field(default_factory=dict)
    database_id: Optional[str] = None
    conversation_mode: bool = False

    def __post_init__(self) -> None:
        """Validate task fields after dataclass initialization.
        
        Raises:
            ValueError: If prompt is empty, mcp is None, or max_steps is non-positive.
        """
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

