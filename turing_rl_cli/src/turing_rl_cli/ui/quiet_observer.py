"""Quiet observer - minimal output, no streaming logs."""

from typing import Any, Optional

from turing_rl_sdk.agents.runtime import RunObserver


class QuietObserver(RunObserver):
    """Observer that produces no output - for quiet/progress-only mode."""

    async def on_message(self, role: str, content: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """Suppress message output."""
        pass

    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Suppress tool call output."""
        pass

    async def on_status(self, message: str, level: str = "info") -> None:
        """Suppress status output."""
        pass

