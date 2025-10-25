"""Helper classes for directing run output."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Protocol

from rich.console import Console


class RunLogger(Protocol):
    """Minimal interface used by the scenario executor."""

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        ...

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        ...


@dataclass
class ConsoleRunLogger:
    """Run logger that writes directly to a Rich console."""

    console: Console
    prefix: Optional[str] = None

    def _emit_prefix(self) -> None:
        if self.prefix:
            self.console.print(f"[bold][{self.prefix}][/bold]")

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._emit_prefix()
        self.console.print(*args, **kwargs)

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._emit_prefix()
        self.console.rule(*args, **kwargs)


class TextualRunLogger:
    """Run logger that enqueues rendered strings for the Textual UI."""

    def __init__(self, queue: asyncio.Queue[str], width: int = 100) -> None:
        self._queue = queue
        self._capture_console = Console(
            color_system=None,
            force_terminal=False,
            record=True,
            width=width,
        )

    def _enqueue_render(self) -> None:
        text = self._capture_console.export_text(clear=True)
        if not text.endswith("\n"):
            text = f"{text}\n"
        self._queue.put_nowait(text)

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._capture_console.print(*args, **kwargs)
        self._enqueue_render()

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._capture_console.rule(*args, **kwargs)
        self._enqueue_render()
