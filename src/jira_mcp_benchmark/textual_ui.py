"""Textual user interface for displaying parallel run output."""

from __future__ import annotations

from typing import Dict, Iterable

from asyncio import QueueEmpty

import asyncio

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, RichLog, TabbedContent, TabPane


class MultiRunApp(App[None]):
    """Simple Textual app that renders log panes for each run."""

    CSS = """
    TabbedContent {
        height: 1fr;
    }
    TextLog {
        height: 1fr;
    }
    """

    def __init__(self, run_labels: Iterable[str], queues: Dict[str, asyncio.Queue[str]]) -> None:
        self._run_labels = list(run_labels)
        self._queues = queues
        super().__init__()

    def compose(self) -> ComposeResult:
        with TabbedContent():
            for label in self._run_labels:
                with TabPane(f"Run {label}", id=f"tab-{label}"):
                    with Vertical():
                        yield RichLog(
                            id=f"log-{label}",
                            auto_scroll=True,
                            highlight=False,
                        )
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(0.1, self._drain_all_queues)

    def _drain_all_queues(self) -> None:
        for label in self._run_labels:
            log_widget = self.query_one(f"#log-{label}", RichLog)
            queue = self._queues[label]
            while True:
                try:
                    message = queue.get_nowait()
                except QueueEmpty:
                    break
                else:
                    log_widget.write(message)
