"""Textual user interface for displaying parallel run output."""

from __future__ import annotations

from typing import Dict, Iterable, Union

from asyncio import QueueEmpty

import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, RichLog, TabbedContent, TabPane


class MultiRunApp(App[None]):
    """Simple Textual app that renders log panes for each run."""

    CSS = """
    TabbedContent, Horizontal {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
    }
    DataTable {
        width: 48;
    }
    """

    def __init__(self, run_labels: Iterable[str], queues: Dict[str, asyncio.Queue[object]]) -> None:
        self._run_labels = list(run_labels)
        self._queues = queues
        super().__init__()

    def compose(self) -> ComposeResult:
        with TabbedContent():
            for label in self._run_labels:
                with TabPane(f"Run {label}", id=f"tab-{label}"):
                    with Horizontal():
                        yield RichLog(
                            id=f"log-{label}",
                            auto_scroll=True,
                            highlight=False,
                        )
                        table = DataTable(
                            id=f"status-{label}",
                            zebra_stripes=True,
                            show_header=True,
                        )
                        table.add_columns("Verifier", "Comparison", "Expected", "Actual", "Status", "Error")
                        yield table
        yield Footer()

    async def on_mount(self) -> None:
        self.set_interval(0.1, self._drain_all_queues)

    def _drain_all_queues(self) -> None:
        for label in self._run_labels:
            log_widget = self.query_one(f"#log-{label}", RichLog)
            queue = self._queues[label]
            while True:
                try:
                    message: Union[str, dict] = queue.get_nowait()
                except QueueEmpty:
                    break
                else:
                    if isinstance(message, str):
                        log_widget.write(message)
                    elif isinstance(message, dict) and message.get("type") == "status":
                        table = self.query_one(f"#status-{label}", DataTable)
                        table.clear()
                        for row in message.get("data", []):
                            status = Text(row["status"], style="green" if row["status"] == "PASS" else "red")
                            error_text = Text(row["error"] or "", style="dim")
                            table.add_row(
                                row["name"],
                                row["comparison"],
                                row["expected"],
                                row["actual"],
                                status,
                                error_text,
                            )
                    else:
                        log_widget.write(str(message))
