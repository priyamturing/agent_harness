"""Textual user interface for displaying parallel run output."""

from __future__ import annotations

import asyncio
import re
from asyncio import QueueEmpty
from typing import Dict, Iterable, Union

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
        self._id_map = self._build_id_map(self._run_labels)
        super().__init__()

    @staticmethod
    def _build_id_map(labels: Iterable[str]) -> Dict[str, str]:
        def sanitize(value: str) -> str:
            cleaned = re.sub(r"[^0-9A-Za-z_-]+", "-", value).strip("-")
            cleaned = re.sub(r"-{2,}", "-", cleaned)
            if not cleaned:
                cleaned = "run"
            if cleaned[0].isdigit():
                cleaned = f"run-{cleaned}"
            return cleaned

        id_map: Dict[str, str] = {}
        seen: Dict[str, int] = {}
        for label in labels:
            base = sanitize(label)
            idx = seen.get(base, 0)
            seen[base] = idx + 1
            safe = f"{base}-{idx}" if idx else base
            id_map[label] = safe
        return id_map

    def compose(self) -> ComposeResult:
        with TabbedContent():
            for label in self._run_labels:
                safe = self._id_map[label]
                with TabPane(f"Run {label}", id=f"tab-{safe}"):
                    with Horizontal():
                        yield RichLog(
                            id=f"log-{safe}",
                            auto_scroll=True,
                            highlight=False,
                        )
                        table = DataTable(
                            id=f"status-{safe}",
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
            safe = self._id_map[label]
            log_widget = self.query_one(f"#log-{safe}", RichLog)
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
                        table = self.query_one(f"#status-{safe}", DataTable)
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
