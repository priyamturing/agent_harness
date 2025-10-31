"""Textual live viewer for background runs."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from . import background_manager


class BackgroundRunViewer(App[Optional[Path]]):
    """Live viewer for in-progress background runs with auto-switch to replay."""

    CSS = """
    #status-panel {
        height: auto;
        padding: 1;
        background: $panel;
        margin-bottom: 1;
    }
    
    #verifier-panel {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }
    
    #log-panel {
        height: 1fr;
    }
    
    RichLog {
        height: 1fr;
    }
    
    DataTable {
        height: auto;
        max-height: 15;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "force_replay", "Force Replay Mode"),
    ]

    def __init__(self, run_id: str, session_dir: Path) -> None:
        self.run_id = run_id
        self.session_dir = session_dir
        self._file_positions: dict[Path, int] = {}
        self._completed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Static("", id="status-panel")
            with Vertical(id="verifier-panel"):
                table = DataTable(
                    id="verifier-table",
                    zebra_stripes=True,
                    show_header=True,
                )
                table.add_columns("Verifier", "Comparison", "Expected", "Actual", "Status", "Error")
                yield table
            with Vertical(id="log-panel"):
                yield RichLog(
                    id="live-log",
                    auto_scroll=True,
                    highlight=True,
                )
        yield Footer()

    async def on_mount(self) -> None:
        """Start the polling and log tailing tasks."""
        self.set_interval(0.5, self._poll_state)
        self.set_interval(0.2, self._tail_logs)
        self.set_interval(0.5, self._update_verifiers)
        await self._update_status()

    async def _poll_state(self) -> None:
        """Poll the background run state and check if completed."""
        run_state = background_manager.read_run_state(self.run_id)
        if not run_state:
            await self._update_status_text("[red]Run not found[/red]")
            return

        await self._update_status_text(
            f"[bold]Run ID:[/bold] {self.run_id}\n"
            f"[bold]Status:[/bold] {run_state.status}\n"
            f"[bold]Progress:[/bold] {run_state.progress['passed']}/{run_state.progress['total']} verifiers passed\n"
            f"[bold]Session:[/bold] {run_state.session_dir}"
        )

        # Check if completed
        if run_state.status in ("completed", "failed") and not self._completed:
            self._completed = True
            log = self.query_one("#live-log", RichLog)
            
            if run_state.status == "completed":
                log.write("\n")
                log.write(Text("╭" + "─" * 78 + "╮", style="green"))
                log.write(Text("│" + " " * 20 + "🎉 Background Run Completed! 🎉" + " " * 26 + "│", style="bold green"))
                log.write(Text("│" + " " * 78 + "│", style="green"))
                log.write(Text("│  Switching to replay mode in 3 seconds...                                    │", style="green"))
                log.write(Text("│  Press 'r' to switch immediately or 'q' to quit                              │", style="green"))
                log.write(Text("╰" + "─" * 78 + "╯", style="green"))
            else:
                error_msg = run_state.error or "Unknown error"
                log.write("\n")
                log.write(Text("╭" + "─" * 78 + "╮", style="red"))
                log.write(Text("│" + " " * 22 + "❌ Background Run Failed ❌" + " " * 29 + "│", style="bold red"))
                log.write(Text("│" + " " * 78 + "│", style="red"))
                log.write(Text(f"│  Error: {error_msg[:67]:<67}│", style="red"))
                log.write(Text("│" + " " * 78 + "│", style="red"))
                log.write(Text("│  Press 'q' to quit                                                           │", style="red"))
                log.write(Text("╰" + "─" * 78 + "╯", style="red"))
            
            # Auto-switch to replay mode after 3 seconds if completed successfully
            if run_state.status == "completed":
                await asyncio.sleep(3)
                self.exit(self.session_dir)

    async def _update_status(self) -> None:
        """Update the status panel."""
        run_state = background_manager.read_run_state(self.run_id)
        if not run_state:
            await self._update_status_text("[red]Run not found[/red]")
            return

        await self._update_status_text(
            f"[bold]Run ID:[/bold] {self.run_id}\n"
            f"[bold]Status:[/bold] {run_state.status}\n"
            f"[bold]Progress:[/bold] {run_state.progress['passed']}/{run_state.progress['total']} verifiers passed\n"
            f"[bold]Session:[/bold] {run_state.session_dir}"
        )

    async def _update_status_text(self, text: str) -> None:
        """Update the status panel text."""
        status = self.query_one("#status-panel", Static)
        status.update(text)
    
    async def _update_verifiers(self) -> None:
        """Load artifact files and update verifier table."""
        artifact_files = list(self.session_dir.glob("run_*.json"))
        if not artifact_files:
            return
        
        # Aggregate all verifier results from all runs
        all_verifiers: list[dict] = []
        for artifact_file in artifact_files:
            try:
                artifact = json.loads(artifact_file.read_text(encoding="utf-8"))
                status_stream = artifact.get("status_stream", [])
                if status_stream:
                    # Use the latest status snapshot
                    all_verifiers.extend(status_stream[-1])
            except (json.JSONDecodeError, FileNotFoundError, IndexError):
                continue
        
        if not all_verifiers:
            return
        
        # Update the DataTable
        table = self.query_one("#verifier-table", DataTable)
        table.clear()
        
        for item in all_verifiers:
            status_text = Text(item.get("status", ""), style="green" if item.get("status") == "PASS" else "red")
            error_text = Text(item.get("error") or "", style="dim")
            table.add_row(
                item.get("name", ""),
                item.get("comparison", ""),
                item.get("expected", ""),
                item.get("actual", ""),
                status_text,
                error_text,
            )

    async def _tail_logs(self) -> None:
        """Tail log files and append new content to the log widget."""
        log_files = list(self.session_dir.glob("run_*.log"))
        log = self.query_one("#live-log", RichLog)

        for log_file in log_files:
            if not log_file.exists():
                continue

            # Initialize file position if new
            if log_file not in self._file_positions:
                self._file_positions[log_file] = 0
                log.write(Text(f"\n{'='*80}\n", style="dim"))
                log.write(Text(f"Starting log: {log_file.name}\n", style="bold cyan"))
                log.write(Text(f"{'='*80}\n", style="dim"))

            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    f.seek(self._file_positions[log_file])
                    new_content = f.read()
                    if new_content:
                        log.write(new_content)
                    self._file_positions[log_file] = f.tell()
            except Exception:
                # Silently ignore read errors
                pass

    def action_force_replay(self) -> None:
        """Force switch to replay mode."""
        if self.session_dir.exists():
            self.exit(self.session_dir)
        else:
            self.exit(None)

    def action_quit(self) -> None:
        """Quit without switching to replay."""
        self.exit(None)


__all__ = ["BackgroundRunViewer"]

