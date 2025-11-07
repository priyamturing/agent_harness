"""Textual app for displaying parallel run output."""

import asyncio
import re
from asyncio import QueueEmpty
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Union

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Center, Container, Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static, TabbedContent, TabPane


class SummaryScreen(Screen):
    """Summary screen showing run results."""

    def __init__(self, run_summaries: list[dict[str, Any]], session_dir: str):
        """Initialize summary screen.

        Args:
            run_summaries: List of run summary dicts
            session_dir: Session directory path
        """
        super().__init__()
        self.run_summaries = run_summaries
        self.session_dir = session_dir

    def compose(self) -> ComposeResult:
        """Compose the summary screen."""
        successful = sum(1 for r in self.run_summaries if r.get("success", False))
        failed = len(self.run_summaries) - successful

        with Vertical():
            yield Label("\n[bold cyan]Benchmark Run Complete[/bold cyan]\n", id="title")
            yield Label(f"Session saved to: [dim]{self.session_dir}[/dim]\n")
            yield Label(f"Total runs: {len(self.run_summaries)}")
            yield Label(f"Successful: [green]{successful}[/green]")
            yield Label(f"Failed: [red]{failed}[/red]\n")

            # Results table
            table = DataTable(zebra_stripes=True, show_header=True)
            table.add_columns("Model", "Scenario", "Run", "Status", "File")

            for summary in self.run_summaries:
                status = Text("✓ PASS", style="green") if summary.get("success") else Text("✗ FAIL", style="red")
                table.add_row(
                    summary.get("model", ""),
                    summary.get("scenario_id", ""),
                    str(summary.get("run_number", 1)),
                    status,
                    summary.get("file", ""),
                )

            yield table
            yield Label("\n[dim]Press 'q' to exit or wait 10 seconds for auto-close[/dim]")
        yield Footer()

    async def on_mount(self) -> None:
        """Auto-close after 10 seconds."""
        await asyncio.sleep(10)
        self.app.exit()


class RunDetailScreen(Screen):
    """Detailed view of a single run's conversation history."""

    CSS = """
    RunDetailScreen {
        layout: vertical;
    }
    
    #detail_header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }
    
    #detail_log {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
    
    #detail_status {
        height: 12;
        border: solid $secondary;
        padding: 1;
    }
    
    #detail_verifier_table {
        width: 100%;
    }
    """

    BINDINGS = [
        ("escape", "back", "Back to Dashboard"),
        ("b", "back", "Back"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, run_label: str, run_messages: list[Union[str, dict]], run_status: Dict[str, Any]):
        """Initialize detail screen.
        
        Args:
            run_label: Label of the run to display
            run_messages: List of messages (shared with parent, updates live)
            run_status: Status dict for the run (shared with parent, updates live)
        """
        super().__init__()
        self.run_label = run_label
        self.run_messages = run_messages  # Reference to parent's message list
        self.run_status = run_status
        self._last_message_count = 0  # Track how many messages we've displayed

    def compose(self) -> ComposeResult:
        """Compose detail screen UI."""
        yield Header()
        
        with Container(id="detail_header"):
            yield Label(f"[bold cyan]Run Details: {self.run_label}[/bold cyan]", id="detail_title")
            yield Label("", id="detail_status_line")
        
        with VerticalScroll(id="detail_log"):
            yield RichLog(
                id="detail_richlog",
                auto_scroll=True,
                highlight=True,
                markup=True,
                wrap=True,
            )
        
        with Container(id="detail_status"):
            yield Label("[bold]Verifier Status[/bold]")
            yield DataTable(id="detail_verifier_table", zebra_stripes=True, show_header=True)
        
        yield Footer()

    def on_mount(self) -> None:
        """Setup table and start updates."""
        table = self.query_one("#detail_verifier_table", DataTable)
        table.add_columns("Status", "Verifier", "Comparison", "Expected", "Actual", "Error")
        
        # Start periodic updates
        self.set_interval(0.25, self.update_display)

    def update_display(self) -> None:
        """Update display with latest messages and status."""
        # Update status line
        status_label = self.query_one("#detail_status_line", Label)
        run_status = self.run_status.get("status", "unknown")
        start_time = self.run_status.get("start_time", "")
        progress = self.run_status.get("progress", "")
        
        status_parts = []
        if run_status == "running":
            status_parts.append("[yellow]▶ Running[/yellow]")
        elif run_status == "completed":
            if self.run_status.get("success"):
                status_parts.append("[green]✓ Completed Successfully[/green]")
            else:
                status_parts.append("[red]✗ Failed[/red]")
        elif run_status == "queued":
            status_parts.append("[cyan]⏸ Queued[/cyan]")
        
        if start_time:
            display_time = start_time.split("T")[1][:8] if "T" in start_time else start_time[:8]
            status_parts.append(f"Started: {display_time}")
        
        if progress:
            status_parts.append(f"Progress: {progress}")
        
        status_label.update(" | ".join(status_parts))
        
        # Process new messages from shared list
        log_widget = self.query_one("#detail_richlog", RichLog)
        current_message_count = len(self.run_messages)
        
        # Only process messages we haven't seen yet
        if current_message_count > self._last_message_count:
            new_messages = self.run_messages[self._last_message_count:current_message_count]
            
            for message in new_messages:
                if isinstance(message, str):
                    # Display string message
                    log_widget.write(message)
                
                elif isinstance(message, dict) and message.get("type") == "status":
                    # Update verifier table
                    table = self.query_one("#detail_verifier_table", DataTable)
                    table.clear()
                    
                    for row in message.get("data", []):
                        status_text = Text(
                            row["status"],
                            style="green" if row["status"] == "PASS" else "red",
                        )
                        error_text = Text(row.get("error") or "", style="dim")
                        
                        table.add_row(
                            status_text,
                            row["name"],
                            row["comparison"],
                            row["expected"],
                            row["actual"],
                            error_text,
                        )
            
            # Update counter
            self._last_message_count = current_message_count

    def action_back(self) -> None:
        """Go back to dashboard."""
        self.app.pop_screen()


class DashboardScreen(Screen):
    """Enhanced dashboard showing real-time progress and active runs."""

    CSS = """
    DashboardScreen {
        layout: vertical;
    }
    
    #stats_container {
        height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $primary;
    }
    
    #progress_container {
        height: auto;
        padding: 1 2;
        margin: 1 0;
    }
    
    #active_runs {
        height: 1fr;
        border: solid $accent;
        padding: 1;
    }
    
    #status_table {
        width: 100%;
    }
    
    #recent_log {
        height: 15;
        border: solid $secondary;
        padding: 1;
    }
    
    .stat_label {
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        total_runs: int,
        run_status: Dict[str, Dict[str, Any]],
        global_log_queue: asyncio.Queue,
        run_queues: Dict[str, asyncio.Queue],
        run_messages: Dict[str, list[Union[str, dict]]],
    ):
        super().__init__()
        self.total_runs = total_runs
        self.run_status = run_status
        self.global_log_queue = global_log_queue
        self.run_queues = run_queues
        self._run_messages = run_messages
        self.start_time = datetime.now()
        self._run_labels_order: list[str] = []  # Track order of runs in table

    def compose(self) -> ComposeResult:
        """Compose dashboard UI."""
        yield Header()
        
        with Container(id="stats_container"):
            yield Label("[bold cyan]Benchmark Dashboard[/bold cyan]", id="title")
            yield Label("Initializing...", id="stats_summary")
            yield Label("Elapsed: 00:00", id="timing_info")
        
        with Container(id="progress_container"):
            yield Label("Overall Progress", classes="stat_label")
            yield ProgressBar(total=self.total_runs, show_eta=True, id="overall_progress")
        
        with VerticalScroll(id="active_runs"):
            yield Label("[bold]Active & Queued Runs[/bold] [dim](↑↓ to select, Enter to view details)[/dim]")
            yield DataTable(id="status_table", zebra_stripes=True, show_header=True)
        
        with Container(id="recent_log"):
            yield Label("[bold]Recent Activity[/bold]")
            yield RichLog(id="activity_log", auto_scroll=True, highlight=True, markup=True, wrap=True)
        
        yield Footer()

    def on_mount(self) -> None:
        """Setup table and start updates."""
        table = self.query_one("#status_table", DataTable)
        table.add_columns("Status", "Harness", "Model", "Scenario", "Run", "Started", "Progress")
        table.cursor_type = "row"  # Enable row selection
        table.focus()  # Focus table for keyboard navigation
        
        # Log dashboard initialization
        try:
            self.global_log_queue.put_nowait(
                f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                f"[cyan]Dashboard initialized - monitoring {self.total_runs} runs[/cyan]"
            )
        except:
            pass
        
        # Initial update to show starting state
        self.update_display()
        
        # Start periodic updates
        self.set_interval(0.5, self.update_display)
        self.set_interval(0.25, self.drain_log_queue)

    def drain_log_queue(self) -> None:
        """Drain global log queue and show recent activity."""
        log_widget = self.query_one("#activity_log", RichLog)
        
        message_count = 0
        while message_count < 10:  # Limit per update
            try:
                message = self.global_log_queue.get_nowait()
                log_widget.write(message)
                message_count += 1
            except QueueEmpty:
                break

    def update_display(self) -> None:
        """Update dashboard with current status."""
        try:
            # Count statuses (successful vs failed, not overlapping)
            successful = sum(1 for s in self.run_status.values() 
                            if s.get("status") == "completed" and s.get("success", False))
            failed = sum(1 for s in self.run_status.values() 
                        if s.get("status") == "completed" and not s.get("success", True))
            running = sum(1 for s in self.run_status.values() if s.get("status") == "running")
            queued = sum(1 for s in self.run_status.values() if s.get("status") == "queued")
            
            # Update summary labels
            stats_label = self.query_one("#stats_summary", Label)
            stats_label.update(
                f"[green]Successful: {successful}[/green]  "
                f"[red]Failed: {failed}[/red]  "
                f"[yellow]Running: {running}[/yellow]  "
                f"[cyan]Queued: {queued}[/cyan]"
            )
            
            # Update timing
            elapsed = (datetime.now() - self.start_time).total_seconds()
            timing_label = self.query_one("#timing_info", Label)
            timing_label.update(f"Elapsed: {self._format_time(elapsed)}")
            
            # Update progress bar (total finished = successful + failed)
            progress = self.query_one("#overall_progress", ProgressBar)
            total_finished = successful + failed
            progress.update(progress=total_finished)
            
            # Update status table - show running and recent
            table = self.query_one("#status_table", DataTable)
            
            # Save cursor position
            current_cursor = table.cursor_row if table.cursor_row is not None else 0
            
            table.clear()
            
            # Sort: running first, then queued, then recently completed
            sorted_runs = sorted(
                self.run_status.items(),
                key=lambda x: (
                    0 if x[1].get("status") == "running" else
                    1 if x[1].get("status") == "queued" else 2,
                    x[1].get("start_time", "")
                ),
                reverse=False
            )
            
            # Track order for selection
            self._run_labels_order = []
            
            # Show up to 20 most relevant runs
            for label, status in sorted_runs[:20]:
                self._run_labels_order.append(label)
                run_status = status.get("status", "unknown")
                
                # Status icon
                if run_status == "completed":
                    if status.get("success", True):
                        status_text = Text("✓ Done", style="green")
                    else:
                        status_text = Text("✗ Failed", style="red")
                elif run_status == "running":
                    status_text = Text("▶ Running", style="yellow bold")
                elif run_status == "queued":
                    status_text = Text("⏸ Queued", style="cyan")
                else:
                    status_text = Text("? Unknown", style="dim")
                
                # Parse label to extract harness, model, scenario, run
                # Label format: {model}_{harness}_{scenario}_r{num} or {model}_{scenario}_r{num}
                parts = label.split("_")
                model = parts[0] if len(parts) > 0 else ""
                
                # Check if last part is run number (starts with 'r')
                has_run_num = len(parts) > 0 and parts[-1].startswith("r") and parts[-1][1:].isdigit()
                run_num = parts[-1] if has_run_num else ""
                
                # Determine if we have harness name (multiple files scenario)
                # If 4+ parts with run_num or 3+ parts without run_num, we have harness
                end_idx = -1 if has_run_num else len(parts)
                middle_parts = parts[1:end_idx]
                
                if len(middle_parts) >= 2:
                    # Has harness: first middle part is harness, rest is scenario
                    harness = middle_parts[0]
                    scenario = "_".join(middle_parts[1:])
                elif len(middle_parts) == 1:
                    # No harness: just scenario
                    harness = "-"
                    scenario = middle_parts[0]
                else:
                    harness = "-"
                    scenario = ""
                
                start_time = status.get("start_time", "")
                if start_time:
                    start_time = start_time.split("T")[1][:8] if "T" in start_time else start_time[:8]
                
                progress_text = status.get("progress", "")
                
                table.add_row(
                    status_text,
                    harness[:20],
                    model[:12],
                    scenario[:25],
                    run_num,
                    start_time,
                    progress_text[:30],
                )
            
            # Restore cursor position (clamped to valid range)
            if len(self._run_labels_order) > 0:
                table.move_cursor(row=min(current_cursor, len(self._run_labels_order) - 1))
                
        except Exception as e:
            # Log error to activity log for debugging
            try:
                self.global_log_queue.put_nowait(
                    f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] "
                    f"[red]Dashboard update error: {e}[/red]"
                )
            except:
                pass  # Even logging failed, give up silently
    
    def on_screen_resume(self) -> None:
        """Called when screen is resumed (e.g., after popping detail screen)."""
        # Refocus the table
        table = self.query_one("#status_table", DataTable)
        table.focus()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (Enter key on a row)."""
        # Get the row index
        row_index = event.cursor_row
        
        # Check if we have a valid selection
        if row_index is not None and row_index < len(self._run_labels_order):
            selected_label = self._run_labels_order[row_index]
            
            # Create and show detail screen with stored messages
            detail_screen = RunDetailScreen(
                run_label=selected_label,
                run_messages=self._run_messages[selected_label],
                run_status=self.run_status[selected_label],
            )
            self.app.push_screen(detail_screen)

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


class MultiRunApp(App[None]):
    """Enhanced Textual app with dashboard and selective detail views.

    Displays:
    - Dashboard with overall progress
    - Active runs status
    - Recent activity log
    - Individual run details (on demand)
    """

    CSS = """
    TabbedContent, Horizontal {
        height: 1fr;
    }
    RichLog {
        height: 1fr;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        run_labels: Iterable[str],
        queues: Dict[str, asyncio.Queue],
        completion_event: Optional[asyncio.Event] = None,
    ) -> None:
        """Initialize multi-run app.

        Args:
            run_labels: List of run labels (one per tab)
            queues: Mapping of run label -> asyncio queue for messages
            completion_event: Event to set when runs complete
        """
        self._run_labels = list(run_labels)
        self._queues = queues
        self._id_map = self._build_id_map(self._run_labels)
        self._summary_data: list[dict[str, Any]] = []
        self._session_dir: str = ""
        self._completion_event = completion_event
        self._completed = False
        
        # Enhanced status tracking
        self._run_status: Dict[str, Dict[str, Any]] = {
            label: {
                "status": "queued",
                "success": None,
                "start_time": "",
                "end_time": "",
                "progress": "",
            }
            for label in run_labels
        }
        self._global_log_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        
        # Store all messages for each run so detail view can access them
        self._run_messages: Dict[str, list[Union[str, dict]]] = {
            label: [] for label in run_labels
        }
        
        super().__init__()

    def set_summary_data(self, summaries: list[dict[str, Any]], session_dir: str) -> None:
        """Set summary data to display when runs complete.

        Args:
            summaries: List of run summaries
            session_dir: Session directory path
        """
        self._summary_data = summaries
        self._session_dir = session_dir

    def show_summary(self) -> None:
        """Show summary screen."""
        summary_screen = SummaryScreen(self._summary_data, self._session_dir)
        self.push_screen(summary_screen)

    @staticmethod
    def _build_id_map(labels: Iterable[str]) -> Dict[str, str]:
        """Build sanitized IDs for widget identification."""

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
        """Compose the UI - show dashboard by default."""
        # Create dashboard as the default screen
        yield Header()
        yield Footer()

    def update_run_status(self, label: str, **kwargs) -> None:
        """Update status for a specific run.
        
        Args:
            label: Run label
            **kwargs: Status fields to update (status, success, start_time, end_time, progress)
        """
        if label in self._run_status:
            self._run_status[label].update(kwargs)
    
    def log_global_activity(self, message: str) -> None:
        """Log a message to global activity log.
        
        Args:
            message: Message to log
        """
        try:
            self._global_log_queue.put_nowait(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim] {message}")
        except asyncio.QueueFull:
            pass  # Drop if queue is full

    async def on_mount(self) -> None:
        """Start polling queues when app mounts and show dashboard."""
        # Show dashboard screen
        dashboard = DashboardScreen(
            total_runs=len(self._run_labels),
            run_status=self._run_status,
            global_log_queue=self._global_log_queue,
            run_queues=self._queues,
            run_messages=self._run_messages,
        )
        self.push_screen(dashboard)
        
        # Start polling queues
        self.set_interval(0.25, self._drain_all_queues)
        self.set_interval(0.5, self._check_completion)

    def _drain_all_queues(self) -> None:
        """Drain all queues and update status tracking."""
        for label in self._run_labels:
            queue = self._queues[label]
            
            message_count = 0
            while message_count < 20:  # Limit messages per run per cycle
                try:
                    message: Union[str, dict] = queue.get_nowait()
                    message_count += 1
                except QueueEmpty:
                    break
                else:
                    # Store message for detail view
                    self._run_messages[label].append(message)
                    
                    # Process message and extract status information
                    if isinstance(message, str):
                        # Parse string messages for status keywords
                        if "Starting" in message or "▶" in message:
                            self.update_run_status(label, status="running", start_time=datetime.now().isoformat())
                            self.log_global_activity(f"{label}: [yellow]Started[/yellow]")
                        elif "✓ Completed successfully" in message:
                            self.update_run_status(label, status="completed", success=True, end_time=datetime.now().isoformat(), progress="")
                            self.log_global_activity(f"{label}: [green]Success ✓[/green]")
                        elif "✗ Failed" in message or ("Failed:" in message and "✗" in message):
                            self.update_run_status(label, status="completed", success=False, end_time=datetime.now().isoformat(), progress="")
                            self.log_global_activity(f"{label}: [red]Failed ✗[/red]")
                        elif "Agent completed" in message:
                            self.update_run_status(label, progress="Agent done, verifying...")
                        elif "Invoking tool" in message:
                            # Extract tool name if possible
                            if "tool" in message.lower():
                                self.update_run_status(label, progress="Calling tools...")
                        elif "Assistant:" in message:
                            self.update_run_status(label, progress="Agent responding...")
                    
                    elif isinstance(message, dict) and message.get("type") == "status":
                        # Verifier status update
                        data = message.get("data", [])
                        if data:
                            passed = sum(1 for r in data if r.get("status") == "PASS")
                            total = len(data)
                            self.update_run_status(label, progress=f"Verifying {passed}/{total}")
                            
                            # Log significant updates
                            if passed == total:
                                self.log_global_activity(f"{label}: [green]All {total} verifiers passed[/green]")

    def _check_completion(self) -> None:
        """Check if runs are complete and show summary."""
        if self._completion_event and self._completion_event.is_set() and not self._completed:
            self._completed = True
            
            # Always show summary if we have data
            if self._summary_data and len(self._summary_data) > 0:
                # Add a small delay then show summary
                self.set_timer(0.5, self.show_summary)
            else:
                # No summary data, just exit
                self.set_timer(2, self.exit)

