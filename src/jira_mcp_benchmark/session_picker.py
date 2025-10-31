"""Textual picker used to select saved run sessions."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Header, Label, Tree
from textual.widgets.tree import TreeNode


@dataclass(frozen=True)
class SessionDisplay:
    """Lightweight view model for listing sessions."""

    path: Path
    display_name: str
    runs: int
    model_summary: str
    verifier_summary: str
    harness_name: str = ""
    status: str = "completed"  # "completed" or "failed"
    

@dataclass(frozen=True)
class BackgroundRunDisplay:
    """Lightweight view model for listing background runs."""

    run_id: str
    display_name: str
    status: str
    progress: str
    started_at: str
    harness_name: str = ""
    model_summary: str = ""
    run_configs: list = field(default_factory=list)
    scenario_batches: list = field(default_factory=list)


class SessionPickerApp(App[Optional[Union[Path, str]]]):
    """Hierarchical Textual TUI for browsing sessions by harness and model."""

    CSS = """
    Screen {
        align: center middle;
    }

    #content {
        width: 90%;
        height: 85%;
        border: round $accent;
        padding: 1 2;
    }

    Tree {
        height: 1fr;
        margin-top: 1;
    }

    .title {
        text-style: bold;
        content-align: center middle;
        color: $accent;
        margin-bottom: 1;
    }
    
    .subtitle {
        content-align: center middle;
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
        ("enter", "select", "Select"),
    ]

    def __init__(
        self,
        sessions: Iterable[SessionDisplay],
        background_runs: Optional[Iterable[BackgroundRunDisplay]] = None,
    ) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._background_runs = list(background_runs) if background_runs else []
        self._selection_future: asyncio.Future[Optional[Union[Path, str]]] = (
            asyncio.get_event_loop().create_future()
        )

    async def wait_for_selection(self) -> Optional[Union[Path, str]]:
        return await self._selection_future

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="content"):
            yield Label("🎯 Benchmark Run Browser", classes="title")
            yield Label("Navigate with ↑↓ arrows, Enter to select, Space to expand/collapse", classes="subtitle")
            tree: Tree[dict] = Tree("Runs", id="run-tree")
            tree.show_root = False
            tree.guide_depth = 4
            yield tree
        yield Footer()

    def on_mount(self) -> None:
        tree = self.query_one("#run-tree", Tree)
        self._populate_tree(tree)
        tree.focus()

    def _load_manifest(self, session_path: Path) -> Optional[dict]:
        """Load session manifest from path."""
        manifest_path = session_path / "session.json"
        if not manifest_path.exists():
            return None
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _populate_tree(self, tree: Tree) -> None:
        """Populate the tree with hierarchical session data: Session → Harness → Model → Runs."""
        
        # Check if we have any data
        if not self._sessions and not self._background_runs:
            tree.root.add_leaf(
                Text("No runs found", style="dim italic"),
                data={"type": "empty"}
            )
            return
        
        # Process completed sessions
        for session in sorted(self._sessions, key=lambda s: s.display_name, reverse=True):
            # Load manifest to get individual runs
            manifest = self._load_manifest(session.path)
            if not manifest:
                continue
            
            # Create session node
            session_node = tree.root.add(
                Text(f"📊 {session.display_name}", style="bold cyan"),
                data={"type": "session_group"},
                expand=False
            )
            
            # Group runs by harness, then model
            by_harness: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
            for run in manifest.get("runs", []):
                # Extract harness name from prompt_file
                prompt_file = run.get("prompt_file", "")
                harness = Path(prompt_file).stem if prompt_file else "Unknown"
                
                # Format model as "provider:model" or just "provider"
                provider = run.get("provider", "")
                model_name = run.get("model")
                if model_name:
                    model = f"{provider}:{model_name}"
                else:
                    model = provider or "Unknown"
                
                by_harness[harness][model].append(run)
            
            # Build tree: harness -> model -> runs
            for harness in sorted(by_harness.keys()):
                harness_node = session_node.add(
                    Text(f"📁 {harness}", style="bold cyan"),
                    data={"type": "harness"},
                    expand=False
                )
                
                for model in sorted(by_harness[harness].keys()):
                    runs = by_harness[harness][model]
                    model_node = harness_node.add(
                        Text(f"🤖 {model} ({len(runs)} runs)", style="bold magenta"),
                        data={"type": "model"},
                        expand=False
                    )
                    
                    # Add individual runs
                    for run in runs:
                        run_label_text = run.get("label", "Run")
                        run_status = run.get("status", "completed")
                        
                        # Calculate verifier stats from scenarios
                        total_verifiers = 0
                        passed_verifiers = 0
                        for scenario in run.get("scenarios", []):
                            total_verifiers += scenario.get("verifiers_total", 0)
                            passed_verifiers += scenario.get("verifiers_passed", 0)
                        
                        # Determine status icon
                        if run_status == "failed":
                            status_icon = "❌"
                            status_style = "red"
                        else:
                            status_icon = "✅"
                            status_style = "green"
                        
                        run_label = Text()
                        run_label.append(f"{status_icon} ", style=status_style)
                        run_label.append(f"{run_label_text} ", style="white")
                        run_label.append(f"• {passed_verifiers}/{total_verifiers} verifiers", style="dim")
                        
                        # Store path to run artifacts for selection
                        artifact_path = run.get("artifact_path")
                        run_data = {
                            "type": "run",
                            "session_path": session.path,
                            "artifact_path": artifact_path,
                            "run_label": run_label_text
                        }
                        
                        model_node.add_leaf(run_label, data=run_data)
        
        # Process background runs (in-progress)
        if self._background_runs:
            for bg_run in sorted(self._background_runs, key=lambda r: r.started_at, reverse=True):
                # Create session node for background run
                session_node = tree.root.add(
                    Text(f"🔄 {bg_run.display_name}", style="bold yellow"),
                    data={"type": "session_group"},
                    expand=False
                )
                
                # Parse run_configs and scenario_batches to build proper hierarchy
                run_configs = bg_run.run_configs or []
                scenario_batches = bg_run.scenario_batches or []
                
                # Group run_configs by harness and model
                # Each scenario_batch is a harness file
                # Each run_config is an individual run (with label, provider, model)
                
                for batch in scenario_batches:
                    # Extract harness name from source path
                    harness_source = batch.get("source", "")
                    harness_name = Path(harness_source).stem if harness_source else batch.get("alias", "Unknown")
                    
                    harness_node = session_node.add(
                        Text(f"📁 {harness_name}", style="bold cyan"),
                        data={"type": "harness"},
                        expand=False
                    )
                    
                    # Group run_configs by model
                    by_model: dict[str, list] = defaultdict(list)
                    for run_config in run_configs:
                        provider = run_config.get("provider", "")
                        model_name = run_config.get("model")
                        if model_name:
                            model = f"{provider}:{model_name}"
                        else:
                            model = provider or "Unknown"
                        by_model[model].append(run_config)
                    
                    # Add model nodes with individual runs
                    for model in sorted(by_model.keys()):
                        runs = by_model[model]
                        model_node = harness_node.add(
                            Text(f"🤖 {model} ({len(runs)} runs)", style="bold magenta"),
                            data={"type": "model"},
                            expand=False
                        )
                        
                        # Add individual run leaves
                        for run_config in runs:
                            run_label_text = run_config.get("label", "Run")
                            
                            status_icon = "🔄" if bg_run.status == "running" else "⏸️"
                            status_style = "yellow" if bg_run.status == "running" else "blue"
                            run_label = Text()
                            run_label.append(f"{status_icon} ", style=status_style)
                            run_label.append(f"{run_label_text} ", style="white")
                            # Format progress as "passed/total verifiers"
                            run_label.append(f"• {bg_run.progress} verifiers", style="dim")
                            run_label.append(f" • {bg_run.started_at}", style="dim italic")
                            
                            model_node.add_leaf(
                                run_label,
                                data={"type": "background_run", "run_id": bg_run.run_id}
                            )

    async def action_quit(self) -> None:
        """Quit the app."""
        if not self._selection_future.done():
            self._selection_future.set_result(None)
        self.exit(None)

    def action_select(self) -> None:
        """Select the current item."""
        tree = self.query_one("#run-tree", Tree)
        cursor_node = tree.cursor_node
        
        if not cursor_node:
            return
        
        node_data = cursor_node.data
        if not isinstance(node_data, dict):
            return
        
        node_type = node_data.get("type")
        
        # Only leaf nodes (actual runs) are selectable
        if node_type == "background_run":
            run_id = node_data.get("run_id")
            if run_id:
                if not self._selection_future.done():
                    self._selection_future.set_result(run_id)
                self.exit(run_id)
        elif node_type == "run":
            # Individual completed run - return session path for now
            # TODO: Could return specific run artifact path in future
            session_path = node_data.get("session_path")
            if session_path:
                if not self._selection_future.done():
                    self._selection_future.set_result(session_path)
                self.exit(session_path)
        elif node_type == "session":
            path = node_data.get("path")
            if path:
                if not self._selection_future.done():
                    self._selection_future.set_result(path)
                self.exit(path)
        elif node_type in {"harness", "model", "session_group"}:
            # Toggle expand/collapse for non-leaf nodes
            cursor_node.toggle()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        node_data = event.node.data
        if not isinstance(node_data, dict):
            return
        
        node_type = node_data.get("type")
        
        # Only leaf nodes (actual runs) are selectable
        if node_type == "background_run":
            run_id = node_data.get("run_id")
            if run_id:
                if not self._selection_future.done():
                    self._selection_future.set_result(run_id)
                self.exit(run_id)
        elif node_type == "run":
            # Individual completed run - return session path for now
            session_path = node_data.get("session_path")
            if session_path:
                if not self._selection_future.done():
                    self._selection_future.set_result(session_path)
                self.exit(session_path)
        elif node_type == "session":
            path = node_data.get("path")
            if path:
                if not self._selection_future.done():
                    self._selection_future.set_result(path)
                self.exit(path)


__all__ = ["SessionPickerApp", "SessionDisplay", "BackgroundRunDisplay"]
