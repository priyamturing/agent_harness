"""Textual picker used to select saved run sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.events import Key
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ListItem, ListView


@dataclass(frozen=True)
class SessionDisplay:
    """Lightweight view model for listing sessions."""

    path: Path
    display_name: str
    runs: int
    model_summary: str


class SessionPickerApp(App[Optional[Path]]):
    """Simple Textual TUI that lets the user choose a session with arrow keys."""

    CSS = """
    Screen {
        align: center middle;
    }

    #content {
        width: 80%;
        height: 80%;
        border: round $accent;
        padding: 1 2;
    }

    ListView {
        height: 1fr;
    }

    .title {
        text-style: bold;
        content-align: center middle;
    }
    """

    selected: reactive[Optional[Path]] = reactive(None)

    def __init__(self, sessions: Iterable[SessionDisplay]) -> None:
        super().__init__()
        self._sessions = list(sessions)
        self._selection_future: asyncio.Future[Optional[Path]] = asyncio.get_event_loop().create_future()

    async def wait_for_selection(self) -> Optional[Path]:
        return await self._selection_future

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="content"):
            yield Label("Select a session to replay", id="title", classes="title")
            entries = []
            for idx, session in enumerate(self._sessions, start=1):
                label = f"{session.display_name}"
                runs = session.runs
                models = session.model_summary or "-"
                item = ListItem(Label(f"{idx}. {label} | Runs: {runs} | Models: {models}"))
                item.data = session.path
                entries.append(item)
            if not entries:
                entries.append(ListItem(Label("No saved sessions found.")))
            list_view = ListView(*entries, id="session-list")
            if not self._sessions:
                list_view.disabled = True
            yield list_view
        yield Footer()

    def on_mount(self) -> None:
        list_view = self.query_one("#session-list", ListView)
        if self._sessions:
            list_view.index = 0
            list_view.focus()

    def action_quit(self) -> None:  # type: ignore[override]
        if not self._selection_future.done():
            self._selection_future.set_result(None)
        super().action_quit()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if not self._sessions:
            if not self._selection_future.done():
                self._selection_future.set_result(None)
            self.exit(None)
            return
        selected_item = event.item
        session_path = getattr(selected_item, "data", None)
        if not self._selection_future.done():
            self._selection_future.set_result(session_path)
        self.exit(session_path)

    async def on_key(self, event: Key) -> None:  # type: ignore[override]
        if event.key in {"q", "Q", "escape"}:
            if not self._selection_future.done():
                self._selection_future.set_result(None)
            self.exit(None)
            return
        if event.key in {"enter", "return", "space"} and self._sessions:
            list_view = self.query_one("#session-list", ListView)
            if list_view.disabled:
                return
            index = getattr(list_view, "index", 0) or 0
            children = list(list_view.children)
            if not children:
                return
            index = max(0, min(index, len(children) - 1))
            item = children[index]
            session_path = getattr(item, "data", None)
            if not self._selection_future.done():
                self._selection_future.set_result(session_path)
            self.exit(session_path)
            return
