"""Helper classes for directing run output."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from rich.console import Console
from rich.table import Table

from .verifier import VerifierResult


class RunLogger(Protocol):
    """Minimal interface used by the scenario executor."""

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        ...

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        ...

    def update_verifier_status(self, results: Sequence[VerifierResult]) -> None:
        ...

    def log_tool_call(
        self, name: str, args: object, response: object | None = None, *, error: bool = False
    ) -> None:
        ...

    def log_message(
        self, role: str, content: str, *, reasoning: Sequence[str] | None = None
    ) -> None:
        ...

    def get_artifacts(self) -> dict:
        ...


def _json_safe(value: object) -> object:
    """Return a JSON-serializable representation of a value."""

    try:
        json.dumps(value)
        return value
    except TypeError:
        try:
            return json.loads(json.dumps(value, default=str))
        except TypeError:
            return repr(value)


@dataclass
class ConsoleRunLogger:
    """Run logger that writes directly to a Rich console."""

    console: Console
    prefix: Optional[str] = None
    conversation: List[dict] = field(default_factory=list)
    verifier_history: List[List[dict]] = field(default_factory=list)

    def _emit_prefix(self) -> None:
        if self.prefix:
            self.console.print(f"[bold][{self.prefix}][/bold]")

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._emit_prefix()
        self.console.print(*args, **kwargs)

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._emit_prefix()
        self.console.rule(*args, **kwargs)

    def update_verifier_status(self, results: Sequence[VerifierResult]) -> None:
        if not results:
            return

        table = Table(title="Verifier status", show_lines=True)
        table.add_column("Verifier")
        table.add_column("Comparison")
        table.add_column("Expected")
        table.add_column("Actual")
        table.add_column("Status")

        snapshot: List[dict] = []
        for result in results:
            label = result.verifier.name or result.verifier.verifier_type
            comparison = result.comparison_type or "-"
            expected = repr(result.expected_value)
            actual = repr(result.actual_value) if result.error is None else "-"
            status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
            if result.error and not result.success:
                status = f"[red]FAIL[/red]\n[dim]{result.error}[/dim]"
            table.add_row(label, comparison, expected, actual, status)

            snapshot.append(
                {
                    "name": label,
                    "comparison": result.comparison_type,
                    "expected": _json_safe(result.expected_value),
                    "actual": _json_safe(result.actual_value),
                    "success": result.success,
                    "error": result.error,
                }
            )

        self._emit_prefix()
        self.console.print(table)
        snapshot_copy = [dict(item) for item in snapshot]
        self.verifier_history.append(snapshot_copy)

        verifier_entry = {
            "type": "verifier_status",
            "results": [dict(item) for item in snapshot_copy],
        }
        self.conversation.append(verifier_entry)

    def log_tool_call(
        self, name: str, args: object, response: object | None = None, *, error: bool = False
    ) -> None:
        call_entry = {
            "type": "tool_call",
            "tool": name,
            "args": _json_safe(args),
        }
        result_entry: Dict[str, Any] = {
            "type": "tool_result",
            "tool": name,
            "output": _json_safe(response),
        }
        if error:
            result_entry["error"] = True
        self.conversation.extend([call_entry, result_entry])

    def log_message(
        self, role: str, content: str, *, reasoning: Sequence[str] | None = None
    ) -> None:
        if not content and not reasoning:
            return
        entry: Dict[str, Any] = {"type": "message", "role": role}
        if content:
            entry["content"] = content
        if reasoning:
            entry["reasoning"] = list(reasoning)
        if self.conversation and self.conversation[-1] == entry:
            return
        self.conversation.append(entry)

    def get_artifacts(self) -> dict:
        return {
            "conversation": list(self.conversation),
            "verifier_history": list(self.verifier_history),
        }


class TextualRunLogger:
    """Run logger that enqueues rendered strings for the Textual UI."""

    def __init__(self, queue: asyncio.Queue[object], width: int = 100) -> None:
        self._queue = queue
        self._capture_console = Console(
            color_system=None,
            force_terminal=False,
            record=True,
            width=width,
        )
        self._artifacts: Dict[str, list] = {
            "conversation": [],
            "verifier_history": [],
            "log_stream": [],
            "status_stream": [],
        }

    def _enqueue_render(self) -> None:
        text = self._capture_console.export_text(clear=True)
        if not text.endswith("\n"):
            text = f"{text}\n"
        self._queue.put_nowait(text)
        self._artifacts["log_stream"].append(text)

    def print(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._capture_console.print(*args, **kwargs)
        self._enqueue_render()

    def rule(self, *args, **kwargs) -> None:  # noqa: ANN001
        self._capture_console.rule(*args, **kwargs)
        self._enqueue_render()

    def update_verifier_status(self, results: Sequence[VerifierResult]) -> None:
        payload = []
        snapshot: List[dict] = []
        for result in results:
            payload_entry = {
                "name": result.verifier.name or result.verifier.verifier_type,
                "comparison": result.comparison_type or "-",
                "expected": repr(result.expected_value),
                "actual": repr(result.actual_value)
                if result.error is None
                else "-",
                "status": "PASS" if result.success else "FAIL",
                "error": result.error,
            }
            payload.append(payload_entry)
            snapshot.append(
                {
                    "name": result.verifier.name or result.verifier.verifier_type,
                    "comparison": result.comparison_type,
                    "expected": _json_safe(result.expected_value),
                    "actual": _json_safe(result.actual_value),
                    "success": result.success,
                    "error": result.error,
                }
            )
        snapshot_copy = [dict(item) for item in snapshot]
        self._artifacts["verifier_history"].append(snapshot_copy)
        self._queue.put_nowait({"type": "status", "data": payload})
        self._artifacts["status_stream"].append(payload)

        self._artifacts["conversation"].append(
            {
                "type": "verifier_status",
                "results": [dict(item) for item in snapshot_copy],
            }
        )

    def log_tool_call(
        self, name: str, args: object, response: object | None = None, *, error: bool = False
    ) -> None:
        call_entry = {
            "type": "tool_call",
            "tool": name,
            "args": _json_safe(args),
        }
        result_entry: Dict[str, Any] = {
            "type": "tool_result",
            "tool": name,
            "output": _json_safe(response),
        }
        if error:
            result_entry["error"] = True
        self._artifacts["conversation"].extend([call_entry, result_entry])

    def log_message(
        self, role: str, content: str, *, reasoning: Sequence[str] | None = None
    ) -> None:
        if not content and not reasoning:
            return
        entry: Dict[str, Any] = {"type": "message", "role": role}
        if content:
            entry["content"] = content
        if reasoning:
            entry["reasoning"] = list(reasoning)
        if self._artifacts["conversation"] and self._artifacts["conversation"][-1] == entry:
            return
        self._artifacts["conversation"].append(entry)

    def get_artifacts(self) -> dict:
        return {
            "conversation": list(self._artifacts["conversation"]),
            "verifier_history": list(self._artifacts["verifier_history"]),
            "log_stream": list(self._artifacts["log_stream"]),
            "status_stream": list(self._artifacts["status_stream"]),
        }
