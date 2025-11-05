"""Textual UI observer for displaying agent execution."""

import asyncio
import json
from typing import Any

from mcp_benchmark_sdk import RunObserver, VerifierResult


def _json_safe(value: object) -> object:
    """Return JSON-serializable representation."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        try:
            return json.loads(json.dumps(value, default=str))
        except TypeError:
            return repr(value)


class TextualObserver(RunObserver):
    """Observer that enqueues Rich markup strings for Textual UI.

    Feeds data to a Textual app via an asyncio queue.
    RichLog widget will render the markup.
    """

    def __init__(self, queue: asyncio.Queue, width: int = 100):
        """Initialize textual observer.

        Args:
            queue: Queue to send messages to Textual app
            width: Console width (not used, kept for compatibility)
        """
        self._queue = queue
        self._artifacts: dict[str, list] = {
            "conversation": [],
            "verifier_history": [],
            "log_stream": [],
            "status_stream": [],
        }
        self._drop_count = 0
        self._last_drop_warning = 0

    def _enqueue(self, text: str) -> None:
        """Send text directly to queue (RichLog handles markup)."""
        # Ensure proper line breaks
        if not text.endswith("\n"):
            text = f"{text}\n"
        
        # Always save to artifacts (even if UI queue is full)
        self._artifacts["log_stream"].append(text)
        
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            # Track drops and warn periodically
            self._drop_count += 1
            
            # Warn every 100 drops to avoid spamming
            if self._drop_count - self._last_drop_warning >= 100:
                self._last_drop_warning = self._drop_count
                warning = f"[yellow]⚠ UI queue full - {self._drop_count} messages dropped (saved to artifacts)[/yellow]\n"
                try:
                    # Try to send warning (might also be dropped)
                    self._queue.put_nowait(warning)
                except asyncio.QueueFull:
                    pass

    async def on_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Log message with Rich markup."""
        # Add to artifacts
        entry: dict[str, Any] = {"type": "message", "role": role}
        if content:
            entry["content"] = content
        if metadata and "reasoning" in metadata:
            entry["reasoning"] = metadata["reasoning"]

        # Avoid duplicates
        if self._artifacts["conversation"] and self._artifacts["conversation"][-1] == entry:
            return
        self._artifacts["conversation"].append(entry)

        # Send Rich markup directly to queue with proper line breaks
        if role == "system":
            self._enqueue(f"[dim]System: {content[:100]}...[/dim]\n")
        elif role == "user":
            self._enqueue(f"[bold magenta]User:[/bold magenta] {content}\n\n")
        elif role == "assistant":
            self._enqueue(f"[bold cyan]Assistant:[/bold cyan] {content}\n")
            if metadata and "reasoning" in metadata:
                reasoning = metadata["reasoning"]
                if reasoning:
                    # Show full reasoning, not truncated
                    self._enqueue(f"[yellow]Reasoning:[/yellow]\n{reasoning}\n\n")

    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Log tool call with Rich markup."""
        # Add to artifacts
        call_entry = {
            "type": "tool_call",
            "tool": tool_name,
            "args": _json_safe(arguments),
        }
        result_entry = {
            "type": "tool_result",
            "tool": tool_name,
            "output": _json_safe(result),
        }
        if is_error:
            result_entry["error"] = True

        self._artifacts["conversation"].extend([call_entry, result_entry])

        # Format arguments nicely
        if isinstance(arguments, dict):
            args_str = json.dumps(arguments, indent=2, ensure_ascii=False)
        else:
            args_str = str(arguments)
        
        self._enqueue(f"[green]→ Invoking tool[/green] [bold]{tool_name}[/bold] with args:\n")
        self._enqueue(f"[italic]{args_str}[/italic]\n")

        # Format result nicely
        if is_error:
            self._enqueue(f"[bold red]← Tool error:[/bold red] {result}\n\n")
        else:
            if isinstance(result, (dict, list)):
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
            else:
                result_str = str(result)
            
            self._enqueue(f"[magenta]← Tool response:[/magenta]\n")
            self._enqueue(f"{result_str}\n\n")

    async def on_verifier_update(self, verifier_results: list[Any]) -> None:
        """Log verifier results to status table."""
        if not verifier_results:
            return

        payload = []
        snapshot: list[dict] = []

        for result in verifier_results:
            if isinstance(result, VerifierResult):
                payload_entry = {
                    "name": result.name,
                    "comparison": result.comparison_type or "-",
                    "expected": repr(result.expected_value),
                    "actual": repr(result.actual_value) if result.actual_value is not None else "-",
                    "status": "PASS" if result.success else "FAIL",
                    "error": result.error,
                }
                payload.append(payload_entry)

                snapshot.append({
                    "name": result.name,
                    "comparison": result.comparison_type,
                    "expected": _json_safe(result.expected_value),
                    "actual": _json_safe(result.actual_value),
                    "success": result.success,
                    "error": result.error,
                    "metadata": result.metadata,
                })

        snapshot_copy = [dict(item) for item in snapshot]
        self._artifacts["verifier_history"].append(snapshot_copy)

        # Send status update to Textual UI
        self._queue.put_nowait({"type": "status", "data": payload})
        self._artifacts["status_stream"].append(payload)

        # Also add to conversation artifacts
        self._artifacts["conversation"].append({
            "type": "verifier_status",
            "results": snapshot_copy,
        })

    async def on_status(self, message: str, level: str = "info") -> None:
        """Log status message with Rich markup."""
        if level == "error":
            self._enqueue(f"[bold red]Error:[/bold red] {message}\n")
        elif level == "warning":
            self._enqueue(f"[yellow]Warning:[/yellow] {message}\n")
        else:
            self._enqueue(f"[dim]{message}[/dim]\n")

    def get_artifacts(self) -> dict:
        """Get all logged artifacts for session persistence."""
        return {
            "conversation": list(self._artifacts["conversation"]),
            "verifier_history": list(self._artifacts["verifier_history"]),
            "log_stream": list(self._artifacts["log_stream"]),
            "status_stream": list(self._artifacts["status_stream"]),
        }
