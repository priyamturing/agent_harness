"""Console observer for displaying agent execution."""

from typing import Any, Optional, TYPE_CHECKING

from turing_rl_sdk.agents.runtime import RunObserver
from turing_rl_sdk.harness.verifiers import VerifierResult
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from turing_rl_sdk.harness.orchestrator import VerifierRunner


class ConsoleObserver(RunObserver):
    """Observer that prints agent execution to console using Rich."""

    def __init__(
        self,
        console: Optional[Console] = None,
        prefix: Optional[str] = None,
        verifier_runner: Optional["VerifierRunner"] = None,
    ):
        """Initialize console observer.

        Args:
            console: Rich console instance (created if not provided)
            prefix: Optional prefix for all output
            verifier_runner: Optional verifier runner for continuous verification
        """
        self.console = console or Console()
        self.prefix = prefix
        
        # Verification support (optional, injected)
        self.verifier_runner = verifier_runner

    def _emit_prefix(self) -> None:
        """Emit prefix if set."""
        if self.prefix:
            self.console.print(f"[bold][{self.prefix}][/bold]")

    async def on_message(
        self, role: str, content: str, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """Display message."""
        self._emit_prefix()

        if role == "system":
            self.console.print(f"[dim]System: {content[:100]}...[/dim]")
        elif role == "user":
            self.console.print(f"[bold magenta]User:[/bold magenta] {content}")
        elif role == "assistant":
            self.console.print(f"[bold cyan]Assistant:[/bold cyan] {content}")

            # Show reasoning if present
            if metadata and "reasoning" in metadata:
                reasoning = metadata["reasoning"]
                if isinstance(reasoning, list):
                    for idx, r in enumerate(reasoning, 1):
                        self.console.print(f"[yellow]Reasoning {idx}:[/yellow] {r[:200]}...")
                elif reasoning:
                    self.console.print(f"[yellow]Reasoning:[/yellow] {reasoning[:200]}...")

    async def on_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        is_error: bool = False,
    ) -> None:
        """Display tool call and optionally run verifiers."""
        self._emit_prefix()

        # Format arguments
        args_str = str(arguments)[:100]
        self.console.print(f"[green]→ Tool:[/green] [bold]{tool_name}[/bold] ({args_str}...)")

        # Format result
        if is_error:
            self.console.print(f"[bold red]← Error:[/bold red] {result}")
        else:
            result_str = str(result)[:200]
            self.console.print(f"[magenta]← Result:[/magenta] {result_str}...")
        
        # Run verifiers if configured (optional)
        if self.verifier_runner and not is_error:
            verifier_results = await self.verifier_runner.run_verifiers()
            self._display_verifier_results(verifier_results)

    def _display_verifier_results(self, verifier_results: list[Any]) -> None:
        """Display verifier results."""
        if not verifier_results:
            return

        self._emit_prefix()

        table = Table(title="Verifier Results", show_lines=True)
        table.add_column("Verifier")
        table.add_column("Expected")
        table.add_column("Actual")
        table.add_column("Status")

        for result in verifier_results:
            if isinstance(result, VerifierResult):
                status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
                if result.error:
                    status = f"[red]FAIL[/red]\n[dim]{result.error}[/dim]"

                table.add_row(
                    result.name,
                    repr(result.expected_value),
                    repr(result.actual_value) if not result.error else "-",
                    status,
                )

        self.console.print(table)

    async def on_status(self, message: str, level: str = "info") -> None:
        """Display status message."""
        self._emit_prefix()

        if level == "error":
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        elif level == "warning":
            self.console.print(f"[yellow]Warning:[/yellow] {message}")
        else:
            self.console.print(f"[dim]{message}[/dim]")

