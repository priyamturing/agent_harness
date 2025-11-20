"""Textual UI mode runner with multi-pane interface."""

import asyncio
import sys
import time
from contextlib import nullcontext, contextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
from turing_rl_sdk.agents.mcp import MCPConfig
from turing_rl_sdk.agents.runtime import RunContext
from rich.console import Console

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..tracing import langfuse_run_context
from ..ui import TextualObserver, MultiRunApp
from turing_rl_sdk.harness.orchestrator import RunResult
from .base import persist_model_reports, create_failed_run_result, run_single_benchmark, build_run_summaries


@contextmanager
def _suppress_grpc_warnings():
    """Suppress gRPC and absl warnings that break the Textual UI."""
    class FilteredStderr:
        def __init__(self, original_stderr):
            self.original = original_stderr
            
        def write(self, text):
            if isinstance(text, str):
                lines_to_filter = [
                    "fork_posix.cc",
                    "absl::InitializeLog()",
                    "WARNING: All log messages before",
                    "Other threads are currently calling into gRPC",
                ]
                if not any(pattern in text for pattern in lines_to_filter):
                    return self.original.write(text)
            return len(text)
            
        def flush(self):
            return self.original.flush()
            
        def __getattr__(self, name):
            return getattr(self.original, name)
    
    old_stderr = sys.stderr
    sys.stderr = FilteredStderr(old_stderr)
    try:
        yield
    finally:
        sys.stderr = old_stderr


async def run_all_with_textual(
    run_configs: list[dict],
    session_dir: Path,
    session_name: str,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    max_steps: int,
    max_concurrent_runs: int,
    tool_call_limit: int,
    session_id: str,
    console: Console,
    langfuse_tracing: bool,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> None:
    """Run all configs with Textual UI (all files batched together).
    
    Args:
        run_configs: List of run configuration dicts
        session_dir: Session directory path
        session_name: Session name
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        max_steps: Maximum agent steps
        max_concurrent_runs: Maximum concurrent runs
        tool_call_limit: Maximum tool calls per run
        session_id: Benchmark session ID for LangSmith thread grouping
        console: Rich console instance
    """
    session_mgr = SessionManager(RESULTS_ROOT)

    # Create bounded queues and completion event for Textual UI
    # Bounded to prevent memory issues with verbose agents
    queues = {cfg["label"]: asyncio.Queue(maxsize=1000) for cfg in run_configs}
    completion_event = asyncio.Event()

    # Start Textual app in background
    textual_app = MultiRunApp(
        run_labels=[cfg["label"] for cfg in run_configs],
        queues=queues,
        completion_event=completion_event,
    )

    # Run app and tasks concurrently
    async def run_textual_app():
        await textual_app.run_async()

    async def run_all_tasks():
        # Create shared HTTP client for all verification runs
        async with httpx.AsyncClient(timeout=30.0) as shared_http_client:
            # Limit concurrency to prevent resource exhaustion
            semaphore = asyncio.Semaphore(max_concurrent_runs)
            tasks = []

            for cfg in run_configs:
                task = _run_single_textual(
                    cfg=cfg,
                    queue=queues[cfg["label"]],
                    mcp_config=mcp_config,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    semaphore=semaphore,
                    max_steps=max_steps,
                    tool_call_limit=tool_call_limit,
                    shared_http_client=shared_http_client,
                    session_id=session_id,
                    langfuse_tracing=langfuse_tracing,
                    timeout=timeout,
                    max_retries=max_retries,
                )
                tasks.append(task)

            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build run summaries
        run_results, run_summaries = build_run_summaries(run_configs, results, session_name=session_name)

        file_mapping = persist_model_reports(
            session_mgr=session_mgr,
            session_dir=session_dir,
            run_results=run_results,
            default_harness_name=session_name,
        )

        for summary in run_summaries:
            file_key = summary.pop("file_key", None)
            if "file" not in summary:
                summary["file"] = file_mapping.get(file_key, "")

        # Save session manifest
        session_mgr.save_session_manifest(
            session_dir,
            run_summaries,
            name=session_name,
        )

        # Give UI time to show final messages
        await asyncio.sleep(2)
        
        # Set summary data BEFORE signaling completion
        textual_app.set_summary_data(run_summaries, str(session_dir))
        
        # Wait a bit to ensure data is set
        await asyncio.sleep(0.5)
        
        # Now signal completion
        completion_event.set()

    # Run both concurrently with stderr filtering to prevent gRPC warnings from breaking the UI
    with _suppress_grpc_warnings():
        await asyncio.gather(
            run_textual_app(),
            run_all_tasks(),
        )
    
    # After Textual UI closes, print summary to console
    console.print("\n")
    console.rule("[bold cyan]Benchmark Complete[/bold cyan]")
    console.print(f"\nSession saved to: [dim]{session_dir}[/dim]")
    
    # Build run summaries list (should be available from run_all_tasks closure)
    # We need to get the summaries that were saved
    import json
    manifest_path = session_dir / "session.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            manifest = json.load(f)
            saved_summaries = manifest.get("runs", [])
        
        console.print(f"\nTotal runs: {len(saved_summaries)}")
        successful = sum(1 for r in saved_summaries if r.get("success", False))
        console.print(f"Successful: [green]{successful}[/green]")
        console.print(f"Failed: [red]{len(saved_summaries) - successful}[/red]\n")
        
        # Print results table
        from rich.table import Table
        table = Table(title="Results Summary", show_header=True, header_style="bold")
        table.add_column("Model", style="cyan")
        table.add_column("Scenario", style="magenta")
        table.add_column("Run", justify="right")
        table.add_column("Status")
        table.add_column("File", style="dim")
        
        for summary in saved_summaries:
            status = "[green]✓ PASS[/green]" if summary.get("success") else "[red]✗ FAIL[/red]"
            table.add_row(
                summary.get("model", ""),
                summary.get("scenario_id", ""),
                str(summary.get("run_number", 1)),
                status,
                summary.get("file", ""),
            )
        
        console.print(table)
        console.print(f"\n[dim]Session directory: {session_dir}[/dim]\n")


async def _run_single_textual(
    cfg: dict,
    queue: asyncio.Queue,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    semaphore: asyncio.Semaphore,
    max_steps: int,
    tool_call_limit: int,
    shared_http_client: httpx.AsyncClient,
    session_id: str,
    langfuse_tracing: bool,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> RunResult:
    """Run a single benchmark with textual observer."""
    async with semaphore:
        context = (
            langfuse_run_context(
                session_id=session_id,
                run_label=cfg["label"],
                model_name=cfg["model"],
                scenario_id=cfg["scenario"].scenario_id,
                run_number=cfg["run_num"],
                batch_alias=cfg["batch_alias"],
            )
            if langfuse_tracing
            else nullcontext()
        )

        with context:
            start_time = time.perf_counter()
            try:
                run_context = RunContext()

                def create_observer(verifier_runner):
                    return TextualObserver(
                        queue=queue,
                        width=100,
                        verifier_runner=verifier_runner,
                    )

                # Send initial status
                await queue.put(f"[bold]Starting {cfg['label']}...[/bold]\n")

                run_result = await run_single_benchmark(
                    model_name=cfg["model"],
                    scenario=cfg["scenario"],
                    mcp_config=mcp_config,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    max_steps=max_steps,
                    tool_call_limit=tool_call_limit,
                    session_id=session_id,
                    run_num=cfg["run_num"],
                    run_context=run_context,
                    shared_http_client=shared_http_client,
                    batch_alias=cfg["batch_alias"],
                    timeout=timeout,
                    max_retries=max_retries,
                    observer_factory=create_observer,
                )

                if run_result.success:
                    await queue.put("[bold green]✓ Completed successfully[/bold green]\n")
                else:
                    await queue.put(f"[bold red]✗ Failed: {run_result.error or 'Unknown'}[/bold red]\n")

                return run_result

            except Exception as exc:
                await queue.put(f"[bold red]✗ Failed: {exc}[/bold red]\n")
                import traceback
                await queue.put(traceback.format_exc())
                
                return create_failed_run_result(
                    cfg=cfg,
                    error=str(exc),
                    start_time=start_time,
                    temperature=temperature
                )
