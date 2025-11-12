"""Textual UI mode runner with multi-pane interface."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import httpx
from mcp_benchmark_sdk.agents.mcp import MCPConfig
from mcp_benchmark_sdk.agents.runtime import RunContext
from rich.console import Console

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..tracing import langfuse_run_context
from ..ui import TextualObserver, MultiRunApp
from mcp_benchmark_sdk.harness.orchestrator import RunResult
from .base import save_benchmark_result


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
                    session_mgr=session_mgr,
                    session_dir=session_dir,
                    semaphore=semaphore,
                    max_steps=max_steps,
                    tool_call_limit=tool_call_limit,
                    shared_http_client=shared_http_client,
                    session_id=session_id,
                    langfuse_tracing=langfuse_tracing,
                )
                tasks.append(task)

            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build run summaries
        run_summaries = []
        for cfg, result in zip(run_configs, results):
            if isinstance(result, Exception):
                run_summaries.append({
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": False,
                    "error": str(result),
                    "file": "",
                })
            elif isinstance(result, dict):
                run_summaries.append(result)

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

    # Run both concurrently
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
    session_mgr: SessionManager,
    session_dir: Path,
    semaphore: asyncio.Semaphore,
    max_steps: int,
    tool_call_limit: int,
    shared_http_client: httpx.AsyncClient,
    session_id: str,
    langfuse_tracing: bool,
) -> dict:
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
            try:
                # Import here to avoid circular dependency
                from mcp_benchmark_sdk.harness.loader import scenario_to_task
                from mcp_benchmark_sdk.harness.orchestrator import VerifierRunner

                # Create RunContext (automatically generates database_id for isolation)
                run_context = RunContext()

                # Convert scenario to task to get verifier definitions
                # Pass database_id for proper database isolation
                task, verifier_defs = scenario_to_task(
                    cfg["scenario"],
                    mcp_config,
                    database_id=run_context.database_id,
                )

                # Create verifier runner for continuous verification
                verifier_runner = VerifierRunner(
                    verifier_defs,
                    run_context,
                    http_client=shared_http_client,
                    mcp_url=mcp_config.url,
                )

                # Add Textual observer with verifier runner for continuous verification
                observer = TextualObserver(
                    queue=queue,
                    width=100,
                    verifier_runner=verifier_runner,
                )
                run_context.add_observer(observer)

                # Send initial status
                await queue.put(f"[bold]Starting {cfg['label']}...[/bold]\n")

                # Run benchmark using the task we already created
                from ..agent_factory import create_agent_from_string

                agent = create_agent_from_string(
                    cfg["model"],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    tool_call_limit=tool_call_limit,
                )

                # Add session metadata
                task.metadata["session_id"] = session_id
                task.metadata["run_number"] = cfg["run_num"]

                result = await agent.run(task, max_steps=max_steps, run_context=run_context)

                verifier_results = await verifier_runner.run_verifiers()

                all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
                final_success = result.success and all_verifiers_passed

                if not all_verifiers_passed:
                    failed = [v.name for v in verifier_results if not v.success]
                    error = f"Verifiers failed: {', '.join(failed)}"
                else:
                    error = result.error

                run_result = RunResult(
                    model=cfg["model"],
                    scenario_id=cfg["scenario"].scenario_id,
                    scenario_name=cfg["scenario"].name,
                    run_number=cfg["run_num"],
                    success=final_success,
                    result=result,
                    verifier_results=verifier_results,
                    error=error,
                    metadata={
                        "temperature": temperature,
                    }
                )

                result_file = save_benchmark_result(
                    session_mgr=session_mgr,
                    session_dir=session_dir,
                    run_result=run_result,
                    batch_alias=cfg["batch_alias"],
                )

                if run_result.success:
                    await queue.put("[bold green]✓ Completed successfully[/bold green]\n")
                else:
                    await queue.put(f"[bold red]✗ Failed: {run_result.error or 'Unknown'}[/bold red]\n")

                return {
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": run_result.success,
                    "file": str(result_file.relative_to(session_dir)),
                }

            except Exception as exc:
                await queue.put(f"[bold red]Error: {exc}[/bold red]\n")
                import traceback
                await queue.put(traceback.format_exc())
                raise
