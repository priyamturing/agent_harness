"""Quiet mode runner with progress bar only."""

import asyncio
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import httpx
from mcp_benchmark_sdk.agents.mcp import MCPConfig
from mcp_benchmark_sdk.agents.runtime import RunContext
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..tracing import langfuse_run_context
from ..ui import QuietObserver
from .base import run_single_benchmark, save_benchmark_result


async def run_all_quiet(
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
    """Run all configs in quiet mode with progress bar only.
    
    Args:
        run_configs: List of run configuration dicts
        session_dir: Session directory
        session_name: Session name
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        max_steps: Maximum steps
        max_concurrent_runs: Maximum concurrent runs
        tool_call_limit: Tool call limit
        session_id: Benchmark session ID for LangSmith thread grouping
        console: Rich console instance
    """
    session_mgr = SessionManager(RESULTS_ROOT)
    run_summaries = []
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running benchmarks...", total=len(run_configs))
        
        # Create shared HTTP client for all verification runs
        async with httpx.AsyncClient(timeout=30.0) as shared_http_client:
            # Limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent_runs)
            tasks = []
            
            for cfg in run_configs:
                task = _run_single_quiet(
                    cfg=cfg,
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
                    progress=progress,
                    task_id=task_id,
                    langfuse_tracing=langfuse_tracing,
                )
                tasks.append(task)
            
            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build run summaries
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
    
    # Print final summary
    console.print("\n")
    console.rule("[bold]Benchmark Complete[/bold]")
    console.print(f"\nTotal runs: {len(run_summaries)}")
    successful = sum(1 for r in run_summaries if r.get("success", False))
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{len(run_summaries) - successful}[/red]")
    console.print(f"\n[dim]Results saved to: {session_dir}[/dim]\n")


async def _run_single_quiet(
    cfg: dict,
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
    progress: Progress,
    task_id: int,
    langfuse_tracing: bool,
) -> dict:
    """Run a single benchmark in quiet mode."""
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
                # Create RunContext with QuietObserver
                run_context = RunContext()
                observer = QuietObserver()
                run_context.add_observer(observer)

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
                )

                result_file = save_benchmark_result(
                    session_mgr=session_mgr,
                    session_dir=session_dir,
                    run_result=run_result,
                    batch_alias=cfg["batch_alias"],
                )

                progress.advance(task_id)

                return {
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": run_result.success,
                    "file": str(result_file.relative_to(session_dir)),
                }

            except Exception as exc:
                # Update progress even on error
                progress.advance(task_id)

                return {
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": False,
                    "error": str(exc),
                    "file": "",
                }
