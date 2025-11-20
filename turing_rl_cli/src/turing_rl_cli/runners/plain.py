"""Plain console mode runner with streaming output."""

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import httpx
from turing_rl_sdk.agents.mcp import MCPConfig
from turing_rl_sdk.agents.runtime import RunContext
from rich.console import Console

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..tracing import langfuse_run_context
from ..ui import ConsoleObserver
from turing_rl_sdk.harness.orchestrator import RunResult
from .base import run_single_benchmark, persist_model_reports, create_failed_run_result


async def run_all_plain(
    run_configs: list[dict],
    session_dir: Path,
    session_name: str,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    max_steps: int,
    tool_call_limit: int,
    session_id: str,
    console: Console,
    langfuse_tracing: bool,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> None:
    """Run all configs in plain console mode.
    
    Args:
        run_configs: List of run configuration dicts
        session_dir: Session directory
        session_name: Session name
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        session_id: Benchmark session ID for LangSmith thread grouping
        max_steps: Maximum steps
        tool_call_limit: Tool call limit
        console: Rich console instance
    """
    session_mgr = SessionManager(RESULTS_ROOT)
    run_summaries = []
    run_results = []
    
    # Create shared HTTP client for all verification runs
    async with httpx.AsyncClient(timeout=30.0) as shared_http_client:
        for cfg in run_configs:
            console.print(f"\n[bold cyan]Running:[/bold cyan] {cfg['label']}")
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
                        return ConsoleObserver(
                            console=console,
                            prefix=cfg["label"],
                            verifier_runner=verifier_runner,
                        )

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
                    
                    run_results.append(run_result)
                    run_summaries.append({
                        "model": cfg["model"],
                        "scenario_id": cfg["scenario"].scenario_id,
                        "run_number": cfg["run_num"],
                        "success": run_result.success,
                        "file_key": (session_name, cfg["model"]),
                    })

                    if run_result.success:
                        console.print("[bold green]✓ Success[/bold green]")
                    else:
                        console.print(f"[bold red]✗ Failed:[/bold red] {run_result.error}")

                except Exception as exc:
                    console.print(f"[red]Error:[/red] {exc}")
                    failure_result = create_failed_run_result(
                        cfg=cfg,
                        error=str(exc),
                        start_time=start_time,
                        temperature=temperature
                    )
                    run_results.append(failure_result)
                    run_summaries.append({
                        "model": cfg["model"],
                        "scenario_id": cfg["scenario"].scenario_id,
                        "run_number": cfg["run_num"],
                        "success": False,
                        "file_key": (session_name, cfg["model"]),
                        "error": str(exc),
                    })
    
    # Persist model reports
    file_mapping = persist_model_reports(
        session_mgr=session_mgr,
        session_dir=session_dir,
        run_results=run_results,
        default_harness_name=session_name,
    )
    
    # Update summaries with file paths
    for summary in run_summaries:
        file_key = summary.pop("file_key", None)
        summary["file"] = file_mapping.get(file_key, "")
    
    # Save manifest and print summary
    session_mgr.save_session_manifest(session_dir, run_summaries, name=session_name)
    
    console.print("\n")
    console.rule("[bold]Summary[/bold]")
    console.print(f"\nTotal runs: {len(run_summaries)}")
    successful = sum(1 for r in run_summaries if r["success"])
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{len(run_summaries) - successful}[/red]")
