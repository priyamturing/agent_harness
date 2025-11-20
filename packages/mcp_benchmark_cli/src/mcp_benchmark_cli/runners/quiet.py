"""Quiet mode runner that only displays aggregate progress."""

import asyncio
import sys
import time
from contextlib import nullcontext, suppress, contextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
from mcp_benchmark_sdk.agents.mcp import MCPConfig
from mcp_benchmark_sdk.agents.runtime import RunContext
from rich.console import Console
from tqdm import tqdm

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..tracing import langfuse_run_context
from ..ui import QuietObserver
from mcp_benchmark_sdk.harness.orchestrator import RunResult
from .base import run_single_benchmark, persist_model_reports


@contextmanager
def _suppress_grpc_warnings():
    """Suppress gRPC and absl warnings that break the progress bar."""
    class FilteredStderr:
        def __init__(self, original_stderr):
            self.original = original_stderr
            self.buffer = ""
            
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
    runs_per_prompt: int,
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
    run_summaries: list[dict[str, Any]] = []
    run_results: list[RunResult] = []
    
    active_runs = {"count": 0}
    summary_counts = {"success": 0, "failure": 0}
    
    # Create progress indicator with stderr filtering to prevent gRPC warnings from breaking the display
    with _suppress_grpc_warnings(), tqdm(
        total=len(run_configs),
        desc="Running benchmarks...",
        unit="run",
        dynamic_ncols=True,
        leave=True,
    ) as progress_bar:
        progress_lock = asyncio.Lock()
        refresh_task = asyncio.create_task(
            _refresh_progress(progress_bar, active_runs, summary_counts, progress_lock)
        )

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
                    semaphore=semaphore,
                    max_steps=max_steps,
                    tool_call_limit=tool_call_limit,
                    shared_http_client=shared_http_client,
                    session_id=session_id,
                    progress_bar=progress_bar,
                    progress_lock=progress_lock,
                    summary_counts=summary_counts,
                    langfuse_tracing=langfuse_tracing,
                    active_runs=active_runs,
                )
                tasks.append(task)

            try:
                # Run all tasks in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                refresh_task.cancel()
                with suppress(asyncio.CancelledError):
                    await refresh_task
        
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
                continue

            run_results.append(result)
            summary = {
                "model": result.model,
                "scenario_id": result.scenario_id,
                "run_number": result.run_number,
                "success": result.success,
                "file_key": ((result.metadata or {}).get("file_stem") or cfg["batch_alias"], result.model),
            }
            if not result.success and (result.error or (result.result and result.result.error)):
                summary["error"] = result.error or (result.result.error if result.result else None)
            run_summaries.append(summary)

    file_mapping = persist_model_reports(
        session_mgr=session_mgr,
        session_dir=session_dir,
        run_results=run_results,
        runs_per_prompt=runs_per_prompt,
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
    semaphore: asyncio.Semaphore,
    max_steps: int,
    tool_call_limit: int,
    shared_http_client: httpx.AsyncClient,
    session_id: str,
    progress_bar: tqdm,
    progress_lock: asyncio.Lock,
    summary_counts: dict[str, int],
    langfuse_tracing: bool,
    active_runs: dict,
) -> RunResult:
    """Run a single benchmark in quiet mode."""
    async with semaphore:
        await _update_active_runs(active_runs, progress_lock, delta=1)
        run_result: Optional[RunResult] = None
        
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
            run_context = RunContext()
            observer = QuietObserver()
            run_context.add_observer(observer)

            try:
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
                )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - start_time) * 1000)
                prompt_text = cfg["scenario"].prompts[0].prompt_text if cfg["scenario"].prompts else ""
                run_result = RunResult(
                    model=cfg["model"],
                    scenario_id=cfg["scenario"].scenario_id,
                    scenario_name=cfg["scenario"].name,
                    run_number=cfg["run_num"],
                    success=False,
                    result=None,
                    verifier_results=[],
                    error=str(exc),
                    metadata={
                        "temperature": temperature,
                        "file_stem": cfg["batch_alias"],
                    },
                    prompt_text=prompt_text,
                    execution_time_ms=elapsed_ms,
                )
            finally:
                await _update_active_runs(active_runs, progress_lock, delta=-1)
                await _record_result(
                    summary_counts,
                    bool(run_result and run_result.success),
                    progress_lock,
                )
                await _advance_progress(progress_bar, progress_lock)

        return run_result


async def _advance_progress(progress_bar: tqdm, lock: asyncio.Lock, step: int = 1) -> None:
    """Advance the shared tqdm progress bar safely from async tasks."""
    async with lock:
        progress_bar.update(step)


async def _record_result(summary_counts: dict[str, int], success: bool, lock: asyncio.Lock) -> None:
    """Record success/failure counts under the shared lock."""
    key = "success" if success else "failure"
    async with lock:
        summary_counts[key] += 1


async def _update_active_runs(active_runs: dict, lock: asyncio.Lock, delta: int) -> None:
    """Update the number of active runs safely."""
    async with lock:
        active_runs["count"] += delta


async def _refresh_progress(
    progress_bar: tqdm,
    active_runs: dict,
    summary_counts: dict[str, int],
    lock: asyncio.Lock,
    interval: float = 0.5,
) -> None:
    """Refresh tqdm description with the latest active run count and pass/fail totals."""
    try:
        while True:
            async with lock:
                active = active_runs["count"]
                passed = summary_counts["success"]
                failed = summary_counts["failure"]
                parts = []
                if active > 0:
                    parts.append(f"{active} active")
                parts.append(f"pass {passed}")
                parts.append(f"fail {failed}")
                desc = "Running benchmarks..."
                if parts:
                    desc += f" ({' | '.join(parts)})"
                progress_bar.set_description(desc)
                progress_bar.refresh()
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
