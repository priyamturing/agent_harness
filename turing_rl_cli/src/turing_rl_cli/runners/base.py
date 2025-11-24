"""Base utilities for benchmark runners."""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable, Any

import httpx
from turing_rl_sdk.agents.mcp import MCPConfig
from turing_rl_sdk.agents.runtime import RunContext, RunObserver
from turing_rl_sdk.harness.loader import scenario_to_task
from turing_rl_sdk.harness.orchestrator import RunResult, VerifierRunner, ResultBundle

from ..agent_factory import create_agent_from_string
from ..session.manager import SessionManager


def create_failed_run_result(
    cfg: dict,
    error: str,
    start_time: float,
    temperature: float
) -> RunResult:
    """Create a failed RunResult from an exception.
    
    Args:
        cfg: Run configuration dictionary
        error: Error message string
        start_time: Start time of the run (time.perf_counter())
        temperature: Sampling temperature used
        
    Returns:
        RunResult initialized with failure details
    """
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    prompt_text = cfg["scenario"].prompts[0].prompt_text if cfg["scenario"].prompts else ""
    
    return RunResult(
        model=cfg["model"],
        scenario_id=cfg["scenario"].scenario_id,
        scenario_name=cfg["scenario"].name,
        run_number=cfg["run_num"],
        success=False,
        result=None,
        verifier_results=[],
        error=error,
        metadata={
            "temperature": temperature,
            "file_stem": cfg["batch_alias"],
        },
        prompt_text=prompt_text,
        execution_time_ms=elapsed_ms,
    )


def build_run_summaries(
    run_configs: list[dict],
    results: list[Any],
    session_name: Optional[str] = None,
) -> tuple[list[RunResult], list[dict[str, Any]]]:
    """Build run summaries and filter successful results.
    
    Args:
        run_configs: List of run configuration dicts
        results: List of results (RunResult or Exception) from asyncio.gather
        session_name: Optional session name to use for file key mapping (matches persist_model_reports)
        
    Returns:
        Tuple of (successful_run_results, all_run_summaries)
    """
    run_results: list[RunResult] = []
    run_summaries: list[dict[str, Any]] = []
    
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

        if not isinstance(result, RunResult):
            # Should not happen given type hints but good for safety
            continue

        run_results.append(result)
        
        # Determine file key based on session_name if provided, otherwise fallback to batch_alias
        file_stem = session_name if session_name else ((result.metadata or {}).get("file_stem") or cfg["batch_alias"])
        
        summary = {
            "model": result.model,
            "scenario_id": result.scenario_id,
            "run_number": result.run_number,
            "success": result.success,
            "file_key": (file_stem, result.model),
        }
        
        if not result.success and (result.error or (result.result and result.result.error)):
            summary["error"] = result.error or (result.result.error if result.result else None)
            
        run_summaries.append(summary)
        
    return run_results, run_summaries


async def run_single_benchmark(
    model_name: str,
    scenario,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    max_steps: int,
    tool_call_limit: int,
    session_id: str,
    run_num: int,
    run_context: RunContext,
    shared_http_client: httpx.AsyncClient,
    batch_alias: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    observer_factory: Optional[Callable[[VerifierRunner], RunObserver]] = None,
) -> RunResult:
    """Run a single benchmark and return result.
    
    Args:
        model_name: Model identifier
        scenario: Scenario to run
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        max_steps: Maximum agent steps
        tool_call_limit: Maximum tool calls
        session_id: Session ID for LangSmith grouping
        run_num: Run number
        run_context: Runtime context with observers
        shared_http_client: Shared HTTP client
        timeout: Timeout in seconds for LLM API calls
        max_retries: Maximum retries for failed LLM calls
        observer_factory: Optional factory to create observer with access to verifier runner
        
    Returns:
        RunResult containing agent result, verifier results, and metadata
    """
    start_time = time.perf_counter()
    agent = create_agent_from_string(
        model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_call_limit=tool_call_limit,
        timeout=timeout,
        max_retries=max_retries,
    )
    
    task, verifier_defs = scenario_to_task(
        scenario, 
        mcp_config,
        database_id=run_context.database_id
    )
    
    task.metadata["session_id"] = session_id
    task.metadata["run_number"] = run_num
    
    verifier_runner = VerifierRunner(
        verifier_defs,
        run_context,
        http_client=shared_http_client,
        mcp_url=mcp_config.url,
    )

    if observer_factory:
        observer = observer_factory(verifier_runner)
        run_context.add_observer(observer)
    
    result = await agent.run(task, max_steps=max_steps, run_context=run_context)
    
    verifier_results = await verifier_runner.run_verifiers()
    
    all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
    final_success = result.success and all_verifiers_passed
    
    if not all_verifiers_passed:
        failed = [v.name for v in verifier_results if not v.success]
        error = f"Verifiers failed: {', '.join(failed)}"
    else:
        error = result.error
    
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    prompt_text = scenario.prompts[0].prompt_text if scenario.prompts else ""

    return RunResult(
        model=model_name,
        scenario_id=scenario.scenario_id,
        scenario_name=scenario.name,
        run_number=run_num,
        success=final_success,
        result=result,
        verifier_results=verifier_results,
        error=error,
        metadata={
            "temperature": temperature,
            "file_stem": batch_alias,
        }
        if batch_alias
        else {
            "temperature": temperature,
        },
        prompt_text=prompt_text,
        execution_time_ms=elapsed_ms,
    )


def persist_model_reports(
    session_mgr: SessionManager,
    session_dir: Path,
    run_results: list[RunResult],
    default_harness_name: str,
) -> dict[tuple[str, str], str]:
    """Persist aggregated result files and return mapping for manifests."""
    if not run_results:
        return {}

    bundle = ResultBundle(
        harness_name=default_harness_name,
        run_results=run_results,
    )
    model_files = bundle.build_model_reports(
        reset_database_between_runs=False,
    )

    mapping: dict[tuple[str, str], str] = {}
    for model_file in model_files:
        path = session_mgr.save_model_result(session_dir, model_file)
        mapping[(model_file.harness_name, model_file.model_name)] = str(
            path.relative_to(session_dir)
        )

    return mapping
