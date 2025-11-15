"""Base utilities for benchmark runners."""

import asyncio
from pathlib import Path
from typing import Optional

import httpx
from mcp_benchmark_sdk.agents.mcp import MCPConfig
from mcp_benchmark_sdk.agents.runtime import RunContext
from mcp_benchmark_sdk.harness.loader import scenario_to_task
from mcp_benchmark_sdk.harness.orchestrator import RunResult, VerifierRunner

from ..agent_factory import create_agent_from_string
from ..session.manager import SessionManager


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
        
    Returns:
        RunResult containing agent result, verifier results, and metadata
    """
    agent = create_agent_from_string(
        model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_call_limit=tool_call_limit,
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
    
    result = await agent.run(task, max_steps=max_steps, run_context=run_context)
    
    verifier_results = await verifier_runner.run_verifiers()
    
    all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
    final_success = result.success and all_verifiers_passed
    
    if not all_verifiers_passed:
        failed = [v.name for v in verifier_results if not v.success]
        error = f"Verifiers failed: {', '.join(failed)}"
    else:
        error = result.error
    
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
        }
    )


def save_benchmark_result(
    session_mgr: SessionManager,
    session_dir: Path,
    run_result: RunResult,
    batch_alias: str,
) -> Path:
    """Save benchmark result to file.
    
    Args:
        session_mgr: Session manager
        session_dir: Session directory
        run_result: RunResult from benchmark execution
        batch_alias: Batch alias for filename
        
    Returns:
        Path to saved result file
    """
    scenario_id = f"{batch_alias}_{run_result.scenario_id}_run{run_result.run_number}"
    
    result_file = session_mgr.save_result(
        session_dir,
        run_result,
        scenario_id,
        metadata={
            "batch_alias": batch_alias,
        },
    )
    return result_file

