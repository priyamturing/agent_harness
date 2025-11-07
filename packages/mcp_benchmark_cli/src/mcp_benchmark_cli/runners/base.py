"""Base utilities for benchmark runners."""

import asyncio
from pathlib import Path
from typing import Optional

import httpx
from mcp_benchmark_harness.orchestrator import VerifierRunner
from mcp_benchmark_sdk import MCPConfig, RunContext, scenario_to_task

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
) -> tuple:
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
        Tuple of (result, verifier_results)
    """
    # Create agent
    agent = create_agent_from_string(
        model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        tool_call_limit=tool_call_limit,
    )
    
    # Convert scenario to task and extract verifier definitions
    # Pass database_id from run_context for proper database isolation
    task, verifier_defs = scenario_to_task(
        scenario, 
        mcp_config,
        database_id=run_context.database_id
    )
    
    # Add session and run metadata for LangSmith grouping
    task.metadata["session_id"] = session_id
    task.metadata["run_number"] = run_num
    
    # Create verifier runner with shared HTTP client
    verifier_runner = VerifierRunner(
        verifier_defs,
        run_context,
        http_client=shared_http_client,
        mcp_url=mcp_config.url,
    )
    
    # Run agent
    result = await agent.run(task, max_steps=max_steps, run_context=run_context)
    
    # Run final verification
    verifier_results = await verifier_runner.run_verifiers()
    
    # Determine final success based on agent result AND verifiers
    all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
    final_success = result.success and all_verifiers_passed
    
    # Update result with final status
    if not all_verifiers_passed:
        failed = [v.name for v in verifier_results if not v.success]
        result.error = f"Verifiers failed: {', '.join(failed)}"
    
    result.success = final_success
    
    return result, verifier_results


def save_benchmark_result(
    session_mgr: SessionManager,
    session_dir: Path,
    result,
    verifier_results: list,
    model_name: str,
    scenario,
    batch_alias: str,
    run_num: int,
    temperature: float,
) -> Path:
    """Save benchmark result to file.
    
    Args:
        session_mgr: Session manager
        session_dir: Session directory
        result: Agent execution result
        verifier_results: List of verifier results from harness
        model_name: Model name
        scenario: Scenario
        batch_alias: Batch alias
        run_num: Run number
        temperature: Temperature setting
        
    Returns:
        Path to saved result file
    """
    result_file = session_mgr.save_result(
        session_dir,
        result,
        model_name,
        f"{batch_alias}_{scenario.scenario_id}_run{run_num}",
        metadata={
            "scenario_name": scenario.name,
            "temperature": temperature,
            "run_number": run_num,
            "batch_alias": batch_alias,
        },
        verifier_results=verifier_results,
    )
    return result_file
