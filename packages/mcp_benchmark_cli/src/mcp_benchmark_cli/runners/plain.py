"""Plain console mode runner with streaming output."""

from pathlib import Path
from typing import Optional

import httpx
from mcp_benchmark_sdk import MCPConfig, RunContext
from rich.console import Console

from ..config import RESULTS_ROOT
from ..session.manager import SessionManager
from ..ui import ConsoleObserver
from .base import save_benchmark_result


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
    
    # Create shared HTTP client for all verification runs
    async with httpx.AsyncClient(timeout=30.0) as shared_http_client:
        for cfg in run_configs:
            console.print(f"\n[bold cyan]Running:[/bold cyan] {cfg['label']}")
            
            try:
                # Import here to avoid circular dependency
                from mcp_benchmark_sdk import scenario_to_task
                from mcp_benchmark_harness.orchestrator import VerifierRunner
                
                # Create RunContext
                run_context = RunContext()
                
                # Convert scenario to task to get verifier definitions
                # Convert scenario to task with database_id for isolation
                task, verifier_defs = scenario_to_task(
                    cfg["scenario"], 
                    mcp_config,
                    database_id=run_context.database_id
                )
                
                # Create verifier runner for continuous verification
                verifier_runner = VerifierRunner(
                    verifier_defs,
                    run_context,
                    http_client=shared_http_client,
                    mcp_url=mcp_config.url,
                )
                
                # Add console observer for output with verifier runner
                observer = ConsoleObserver(
                    console=console,
                    prefix=cfg["label"],
                    verifier_runner=verifier_runner,
                )
                run_context.add_observer(observer)
                
                # Create agent and run
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
                
                # Run the agent
                result = await agent.run(task, max_steps=max_steps, run_context=run_context)
                
                # Run final verification
                verifier_results = await verifier_runner.run_verifiers()
                
                # Determine final success
                all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
                final_success = result.success and all_verifiers_passed
                
                if not all_verifiers_passed:
                    failed = [v.name for v in verifier_results if not v.success]
                    result.error = f"Verifiers failed: {', '.join(failed)}"
                
                result.success = final_success
                
                # Save result
                result_file = save_benchmark_result(
                    session_mgr=session_mgr,
                    session_dir=session_dir,
                    result=result,
                    verifier_results=verifier_results,
                    model_name=cfg["model"],
                    scenario=cfg["scenario"],
                    batch_alias=cfg["batch_alias"],
                    run_num=cfg["run_num"],
                    temperature=temperature,
                )
                
                run_summaries.append({
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": result.success,
                    "file": str(result_file.relative_to(session_dir)),
                })
                
                if result.success:
                    console.print("[bold green]✓ Success[/bold green]")
                else:
                    console.print(f"[bold red]✗ Failed:[/bold red] {result.error}")
                    
            except Exception as exc:
                console.print(f"[red]Error:[/red] {exc}")
                run_summaries.append({
                    "model": cfg["model"],
                    "scenario_id": cfg["scenario"].scenario_id,
                    "run_number": cfg["run_num"],
                    "success": False,
                    "file": "",
                    "error": str(exc),
                })
    
    # Save manifest and print summary
    session_mgr.save_session_manifest(session_dir, run_summaries, name=session_name)
    
    console.print("\n")
    console.rule("[bold]Summary[/bold]")
    console.print(f"\nTotal runs: {len(run_summaries)}")
    successful = sum(1 for r in run_summaries if r["success"])
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{len(run_summaries) - successful}[/red]")
