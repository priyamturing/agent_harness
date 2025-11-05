"""Main CLI entry point for MCP Benchmark."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Suppress transformers warning (imported by langchain_anthropic)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress gRPC fork warnings (common with Google libraries + asyncio)
# This is safe across all platforms and prevents the fork() warning messages
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "1")

import typer
from dotenv import load_dotenv
from mcp_benchmark_sdk import MCPConfig, RunContext
from rich.console import Console

from .agent_factory import create_agent_from_string
from .config import DEFAULT_JIRA_MCP, DEFAULT_SQL_RUNNER_URL, RESULTS_ROOT, MAX_CONCURRENT_RUNS
from .harness_loader import load_harness_file, scenario_to_task
from .session.manager import SessionManager
from .ui import ConsoleObserver, TextualObserver, MultiRunApp
from .verifier_runner import VerifierRunner

app = typer.Typer(help="Run MCP agent benchmarks with various LLM providers")
console = Console()


@app.command()
def main(
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        help="Path to benchmark JSON file or directory containing JSON files",
    ),
    harness_file: Optional[Path] = typer.Option(
        None,
        "--harness-file",
        help="Alternative to --prompt-file (supports files and directories)",
    ),
    model: list[str] = typer.Option(
        ["gpt-5-high"],
        "--model",
        help="Model to run (can be specified multiple times)",
    ),
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        help="Sampling temperature",
    ),
    max_output_tokens: Optional[int] = typer.Option(
        None,
        "--max-output-tokens",
        help="Maximum output tokens",
    ),
    runs: int = typer.Option(
        1,
        "--runs",
        help="Number of parallel runs per model/scenario",
    ),
    ui: str = typer.Option(
        "auto",
        "--ui",
        help="UI mode: auto, plain, textual",
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        help="Path to .env file",
    ),
    mcp_url: Optional[str] = typer.Option(
        None,
        "--mcp-url",
        help="MCP server URL (overrides default JIRA MCP)",
    ),
    sql_runner_url: Optional[str] = typer.Option(
        None,
        "--sql-runner-url",
        help="SQL runner endpoint URL",
    ),
    max_steps: int = typer.Option(
        1000,
        "--max-steps",
        help="Maximum number of agent steps per run",
    ),
    max_concurrent_runs: int = typer.Option(
        20,
        "--max-concurrent-runs",
        help="Maximum number of concurrent runs",
    ),
    tool_call_limit: int = typer.Option(
        1000,
        "--tool-call-limit",
        help="Maximum number of tool calls per run",
    ),
) -> None:
    """Run benchmarks using the MCP Benchmark SDK."""
    # Load environment
    load_dotenv(override=False)
    if env_file:
        load_dotenv(env_file, override=True)

    # Determine harness file or directory
    harness_path = prompt_file or harness_file
    if not harness_path:
        console.print("[red]Error:[/red] --prompt-file or --harness-file required")
        raise typer.Exit(1)

    if not harness_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {harness_path}")
        raise typer.Exit(1)
    
    # Load all scenarios from file(s) or directory
    scenario_batches = []
    
    if harness_path.is_dir():
        # Find all JSON files in directory
        json_files = sorted(harness_path.glob("*.json"))
        if not json_files:
            console.print(f"[red]Error:[/red] No JSON files found in directory: {harness_path}")
            raise typer.Exit(1)
        
        console.print(f"[bold]Found {len(json_files)} harness file(s) in directory[/bold]")
        
        # Load scenarios from all files
        for json_file in json_files:
            try:
                scenarios = load_harness_file(json_file)
                if scenarios:
                    scenario_batches.append({
                        "file": json_file,
                        "alias": json_file.stem,
                        "scenarios": scenarios,
                    })
            except Exception as exc:
                console.print(f"[yellow]Warning: Failed to load {json_file.name}: {exc}[/yellow]")
        
        if not scenario_batches:
            console.print(f"[red]Error:[/red] No valid scenarios found in directory")
            raise typer.Exit(1)
            
        session_name = harness_path.name
    else:
        # Single file
        scenarios = load_harness_file(harness_path)
        scenario_batches.append({
            "file": harness_path,
            "alias": harness_path.stem,
            "scenarios": scenarios,
        })
        session_name = harness_path.stem

    # Run async main (all files together)
    try:
        asyncio.run(
            run_benchmark_batched(
                scenario_batches=scenario_batches,
                session_name=session_name,
                models=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                runs=runs,
                ui_mode=ui,
                mcp_url=mcp_url,
                sql_runner_url=sql_runner_url or DEFAULT_SQL_RUNNER_URL,
                max_steps=max_steps,
                max_concurrent_runs=max_concurrent_runs,
                tool_call_limit=tool_call_limit,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as exc:
        console.print(f"[red]Fatal error:[/red] {exc}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


async def run_benchmark_batched(
    scenario_batches: list[dict],
    session_name: str,
    models: list[str],
    temperature: float,
    max_output_tokens: Optional[int],
    runs: int,
    ui_mode: str,
    mcp_url: Optional[str],
    sql_runner_url: str,
    max_steps: int,
    max_concurrent_runs: int,
    tool_call_limit: int,
) -> None:
    """Run all scenario batches together in one session.
    
    Args:
        scenario_batches: List of dicts with 'file', 'alias', 'scenarios'
        session_name: Name for the session
        models: List of model names
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        runs: Number of runs per scenario
        ui_mode: UI mode
        mcp_url: Optional MCP URL
        sql_runner_url: SQL runner endpoint
        max_steps: Maximum steps
        max_concurrent_runs: Max concurrent runs
        tool_call_limit: Tool call limit
    """
    # Create MCP config
    if mcp_url:
        mcp_config = MCPConfig(name="custom", url=mcp_url, transport="streamable_http")
    else:
        mcp_config = DEFAULT_JIRA_MCP
    
    # Create single session directory for all files
    session_mgr = SessionManager(RESULTS_ROOT)
    session_dir = session_mgr.create_session_dir(session_name)
    
    # Determine UI mode
    total_scenarios = sum(len(batch["scenarios"]) for batch in scenario_batches)
    total_runs = len(models) * total_scenarios * runs
    use_textual = ui_mode == "textual" or (ui_mode == "auto" and total_runs > 1)
    
    # Build all run configs upfront
    all_run_configs = []
    for batch in scenario_batches:
        for model_name in models:
            for scenario in batch["scenarios"]:
                for run_num in range(1, runs + 1):
                    label_parts = [model_name]
                    if len(scenario_batches) > 1:
                        label_parts.append(batch["alias"])
                    label_parts.append(scenario.scenario_id)
                    if runs > 1:
                        label_parts.append(f"r{run_num}")
                    
                    run_label = "_".join(label_parts)
                    all_run_configs.append({
                        "label": run_label,
                        "model": model_name,
                        "scenario": scenario,
                        "batch_alias": batch["alias"],
                        "run_num": run_num,
                    })
    
    console.print(f"[dim]Total runs: {len(all_run_configs)} ({total_runs} tasks)[/dim]")
    console.print(f"[dim]Session directory: {session_dir}[/dim]\n")
    
    if use_textual:
        # Run with Textual UI - all files together
        await run_all_with_textual(
            run_configs=all_run_configs,
            session_dir=session_dir,
            session_name=session_name,
            mcp_config=mcp_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            sql_runner_url=sql_runner_url,
            max_steps=max_steps,
            max_concurrent_runs=max_concurrent_runs,
            tool_call_limit=tool_call_limit,
        )
    else:
        # Run in plain mode
        await run_all_plain(
            run_configs=all_run_configs,
            session_dir=session_dir,
            session_name=session_name,
            mcp_config=mcp_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            sql_runner_url=sql_runner_url,
            max_steps=max_steps,
            tool_call_limit=tool_call_limit,
        )


async def run_all_plain(
    run_configs: list[dict],
    session_dir: Path,
    session_name: str,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    sql_runner_url: str,
    max_steps: int,
    tool_call_limit: int,
) -> None:
    """Run all configs in plain console mode.
    
    Args:
        run_configs: List of run configuration dicts
        session_dir: Session directory
        session_name: Session name
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        sql_runner_url: SQL runner URL
        max_steps: Maximum steps
        tool_call_limit: Tool call limit
    """
    session_mgr = SessionManager(RESULTS_ROOT)
    run_summaries = []
    
    for cfg in run_configs:
        console.print(f"\n[bold cyan]Running:[/bold cyan] {cfg['label']}")
        
        try:
            # Create agent
            agent = create_agent_from_string(
                cfg["model"],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
            )
            
            # Convert scenario to task and extract verifiers separately
            task, verifiers = scenario_to_task(cfg["scenario"], [mcp_config])
            
            # Create RunContext
            run_context = RunContext(sql_runner_url=sql_runner_url)
            
            # Add console observer for output (with optional verifiers)
            observer = ConsoleObserver(
                console=console,
                prefix=cfg["label"],
                verifiers=verifiers,
                run_context=run_context,
            )
            run_context.add_observer(observer)
            
            # Run agent (with continuous verification via observer)
            result = await agent.run(task, max_steps=max_steps, run_context=run_context)
            
            # Run verification separately (CLI orchestrates this)
            verifier_runner = VerifierRunner()
            verifier_results = await verifier_runner.run_verifiers(verifiers, run_context)
            
            # Enrich result with verifier results
            result.verifier_results = verifier_results
            
            # Determine final success
            all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
            final_success = result.success and all_verifiers_passed
            
            # Update result with final status
            if not all_verifiers_passed:
                failed = [v.name for v in verifier_results if not v.success]
                result.error = f"Verifiers failed: {', '.join(failed)}"
            
            result.success = final_success
            
            # Save result
            result_file = session_mgr.save_result(
                session_dir,
                result,
                cfg["model"],
                f"{cfg['batch_alias']}_{cfg['scenario'].scenario_id}_run{cfg['run_num']}",
                metadata={
                    "scenario_name": cfg["scenario"].name,
                    "temperature": temperature,
                    "run_number": cfg["run_num"],
                },
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


async def run_benchmark_plain(
    harness_path: Path,
    models: list[str],
    temperature: float,
    max_output_tokens: Optional[int],
    runs: int,
    mcp_url: Optional[str],
    sql_runner_url: str,
    max_steps: int,
    tool_call_limit: int,
) -> None:
    """Run benchmarks in plain console mode."""
    if runs > 1:
        console.print(f"[yellow]Note: Running {runs} times per scenario[/yellow]")
    
    console.print(f"[bold]Loading harness:[/bold] {harness_path}")

    # Load scenarios
    try:
        scenarios = load_harness_file(harness_path)
    except Exception as exc:
        console.print(f"[red]Failed to load harness:[/red] {exc}")
        raise

    console.print(f"[green]Loaded {len(scenarios)} scenario(s)[/green]\n")

    # Create MCP config
    if mcp_url:
        mcp_config = MCPConfig(
            name="custom",
            url=mcp_url,
            transport="streamable_http",
        )
    else:
        mcp_config = DEFAULT_JIRA_MCP

    # Create session directory
    session_mgr = SessionManager(RESULTS_ROOT)
    session_dir = session_mgr.create_session_dir(harness_path.stem)
    console.print(f"[dim]Results will be saved to: {session_dir}[/dim]\n")

    run_summaries = []

    # Run each model
    for model_name in models:
        console.rule(f"[bold cyan]Model: {model_name}[/bold cyan]")

        try:
            # Create agent
            agent = create_agent_from_string(
                model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
            )
            console.print(f"[green]Created agent:[/green] {agent.__class__.__name__}\n")

        except Exception as exc:
            console.print(f"[red]Failed to create agent:[/red] {exc}")
            continue

        # Run each scenario (multiple times if runs > 1)
        for scenario in scenarios:
            for run_num in range(1, runs + 1):
                run_suffix = f" (run {run_num}/{runs})" if runs > 1 else ""
                console.print(f"\n[bold]Scenario:[/bold] {scenario.name}{run_suffix}")
                console.print(f"[dim]{scenario.description}[/dim]\n")

                try:
                    # Convert to Task and extract verifiers separately
                    task, verifiers = scenario_to_task(scenario, [mcp_config])

                    # Create RunContext with observer and SQL runner
                    run_context = RunContext(
                        sql_runner_url=sql_runner_url,
                    )

                    # Add console observer (with optional verifiers)
                    observer = ConsoleObserver(
                        console=console,
                        prefix=model_name,
                        verifiers=verifiers,
                        run_context=run_context,
                    )
                    run_context.add_observer(observer)

                    # Run agent (with continuous verification via observer)
                    result = await agent.run(task, max_steps=max_steps, run_context=run_context)
                    
                    # Run verification separately (CLI orchestrates this)
                    verifier_runner = VerifierRunner()
                    verifier_results = await verifier_runner.run_verifiers(verifiers, run_context)
                    
                    # Enrich result with verifier results
                    result.verifier_results = verifier_results
                    
                    # Determine final success
                    all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
                    final_success = result.success and all_verifiers_passed
                    
                    # Update result with final status
                    if not all_verifiers_passed:
                        failed = [v.name for v in verifier_results if not v.success]
                        result.error = f"Verifiers failed: {', '.join(failed)}"
                    
                    result.success = final_success

                    # Display result
                    if result.success:
                        console.print("\n[bold green]✓ Success[/bold green]")
                    else:
                        console.print(f"\n[bold red]✗ Failed:[/bold red] {result.error or 'Unknown error'}")

                    # Save result with run number in filename
                    result_file = session_mgr.save_result(
                        session_dir,
                        result,
                        model_name,
                        f"{scenario.scenario_id}_run{run_num}" if runs > 1 else scenario.scenario_id,
                        metadata={
                            "scenario_name": scenario.name,
                            "temperature": temperature,
                            "run_number": run_num,
                            "total_runs": runs,
                        },
                    )
                    console.print(f"[dim]Saved to: {result_file}[/dim]")

                    run_summaries.append({
                        "model": model_name,
                        "scenario_id": scenario.scenario_id,
                        "scenario_name": scenario.name,
                        "run_number": run_num,
                        "success": result.success,
                        "file": str(result_file.relative_to(session_dir)),
                    })

                except Exception as exc:
                    console.print(f"\n[red]Error running scenario:[/red] {exc}")
                    import traceback
                    traceback.print_exc()
                    continue

    # Save session manifest
    console.print("\n")
    console.rule("[bold]Summary[/bold]")

    manifest_file = session_mgr.save_session_manifest(
        session_dir,
        run_summaries,
        name=harness_path.stem,
    )

    console.print(f"\nSession manifest: {manifest_file}")
    console.print(f"\nTotal runs: {len(run_summaries)}")
    successful = sum(1 for r in run_summaries if r["success"])
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{len(run_summaries) - successful}[/red]")


async def run_all_with_textual(
    run_configs: list[dict],
    session_dir: Path,
    session_name: str,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    sql_runner_url: str,
    max_steps: int,
    max_concurrent_runs: int,
    tool_call_limit: int,
) -> None:
    """Run all configs with Textual UI (all files batched together).
    
    Args:
        run_configs: List of run configuration dicts
        session_dir: Session directory path
        session_name: Session name
        mcp_config: MCP configuration
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens
        sql_runner_url: SQL runner endpoint
        max_steps: Maximum agent steps
        max_concurrent_runs: Maximum concurrent runs
        tool_call_limit: Maximum tool calls per run
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
        # Limit concurrency to prevent resource exhaustion
        semaphore = asyncio.Semaphore(max_concurrent_runs)
        tasks = []

        for cfg in run_configs:
            task = run_single_task(
                label=cfg["label"],
                model_name=cfg["model"],
                scenario=cfg["scenario"],
                batch_alias=cfg["batch_alias"],
                run_num=cfg["run_num"],
                queue=queues[cfg["label"]],
                mcp_config=mcp_config,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                sql_runner_url=sql_runner_url,
                session_mgr=session_mgr,
                session_dir=session_dir,
                semaphore=semaphore,
                max_steps=max_steps,
                tool_call_limit=tool_call_limit,
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


async def run_single_task(
    label: str,
    model_name: str,
    scenario,
    batch_alias: str,
    run_num: int,
    queue: asyncio.Queue,
    mcp_config: MCPConfig,
    temperature: float,
    max_output_tokens: Optional[int],
    sql_runner_url: str,
    session_mgr: SessionManager,
    session_dir: Path,
    semaphore: asyncio.Semaphore,
    max_steps: int,
    tool_call_limit: int,
):
    """Run a single task with Textual observer."""
    async with semaphore:
        try:
            # Create agent
            agent = create_agent_from_string(
                model_name,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                tool_call_limit=tool_call_limit,
            )

            # Convert to Task and extract verifiers separately
            task, verifiers = scenario_to_task(scenario, [mcp_config])

            # Create RunContext with Textual observer
            run_context = RunContext(
                sql_runner_url=sql_runner_url,
            )

            # Add Textual observer for UI display (with optional verifiers)
            observer = TextualObserver(
                queue=queue,
                width=100,
                verifiers=verifiers,
                run_context=run_context,
            )
            run_context.add_observer(observer)

            # Send initial status
            await queue.put(f"[bold]Starting {label}...[/bold]\n")

            # Run agent (with continuous verification via observer)
            result = await agent.run(task, max_steps=max_steps, run_context=run_context)

            # Run verification separately (CLI orchestrates this)
            verifier_runner = VerifierRunner()
            verifier_results = await verifier_runner.run_verifiers(verifiers, run_context)

            # Enrich result with verifier results
            result.verifier_results = verifier_results

            # Determine final success
            all_verifiers_passed = all(v.success for v in verifier_results) if verifier_results else True
            final_success = result.success and all_verifiers_passed
            
            # Update result with final status
            if not all_verifiers_passed:
                failed = [v.name for v in verifier_results if not v.success]
                result.error = f"Verifiers failed: {', '.join(failed)}"
            
            result.success = final_success

            # Save result with batch alias in filename
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
            )

            # Send completion status
            if result.success:
                await queue.put("[bold green]✓ Completed successfully[/bold green]\n")
            else:
                await queue.put(f"[bold red]✗ Failed: {result.error or 'Unknown'}[/bold red]\n")

            return {
                "model": model_name,
                "scenario_id": scenario.scenario_id,
                "run_number": run_num,
                "success": result.success,
                "file": str(result_file.relative_to(session_dir)),
            }

        except Exception as exc:
            await queue.put(f"[bold red]Error: {exc}[/bold red]\n")
            import traceback
            await queue.put(traceback.format_exc())
            raise


if __name__ == "__main__":
    app()

