"""Main CLI entry point for MCP Benchmark - Simplified and refactored."""

import asyncio
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Suppress transformers warning (imported by langchain_anthropic)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress gRPC fork warnings (common with Google libraries + asyncio)
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "1"
os.environ["GRPC_POLL_STRATEGY"] = "poll"
os.environ["GRPC_VERBOSITY"] = "ERROR"

# Suppress absl logging warnings (Google libraries)
os.environ["ABSL_LOGGING_LEVEL"] = "ERROR"

# Suppress Google GenAI schema warnings (additionalProperties not supported)
logging.getLogger("google.ai.generativelanguage_v1beta.services.generative_service").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*additionalProperties.*")

import typer
from dotenv import load_dotenv
from turing_rl_sdk.agents.mcp import MCPConfig
from turing_rl_sdk.harness.loader import load_harness_file
from rich.console import Console

from .config import DEFAULT_JIRA_MCP
from .runners import run_all_quiet, run_all_plain, run_all_with_textual
from .session.manager import SessionManager
from .config import RESULTS_ROOT

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
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature (defaults to 0.1, or 1.0 for Claude models)",
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
        "quiet",
        "--ui",
        help="UI mode: auto, plain, textual, quiet (progress bar + summary only)",
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
    timeout: Optional[float] = typer.Option(
        None,
        "--timeout",
        help="Timeout in seconds for LLM API calls (default: 600)",
    ),
    max_retries: Optional[int] = typer.Option(
        None,
        "--max-retries",
        help="Maximum number of retries for failed LLM calls (default: 3)",
    ),
    langsmith: bool = typer.Option(
        False,
        "--langsmith",
        help="Enable LangSmith tracing for observability (requires LANGCHAIN_API_KEY)",
    ),
    langsmith_project: Optional[str] = typer.Option(
        None,
        "--langsmith-project",
        help="LangSmith project name (defaults to MCP_BENCHMARK)",
    ),
    langfuse: bool = typer.Option(
        False,
        "--langfuse",
        help="Enable Langfuse tracing (requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)",
    ),
    langfuse_public_key: Optional[str] = typer.Option(
        None,
        "--langfuse-public-key",
        help="Langfuse public API key (overrides LANGFUSE_PUBLIC_KEY)",
    ),
    langfuse_secret_key: Optional[str] = typer.Option(
        None,
        "--langfuse-secret-key",
        help="Langfuse secret API key (overrides LANGFUSE_SECRET_KEY)",
    ),
    langfuse_base_url: Optional[str] = typer.Option(
        None,
        "--langfuse-base-url",
        help="Langfuse base URL (overrides LANGFUSE_BASE_URL)",
    ),
    langfuse_environment: Optional[str] = typer.Option(
        None,
        "--langfuse-environment",
        help="Langfuse tracing environment label (overrides LANGFUSE_TRACING_ENVIRONMENT)",
    ),
    langfuse_release: Optional[str] = typer.Option(
        None,
        "--langfuse-release",
        help="Release identifier reported to Langfuse (overrides LANGFUSE_RELEASE)",
    ),
) -> None:
    """Run benchmarks using the MCP Benchmark SDK."""
    # Load environment
    load_dotenv(override=False)
    if env_file:
        load_dotenv(env_file, override=True)
    
    # Generate unique session_id for this benchmark invocation
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    
    langfuse_active = False

    # Configure LangSmith tracing if requested
    if langsmith:
        from turing_rl_sdk import configure_langsmith
        
        try:
            env_vars = configure_langsmith(
                project_name=langsmith_project,
                enabled=True
            )
            console.print(f"[green]✓ LangSmith tracing enabled[/green]")
            console.print(f"  Project: [cyan]{env_vars.get('LANGCHAIN_PROJECT')}[/cyan]")
            console.print(f"  Session: [dim]{session_id}[/dim]")
            console.print(f"  View traces at: [dim]https://smith.langchain.com[/dim]")
        except ValueError as e:
            console.print(f"[yellow]⚠ LangSmith setup failed: {e}[/yellow]")
            console.print(f"[dim]Continuing without tracing...[/dim]")
    elif os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
        console.print(f"[green]✓ LangSmith tracing enabled via environment[/green]")
        project = os.environ.get("LANGCHAIN_PROJECT", "default")
        console.print(f"  Project: [cyan]{project}[/cyan]")
        console.print(f"  Session: [dim]{session_id}[/dim]")

    # Configure Langfuse tracing if requested
    if langfuse:
        from turing_rl_sdk import configure_langfuse

        try:
            env_vars = configure_langfuse(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                base_url=langfuse_base_url,
                environment=langfuse_environment,
                release=langfuse_release,
                enabled=True,
            )
            _print_langfuse_status(session_id, env_vars)
            langfuse_active = True
        except ValueError as e:
            console.print(f"[yellow]⚠ Langfuse setup failed: {e}[/yellow]")
            console.print(f"[dim]Continuing without Langfuse tracing...[/dim]")
    else:
        try:
            from turing_rl_sdk import is_langfuse_enabled
        except ImportError:
            is_langfuse_enabled = lambda: False  # type: ignore

        if is_langfuse_enabled():
            _print_langfuse_status(session_id)
            langfuse_active = True

    # Determine harness file or directory
    harness_path = prompt_file or harness_file
    if not harness_path:
        console.print("[red]Error:[/red] --prompt-file or --harness-file required")
        raise typer.Exit(1)

    if not harness_path.exists():
        console.print(f"[red]Error:[/red] Path not found: {harness_path}")
        raise typer.Exit(1)
    
    # Load all scenarios from file(s) or directory
    scenario_batches = _load_scenarios(harness_path)
    if not scenario_batches:
        console.print(f"[red]Error:[/red] No valid scenarios found")
        raise typer.Exit(1)
    
    session_name = harness_path.name if harness_path.is_dir() else harness_path.stem

    contains_claude_model = _contains_claude_model(model)
    if temperature is None:
        resolved_temperature = 1.0 if contains_claude_model else 0.1
        if contains_claude_model:
            console.print("[dim]Detected Claude model(s); defaulting temperature to 1.0[/dim]")
    else:
        resolved_temperature = temperature

    # Run async main (all files together)
    try:
        asyncio.run(
            run_benchmark_batched(
                scenario_batches=scenario_batches,
                session_name=session_name,
                models=model,
                temperature=resolved_temperature,
                max_output_tokens=max_output_tokens,
                runs=runs,
                ui_mode=ui,
                mcp_url=mcp_url,
                max_steps=max_steps,
                max_concurrent_runs=max_concurrent_runs,
                tool_call_limit=tool_call_limit,
                session_id=session_id,
                langfuse_tracing=langfuse_active,
                timeout=timeout,
                max_retries=max_retries,
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


def _contains_claude_model(models: list[str]) -> bool:
    """Return True if any requested model targets Claude."""
    return any(_is_claude_model(model_name) for model_name in models)


def _is_claude_model(model_name: str) -> bool:
    """Normalize provider prefixes and check if the underlying model is Claude."""
    normalized = model_name.lower()
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1]
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized.startswith("claude")


def _load_scenarios(harness_path: Path) -> list[dict]:
    """Load scenarios from file or directory.
    
    Returns:
        List of dicts with 'file', 'alias', 'scenarios' keys
    """
    scenario_batches = []
    
    if harness_path.is_dir():
        # Find all JSON files in directory
        json_files = sorted(harness_path.glob("*.json"))
        if not json_files:
            return []
        
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
    else:
        # Single file
        scenarios = load_harness_file(harness_path)
        scenario_batches.append({
            "file": harness_path,
            "alias": harness_path.stem,
            "scenarios": scenarios,
        })
    
    return scenario_batches


def _print_langfuse_status(session_id: str, env_overrides: Optional[dict[str, str]] = None) -> None:
    env_overrides = env_overrides or {}

    def _get(name: str, default: Optional[str] = None) -> Optional[str]:
        return env_overrides.get(name) or os.environ.get(name) or default

    base_url = _get("LANGFUSE_BASE_URL") or _get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    environment = _get("LANGFUSE_TRACING_ENVIRONMENT", "default")

    console.print("[green]✓ Langfuse tracing enabled[/green]")
    console.print(f"  Environment: [cyan]{environment}[/cyan]")
    console.print(f"  Session: [dim]{session_id}[/dim]")
    console.print(f"  View traces at: [dim]{base_url}[/dim]")




async def run_benchmark_batched(
    scenario_batches: list[dict],
    session_name: str,
    models: list[str],
    temperature: float,
    max_output_tokens: Optional[int],
    runs: int,
    ui_mode: str,
    mcp_url: Optional[str],
    max_steps: int,
    max_concurrent_runs: int,
    tool_call_limit: int,
    session_id: str,
    langfuse_tracing: bool,
    timeout: Optional[float],
    max_retries: Optional[int],
) -> None:
    """Run all scenario batches together in one session."""
    # Create MCP config
    if mcp_url:
        mcp_config = MCPConfig(name="custom", url=mcp_url, transport="streamable_http")
    else:
        mcp_config = DEFAULT_JIRA_MCP
    
    # Create single session directory for all files
    session_mgr = SessionManager(RESULTS_ROOT)
    session_dir = session_mgr.create_session_dir(session_name)
    
    # Build all run configs upfront
    all_run_configs = _build_run_configs(scenario_batches, models, runs)
    
    # Select and run UI mode
    if ui_mode == "quiet":
        console.print(f"[dim]Running {len(all_run_configs)} benchmarks in quiet mode...[/dim]")
        console.print(f"[dim]Session directory: {session_dir}[/dim]\n")
        
        await run_all_quiet(
            run_configs=all_run_configs,
            session_dir=session_dir,
            session_name=session_name,
            mcp_config=mcp_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_steps=max_steps,
            max_concurrent_runs=max_concurrent_runs,
            tool_call_limit=tool_call_limit,
            session_id=session_id,
            console=console,
            langfuse_tracing=langfuse_tracing,
            timeout=timeout,
            max_retries=max_retries,
        )
    elif ui_mode == "textual" or (ui_mode == "auto" and len(all_run_configs) > 1):
        console.print(f"[dim]Total runs: {len(all_run_configs)}[/dim]")
        console.print(f"[dim]Session directory: {session_dir}[/dim]\n")
        
        await run_all_with_textual(
            run_configs=all_run_configs,
            session_dir=session_dir,
            session_name=session_name,
            mcp_config=mcp_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_steps=max_steps,
            max_concurrent_runs=max_concurrent_runs,
            tool_call_limit=tool_call_limit,
            session_id=session_id,
            console=console,
            langfuse_tracing=langfuse_tracing,
            timeout=timeout,
            max_retries=max_retries,
        )
    else:
        console.print(f"[dim]Total runs: {len(all_run_configs)}[/dim]")
        console.print(f"[dim]Session directory: {session_dir}[/dim]\n")
        
        await run_all_plain(
            run_configs=all_run_configs,
            session_dir=session_dir,
            session_name=session_name,
            mcp_config=mcp_config,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_steps=max_steps,
            tool_call_limit=tool_call_limit,
            session_id=session_id,
            console=console,
            langfuse_tracing=langfuse_tracing,
            timeout=timeout,
            max_retries=max_retries,
        )


def _build_run_configs(
    scenario_batches: list[dict],
    models: list[str],
    runs: int,
) -> list[dict]:
    """Build run configurations for all model/scenario combinations."""
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
    
    return all_run_configs


if __name__ == "__main__":
    app()
