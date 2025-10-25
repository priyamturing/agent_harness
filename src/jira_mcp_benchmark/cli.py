"""Command line interface for running MCP-backed LangChain benchmarks."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, Dict, Optional
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import execute_scenario, extract_ai_message_content
from .mcp_loader import MCPConfig, load_tools_from_mcp
from .prompts import load_scenarios
from .providers import PROVIDERS, create_chat_model
from .run_logging import ConsoleRunLogger, RunLogger, TextualRunLogger
from .textual_ui import MultiRunApp
from .verifier import render_verifier_summary, run_verifiers
from dotenv import load_dotenv

console = Console()
app = typer.Typer(help="Run Jira MCP benchmarks with LangChain agents.")

DEFAULT_MCP_URL = "http://localhost:8015/mcp"
DEFAULT_MCP_TRANSPORT = "streamable_http"
DEFAULT_TOOL_CALL_LIMIT = 1000

_mcp_base = DEFAULT_MCP_URL.rstrip("/")
if _mcp_base.endswith("/mcp"):
    _mcp_base = _mcp_base[: -len("/mcp")]
DEFAULT_SQL_RUNNER_URL = f"{_mcp_base}/api/sql-runner"


def _load_environment(env_file: Optional[Path]) -> None:
    """Load environment variables from .env files before running."""

    load_dotenv(override=False)
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)


async def _execute_run(
    *,
    run_label: str,
    logger_factory: Callable[[str, str], RunLogger],
    scenarios,
    provider: str,
    model: Optional[str],
    temperature: float,
    max_output_tokens: Optional[int],
) -> dict:
    run_database_id = str(uuid4())
    logger = logger_factory(run_label, run_database_id)

    logger.print(
        Panel(
            f"Loading tools from MCP server at [bold]{DEFAULT_MCP_URL}[/bold] "
            f"using transport [bold]{DEFAULT_MCP_TRANSPORT}[/bold]\n"
            f"[dim]x-database-id={run_database_id}[/dim]",
            title=f"Run {run_label} setup",
        )
    )

    tools = await load_tools_from_mcp(
        MCPConfig(
            name="jira-mcp-local",
            transport=DEFAULT_MCP_TRANSPORT,
            url=DEFAULT_MCP_URL,
            headers={"x-database-id": run_database_id},
        )
    )

    model_instance = create_chat_model(
        provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    logger.print(
        Panel(
            f"Using provider [bold]{provider}[/bold] with model "
            f"[bold]{model or PROVIDERS[provider].default_model}[/bold]",
            title=f"Run {run_label} LLM",
        )
    )

    run_summary = []

    for scenario in scenarios:
        responses = await execute_scenario(
            model_instance,
            tools,
            scenario,
            tool_call_limit=DEFAULT_TOOL_CALL_LIMIT,
            console=logger,  # type: ignore[arg-type]
        )
        final_response = responses[-1] if responses else None
        if final_response:
            logger.rule(f"Scenario {scenario.scenario_id} final response")
            final_text, final_reasoning, _ = extract_ai_message_content(final_response)
            for idx, chunk in enumerate(final_reasoning, start=1):
                label_text = "Reasoning" if len(final_reasoning) == 1 else f"Reasoning {idx}"
                logger.print(f"[yellow]{label_text}[/yellow]: {chunk}")
            if final_text:
                logger.print(final_text)

        verifier_results = await run_verifiers(
            scenario,
            sql_runner_url=DEFAULT_SQL_RUNNER_URL,
            database_id=run_database_id,
            console=logger,  # type: ignore[arg-type]
        )
        if verifier_results:
            logger.rule(f"Scenario {scenario.scenario_id} verifications")
            render_verifier_summary(verifier_results, console=logger)  # type: ignore[arg-type]
        run_summary.append((scenario.scenario_id, verifier_results))

    return {
        "run": run_label,
        "database_id": run_database_id,
        "summary": run_summary,
    }


async def _run_plain(
    *,
    runs: int,
    scenarios,
    provider: str,
    model: Optional[str],
    temperature: float,
    max_output_tokens: Optional[int],
) -> None:
    run_labels = [str(i + 1) for i in range(runs)]

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        prefix = f"Run {label}" if runs > 1 else None
        return ConsoleRunLogger(console, prefix=prefix)

    tasks = [
        _execute_run(
            run_label=label,
            logger_factory=logger_factory,
            scenarios=scenarios,
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        for label in run_labels
    ]

    results = [await tasks[0]] if runs == 1 else await asyncio.gather(*tasks)
    _render_summary(results)


async def _run_textual(
    *,
    runs: int,
    scenarios,
    provider: str,
    model: Optional[str],
    temperature: float,
    max_output_tokens: Optional[int],
) -> None:
    run_labels = [str(i + 1) for i in range(runs)]
    queues: Dict[str, asyncio.Queue[str]] = {label: asyncio.Queue() for label in run_labels}

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        return TextualRunLogger(queues[label])

    app = MultiRunApp(run_labels, queues)
    app_task = asyncio.create_task(app.run_async())

    try:
        results = await asyncio.gather(
            *[
                _execute_run(
                    run_label=label,
                    logger_factory=logger_factory,
                    scenarios=scenarios,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                for label in run_labels
            ]
        )
    finally:
        await app.action_quit()
        await app_task

    _render_summary(results)


def _render_summary(results: list[dict]) -> None:
    summary_table = Table(title="Run summary")
    summary_table.add_column("Run", justify="right")
    summary_table.add_column("Database ID")
    summary_table.add_column("Scenarios")
    summary_table.add_column("Verifiers")

    for result in results:
        total_scenarios = len(result["summary"])
        total_verifiers = sum(len(item[1]) for item in result["summary"])
        passed = sum(
            1 for _, verifier_results in result["summary"] for vr in verifier_results if vr.success
        )
        summary_table.add_row(
            result["run"],
            result["database_id"],
            str(total_scenarios),
            f"{passed}/{total_verifiers}" if total_verifiers else "0",
        )

    console.print(summary_table)


@app.command()
def run(  # noqa: D417 - typer handles CLI docs
    prompt_file: Path = typer.Option(
        Path("old_sample_new_system_1_benchmark.json"),
        exists=True,
        readable=True,
        help="Benchmark JSON file containing scenarios.",
    ),
    harness_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Optional harness JSON file. Overrides --prompt-file when provided.",
    ),
    provider: str = typer.Option(
        "openai",
        case_sensitive=False,
        help="LLM provider to use (openai, anthropic, xai).",
    ),
    model: Optional[str] = typer.Option(
        None,
        help="Override the default model for the selected provider.",
    ),
    temperature: float = typer.Option(
        0.1,
        min=0.0,
        max=1.0,
        help="Sampling temperature for the chat model.",
    ),
    max_output_tokens: Optional[int] = typer.Option(
        None,
        help="Optional maximum number of output tokens.",
    ),
    env_file: Optional[Path] = typer.Option(
        None,
        exists=True,
        readable=True,
        help="Load environment variables from this .env file before execution.",
    ),
    runs: int = typer.Option(
        1,
        min=1,
        help="Number of parallel runs to execute with unique MCP databases.",
    ),
    ui: str = typer.Option(
        "auto",
        case_sensitive=False,
        help="User interface mode: 'auto', 'plain', or 'textual'.",
    ),
) -> None:
    _load_environment(env_file)

    provider_key = provider.lower()
    if provider_key not in PROVIDERS:
        raise typer.BadParameter(
            f"Unknown provider '{provider}'. "
            f"Expected one of: {', '.join(sorted(PROVIDERS))}"
        )

    prompt_source = harness_file or prompt_file

    console.print(
        Panel(
            f"Loading scenarios from [bold]{prompt_source}[/bold]",
            title="Harness",
        )
    )
    scenarios = load_scenarios(prompt_source)
    if not scenarios:
        raise typer.BadParameter(f"No scenarios found in {prompt_source}")

    ui_mode = ui.lower()
    valid_ui = {"auto", "plain", "textual"}
    if ui_mode not in valid_ui:
        raise typer.BadParameter(
            f"Invalid ui mode '{ui}'. Expected one of: {', '.join(sorted(valid_ui))}"
        )
    if ui_mode == "auto":
        ui_mode = "textual" if runs > 1 else "plain"

    asyncio.run(
        _run_textual(
            runs=runs,
            scenarios=scenarios,
            provider=provider_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if ui_mode == "textual"
        else _run_plain(
            runs=runs,
            scenarios=scenarios,
            provider=provider_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    )


def main() -> None:
    """Entry point for `python -m jira_mcp_benchmark`."""
    app()
