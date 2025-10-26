"""Command line interface for running MCP-backed LangChain benchmarks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import execute_scenario, extract_ai_message_content
from .mcp_loader import MCPConfig, load_tools_from_mcp
from .prompts import load_scenarios
from .providers import PROVIDERS, create_chat_model, resolve_provider_for_model
from .run_logging import ConsoleRunLogger, RunLogger, TextualRunLogger
from .textual_ui import MultiRunApp
from .verifier import evaluate_verifiers
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
    artifact_dir: Path,
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

    resolved_model_name = model or PROVIDERS[provider].default_model
    model_instance = create_chat_model(
        provider,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    actual_model = getattr(model_instance, "model_name", resolved_model_name)
    logger.print(
        Panel(
            f"Using provider [bold]{provider}[/bold] with model "
            f"[bold]{actual_model}[/bold]",
            title=f"Run {run_label} LLM",
        )
    )

    run_summary = []
    verifier_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    try:
        for scenario in scenarios:
            responses = await execute_scenario(
                model_instance,
                tools,
                scenario,
                tool_call_limit=DEFAULT_TOOL_CALL_LIMIT,
                logger=logger,
                sql_runner_url=DEFAULT_SQL_RUNNER_URL,
                database_id=run_database_id,
                verifier_client=verifier_client,
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

            final_results = await evaluate_verifiers(
                scenario,
                sql_runner_url=DEFAULT_SQL_RUNNER_URL,
                database_id=run_database_id,
                client=verifier_client,
            )
            if final_results:
                logger.update_verifier_status(final_results)
            run_summary.append((scenario.scenario_id, final_results))
    finally:
        await verifier_client.aclose()

    artifacts = logger.get_artifacts()
    artifacts["run_label"] = run_label
    artifacts["database_id"] = run_database_id
    artifacts["provider"] = provider
    artifacts["model"] = actual_model
    artifacts["scenarios"] = []
    for scenario_id, final_results in run_summary:
        artifacts["scenarios"].append(
            {
                "scenario_id": scenario_id,
                "verifiers": [
                    {
                        "name": result.verifier.name or result.verifier.verifier_type,
                        "comparison": result.comparison_type,
                        "expected": result.expected_value,
                        "actual": result.actual_value,
                        "success": result.success,
                        "error": result.error,
                    }
                    for result in final_results
                ],
            }
        )

    artifact_file = artifact_dir / f"run_{run_label}.json"
    artifact_file.write_text(json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run": run_label,
        "database_id": run_database_id,
        "summary": run_summary,
        "provider": provider,
        "model": actual_model,
        "artifact_path": str(artifact_file),
    }


async def _run_plain(
    *,
    run_configs: list[dict[str, Optional[str]]],
    scenarios,
    temperature: float,
    max_output_tokens: Optional[int],
    artifact_dir: Path,
) -> None:
    multiple_runs = len(run_configs) > 1

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        prefix = f"Run {label}" if multiple_runs else None
        return ConsoleRunLogger(console, prefix=prefix)

    tasks = [
        _execute_run(
            run_label=config["label"],
            logger_factory=logger_factory,
            scenarios=scenarios,
            provider=config["provider"],
            model=config["model"],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            artifact_dir=artifact_dir,
        )
        for config in run_configs
    ]

    results = [await tasks[0]] if len(tasks) == 1 else await asyncio.gather(*tasks)
    _render_summary(results)


async def _run_textual(
    *,
    run_configs: list[dict[str, Optional[str]]],
    scenarios,
    temperature: float,
    max_output_tokens: Optional[int],
    artifact_dir: Path,
) -> None:
    run_labels = [config["label"] for config in run_configs]
    queues: Dict[str, asyncio.Queue[object]] = {label: asyncio.Queue() for label in run_labels}

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        return TextualRunLogger(queues[label])

    app = MultiRunApp(run_labels, queues)
    app_task = asyncio.create_task(app.run_async())

    try:
        results = await asyncio.gather(
            *[
                _execute_run(
                    run_label=config["label"],
                    logger_factory=logger_factory,
                    scenarios=scenarios,
                    provider=config["provider"],
                    model=config["model"],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    artifact_dir=artifact_dir,
                )
                for config in run_configs
            ]
        )
    finally:
        await app.action_quit()
        await app_task

    _render_summary(results)


def _prepare_session_dir(base_name: str) -> Path:
    idx = 1
    while True:
        candidate = Path(f"{base_name}_{idx}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        idx += 1


def _build_run_configs(
    model_entries: list[tuple[str, Optional[str]]], runs: int
) -> list[dict[str, Optional[str]]]:
    configs: list[dict[str, Optional[str]]] = []
    multiple_models = len(model_entries) > 1
    for provider_name, model_name in model_entries:
        resolved_model = model_name or PROVIDERS[provider_name].default_model
        alias_base = resolved_model.replace("/", "_")
        if multiple_models:
            alias_base = f"{provider_name}-{alias_base}"
        for idx in range(1, runs + 1):
            label = alias_base if runs == 1 else f"{alias_base}-{idx}"
            configs.append(
                {
                    "label": label,
                    "provider": provider_name,
                    "model": model_name,
                }
            )
    return configs


def _render_summary(results: list[dict]) -> None:
    summary_table = Table(title="Run summary")
    summary_table.add_column("Run", justify="right")
    summary_table.add_column("Provider")
    summary_table.add_column("Model")
    summary_table.add_column("Database ID")
    summary_table.add_column("Scenarios")
    summary_table.add_column("Verifiers")
    summary_table.add_column("Artifact")

    for result in results:
        total_scenarios = len(result["summary"])
        total_verifiers = sum(len(item[1]) for item in result["summary"])
        passed = sum(
            1 for _, verifier_results in result["summary"] for vr in verifier_results if vr.success
        )
        summary_table.add_row(
            result["run"],
            result.get("provider", "-"),
            result.get("model", "-"),
            result["database_id"],
            str(total_scenarios),
            f"{passed}/{total_verifiers}" if total_verifiers else "0",
            result.get("artifact_path", "-"),
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
    model: List[str] = typer.Option(  # type: ignore[assignment]
        None,
        "--model",
        help="Run a specific model (provider inferred automatically). Repeat to compare models; optionally prefix with '<provider>:' to force a provider.",
        show_default=False,
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
    raw_models = list(model) if model else []
    model_entries: list[tuple[str, Optional[str]]] = []
    if not raw_models:
        model_entries.append((resolve_provider_for_model(None), None))
    else:
        for raw_model in raw_models:
            provider_hint: Optional[str] = None
            model_name: Optional[str] = raw_model
            if raw_model and ":" in raw_model:
                hint, value = raw_model.split(":", 1)
                provider_hint = hint.strip() or None
                model_name = value.strip() or None
            try:
                provider_key = resolve_provider_for_model(model_name, provider_hint=provider_hint)
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            model_entries.append((provider_key, model_name))

    run_configs = _build_run_configs(model_entries, runs)

    ui_mode = ui.lower()
    valid_ui = {"auto", "plain", "textual"}
    if ui_mode not in valid_ui:
        raise typer.BadParameter(
            f"Invalid ui mode '{ui}'. Expected one of: {', '.join(sorted(valid_ui))}"
        )
    if ui_mode == "auto":
        ui_mode = "textual" if len(run_configs) > 1 else "plain"

    session_dir = _prepare_session_dir(prompt_source.stem)
    console.print(Panel(f"Storing run artifacts in [bold]{session_dir}[/bold]"))

    asyncio.run(
        _run_textual(
            run_configs=run_configs,
            scenarios=scenarios,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            artifact_dir=session_dir,
        )
        if ui_mode == "textual"
        else _run_plain(
            run_configs=run_configs,
            scenarios=scenarios,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            artifact_dir=session_dir,
        )
    )


def main() -> None:
    """Entry point for `python -m jira_mcp_benchmark`."""
    app()
