"""Command line interface for running MCP-backed LangChain benchmarks."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Union
from uuid import uuid4

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import execute_scenario, extract_ai_message_content
from .mcp_loader import MCPConfig, load_tools_from_mcp
from .prompts import Scenario, load_scenarios
from .providers import PROVIDERS, create_chat_model, resolve_provider_for_model
from .run_logging import ConsoleRunLogger, RunLogger, TextualRunLogger
from .session_picker import BackgroundRunDisplay, SessionDisplay, SessionPickerApp
from .textual_ui import MultiRunApp
from .background_viewer import BackgroundRunViewer
from .verifier import evaluate_verifiers
from .gym_loader import get_gym_registry, GymConfig
from dotenv import load_dotenv
from . import background_manager

console = Console()
app = typer.Typer(help="Run Jira MCP benchmarks with LangChain agents.")

DEFAULT_TOOL_CALL_LIMIT = 1000
SESSION_MANIFEST_FILENAME = "session.json"
VIEW_SELECT_SENTINEL = "__SELECT__"
RESULTS_ROOT = Path("results")


def _sanitize_label(raw: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
    sanitized = sanitized.strip("_") or "run"
    return sanitized


class ScenarioBatch(NamedTuple):
    alias: str
    source: Path
    scenarios: Sequence[Scenario]


def _compose_run_label(*, config_label: str, batch_alias: str, multiple_batches: bool) -> str:
    if multiple_batches:
        base = f"{batch_alias}-{config_label}" if config_label else batch_alias
    else:
        base = config_label or batch_alias
    return _sanitize_label(base)


def _prepare_artifacts_for_export(raw_artifacts: dict) -> dict:
    """Strip render-only fields from artifacts before exporting them."""

    artifacts = deepcopy(raw_artifacts)
    artifacts.pop("verifier_history", None)
    artifacts.pop("status_stream", None)
    conversation = artifacts.get("conversation")
    if isinstance(conversation, list):
        artifacts["conversation"] = [
            entry
            for entry in conversation
            if not (isinstance(entry, dict) and entry.get("type") == "verifier_status")
        ]
    return artifacts


# Ensure bare "--view" is treated as interactive selection by injecting a sentinel value
if "--view" in sys.argv:
    try:
        idx = sys.argv.index("--view")
    except ValueError:
        idx = -1
    if idx != -1:
        needs_value = idx == len(sys.argv) - 1 or sys.argv[idx + 1].startswith("-")
        if needs_value:
            sys.argv.insert(idx + 1, VIEW_SELECT_SENTINEL)


def _load_environment(env_file: Optional[Path]) -> None:
    """Load environment variables from .env files before running."""

    load_dotenv(override=False)
    if env_file:
        load_dotenv(dotenv_path=env_file, override=True)


async def _execute_run(
    *,
    run_label: str,
    logger_factory: Callable[[str, str], RunLogger],
    scenarios: Sequence[Scenario],
    provider: str,
    model: Optional[str],
    temperature: float,
    max_output_tokens: Optional[int],
    artifact_dir: Path,
    prompt_path: Path,
    prompt_alias: str,
    gym_config: GymConfig,
) -> dict:
    run_database_id = str(uuid4())
    logger = logger_factory(run_label, run_database_id)

    logger.print(
        Panel(
            f"Loading tools from MCP server at [bold]{gym_config.mcp_url}[/bold] "
            f"using transport [bold]{gym_config.mcp_transport}[/bold]\n"
            f"Gym: [bold cyan]{gym_config.name}[/bold cyan] - {gym_config.description}\n"
            f"[dim]x-database-id={run_database_id}[/dim]",
            title=f"Run {run_label} setup",
        )
    )

    logger.print(
        Panel(
            f"Executing scenarios from [bold]{prompt_path.name}[/bold]\n[dim]{prompt_path}[/dim]",
            title=f"Run {run_label} prompts",
        )
    )

    tools = await load_tools_from_mcp(
        MCPConfig(
            name=f"jira-mcp-{gym_config.name}",
            transport=gym_config.mcp_transport,
            url=gym_config.mcp_url,
            headers=gym_config.get_headers_for_run(run_database_id),
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
    run_failed = False
    failure_reason: Optional[str] = None
    failed_scenario_id: Optional[str] = None

    try:
        for scenario in scenarios:
            responses = await execute_scenario(
                model_instance,
                tools,
                scenario,
                tool_call_limit=DEFAULT_TOOL_CALL_LIMIT,
                logger=logger,
                sql_runner_url=gym_config.sql_runner_url,
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
                sql_runner_url=gym_config.sql_runner_url,
                database_id=run_database_id,
                client=verifier_client,
            )
            if final_results:
                logger.update_verifier_status(final_results)
            run_summary.append((scenario.scenario_id, final_results))
    except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
        raise
    except Exception as exc:  # noqa: BLE001
        run_failed = True
        failure_reason = str(exc).strip() or exc.__class__.__name__
        failed_scenario_id = getattr(scenario, "scenario_id", None)
        scenario_label = (
            f" scenario {failed_scenario_id}" if failed_scenario_id is not None else ""
        )
        logger.print(
            "[bold red]Run {label} failed while processing{scenario_label}: "
            "{reason}[/bold red]".format(
                label=run_label,
                scenario_label=scenario_label,
                reason=failure_reason,
            )
        )
        logger.log_message(
            "system",
            f"Run {run_label} failed{scenario_label}: {failure_reason}",
        )
    finally:
        await verifier_client.aclose()

    artifacts = _prepare_artifacts_for_export(logger.get_artifacts())
    artifacts["run_label"] = run_label
    artifacts["database_id"] = run_database_id
    artifacts["provider"] = provider
    artifacts["model"] = actual_model
    artifacts["prompt_file"] = str(prompt_path)
    artifacts["prompt_alias"] = prompt_alias
    artifacts["gym_name"] = gym_config.name
    artifacts["gym_url"] = gym_config.mcp_url
    artifacts["status"] = "failed" if run_failed else "completed"
    if run_failed and failure_reason:
        artifacts["failure_reason"] = failure_reason
    if run_failed and failed_scenario_id is not None:
        artifacts["failed_scenario_id"] = failed_scenario_id
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
        "prompt_file": str(prompt_path),
        "prompt_alias": prompt_alias,
        "artifact_path": str(artifact_file),
        "gym_name": gym_config.name,
        "gym_url": gym_config.mcp_url,
        "status": "failed" if run_failed else "completed",
        "failure_reason": failure_reason,
        "failed_scenario_id": failed_scenario_id,
    }


async def _run_plain(
    *,
    run_configs: list[dict[str, Optional[str]]],
    scenario_batches: Sequence[ScenarioBatch],
    temperature: float,
    max_output_tokens: Optional[int],
    artifact_dir: Path,
    gym_config: GymConfig,
) -> List[dict]:
    run_items: List[dict] = []
    multiple_batches = len(scenario_batches) > 1
    for batch in scenario_batches:
        for config in run_configs:
            config_label = config.get("label") or "run"
            run_label = _compose_run_label(
                config_label=config_label,
                batch_alias=batch.alias,
                multiple_batches=multiple_batches,
            )
            run_items.append(
                {
                    "label": run_label,
                    "config": config,
                    "scenarios": batch.scenarios,
                    "prompt_path": batch.source,
                    "prompt_alias": batch.alias,
                }
            )

    multiple_runs = len(run_items) > 1

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        prefix = f"Run {label}" if multiple_runs else None
        return ConsoleRunLogger(console, prefix=prefix)

    tasks = [
        _execute_run(
            run_label=item["label"],
            logger_factory=logger_factory,
            scenarios=item["scenarios"],
            provider=item["config"]["provider"],
            model=item["config"]["model"],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            artifact_dir=artifact_dir,
            prompt_path=item["prompt_path"],
            prompt_alias=item["prompt_alias"],
            gym_config=gym_config,
        )
        for item in run_items
    ]

    results = [await tasks[0]] if len(tasks) == 1 else await asyncio.gather(*tasks)
    _render_summary(results)
    return list(results)


async def _run_textual(
    *,
    run_configs: list[dict[str, Optional[str]]],
    scenario_batches: Sequence[ScenarioBatch],
    temperature: float,
    max_output_tokens: Optional[int],
    artifact_dir: Path,
    gym_config: GymConfig,
) -> List[dict]:
    run_items: List[dict] = []
    multiple_batches = len(scenario_batches) > 1
    for batch in scenario_batches:
        for config in run_configs:
            config_label = config.get("label") or "run"
            run_label = _compose_run_label(
                config_label=config_label,
                batch_alias=batch.alias,
                multiple_batches=multiple_batches,
            )
            run_items.append(
                {
                    "label": run_label,
                    "config": config,
                    "scenarios": batch.scenarios,
                    "prompt_path": batch.source,
                    "prompt_alias": batch.alias,
                }
            )

    run_labels = [item["label"] for item in run_items]
    queues: Dict[str, asyncio.Queue[object]] = {label: asyncio.Queue() for label in run_labels}

    def logger_factory(label: str, _db_id: str) -> RunLogger:
        return TextualRunLogger(queues[label])

    app = MultiRunApp(run_labels, queues)
    app_task = asyncio.create_task(app.run_async())

    try:
        results = await asyncio.gather(
            *[
                _execute_run(
                    run_label=item["label"],
                    logger_factory=logger_factory,
                    scenarios=item["scenarios"],
                    provider=item["config"]["provider"],
                    model=item["config"]["model"],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    artifact_dir=artifact_dir,
                    prompt_path=item["prompt_path"],
                    prompt_alias=item["prompt_alias"],
                    gym_config=gym_config,
                )
                for item in run_items
            ]
        )
    finally:
        await app.action_quit()
        await app_task

    _render_summary(results)
    return list(results)


def _prepare_session_dir(base_name: str) -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    idx = 1
    while True:
        candidate = RESULTS_ROOT / f"{base_name}_{idx}"
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
        alias_base = _sanitize_label(resolved_model.replace("/", "_"))
        if multiple_models:
            alias_base = _sanitize_label(f"{provider_name}-{alias_base}")
        for idx in range(1, runs + 1):
            label = alias_base if runs == 1 else _sanitize_label(f"{alias_base}-{idx}")
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
    summary_table.add_column("Status")
    summary_table.add_column("Prompt")
    summary_table.add_column("Provider")
    summary_table.add_column("Model")
    summary_table.add_column("Database ID")
    summary_table.add_column("Scenarios")
    summary_table.add_column("Verifiers")
    summary_table.add_column("Artifact")

    prompt_model_stats = defaultdict(lambda: {"total": 0, "passed": 0})

    for result in results:
        total_scenarios = len(result["summary"])
        total_verifiers = sum(len(item[1]) for item in result["summary"])
        passed = sum(
            1 for _, verifier_results in result["summary"] for vr in verifier_results if vr.success
        )
        status_value = (result.get("status") or "completed").lower()
        if status_value == "failed":
            status_text = "[red]FAILED[/red]"
        elif status_value in {"completed", "success"}:
            status_text = "[green]OK[/green]"
        else:
            status_text = status_value.upper()
        prompt_value = result.get("prompt_alias")
        if not prompt_value:
            prompt_path_raw = result.get("prompt_file")
            prompt_value = Path(prompt_path_raw).stem if prompt_path_raw else "-"
        provider_value = result.get("provider") or "-"
        model_value = result.get("model") or "-"
        summary_items = result.get("summary") or []
        all_success = True
        has_verifiers = False
        for _, verifier_results in summary_items:
            for vr in verifier_results or []:
                has_verifiers = True
                if not getattr(vr, "success", False):
                    all_success = False
                    break
            if not all_success:
                break
        passed_all = has_verifiers and all_success and status_value != "failed"
        stats_entry = prompt_model_stats[(prompt_value, provider_value, model_value)]
        stats_entry["total"] += 1
        if passed_all:
            stats_entry["passed"] += 1
        summary_table.add_row(
            result["run"],
            status_text,
            prompt_value,
            result.get("provider", "-"),
            result.get("model", "-"),
            result["database_id"],
            str(total_scenarios),
            f"{passed}/{total_verifiers}" if total_verifiers else "0",
            result.get("artifact_path", "-"),
        )

    console.print(summary_table)

    if prompt_model_stats:
        aggregate_table = Table(title="Full verifier passes by prompt/model")
        aggregate_table.add_column("Prompt")
        aggregate_table.add_column("Provider")
        aggregate_table.add_column("Model")
        aggregate_table.add_column("Full Passes", justify="right")
        for (prompt_label, provider_label, model_label), metrics in sorted(
            prompt_model_stats.items()
        ):
            aggregate_table.add_row(
                prompt_label,
                provider_label,
                model_label,
                f"{metrics['passed']}/{metrics['total']}",
            )
        console.print(aggregate_table)


def _summarize_models(results: List[dict]) -> list[dict]:
    counter: Counter[tuple[Optional[str], Optional[str]]] = Counter()
    for result in results:
        provider = result.get("provider")
        model = result.get("model")
        counter[(provider, model)] += 1
    summary: list[dict] = []
    for (provider, model), count in counter.items():
        summary.append(
            {
                "provider": provider,
                "model": model,
                "count": count,
            }
        )
    summary.sort(key=lambda item: (item.get("provider") or "", item.get("model") or ""))
    return summary


def _format_model_summary(summary: list[dict]) -> str:
    parts: list[str] = []
    for entry in summary:
        provider = entry.get("provider") or "-"
        model = entry.get("model") or "-"
        label = f"{provider}:{model}" if provider != "-" else model
        count = entry.get("count", 1)
        if count and count > 1:
            label = f"{label} ×{count}"
        parts.append(label)
    return ", ".join(parts) if parts else "N/A"


def _extract_verifier_counts_from_runs(runs: Sequence[dict] | None) -> tuple[int, int]:
    total_passed = 0
    total_checks = 0
    if not runs:
        return total_passed, total_checks
    for run in runs:
        if not isinstance(run, dict):
            continue
        scenarios = run.get("scenarios") or []
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            total = scenario.get("verifiers_total")
            passed = scenario.get("verifiers_passed")
            if isinstance(total, int) and total >= 0:
                total_count = max(total, 0)
                total_checks += total_count
                if isinstance(passed, int):
                    total_passed += max(0, min(passed, total_count))
    return total_passed, total_checks


def _format_verifier_summary_text(passed: int, total: int) -> str:
    if total <= 0:
        return "-"
    percentage = (passed / total) * 100 if total else 0.0
    if percentage.is_integer():
        percentage_text = f"{int(percentage)}%"
    else:
        percentage_text = f"{percentage:.1f}%"
    return f"{passed}/{total} ({percentage_text})"


def _write_session_manifest(
    *,
    session_dir: Path,
    prompt_source: Path,
    prompt_files: Sequence[Path] | None,
    ui_mode: str,
    results: List[dict],
) -> None:
    created_at = datetime.now(timezone.utc)
    model_summary = _summarize_models(results)

    runs_payload: list[dict] = []
    for result in results:
        scenario_entries: list[dict] = []
        for scenario_id, verifier_results in result.get("summary", []):
            verifier_list = list(verifier_results or [])
            total = len(verifier_list)
            passed = sum(1 for item in verifier_list if getattr(item, "success", False))
            scenario_entries.append(
                {
                    "scenario_id": scenario_id,
                    "verifiers_total": total,
                    "verifiers_passed": passed,
                }
            )

        runs_payload.append(
            {
                "label": result.get("run"),
                "status": result.get("status", "completed"),
                "failure_reason": result.get("failure_reason"),
                "failed_scenario_id": result.get("failed_scenario_id"),
                "provider": result.get("provider"),
                "model": result.get("model"),
                "artifact_path": result.get("artifact_path"),
                "database_id": result.get("database_id"),
                "prompt_file": result.get("prompt_file"),
                "prompt_alias": result.get("prompt_alias"),
                "scenarios": scenario_entries,
            }
        )

    created_display = created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    prompt_source_name = prompt_source.stem if prompt_source.is_file() else prompt_source.name
    prompt_file_entries = prompt_files or [prompt_source]
    passed_count, total_count = _extract_verifier_counts_from_runs(runs_payload)
    manifest = {
        "created_at": created_at.isoformat(),
        "created_at_display": created_display,
        "harness_file": str(prompt_source),
        "harness_name": prompt_source_name,
        "ui_mode": ui_mode,
        "run_count": len(results),
        "model_summary": model_summary,
        "model_summary_text": _format_model_summary(model_summary),
        "runs": runs_payload,
        "prompt_files": [str(item) for item in prompt_file_entries],
        "verifier_totals": {"passed": passed_count, "total": total_count},
        "verifier_summary_text": _format_verifier_summary_text(passed_count, total_count),
    }
    manifest["display_name"] = (
        f"{created_display} • {prompt_source_name} • {manifest['model_summary_text']}"
    )

    manifest_path = session_dir / "session.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_sessions(base_dir: Path | None = None) -> list[tuple[Path, dict]]:
    if base_dir is not None:
        search_roots = [base_dir]
    else:
        search_roots = []
        if RESULTS_ROOT.exists() and RESULTS_ROOT.is_dir():
            search_roots.append(RESULTS_ROOT)
        search_roots.append(Path("."))

    sessions: list[tuple[Path, dict]] = []
    seen_paths: set[Path] = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            manifest_path = entry / SESSION_MANIFEST_FILENAME
            if not manifest_path.is_file():
                continue
            if entry in seen_paths:
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            created = data.get("created_at")
            sort_key: datetime | None = None
            if isinstance(created, str):
                try:
                    sort_key = datetime.fromisoformat(created)
                except ValueError:
                    sort_key = None
            data["_session_dir"] = str(entry)
            data["_created_at"] = sort_key
            sessions.append((entry, data))
            seen_paths.add(entry)
    sessions.sort(
        key=lambda item: item[1].get("_created_at") or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return sessions


def _print_session_list(sessions: list[tuple[Path, dict]]) -> None:
    if not sessions:
        console.print("No saved sessions found.")
        return

    table = Table(title="Saved Sessions")
    table.add_column("#", justify="right")
    table.add_column("Session")
    table.add_column("Runs", justify="right")
    table.add_column("Models")
    table.add_column("Verifiers", justify="right")
    table.add_column("Path", overflow="fold")

    for idx, (path, manifest) in enumerate(sessions, start=1):
        display = manifest.get("display_name") or path.name
        run_count = manifest.get("run_count", "-")
        models = manifest.get("model_summary_text", "-")
        verifiers = manifest.get("verifier_summary_text", "-")
        table.add_row(str(idx), display, str(run_count), models, verifiers, str(path))

    console.print(table)


def _resolve_session_path(raw: str | None) -> Path:
    if not raw:
        raise typer.BadParameter("Session path is required.")
    candidate = Path(raw).expanduser()
    if not candidate.exists():
        alt = RESULTS_ROOT / Path(raw)
        if alt.exists():
            candidate = alt
    if not candidate.exists():
        raise typer.BadParameter(f"Session directory '{raw}' does not exist.")
    if not candidate.is_dir():
        raise typer.BadParameter(f"'{raw}' is not a directory.")
    manifest_path = candidate / SESSION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise typer.BadParameter(
            f"No session manifest found in '{candidate}'. Expected {SESSION_MANIFEST_FILENAME}."
        )
    return candidate


def _normalize_manifest(manifest: dict) -> dict:
    summary = manifest.get("model_summary")
    if summary and "model_summary_text" not in manifest:
        manifest["model_summary_text"] = _format_model_summary(summary)
    if "model_summary_text" not in manifest:
        manifest["model_summary_text"] = "-"

    totals_raw = manifest.get("verifier_totals")
    passed_count: Optional[int] = None
    total_count: Optional[int] = None
    if isinstance(totals_raw, dict):
        raw_passed = totals_raw.get("passed")
        raw_total = totals_raw.get("total")
        if isinstance(raw_passed, int) and isinstance(raw_total, int):
            passed_count = raw_passed
            total_count = raw_total
    if passed_count is None or total_count is None:
        passed_count, total_count = _extract_verifier_counts_from_runs(manifest.get("runs", []))
        manifest["verifier_totals"] = {
            "passed": passed_count,
            "total": total_count,
        }
    summary_text = manifest.get("verifier_summary_text")
    if not isinstance(summary_text, str) or not summary_text.strip():
        manifest["verifier_summary_text"] = _format_verifier_summary_text(passed_count, total_count)

    created_display = manifest.get("created_at_display")
    if not created_display:
        created_raw = manifest.get("created_at")
        if isinstance(created_raw, str):
            try:
                created_dt = datetime.fromisoformat(created_raw)
                created_display = created_dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z").strip()
            except ValueError:
                created_display = created_raw
    if not created_display:
        created_display = "Unknown time"

    if "display_name" not in manifest:
        harness_name = manifest.get("harness_name") or Path(
            manifest.get("harness_file", "session")
        ).stem
        display = f"{created_display} • {harness_name}"
        model_text = manifest.get("model_summary_text")
        if model_text and model_text != "-":
            display = f"{display} • {model_text}"
        manifest["display_name"] = display

    return manifest


def _load_manifest(session_dir: Path) -> dict:
    manifest_path = session_dir / SESSION_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise typer.BadParameter(
            f"No session manifest found in '{session_dir}'. Expected {SESSION_MANIFEST_FILENAME}."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(
            f"Failed to read session manifest from '{session_dir}': {exc}"
        ) from exc
    return _normalize_manifest(manifest)


def _select_session_interactive(
    sessions: list[tuple[Path, dict]],
    *,
    background_runs: Optional[list[background_manager.BackgroundRun]] = None,
    use_textual: bool
) -> Optional[Union[Path, str]]:
    if not sessions and not background_runs:
        console.print("No saved sessions or background runs found.")
        return None

    if use_textual:
        session_displays = [
            SessionDisplay(
                path=path,
                display_name=manifest.get("display_name") or path.name,
                runs=manifest.get("run_count", 0),
                model_summary=manifest.get("model_summary_text", "-"),
                verifier_summary=manifest.get("verifier_summary_text", "-"),
                harness_name=manifest.get("harness_name", ""),
                status="completed",  # Completed sessions
            )
            for path, manifest in sessions
        ]

        bg_displays = []
        if background_runs:
            bg_displays = [
                BackgroundRunDisplay(
                    run_id=bg_run.run_id,
                    display_name=bg_run.run_id[-8:],  # Short ID for display
                    status=bg_run.status,
                    progress=f"{bg_run.progress['passed']}/{bg_run.progress['total']}",
                    started_at=background_manager.format_time_ago(bg_run.started_at),
                    harness_name=bg_run.harness_name,
                    model_summary=bg_run.model_summary,
                    run_configs=bg_run.run_configs,
                    scenario_batches=bg_run.scenario_batches,
                )
                for bg_run in background_runs
                if bg_run.status == "running"
            ]

        async def _run_picker() -> Optional[Union[Path, str]]:
            picker = SessionPickerApp(session_displays, background_runs=bg_displays)
            wait_task = asyncio.create_task(picker.wait_for_selection())
            await picker.run_async()
            return await wait_task

        try:
            selection = asyncio.run(_run_picker())
            if selection is not None:
                return selection
            console.print("Cancelled.")
            return None
        except Exception as exc:  # noqa: BLE001
            console.print(
                f"[yellow]Failed to launch Textual picker ({exc!r}). Falling back to console selection.[/yellow]"
            )

    # Console fallback
    all_items: list[tuple[str, Union[Path, str]]] = []
    
    # Add background runs
    if background_runs:
        console.print("[bold]In Progress:[/bold]")
        for bg_run in background_runs:
            if bg_run.status == "running":
                console.print(
                    f"  {len(all_items) + 1}. {bg_run.harness_name} • {bg_run.model_summary} "
                    f"({bg_run.progress['passed']}/{bg_run.progress['total']} verifiers) - "
                    f"{background_manager.format_time_ago(bg_run.started_at)}"
                )
                all_items.append(("background", bg_run.run_id))
    
    # Add completed sessions
    if sessions:
        console.print("[bold]Completed:[/bold]")
        for path, manifest in sessions:
            display_name = manifest.get("display_name") or path.name
            console.print(f"  {len(all_items) + 1}. {display_name}")
            all_items.append(("session", path))
    
    max_index = len(all_items)
    if max_index == 0:
        console.print("No items to select.")
        return None

    while True:
        response = console.input(
            f"Select an item [1-{max_index}] (Enter for 1, or 'q' to cancel): "
        ).strip()
        if not response:
            item_type, item_value = all_items[0]
            return item_value
        if response.lower() in {"q", "quit", "exit"}:
            return None
        if response.isdigit():
            value = int(response)
            if 1 <= value <= max_index:
                item_type, item_value = all_items[value - 1]
                return item_value
        console.print("[red]Invalid selection.[/red] Please try again.")


def _artifact_path_for_run(session_dir: Path, label: str, artifact_hint: Optional[str]) -> Path:
    candidate = session_dir / f"run_{label}.json"
    if candidate.exists():
        return candidate
    if artifact_hint:
        hint_path = Path(artifact_hint)
        if hint_path.exists():
            return hint_path
        if not hint_path.is_absolute():
            candidate_try = session_dir / hint_path
            if candidate_try.exists():
                return candidate_try
            candidate_try = session_dir / hint_path.name
            if candidate_try.exists():
                return candidate_try
    return candidate


def _load_run_artifacts(session_dir: Path, manifest: dict) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for run in manifest.get("runs", []):
        label = run.get("label")
        if not label:
            continue
        artifact_path = _artifact_path_for_run(session_dir, label, run.get("artifact_path"))
        if not artifact_path.exists():
            console.print(f"[yellow]Warning[/yellow]: missing artifact for run '{label}'.")
            continue
        try:
            artifacts[label] = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to read artifact for run '{label}': {exc}[/red]")
    return artifacts


def _build_status_payload(snapshot: list[dict]) -> list[dict]:
    payload: list[dict] = []
    for item in snapshot:
        name = item.get("name")
        comparison = item.get("comparison") or "-"
        expected = repr(item.get("expected"))
        actual = "-" if item.get("error") else repr(item.get("actual"))
        status = "PASS" if item.get("success") else "FAIL"
        payload.append(
            {
                "name": name,
                "comparison": comparison,
                "expected": expected,
                "actual": actual,
                "status": status,
                "error": item.get("error"),
            }
        )
    return payload


async def _stream_run_to_queue(label: str, artifact: dict, queue: asyncio.Queue[object]) -> None:
    for chunk in artifact.get("log_stream", []):
        if chunk:
            queue.put_nowait(chunk)
        await asyncio.sleep(0)

    status_stream = artifact.get("status_stream") or []
    if status_stream:
        for payload in status_stream:
            queue.put_nowait({"type": "status", "data": payload})
            await asyncio.sleep(0)
    else:
        for snapshot in artifact.get("verifier_history", []):
            queue.put_nowait({"type": "status", "data": _build_status_payload(snapshot)})
            await asyncio.sleep(0)


async def _replay_textual_session(manifest: dict, artifacts: dict[str, dict]) -> None:
    runs = [run for run in manifest.get("runs", []) if run.get("label") in artifacts]
    if not runs:
        console.print("No run artifacts available to replay.")
        return

    run_labels = [run.get("label") for run in runs if run.get("label")]
    queues: Dict[str, asyncio.Queue[object]] = {label: asyncio.Queue() for label in run_labels}  # type: ignore[index]

    app = MultiRunApp(run_labels, queues)
    app_task = asyncio.create_task(app.run_async())

    try:
        await asyncio.gather(
            *[
                _stream_run_to_queue(label, artifacts[label], queues[label])
                for label in run_labels
            ]
        )
    except Exception:
        if not app_task.done():
            await app.action_quit()
        raise
    finally:
        await app_task


def _render_conversation_plain(conversation: list) -> None:
    for entry in conversation:
        if isinstance(entry, dict):
            entry_type = entry.get("type")
            if entry_type == "message" or (entry_type is None and "role" in entry):
                role = entry.get("role", "unknown")
                content = entry.get("content", "")
                reasoning = entry.get("reasoning")
                if reasoning:
                    content = f"{content}\nReasoning: {' | '.join(reasoning)}"
                console.print(f"[{role}] {content}")
            elif entry_type == "tool_call":
                console.print(
                    f"→ Invoking tool {entry.get('tool')} with args "
                    f"{json.dumps(entry.get('args'))}"
                )
            elif entry_type == "tool_result":
                status = "error" if entry.get("error") else "result"
                console.print(f"← Tool {entry.get('tool')} {status}: {entry.get('output')}")
            elif entry_type == "verifier_status":
                payload = entry.get("results") or []
                console.print("Verifier status snapshot:")
                for item in payload:
                    console.print(
                        f"  - {item.get('name')}: "
                        f"{'PASS' if item.get('success') else 'FAIL'} "
                        f"(expected {item.get('expected')}, actual {item.get('actual')})"
                    )
        else:
            console.print(str(entry))


def _render_status_table_plain(snapshot: list[dict]) -> None:
    if not snapshot:
        return
    table = Table(title="Verifier status", show_lines=True)
    table.add_column("Verifier")
    table.add_column("Comparison")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Status")
    table.add_column("Error")

    for item in snapshot:
        status = "PASS" if item.get("success") else "FAIL"
        table.add_row(
            str(item.get("name")),
            str(item.get("comparison") or "-"),
            repr(item.get("expected")),
            repr(item.get("actual")),
            status,
            str(item.get("error") or ""),
        )
    console.print(table)


def _replay_plain_session(manifest: dict, artifacts: dict[str, dict]) -> None:
    runs = manifest.get("runs", [])
    if not runs:
        console.print("No run artifacts available to replay.")
        return

    console.print(
        Panel(
            f"Viewing session [bold]{manifest.get('display_name', 'session')}[/bold]",
            title="Replay",
        )
    )

    for run in runs:
        label = run.get("label")
        provider = run.get("provider") or "-"
        model = run.get("model") or "-"
        console.rule(f"Run {label} • {provider}:{model}")
        artifact = artifacts.get(label)
        if not artifact:
            console.print(f"[yellow]Artifact missing for run '{label}'.[/yellow]")
            continue

        log_stream = artifact.get("log_stream")
        if log_stream:
            for chunk in log_stream:
                if chunk:
                    console.print(chunk, end="")
        else:
            conversation = artifact.get("conversation", [])
            if conversation:
                _render_conversation_plain(conversation)
            else:
                console.print("[dim]No conversation or logs recorded for this run.[/dim]")

        history = artifact.get("verifier_history")
        if history:
            _render_status_table_plain(history[-1])


def _replay_session(session_dir: Path, manifest: dict, ui_mode: str) -> None:
    artifacts = _load_run_artifacts(session_dir, manifest)
    if not artifacts:
        console.print(f"No replayable data found in '{session_dir}'.")
        return

    if ui_mode == "textual":
        asyncio.run(_replay_textual_session(manifest, artifacts))
    else:
        _replay_plain_session(manifest, artifacts)


def _attach_to_background_run(run_id: str, ui_mode: str) -> None:
    """Attach to a running background job and show live progress."""
    run_state = background_manager.read_run_state(run_id)
    if not run_state:
        console.print(f"[red]Background run {run_id} not found.[/red]")
        return
    
        console.print(
            Panel(
                f"Attaching to background run [bold]{run_id}[/bold]\n"
                f"Status: {run_state.status}\n"
                f"Progress: {run_state.progress['passed']}/{run_state.progress['total']} verifiers\n"
                f"Session: {run_state.session_dir}",
                title="Background Run Viewer",
            )
        )
    
    if run_state.status != "running":
        # Run already completed, just show the session
        session_dir = Path(run_state.session_dir)
        if session_dir.exists():
            manifest = _load_manifest(session_dir)
            _replay_session(session_dir, manifest, ui_mode)
        else:
            console.print(f"[yellow]Session directory not found: {session_dir}[/yellow]")
        return
    
    # For running jobs, tail the logs
    session_dir = Path(run_state.session_dir)
    log_files = list(session_dir.glob("run_*.log"))
    
    if not log_files:
        console.print("[yellow]No log files found yet. Run may still be starting...[/yellow]")
        return
    
    if ui_mode == "plain":
        # Plain mode: just tail the logs to console
        console.print("[dim]Press Ctrl+C to stop watching[/dim]\n")
        import time
        
        # Track file positions
        file_positions: dict[Path, int] = {f: 0 for f in log_files}
        
        try:
            while True:
                # Check if run is still running
                current_state = background_manager.read_run_state(run_id)
                if not current_state or current_state.status != "running":
                    console.print("\n[green]Background run completed.[/green]")
                    
                    # Auto-switch to replay mode
                    if current_state and current_state.status == "completed":
                        console.print("[cyan]Switching to replay mode in 3 seconds... (Ctrl+C to cancel)[/cyan]")
                        try:
                            time.sleep(3)
                            manifest = _load_manifest(session_dir)
                            _replay_session(session_dir, manifest, ui_mode)
                        except KeyboardInterrupt:
                            console.print("\n[dim]Cancelled replay.[/dim]")
                    break
                
                # Read new content from each log file
                for log_file in log_files:
                    if not log_file.exists():
                        continue
                    
                    with open(log_file, "r", encoding="utf-8") as f:
                        f.seek(file_positions[log_file])
                        new_content = f.read()
                        if new_content:
                            console.print(new_content, end="")
                        file_positions[log_file] = f.tell()
                
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\n[dim]Detached from background run.[/dim]")
    else:
        # Textual mode: use the fancy live viewer
        viewer = BackgroundRunViewer(run_id, session_dir)
        result = viewer.run()
        
        # If viewer returns a session directory, switch to replay mode
        if result:
            manifest = _load_manifest(result)
            _replay_session(result, manifest, ui_mode)


def _handle_view_command(view: Optional[str], ui_mode: str) -> None:
    sessions = _collect_sessions()
    background_runs = background_manager.list_background_runs()

    if view == "list":
        # Print background runs
        if background_runs:
            console.print("[bold]In Progress Background Runs:[/bold]")
            for bg_run in background_runs:
                if bg_run.status == "running":
                    console.print(
                        f"  • {bg_run.run_id} - {bg_run.harness_name} | "
                        f"{bg_run.progress['passed']}/{bg_run.progress['total']} verifiers | "
                        f"{background_manager.format_time_ago(bg_run.started_at)}"
                    )
        _print_session_list(sessions)
        return

    if view in {VIEW_SELECT_SENTINEL, "", None}:
        selection = _select_session_interactive(
            sessions,
            background_runs=background_runs,
            use_textual=(ui_mode == "textual")
        )
        if selection is None:
            console.print("Cancelled.")
            return
        
        # Check if selection is a background run (str) or session (Path)
        if isinstance(selection, str):
            # It's a background run ID
            _attach_to_background_run(selection, ui_mode)
            return
        else:
            session_dir = selection
    else:
        session_dir = _resolve_session_path(view)

    manifest = _load_manifest(session_dir)
    _replay_session(session_dir, manifest, ui_mode)


@app.command()
def list_gyms(  # noqa: D417 - typer handles CLI docs
    gym_config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to gyms.json configuration file. If not provided, searches for gyms.json in current directory or project root.",
        show_default=False,
    ),
) -> None:
    """List all available gym configurations."""
    try:
        registry = get_gym_registry(gym_config_file)
        gyms = registry.list_gyms()
        default_gym = registry.get_default_gym_name()
        
        if not gyms:
            console.print("[yellow]No gyms configured.[/yellow]")
            console.print(f"Configuration file: {registry.config_path}")
            return
        
        table = Table(title="Available Gyms")
        table.add_column("Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("MCP URL", style="blue")
        table.add_column("Transport", style="magenta")
        table.add_column("Default", style="green")
        
        for gym in sorted(gyms, key=lambda g: g.name):
            is_default = "✓" if gym.name == default_gym else ""
            table.add_row(
                gym.name,
                gym.description,
                gym.mcp_url,
                gym.mcp_transport,
                is_default,
            )
        
        console.print(table)
        console.print(f"\n[dim]Configuration file: {registry.config_path}[/dim]")
        
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("\n[yellow]Example gyms.json:[/yellow]")
        console.print("""
{
  "gyms": {
    "local": {
      "name": "local",
      "description": "Local development server",
      "mcp_url": "http://localhost:8015/mcp",
      "mcp_transport": "streamable_http",
      "sql_runner_url": "http://localhost:8015/api/sql-runner"
    }
  },
  "default": "local"
}
""")
        raise typer.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error loading gym configuration:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command()
def cleanup(  # noqa: D417 - typer handles CLI docs
    stale_hours: int = typer.Option(
        24,
        "--stale-hours",
        help="Number of hours after which a running job is considered stale.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Show what would be cleaned up without actually doing it.",
    ),
) -> None:
    """Clean up stale background runs stuck in 'running' status."""
    from rich.console import Console
    from rich.table import Table
    from . import background_manager
    
    console = Console()
    
    stale_runs = background_manager.find_stale_runs(stale_hours)
    
    if not stale_runs:
        console.print("[green]✓[/green] No stale runs found.")
        return
    
    table = Table(title=f"Stale Runs (running > {stale_hours}h)")
    table.add_column("Run ID", style="cyan")
    table.add_column("Started", style="yellow")
    table.add_column("Progress", style="magenta")
    table.add_column("Model", style="blue")
    
    for run in stale_runs:
        time_ago = background_manager.format_time_ago(run.started_at)
        progress = f"{run.progress['passed']}/{run.progress['total']}"
        table.add_row(run.run_id, time_ago, progress, run.model_summary)
    
    console.print(table)
    
    if dry_run:
        console.print(
            f"\n[yellow]DRY RUN:[/yellow] Would mark {len(stale_runs)} run(s) as failed."
        )
        console.print("Run with [bold]--no-dry-run[/bold] to actually clean them up.")
    else:
        cleaned = background_manager.cleanup_stale_runs(stale_hours, dry_run=False)
        console.print(
            f"\n[green]✓[/green] Marked {len(cleaned)} stale run(s) as failed."
        )


@app.command()
def run(  # noqa: D417 - typer handles CLI docs
    prompt_file: Optional[Path] = typer.Option(
        None,
        "--prompt-file",
        help="Benchmark JSON file or directory containing scenario files.",
        show_default=False,
    ),
    harness_file: Optional[Path] = typer.Option(
        None,
        help="Optional harness JSON file or directory. Overrides --prompt-file when provided.",
        show_default=False,
    ),
    view: Optional[str] = typer.Option(
        None,
        "--view",
        help=(
            "Replay a saved session. "
            "Use '--view list' to list sessions, '--view' to choose interactively, "
            "or '--view <path>' to open a specific session folder."
        ),
        flag_value=VIEW_SELECT_SENTINEL,
        show_default=False,
    ),
    gym: Optional[str] = typer.Option(
        None,
        "--gym",
        help="Name of the gym (MCP server environment) to run against. If not provided, uses the default gym from gyms.json.",
        show_default=False,
    ),
    gym_config_file: Optional[Path] = typer.Option(
        None,
        "--gym-config",
        help="Path to gyms.json configuration file. If not provided, searches for gyms.json in current directory or project root.",
        show_default=False,
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
    background: bool = typer.Option(
        False,
        "--background",
        help="Run benchmarks in background. Use --view to monitor progress.",
    ),
) -> None:
    _load_environment(env_file)

    ui_mode_raw = ui.lower()
    valid_ui = {"auto", "plain", "textual"}
    if ui_mode_raw not in valid_ui:
        raise typer.BadParameter(
            f"Invalid ui mode '{ui}'. Expected one of: {', '.join(sorted(valid_ui))}"
        )

    raw_cli_args = sys.argv[1:]
    view_without_value = any(arg == "--view" for arg in raw_cli_args)

    if view is not None or view_without_value:
        forbidden = []
        if model:
            forbidden.append("--model")
        if runs != 1:
            forbidden.append("--runs")
        if temperature != 0.1:
            forbidden.append("--temperature")
        if max_output_tokens is not None:
            forbidden.append("--max-output-tokens")
        if harness_file is not None:
            forbidden.append("--harness-file")
        if background:
            forbidden.append("--background")
        if forbidden:
            raise typer.BadParameter(
                f"--view cannot be combined with {', '.join(forbidden)}."
            )
        effective_view = view
        if view is None and view_without_value:
            effective_view = VIEW_SELECT_SENTINEL
        playback_ui = "textual" if ui_mode_raw == "auto" else ui_mode_raw
        _handle_view_command(effective_view, playback_ui)
        return

    if harness_file is None and prompt_file is None:
        raise typer.BadParameter("Provide either --prompt-file or --harness-file.")

    # Load gym configuration
    try:
        registry = get_gym_registry(gym_config_file)
        gym_config = registry.get_gym(gym)
        console.print(
            Panel(
                f"Using gym: [bold cyan]{gym_config.name}[/bold cyan]\n"
                f"{gym_config.description}\n"
                f"MCP URL: [bold]{gym_config.mcp_url}[/bold]",
                title="Gym Configuration",
            )
        )
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("\n[yellow]Tip:[/yellow] Create a gyms.json file or use --gym-config to specify the path.")
        console.print("Run [bold]python -m jira_mcp_benchmark list-gyms --help[/bold] for more information.")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("\nRun [bold]python -m jira_mcp_benchmark list-gyms[/bold] to see available gyms.")
        raise typer.Exit(1) from exc

    prompt_source = harness_file or prompt_file
    param_name = "--harness-file" if harness_file else "--prompt-file"
    if not prompt_source.exists():
        raise typer.BadParameter(f"{param_name}: path '{prompt_source}' does not exist.")

    if prompt_source.is_file():
        console.print(
            Panel(
                f"Loading scenarios from [bold]{prompt_source}[/bold]",
                title="Harness",
            )
        )
        prompt_files = [prompt_source]
    elif prompt_source.is_dir():
        prompt_files = sorted(
            [
                entry
                for entry in prompt_source.iterdir()
                if entry.is_file() and entry.suffix.lower() == ".json"
            ],
            key=lambda path: path.name.lower(),
        )
        if not prompt_files:
            raise typer.BadParameter(
                f"{param_name}: directory '{prompt_source}' does not contain any JSON files."
            )
        console.print(
            Panel(
                f"Loading scenarios from [bold]{prompt_source}[/bold] "
                f"({len(prompt_files)} JSON files)",
                title="Harness",
            )
        )
    else:
        raise typer.BadParameter(
            f"{param_name}: '{prompt_source}' is neither a file nor a directory."
        )

    alias_counts: Dict[str, int] = {}
    scenario_batches: list[ScenarioBatch] = []
    for prompt_path in prompt_files:
        scenarios = load_scenarios(prompt_path)
        if not scenarios:
            raise typer.BadParameter(f"No scenarios found in {prompt_path}")
        base_alias = _sanitize_label(prompt_path.stem)
        occurrence = alias_counts.get(base_alias, 0)
        alias_counts[base_alias] = occurrence + 1
        alias = base_alias if occurrence == 0 else _sanitize_label(
            f"{base_alias}-{occurrence + 1}"
        )
        scenario_batches.append(
            ScenarioBatch(alias=alias, source=prompt_path, scenarios=scenarios)
        )

    if len(scenario_batches) > 1:
        for batch in scenario_batches:
            console.print(
                f"[dim]  • {batch.source} ({len(batch.scenarios)} scenarios)[/dim]"
            )

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

    ui_mode = ui_mode_raw
    if ui_mode == "auto":
        total_runs = len(run_configs) * len(scenario_batches)
        ui_mode = "textual" if total_runs > 1 else "plain"

    session_name_base = prompt_source.stem if prompt_source.is_file() else prompt_source.name
    session_dir = _prepare_session_dir(session_name_base)
    console.print(Panel(f"Storing run artifacts in [bold]{session_dir}[/bold]"))

    # Handle background execution
    if background:
        run_id = background_manager.create_background_run_id()
        harness_name = prompt_source.stem if prompt_source.is_file() else prompt_source.name
        model_summary = _format_model_summary(_summarize_models([
            {"provider": config["provider"], "model": config.get("model")}
            for config in run_configs
        ]))
        
        console.print(
            Panel(
                f"Starting background run [bold]{run_id}[/bold]\n"
                f"Session directory: [bold]{session_dir}[/bold]\n"
                f"Use [bold]--view[/bold] to monitor progress",
                title="Background Run",
            )
        )
        
        try:
            pid = background_manager.spawn_background_process(
                run_id=run_id,
                session_dir=session_dir,
                harness_name=harness_name,
                model_summary=model_summary,
                run_configs=run_configs,
                scenario_batches=scenario_batches,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            console.print(f"[green]Background process started (PID: {pid})[/green]")
        except Exception as exc:
            console.print(f"[red]Failed to start background run: {exc}[/red]")
            raise
        return

    if ui_mode == "textual":
        run_results = asyncio.run(
            _run_textual(
                run_configs=run_configs,
                scenario_batches=scenario_batches,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                artifact_dir=session_dir,
                gym_config=gym_config,
            )
        )
    else:
        run_results = asyncio.run(
            _run_plain(
                run_configs=run_configs,
                scenario_batches=scenario_batches,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                artifact_dir=session_dir,
                gym_config=gym_config,
            )
        )

    _write_session_manifest(
        session_dir=session_dir,
        prompt_source=prompt_source,
        prompt_files=[batch.source for batch in scenario_batches],
        ui_mode=ui_mode,
        results=run_results,
    )


def main() -> None:
    """Entry point for `python -m jira_mcp_benchmark`."""
    app()
