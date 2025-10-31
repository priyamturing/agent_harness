"""Background run management for persistent benchmark execution."""

from __future__ import annotations

import fcntl
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4

BACKGROUND_RUNS_DIR = Path("results/background_runs")


@dataclass
class BackgroundRun:
    """Metadata for a background benchmark run."""

    run_id: str
    status: str  # "running", "completed", "failed"
    started_at: str
    completed_at: Optional[str]
    progress: dict[str, int]  # {"passed": 8, "total": 10} - verifier counts
    session_dir: str
    harness_name: str
    model_summary: str
    error: Optional[str] = None
    run_configs: list[dict] = field(default_factory=list)
    scenario_batches: list[dict] = field(default_factory=list)


def _ensure_background_runs_dir() -> Path:
    """Create and return the background runs directory."""
    BACKGROUND_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return BACKGROUND_RUNS_DIR


def create_background_run_id() -> str:
    """Generate a unique ID for a background run."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid4())[:8]
    return f"bg_{timestamp}_{short_uuid}"


def get_background_run_dir(run_id: str) -> Path:
    """Get the directory path for a specific background run."""
    return BACKGROUND_RUNS_DIR / run_id


def get_state_file_path(run_id: str) -> Path:
    """Get the state.json file path for a background run."""
    return get_background_run_dir(run_id) / "state.json"


def get_lock_file_path(run_id: str) -> Path:
    """Get the lock file path for a background run."""
    return get_background_run_dir(run_id) / "state.lock"


def read_run_state(run_id: str) -> Optional[BackgroundRun]:
    """Read the state of a background run with file locking."""
    state_file = get_state_file_path(run_id)
    if not state_file.exists():
        return None

    lock_file = get_lock_file_path(run_id)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(lock_file, "w") as lock_fd:
            # Try to acquire shared lock (read)
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_SH)
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                return BackgroundRun(**data)
            finally:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, TypeError, KeyError, FileNotFoundError):
        return None


def write_run_state(run: BackgroundRun) -> None:
    """Write the state of a background run with file locking."""
    state_file = get_state_file_path(run.run_id)
    lock_file = get_lock_file_path(run.run_id)

    state_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_file, "w") as lock_fd:
        # Acquire exclusive lock (write)
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        try:
            data = {
                "run_id": run.run_id,
                "status": run.status,
                "started_at": run.started_at,
                "completed_at": run.completed_at,
                "progress": run.progress,
                "session_dir": run.session_dir,
                "harness_name": run.harness_name,
                "model_summary": run.model_summary,
                "error": run.error,
                "run_configs": run.run_configs,
                "scenario_batches": run.scenario_batches,
            }
            state_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)


def list_background_runs() -> list[BackgroundRun]:
    """List all background runs, sorted by start time (newest first)."""
    _ensure_background_runs_dir()

    runs: list[BackgroundRun] = []
    if not BACKGROUND_RUNS_DIR.exists():
        return runs

    for run_dir in BACKGROUND_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run = read_run_state(run_dir.name)
        if run:
            runs.append(run)

    # Sort by started_at, newest first
    runs.sort(
        key=lambda r: r.started_at if r.started_at else "", reverse=True
    )
    return runs


def update_run_progress(run_id: str, passed: int, total: int) -> None:
    """Update the verifier progress of a background run."""
    run = read_run_state(run_id)
    if run:
        run.progress = {"passed": passed, "total": total}
        write_run_state(run)


def mark_run_completed(run_id: str, error: Optional[str] = None) -> None:
    """Mark a background run as completed or failed."""
    run = read_run_state(run_id)
    if run:
        run.status = "failed" if error else "completed"
        run.completed_at = datetime.now(timezone.utc).isoformat()
        run.error = error
        write_run_state(run)


def spawn_background_process(
    *,
    run_id: str,
    session_dir: Path,
    harness_name: str,
    model_summary: str,
    run_configs: list[dict],
    scenario_batches: Sequence,
    temperature: float,
    max_output_tokens: Optional[int],
) -> int:
    """Spawn a background process to run benchmarks.
    
    Returns the PID of the spawned process.
    """
    import subprocess

    # Create initial state
    initial_state = BackgroundRun(
        run_id=run_id,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        completed_at=None,
        progress={"passed": 0, "total": 0},  # Will be updated as verifiers run
        session_dir=str(session_dir),
        harness_name=harness_name,
        model_summary=model_summary,
        run_configs=run_configs,
        scenario_batches=[
            {
                "alias": batch.alias,
                "source": str(batch.source),
                "scenario_count": len(batch.scenarios),
            }
            for batch in scenario_batches
        ],
    )
    write_run_state(initial_state)

    # Prepare command to run in background
    # We'll use the internal _run_background_worker function
    python_executable = sys.executable
    module_path = Path(__file__).parent / "_background_worker.py"

    # Create a simple worker script
    worker_script = get_background_run_dir(run_id) / "_worker.py"
    worker_script.parent.mkdir(parents=True, exist_ok=True)

    # Import necessary data
    import pickle

    worker_data = {
        "run_id": run_id,
        "session_dir": str(session_dir),
        "run_configs": run_configs,
        "scenario_batches": [
            {
                "alias": batch.alias,
                "source": str(batch.source),
                "scenarios": batch.scenarios,
            }
            for batch in scenario_batches
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }

    worker_data_file = get_background_run_dir(run_id) / "_worker_data.pkl"
    with open(worker_data_file, "wb") as f:
        pickle.dump(worker_data, f)

    worker_script.write_text(
        f'''"""Background worker process for run {run_id}."""
import sys
import asyncio
import pickle
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from jira_mcp_benchmark.background_manager import _run_background_worker

if __name__ == "__main__":
    data_file = Path(__file__).parent / "_worker_data.pkl"
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    asyncio.run(_run_background_worker(**data))
''',
        encoding="utf-8",
    )

    # Start background process
    log_file = get_background_run_dir(run_id) / "worker.log"
    with open(log_file, "w") as log_fd:
        process = subprocess.Popen(
            [python_executable, str(worker_script)],
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # Detach from parent
        )

    return process.pid


async def _run_background_worker(
    *,
    run_id: str,
    session_dir: str,
    run_configs: list[dict],
    scenario_batches: list[dict],
    temperature: float,
    max_output_tokens: Optional[int],
) -> None:
    """Internal function to run benchmarks in background process."""
    from .cli import _execute_run
    from .prompts import Scenario
    from .run_logging import BackgroundRunLogger
    from typing import NamedTuple

    session_path = Path(session_dir)

    # Define a simple batch structure
    class ScenarioBatch(NamedTuple):
        alias: str
        source: Path
        scenarios: list

    # Convert scenario_batches back to proper objects
    batches = []
    for batch_data in scenario_batches:
        # Scenarios are already Scenario objects from pickle, not dicts
        scenarios = batch_data["scenarios"]
        if scenarios and not isinstance(scenarios[0], Scenario):
            # Fallback: if they're dicts, reconstruct them
            scenarios = [Scenario(**s) for s in scenarios]
        batches.append(
            ScenarioBatch(
                alias=batch_data["alias"],
                source=Path(batch_data["source"]),
                scenarios=scenarios,
            )
        )

    def logger_factory(label: str, db_id: str) -> BackgroundRunLogger:
        return BackgroundRunLogger(
            run_id=run_id, run_label=label, session_dir=session_path
        )

    results = []
    total_verifiers_passed = 0
    total_verifiers = 0
    error_message: Optional[str] = None

    try:
        for batch in batches:
            for config in run_configs:
                try:
                    result = await _execute_run(
                        run_label=config["label"],
                        logger_factory=logger_factory,
                        scenarios=batch.scenarios,
                        provider=config["provider"],
                        model=config["model"],
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        artifact_dir=session_path,
                        prompt_path=batch.source,
                        prompt_alias=batch.alias,
                    )
                    results.append(result)
                    
                    # Extract verifier counts from this run's results
                    for scenario_id, verifier_results in result.get("summary", []):
                        for vr in verifier_results or []:
                            total_verifiers += 1
                            if getattr(vr, "success", False):
                                total_verifiers_passed += 1
                    
                    # Update progress with cumulative verifier counts
                    update_run_progress(run_id, total_verifiers_passed, total_verifiers)
                except Exception as exc:
                    # Store error but don't mark completed yet - let finally handle it
                    error_message = f"Failed during execution: {exc}"
                    raise

    except Exception as exc:
        # Capture the error for the finally block
        if not error_message:
            error_message = str(exc)
    finally:
        # ALWAYS mark the run as completed, even if the process crashes
        # This ensures no runs are left in perpetual "running" state
        mark_run_completed(run_id, error=error_message)


def get_run_log_file(run_id: str, run_label: str) -> Path:
    """Get the log file path for a specific run within a background job."""
    run_dir = get_background_run_dir(run_id)
    return run_dir / f"run_{run_label}.log"


def format_time_ago(iso_timestamp: str) -> str:
    """Format an ISO timestamp as 'X minutes/hours ago'."""
    try:
        started = datetime.fromisoformat(iso_timestamp)
        now = datetime.now(timezone.utc)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        delta = now - started

        if delta.total_seconds() < 60:
            return "just now"
        elif delta.total_seconds() < 3600:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m ago"
        elif delta.total_seconds() < 86400:
            hours = int(delta.total_seconds() / 3600)
            return f"{hours}h ago"
        else:
            days = int(delta.total_seconds() / 86400)
            return f"{days}d ago"
    except (ValueError, TypeError):
        return "unknown"


def is_run_stale(run: BackgroundRun, stale_hours: int = 24) -> bool:
    """Check if a running background run is stale (hasn't been updated in X hours).
    
    A run is considered stale if:
    1. It's in "running" status
    2. It hasn't been updated in the specified number of hours
    """
    if run.status != "running":
        return False
    
    try:
        started = datetime.fromisoformat(run.started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        hours_running = (now - started).total_seconds() / 3600
        
        return hours_running > stale_hours
    except (ValueError, TypeError):
        return False


def find_stale_runs(stale_hours: int = 24) -> list[BackgroundRun]:
    """Find all stale background runs."""
    all_runs = list_background_runs()
    return [run for run in all_runs if is_run_stale(run, stale_hours)]


def cleanup_stale_runs(stale_hours: int = 24, dry_run: bool = True) -> list[str]:
    """Mark stale runs as failed.
    
    Returns list of run IDs that were marked as failed.
    """
    stale_runs = find_stale_runs(stale_hours)
    cleaned_ids = []
    
    for run in stale_runs:
        if not dry_run:
            mark_run_completed(
                run.run_id,
                error=f"Run marked as stale after {stale_hours}h with no completion"
            )
        cleaned_ids.append(run.run_id)
    
    return cleaned_ids

