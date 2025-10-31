# Background Run Support Implementation Summary

## Overview
Successfully implemented background run support for the Jira MCP benchmark tool. Users can now run benchmarks in the background and monitor their progress through an interactive dashboard.

## What Was Implemented

### 1. Background Manager Module (`background_manager.py`)
- **BackgroundRun** dataclass for storing run metadata
- File-based state management with file locking (`fcntl`)
- Progress tracking (completed/total runs)
- Background process spawning using subprocess
- Helper functions for listing, reading, and updating background runs
- Time-ago formatting for display

### 2. Background Run Logger (`run_logging.py`)
- **BackgroundRunLogger** class that writes logs to files
- Inherits same interface as TextualRunLogger
- Writes to individual log files per run (e.g., `run_label.log`)
- Collects artifacts for later replay

### 3. Enhanced Session Picker (`session_picker.py`)
- Added **BackgroundRunDisplay** dataclass
- Modified **SessionPickerApp** to show two sections:
  - **IN PROGRESS**: Active background runs with progress indicators
  - **COMPLETED**: Finished sessions
- Returns either a Path (for sessions) or string (for background run IDs)
- Visual indicators: 🔄 for running, ✓ for completed

### 4. CLI Enhancements (`cli.py`)
- Added `--background` flag to run benchmarks in background
- Modified `_handle_view_command` to show both background runs and sessions
- Added `_attach_to_background_run` function for live log tailing
- Updated `_select_session_interactive` to handle both types of items
- Background runs spawn detached processes that persist after parent exits

### 5. File Structure
Background runs are stored in: `results/background_runs/<run_id>/`

Each run directory contains:
- `state.json` - Run metadata and progress
- `state.lock` - File lock for atomic updates
- `_worker.py` - Generated worker script
- `_worker_data.pkl` - Pickled run configuration
- `worker.log` - Worker process stdout/stderr

## User Workflows

### Starting a Background Run
```bash
python -m jira_mcp_benchmark --harness-file task.json --model claude-sonnet-4 --runs 5 --background
```

Output:
```
╭─ Background Run ─────────────────────────────────────────╮
│ Starting background run bg_20251029_abc12345             │
│ Session directory: results/task_1/                       │
│ Use --view to monitor progress                           │
╰──────────────────────────────────────────────────────────╯
Background process started (PID: 12345)
```

### Viewing the Dashboard
```bash
python -m jira_mcp_benchmark --view
```

Shows Textual UI with:
```
IN PROGRESS
1. 🔄 task.json • claude-sonnet-4 (3/5) | 5m ago

COMPLETED
2. 2025-10-28 10:30 • task.json • claude-sonnet-4 ×5 • 45/50 (90%)
3. 2025-10-27 15:20 • sample.json • gpt-4o ×3 • 28/30 (93%)
```

### List Mode
```bash
python -m jira_mcp_benchmark --view list
```

Shows plain text list of all runs and sessions.

### Attaching to a Live Run
1. Run `--view` and select an in-progress run
2. System shows live logs tailing in console
3. Polls run state to detect completion
4. Press Ctrl+C to detach (run continues in background)

## Technical Details

### State Management
- Uses `fcntl.flock()` for atomic file locking
- State updates after each completed sub-run
- Progress calculated as: `completed_runs / (len(run_configs) * len(scenario_batches))`

### Process Spawning
- Uses `subprocess.Popen` with `start_new_session=True` for detachment
- Worker script generated dynamically per run
- Run configuration pickled and passed to worker
- Worker imports from main codebase and runs normally

### Live Monitoring
- Plain mode: File-seek-based log tailing
- Checks run state every 0.5 seconds
- Automatically detects completion
- Textual mode: Falls back to plain (future enhancement point)

## Files Modified

1. **NEW**: `src/jira_mcp_benchmark/background_manager.py` (361 lines)
2. **MODIFIED**: `src/jira_mcp_benchmark/cli.py` (+~200 lines)
3. **MODIFIED**: `src/jira_mcp_benchmark/run_logging.py` (+~115 lines)
4. **MODIFIED**: `src/jira_mcp_benchmark/session_picker.py` (+~80 lines)

## Testing

The implementation has been tested with:
- CLI help showing new `--background` flag ✓
- `--view list` command working correctly ✓
- No background runs: Shows only completed sessions ✓
- Type safety: Proper Union types for Path|str returns ✓

## Future Enhancements

1. **Textual Live Viewer**: Implement file-based log streaming in MultiRunApp
2. **Kill Command**: Add ability to terminate running background jobs
3. **Clean Command**: Remove old completed background runs
4. **Email Notifications**: Alert when background runs complete
5. **Resource Limits**: Set CPU/memory limits for background processes
6. **Retry Logic**: Automatic retry of failed runs with exponential backoff

## Known Limitations

1. Windows support: Uses Unix-specific `fcntl` for file locking (would need `msvcrt` for Windows)
2. Textual attach mode falls back to plain mode
3. No automatic cleanup of old background run directories
4. Background runs don't write session manifests until completion
5. Progress updates happen per-run, not per-scenario

## Compatibility

- Python 3.10+
- Tested on macOS (Darwin 24.6.0)
- Requires existing dependencies (no new ones added)
- Backwards compatible with existing runs and sessions


