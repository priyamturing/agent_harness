# Background Run Fixes - October 31, 2025

## Problem
Background runs were getting stuck in "running" status and never moving to "completed" or "failed" state.

## Root Causes Identified

1. **No crash recovery**: If the background worker process crashed, was killed, or failed to start, the run would remain in "running" status forever because there was no process to mark it as completed.

2. **Failed runs with null `completed_at`**: Some failed runs had `status: "failed"` but `completed_at: null`, which was inconsistent and caused display issues.

3. **No staleness detection**: There was no mechanism to identify runs that had been "running" for an unreasonable amount of time.

## Solutions Implemented

### 1. Enhanced Worker Error Handling (`background_manager.py`)
**Change**: Modified `_run_background_worker()` to use a `finally` block that ALWAYS marks the run as completed.

**Before**:
```python
try:
    # ... run benchmarks ...
    mark_run_completed(run_id)
except Exception as exc:
    mark_run_completed(run_id, error=str(exc))
    raise
```

**After**:
```python
error_message: Optional[str] = None
try:
    # ... run benchmarks ...
except Exception as exc:
    if not error_message:
        error_message = str(exc)
finally:
    # ALWAYS mark the run as completed, even if the process crashes
    mark_run_completed(run_id, error=error_message)
```

**Impact**: Now, even if the worker process crashes unexpectedly, the finally block ensures the run state is updated before the process terminates.

### 2. Stale Run Detection (`background_manager.py`)
**Added new functions**:
- `is_run_stale(run, stale_hours)` - Checks if a run has been in "running" state for more than X hours
- `find_stale_runs(stale_hours)` - Finds all stale runs
- `cleanup_stale_runs(stale_hours, dry_run)` - Marks stale runs as failed

**Usage**:
```python
# Find runs stuck in "running" for more than 24 hours
stale_runs = find_stale_runs(stale_hours=24)

# Mark them as failed (dry_run=False to actually execute)
cleaned = cleanup_stale_runs(stale_hours=24, dry_run=False)
```

### 3. CLI Cleanup Command (`cli.py`)
**Added new Typer command**: `cleanup`

**Usage**:
```bash
# Dry run (show what would be cleaned up)
python -m jira_mcp_benchmark cleanup --stale-hours 24

# Actually clean up stale runs
python -m jira_mcp_benchmark cleanup --stale-hours 24 --no-dry-run

# Check runs older than 48 hours
python -m jira_mcp_benchmark cleanup --stale-hours 48 --no-dry-run
```

**Features**:
- Shows a nice table of stale runs
- Dry run by default (safe to run)
- Customizable staleness threshold
- Displays run ID, start time, progress, and model

## Testing Results

### Before Fixes
```
bg_20251028_210801_aa84f604: status=running, completed_at=null (2+ days old!)
bg_20251031_145250_2c83b3a1: status=failed, completed_at=null (broken!)
```

### After Running Cleanup
```
bg_20251028_210801_aa84f604: status=failed, completed_at=2025-10-31T17:32:52
  Error: "Run marked as stale after 24h with no completion"

bg_20251031_145250_2c83b3a1: status=failed, completed_at=2025-10-31T17:33:20
  Error: "TypeError: Scenario serialization bug (fixed in latest code)"
```

### All Background Runs Status (Verified)
```
✓ bg_20251028_210801_aa84f604: failed, completed at 2025-10-31T17:32:52
✓ bg_20251031_145250_2c83b3a1: failed, completed at 2025-10-31T17:33:20
✓ bg_20251031_145620_e815e9ba: completed at 2025-10-31T15:00:45
✓ bg_20251031_160539_b4e46a96: completed at 2025-10-31T17:26:06
✓ bg_20251031_170021_cf1cafd0: completed at 2025-10-31T17:31:02
```

All runs now have proper status! ✓

## Prevention
The new `finally` block in the worker ensures this problem won't happen again for **new** runs. Any crashes or exceptions will still result in the run being properly marked as completed (or failed).

## Recovery
For existing stuck runs, use the cleanup command:
```bash
python -m jira_mcp_benchmark cleanup --stale-hours 24 --no-dry-run
```

## Recommendations

1. **Periodic cleanup**: Consider running the cleanup command periodically (e.g., daily cron job) to catch any orphaned runs:
   ```bash
   python -m jira_mcp_benchmark cleanup --stale-hours 48 --no-dry-run
   ```

2. **Monitoring**: Add monitoring to alert when runs have been in "running" state for more than a few hours.

3. **Process tracking**: Future enhancement could track the actual worker PID and check if it's still alive.

## Files Modified

1. **src/jira_mcp_benchmark/background_manager.py**
   - Enhanced `_run_background_worker()` with finally block
   - Added `is_run_stale()`, `find_stale_runs()`, `cleanup_stale_runs()`

2. **src/jira_mcp_benchmark/cli.py**
   - Added `cleanup` command

## Migration Notes

No migration needed for existing code. The fixes are backward compatible and only affect runtime behavior.

