# Test Harness Refactoring Summary

## Overview

The test harness functionality has been moved from the CLI package into the SDK package, making it available for broader use. This provides a HUD-like experience similar to what's described at https://docs.hud.ai/.

## What Was Moved

### From CLI to SDK

1. **Scenario Data Structures** (`scenario.py`)
   - `Scenario`, `ScenarioPrompt`, `VerifierDefinition`
   - Now in: `mcp_benchmark_sdk/harness/scenario.py`

2. **Harness Loader** (`harness_loader.py`)
   - `load_harness_file()`, `load_harness_directory()`
   - `scenario_to_task()`, `create_verifier_from_definition()`
   - Now in: `mcp_benchmark_sdk/harness/loader.py`

3. **Verifier Runner** (`verifier_runner.py`)
   - `VerifierRunner` class
   - Now in: `mcp_benchmark_sdk/harness/orchestrator.py`

4. **New Additions**
   - `TestHarness` - Main orchestrator class
   - `TestHarnessConfig` - Configuration for test runs
   - `RunResult` - Result of individual runs
   - `create_agent()` - Agent factory for SDK
   - All in: `mcp_benchmark_sdk/harness/`

## Architecture

```
mcp_benchmark_sdk/
└── harness/
    ├── __init__.py           # Public exports
    ├── scenario.py           # Scenario data structures
    ├── loader.py             # Load harness files
    ├── agent_factory.py      # Create agents from strings
    ├── orchestrator.py       # TestHarness + VerifierRunner
    └── README.md             # Usage documentation

mcp_benchmark_cli/
├── agent_factory.py          # Extends SDK with Qwen support
├── scenario.py               # Re-exports from SDK
├── harness_loader.py         # Re-exports from SDK
├── verifier_runner.py        # Re-exports from SDK
├── cli.py                    # CLI-specific orchestration
├── session/                  # Session management
└── ui/                       # UI observers (console, textual)
```

## Usage

### SDK (Simple)

```python
from pathlib import Path
from mcp_benchmark_sdk import (
    TestHarness,
    TestHarnessConfig,
    MCPConfig,
    create_agent,
)

# Configure
mcp_config = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http",
)

# Create harness
harness = TestHarness(
    harness_path=Path("benchmarks/task1.json"),
    config=TestHarnessConfig(
        mcps=[mcp_config],
        sql_runner_url="http://localhost:8015/api/sql-runner",
        runs_per_scenario=3,
    )
)

# Run benchmarks
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5"],
    agent_factory=create_agent,
)

# Analyze
for result in results:
    print(f"{result.model} - {result.scenario_id}: {'✓' if result.success else '✗'}")
```

### CLI (Full Featured)

The CLI now uses SDK components internally but adds:
- Session management (results directories, manifests)
- Multiple UI modes (plain, textual)
- Progress tracking
- Result formatting

```bash
mcp-benchmark-cli --harness-file benchmarks/task1.json \
    --model gpt-4o --model claude-sonnet-4-5 \
    --runs 3 \
    --ui textual
```

## Benefits

1. **Reusable**: Anyone can use the test harness without the CLI
2. **Simple API**: HUD-like experience with minimal boilerplate
3. **Flexible**: Support custom observers, agents, configurations
4. **Backward Compatible**: CLI continues to work as before
5. **Extensible**: Easy to add new verifier types, agents, etc.

## Key Classes

### TestHarness

Main orchestrator that:
- Loads harness files/directories
- Creates agents from model specifications
- Runs tasks with verification
- Manages concurrent execution
- Collects and returns results

### TestHarnessConfig

Configuration for:
- MCP servers
- SQL runner URL
- Max steps, tool call limit
- Temperature, max output tokens
- Concurrent runs, runs per scenario

### RunResult

Result of a single run containing:
- Model name, scenario info
- Success status
- Agent result
- Verifier results
- Error messages (if any)
- Metadata

### VerifierRunner

Orchestrates verifier execution:
- Creates verifiers from definitions
- Reuses HTTP client efficiently
- Runs verifiers after agent execution

## Migration Guide

### For SDK Users (New)

Import from `mcp_benchmark_sdk.harness`:

```python
from mcp_benchmark_sdk import (
    TestHarness,
    TestHarnessConfig,
    create_agent,
    load_harness_file,
    Scenario,
)
```

### For CLI Users (Existing)

No changes needed! The CLI still works exactly as before. The modules now re-export from SDK:

```python
# Still works!
from mcp_benchmark_cli import load_harness_file, scenario_to_task
from mcp_benchmark_cli.scenario import Scenario
from mcp_benchmark_cli.verifier_runner import VerifierRunner
```

## Examples

1. **Simple SDK Example**: `packages/mcp_benchmark_sdk/examples/simple_harness_example.py`
2. **Full SDK Documentation**: `packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/harness/README.md`
3. **CLI Usage**: Unchanged, see `packages/mcp_benchmark_cli/README.md`

## Testing

Run the example:

```bash
cd packages/mcp_benchmark_sdk
python examples/simple_harness_example.py
```

Run CLI (unchanged):

```bash
mcp-benchmark-cli --harness-file task4.json --model gpt-4o
```

## Next Steps

1. Add more verifier types (HTTP, filesystem, etc.)
2. Add more agent providers
3. Enhance observer capabilities
4. Add result exporters (JSON, HTML reports, etc.)
5. Add harness validation tools
