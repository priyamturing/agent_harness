# MCP Benchmark Packages

This directory contains two packages:

## 1. `mcp_benchmark_sdk/` - Core SDK

The SDK for building and running MCP-based LLM agent benchmarks.

### Installation (Development)

```bash
cd packages/mcp_benchmark_sdk
pip install -e .
```

### Quick Example

```python
import asyncio
from pathlib import Path
from mcp_benchmark_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

async def main():
    # Configure MCP server
    mcp_config = MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http"
    )
    
    # Create harness
    harness = TestHarness(
        harness_path=Path("benchmarks/task.json"),
        config=TestHarnessConfig(
            mcp=mcp_config,
            max_steps=1000,
            tool_call_limit=1000,
        )
    )
    
    # Run benchmarks
    results = await harness.run(
        models=["gpt-4o", "claude-3-5-sonnet-20241022"],
        agent_factory=create_agent,
    )
    
    # Print results
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{result.model} - {result.scenario_id}: {status}")

asyncio.run(main())
```

## 2. `mcp_benchmark_cli/` - CLI Tool

Command-line interface that uses the SDK to run benchmarks from JSON harness files.

### Installation (Development)

```bash
cd packages/mcp_benchmark_cli
pip install -e .
```

This installs the `mcp-benchmark` command.

### Quick Example

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run a benchmark
mcp-benchmark --prompt-file test_harness.json --model gpt-4o

# Run with multiple models
mcp-benchmark --prompt-file test_harness.json \
  --model gpt-4o \
  --model claude-3-5-sonnet-20241022
```

## Installation (Both Packages)

To install both packages at once:

```bash
# From the repository root
pip install -e packages/mcp_benchmark_sdk
pip install -e packages/mcp_benchmark_cli
```

## Requirements

- **MCP Server**: You must have an MCP server running (e.g., `http://localhost:8015/mcp`)
- **API Keys**: Set environment variables for the providers you want to use:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `XAI_API_KEY`

## Documentation

See individual package READMEs for detailed documentation:
- [`mcp_benchmark_sdk/README.md`](./mcp_benchmark_sdk/README.md) - Complete SDK reference
- [`mcp_benchmark_cli/README.md`](./mcp_benchmark_cli/README.md) - CLI usage guide

## Key Differences

| Feature | SDK | CLI |
|---------|-----|-----|
| **Use Case** | Programmatic integration | Quick testing |
| **Interface** | Python API | Command-line |
| **UI** | Custom observers | Plain/TUI/Quiet modes |
| **Session Management** | Manual | Automatic |
| **Flexibility** | Full control | Predefined workflows |
| **Result Export** | Custom formats | JSON artifacts |

## Quick Links

- **SDK Quick Start**: [`mcp_benchmark_sdk/QUICKSTART.md`](./mcp_benchmark_sdk/QUICKSTART.md)
- **Examples**: [`mcp_benchmark_sdk/examples/`](./mcp_benchmark_sdk/examples/)
- **Architecture**: [`../SDK_ARCHITECTURE.md`](../SDK_ARCHITECTURE.md)
