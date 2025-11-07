# MCP Benchmark Packages

This directory now contains four packages:

## 1. `mcp_benchmark_agents/` - Agent Runtime Library

Everything required to build MCP-aware agents (LLM providers, parsers, telemetry,
verifiers, runtime helpers).

```bash
cd packages/mcp_benchmark_agents
pip install -e .
```

## 2. `mcp_benchmark_harness/` - Benchmark Harness

Scenario loader + orchestrator that depends on the agent runtime.

```bash
cd packages/mcp_benchmark_harness
pip install -e .
```

## 3. `mcp_benchmark_sdk/` - Meta SDK

The SDK for building and running MCP-based LLM agent benchmarks.

### Installation (Development)

```bash
cd packages/mcp_benchmark_sdk
pip install -e .
```

### Quick Example

```python
import asyncio
from mcp_benchmark_sdk import ClaudeAgent, Task, MCPConfig

async def main():
    agent = ClaudeAgent()
    
    task = Task(
        prompt="Create a bug issue in project DEMO",
        mcps=[
            MCPConfig(
                name="jira",
                url="http://localhost:8015/mcp"
            )
        ]
    )
    
    result = await agent.run(task)
    print(f"Success: {result.success}")

asyncio.run(main())
```

## 4. `mcp_benchmark_cli/` - CLI Tool

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
mcp-benchmark --prompt-file test_harness.json --model gpt-5-high

# Run with multiple models
mcp-benchmark --prompt-file test_harness.json \
  --model gpt-5-high \
  --model claude-sonnet-4-5
```

## Installing everything

To install all packages at once:

```bash
# From the repository root
pip install -e packages/mcp_benchmark_agents
pip install -e packages/mcp_benchmark_harness
pip install -e packages/mcp_benchmark_sdk
pip install -e packages/mcp_benchmark_cli
```

## Requirements

- **MCP Server**: You must have a Jira MCP server running on `http://localhost:8015/mcp`
- **API Keys**: Set environment variables for the providers you want to use:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `XAI_API_KEY`

## Documentation

See individual package READMEs for detailed documentation:
- [`mcp_benchmark_agents/README.md`](./mcp_benchmark_agents/README.md)
- [`mcp_benchmark_harness/README.md`](./mcp_benchmark_harness/README.md)
- [`mcp_benchmark_sdk/README.md`](./mcp_benchmark_sdk/README.md)
- [`mcp_benchmark_cli/README.md`](./mcp_benchmark_cli/README.md)
