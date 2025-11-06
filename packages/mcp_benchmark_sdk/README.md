# MCP Benchmark SDK

A Python SDK for building and running LLM agent benchmarks against MCP (Model Context Protocol) servers.

## Features

- **Multi-level Agent API**: High-level `agent.run()`, mid-level overrides, or low-level manual control
- **Provider Support**: Built-in support for OpenAI, Anthropic, Google, and xAI models
- **Extensible**: Easy to add custom agents, verifiers, and MCP integrations
- **Multi-MCP**: Connect to multiple MCP servers per task
- **Event System**: Observable execution for logging and telemetry
- **Verifiers**: Built-in SQL-based verification, extensible for custom checks

## Installation

```bash
pip install mcp-benchmark-sdk
```

## Quick Start

```python
import asyncio
from mcp_benchmark_sdk import ClaudeAgent, Task, MCPConfig

async def main():
    agent = ClaudeAgent()
    
    task = Task(
        prompt="Create a bug issue in project DEMO",
        mcp=MCPConfig(
            name="jira",
            url="http://localhost:8015/mcp",
            transport="streamable_http"
        )
    )
    
    result = await agent.run(task)
    print(f"Success: {result.success}")

asyncio.run(main())
```

## Documentation

See the main repository README for detailed documentation and examples.

