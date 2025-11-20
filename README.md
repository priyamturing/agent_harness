# Turing RL Tooling

This repository contains two packages:

## 1. `turing_rl_sdk/` - Core SDK

The SDK for building and running MCP-based LLM agent benchmarks.

### Installation (Development)

```bash
cd turing_rl_sdk
pip install -e .
```

### Quick Example

```python
import asyncio
from pathlib import Path
from turing_rl_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

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
        models=["gpt-4o", "claude-sonnet-4-5"],
        agent_factory=create_agent,
    )
    
    # Print results
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{result.model} - {result.scenario_id}: {status}")

asyncio.run(main())
```

## 2. `turing_rl_cli/` - CLI Tool

Command-line interface that uses the SDK to run benchmarks from JSON harness files.

### Installation (Development)

```bash
cd turing_rl_cli
pip install -e .
```

This installs the `turing-rl` command.

### Quick Example

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run a benchmark
turing-rl --prompt-file test_harness.json --model gpt-4o

# Run with multiple models
turing-rl --prompt-file test_harness.json \
  --model gpt-4o \
  --model claude-sonnet-4-5
```

## Installation (Both Packages)

### Option 1: Install Directly from GitHub (No Cloning Needed)

```bash
# Install SDK directly from GitHub (sdk_initial branch)
pip install git+https://github.com/TuringGpt/rl-gym-tooling.git@sdk_initial#subdirectory=turing_rl_sdk

# Install CLI directly from GitHub (sdk_initial branch)
pip install git+https://github.com/TuringGpt/rl-gym-tooling.git@sdk_initial#subdirectory=turing_rl_cli
```

> **Note:** Once merged to `main`, remove `@sdk_initial` from the URLs above.

### Option 2: Install from Local Clone (Development)

```bash
# Clone the repository and checkout the branch
git clone https://github.com/TuringGpt/rl-gym-tooling.git
cd rl-gym-tooling
git checkout sdk_initial

# Install both packages in editable mode
pip install -e turing_rl_sdk
pip install -e turing_rl_cli
```

## Requirements

- **MCP Server**: You must have an MCP server running (e.g., `http://localhost:8015/mcp`)
- **API Keys**: Set environment variables for the providers you want to use:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GOOGLE_API_KEY`
  - `XAI_API_KEY`

## Supported Models

**Important:** Model names must match the exact identifiers expected by each provider. Both packages auto-detect the provider from the model name prefix (e.g., `gpt-`, `claude-`, `gemini-`, `grok-`) or from an explicit provider prefix (`openai:`, `anthropic:`, `google:`, `xai:`).

| Provider | Model Name Examples | Documentation Link |
| --- | --- | --- |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o1`, `o3-mini` | [OpenAI Models](https://platform.openai.com/docs/models) |
| **Anthropic** | `claude-sonnet-4-5`, `claude-opus-4`, `claude-haiku-3-5` | [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models) |
| **Google** | `gemini-2.0-flash-exp`, `gemini-2.5-pro`, `gemini-1.5-pro` | [Google AI Models](https://ai.google.dev/gemini-api/docs/models/gemini) |
| **xAI** | `grok-4`, `grok-vision-beta` | [xAI Models](https://docs.x.ai/docs/models) |

**Usage Examples:**

```python
# SDK - Auto-detection
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash-exp"],
    agent_factory=create_agent,
)

# SDK - Explicit prefix (when auto-detection fails)
results = await harness.run(
    models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"],
    agent_factory=create_agent,
)
```

```bash
# CLI - Auto-detection
turing-rl --prompt-file task.json --model gpt-4o --model claude-sonnet-4-5

# CLI - Explicit prefix
turing-rl --prompt-file task.json --model openai:gpt-4o
```

## Documentation

See individual package READMEs for detailed documentation:
- [`turing_rl_sdk/README.md`](./turing_rl_sdk/README.md) - Complete SDK reference
- [`turing_rl_cli/README.md`](./turing_rl_cli/README.md) - CLI usage guide

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

- **SDK Quick Start**: [`turing_rl_sdk/QUICKSTART.md`](./turing_rl_sdk/QUICKSTART.md)
- **Examples**: [`turing_rl_sdk/examples/`](./turing_rl_sdk/examples/)
