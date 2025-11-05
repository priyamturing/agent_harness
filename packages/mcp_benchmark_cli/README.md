# MCP Benchmark CLI

Command-line interface for running MCP agent benchmarks using the MCP Benchmark SDK.

## Installation

```bash
pip install mcp-benchmark-cli
```

This will also install the SDK (`mcp-benchmark-sdk`) as a dependency.

## Quick Start

```bash
# Run a benchmark harness file
mcp-benchmark --prompt-file test_harness.json --model gpt-5-high

# Run with multiple models
mcp-benchmark --prompt-file test_harness.json \
  --model gpt-5-high \
  --model claude-sonnet-4-5

# View previous results
mcp-benchmark --view
```

## Documentation

See the main repository README for detailed documentation.

