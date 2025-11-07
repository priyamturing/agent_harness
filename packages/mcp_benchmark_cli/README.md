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

## Using OpenRouter models

OpenRouter exposes hundreds of models behind an OpenAI-compatible API. To run any
tool-capable OpenRouter model from the CLI:

1. Create an API key at <https://openrouter.ai/keys> and export it:
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```
2. (Optional but recommended) set attribution headers for the leaderboard as
   described in the [Quickstart guide](https://openrouter.ai/docs/quickstart):
   ```bash
   export OPENROUTER_HTTP_REFERER="https://your-app.example"
   export OPENROUTER_APP_TITLE="Your App Name"
   ```
3. Choose any model slug that advertises tool-calling in
   `https://openrouter.ai/api/v1/models` and prefix it with `openrouter:` when
   invoking the CLI:
   ```bash
   mcp-benchmark --prompt-file test_harness.json \
     --model openrouter:anthropic/claude-3.5-sonnet
   ```

Reasoning tokens are requested automatically for models that support them. You
can override the behavior by setting `OPENROUTER_REASONING` to a JSON blob (for
example `{"effort":"high"}`) or `OPENROUTER_INCLUDE_REASONING=false`. See the
[Reasoning Tokens guide](https://openrouter.ai/docs/use-cases/reasoning-tokens)
for accepted parameters.

## Documentation

See the main repository README for detailed documentation.
