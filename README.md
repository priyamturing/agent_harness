# Jira MCP Benchmark Runner

This project provides a small LangChain-based CLI that drives a locally running Jira MCP server and executes prompt scenarios end-to-end with the LLM of your choice. The workflow was designed around the sample benchmark prompts in `old_sample_new_system_1_benchmark.json`.

## Features

- Connects to a local MCP server (defaults to `http://localhost:8015/mcp`) using `langchain-mcp-adapters`
- Runs each benchmark scenario without any interactive input from the user
- Supports the latest model identifiers for OpenAI, Anthropic, and xAI:
  - OpenAI: `gpt-5-high` (mapped to `gpt-5` with reasoning effort set to `high`)
  - Anthropic: `claude-4.5-sonnet-reasoning`
  - xAI: `grok-4`
- Binds all MCP tools to the selected LLM so it can call them while solving the scenario
- Streams any reasoning traces emitted by the model for easier debugging
- Automatically provisions a unique MCP workspace each run by sending an `x-database-id` header
- Requests OpenAI reasoning summaries and encrypted reasoning tokens so follow-up turns can reference prior cogitation if needed
- After each scenario, runs the harness verifiers against the same database via the MCP SQL runner endpoint (`/api/sql-runner`)
- Optional Textual-based UI so every parallel run can stream into its own terminal tab

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install will pull the required LangChain integrations defined in `pyproject.toml`.

## Environment variables

Environment variables can be loaded automatically from a `.env` file. Copy `.env.example` to `.env` and set the appropriate keys, or point the CLI at a specific file with `--env-file`.

```bash
cp .env.example .env
```

Alternatively, export the variables directly in your shell:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export XAI_API_KEY=...
```

## Running the benchmark

```bash
python -m jira_mcp_benchmark \
  --harness-file old_sample_new_system_1_benchmark.json \
  --provider openai \
  --env-file .env \
  --runs 3
```

Additional useful options:

- `--model`: override the default model identifier for the chosen provider
- `--temperature`: adjust sampling temperature (default `0.1`)
- `--max-output-tokens`: limit the maximum number of generated tokens
- `--harness-file`: select any benchmark JSON file without changing the default
- `--env-file`: load environment variable overrides from a specific `.env` file
- `--runs`: launch multiple parallel executions, each with its own MCP database
- `--ui`: choose `textual` to open a multi-pane terminal UI (default `auto` picks Textual when `--runs > 1`)

When you run the command the CLI will:

1. Load the benchmark scenarios from the JSON file
2. Fetch all available tools from the configured MCP server
3. Instantiate the requested LLM
4. Execute every prompt sequentially without pausing for user confirmation

## Notes

- The CLI automatically targets the local Jira MCP server at `http://localhost:8015/mcp` using the `streamable_http` transport; adjust the code if your deployment differs.
- Each scenario allows up to 1000 MCP tool invocations by default; raise or lower the limit in `src/jira_mcp_benchmark/cli.py` if needed.
- A fresh UUID is used for the `x-database-id` header on every invocation so each run operates against an isolated MCP database instance.
- Verifier queries are executed via `POST /api/sql-runner` using the same `x-database-id`, so ensure that endpoint is reachable on the MCP host.
- When running multiple parallel executions, the default `textual` UI gives each run its own tab so logs never interleave; switch back to `--ui plain` if you prefer classic console output.
- The CLI uses LangChain's tool binding APIs; tool responses and final assistant messages are echoed to the terminal to aid debugging.
- If you add new scenarios, simply update the JSON file—no code changes are required.
