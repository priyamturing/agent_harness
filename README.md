# Jira MCP Benchmark Runner

This project provides a small LangChain-based CLI that drives a locally running Jira MCP server and executes prompt scenarios end-to-end with the LLM of your choice. The workflow was designed around the sample benchmark prompts in `old_sample_new_system_1_benchmark.json`.

## Features

- **Multi-Gym Support**: Configure and switch between multiple MCP server environments (local, staging, production, etc.) using a simple JSON configuration file
- Connects to MCP servers using `langchain-mcp-adapters`
- Runs each benchmark scenario without any interactive input from the user
- Supports the latest model identifiers for OpenAI, Anthropic, xAI, Google, and OpenRouter:
  - OpenAI: `gpt-5-high` (mapped to `gpt-5` with reasoning effort set to `high`) and `gpt-4o` (runs without reasoning traces)
  - Anthropic: `claude-sonnet-4-5` (extended thinking enabled automatically)
  - xAI: `grok-4`
  - Google: `gemini-2.5-pro`
  - OpenRouter (Fireworks/Groq): `meta-llama/llama-4-maverick` with provider routing constrained to tool-capable endpoints
- Launches multiple models (even across providers) in a single CLI invocation; pass `--model` more than once and the CLI auto-selects the correct provider for each model
- Binds all MCP tools to the selected LLM so it can call them while solving the scenario
- Streams any reasoning traces emitted by the model for easier debugging
- Automatically provisions a unique MCP workspace each run by sending an `x-database-id` header
- Requests OpenAI reasoning summaries and encrypted reasoning tokens for reasoning-capable models so follow-up turns can reference prior cogitation if needed
- Replays the harness verifiers after every MCP tool call (and again at the end of each scenario) so progress is visible in real time
- Optional Textual-based UI so every parallel run can stream into its own terminal tab
- Writes per-run JSON artifacts (conversation history, tool calls, verifier snapshots) inside `results/`, under a session folder named after the harness file
- Captures a session manifest so you can revisit prior runs with `--view` in either the Textual UI or plain console mode

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The editable install will pull the required LangChain integrations defined in `pyproject.toml`.

## Gym Configuration

The benchmark runner supports multiple MCP server environments (called "gyms"). Configure your gyms by creating a `gyms.json` file:

```bash
cp gyms.example.json gyms.json
```

Edit `gyms.json` to configure your MCP server environments:

```json
{
  "gyms": {
    "local": {
      "name": "local",
      "description": "Local development Jira MCP server",
      "url": "http://localhost:8015/mcp",
      "transport": {
        "type": "streamable_http"
      },
      "sql_runner_url": "http://localhost:8015/api/sql-runner"
    },
    "teams-management": {
      "name": "teams-management",
      "description": "Teams management MCP server",
      "url": "http://localhost:8000/mcp",
      "headers": {
        "x-teams-access-token": "static_token_alex"
      },
      "transport": {
        "type": "http"
      }
    },
    "staging": {
      "name": "staging",
      "description": "Staging environment",
      "url": "http://staging.example.com:8015/mcp",
      "transport": {
        "type": "streamable_http"
      },
      "headers": {
        "x-api-key": "your-staging-api-key"
      }
    }
  },
  "default": "local"
}
```

### Configuration Fields

Each gym supports the following fields:

- **`name`** (optional): Display name for the gym. Defaults to the key name.
- **`description`** (optional): Human-readable description of the environment.
- **`url`**: The MCP server URL (e.g., `http://localhost:8015/mcp`).
- **`transport`**: Transport configuration object with a `type` field (e.g., `"streamable_http"`, `"http"`).
- **`sql_runner_url`** (optional): URL for the SQL runner endpoint. If not provided, it will be derived from the `url` by replacing `/mcp` with `/api/sql-runner`.
- **`headers`** (optional): Custom HTTP headers to send with every request. The `x-database-id` header is automatically added per-run and should not be included here.

List all configured gyms:

```bash
python -m jira_mcp_benchmark list-gyms
```

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
export OPENROUTER_API_KEY=...
# Optional but recommended when using OpenRouter:
# export OPENROUTER_HTTP_REFERER=https://your-app.example.com
# export OPENROUTER_APP_TITLE="My MCP Benchmark Runner"
# export OPENROUTER_PROVIDER_ONLY=fireworks,groq
# export OPENROUTER_ALLOW_FALLBACKS=true
# export OPENROUTER_USE_RESPONSES_API=false
```

## Running the benchmark

```bash
python -m jira_mcp_benchmark run \
  --harness-file harness/old_sample_new_system_1_benchmark.json \
  --env-file .env \
  --gym local \
  --runs 3
```

### Running against different environments

Use the `--gym` option to specify which environment to run against:

```bash
# Run against local development server (default)
python -m jira_mcp_benchmark run --harness-file harness/test.json --gym local

# Run against staging environment
python -m jira_mcp_benchmark run --harness-file harness/test.json --gym staging

# Run against production (be careful!)
python -m jira_mcp_benchmark run --harness-file harness/test.json --gym production
```

If you don't specify `--gym`, the default gym from `gyms.json` will be used.

### Additional useful options

- `--gym`: specify which gym (MCP server environment) to run against (e.g., `local`, `staging`, `production`)
- `--gym-config`: path to a custom gyms.json file (defaults to searching current directory and project root)
- `--model`: request a specific model; repeat to run multiple models (optionally prefix with `<provider>:` to disambiguate, e.g. `--model anthropic:claude-4.5-sonnet-reasoning`)
- `--temperature`: adjust sampling temperature (default `0.1`)
- `--max-output-tokens`: limit the maximum number of generated tokens
- `--harness-file`: select any benchmark JSON file without changing the default
- `--env-file`: load environment variable overrides from a specific `.env` file
- `--runs`: launch multiple parallel executions, each with its own MCP database
- `--ui`: choose `textual` to open a multi-pane terminal UI (default `auto` picks Textual when `--runs > 1`)
- `--background`: run benchmarks in the background (use `--view` to monitor progress)
- `--view`: replay a saved session (see below)

You must provide either `--harness-file` or `--prompt-file`; the CLI will exit if neither is supplied.

When you run the command the CLI will:

1. Load the benchmark scenarios from the JSON file
2. Fetch all available tools from the configured MCP server
3. Instantiate the requested LLM
4. Execute every prompt sequentially without pausing for user confirmation

## Replaying saved sessions

Each invocation writes a manifest plus per-run artifacts inside `results/` in a folder named `<harness>_<n>`. You can revisit those runs at any time:

```bash
# List all saved sessions (newest first)
python -m jira_mcp_benchmark --view list

# Launch an interactive picker (arrow keys to navigate)
python -m jira_mcp_benchmark --view

# Replay a specific session in Textual UI
python -m jira_mcp_benchmark --view new_sys_task13_7_c1_p1_r8_1_benchmark_4 --ui textual

# Replay the same session in plain console mode
python -m jira_mcp_benchmark --view new_sys_task13_7_c1_p1_r8_1_benchmark_4 --ui plain
```

While replaying, the Textual UI rebuilds the original log panes and live verifier table for each run. Plain mode streams the saved log output back to the console.

## Commands

### `run`
Execute benchmark scenarios against an MCP server.

```bash
python -m jira_mcp_benchmark run --harness-file <file> --gym <gym-name>
```

### `list-gyms`
List all configured gym environments.

```bash
python -m jira_mcp_benchmark list-gyms
```

### `cleanup`
Clean up stale background runs.

```bash
python -m jira_mcp_benchmark cleanup --stale-hours 24 --no-dry-run
```

## Notes

- The CLI uses gym configurations from `gyms.json` to connect to different MCP server environments.
- Each scenario allows up to 1000 MCP tool invocations by default; raise or lower the limit in `src/jira_mcp_benchmark/cli.py` if needed.
- A fresh UUID is used for the `x-database-id` header on every invocation so each run operates against an isolated MCP database instance.
- Verifier queries are executed via `POST /api/sql-runner` using the same `x-database-id`, so ensure that endpoint is reachable on the MCP host.
- When running multiple parallel executions, the default `textual` UI gives each run its own tab so logs never interleave; switch back to `--ui plain` if you prefer classic console output.
- Each CLI invocation creates a new session folder under `results/` named `<harness>_<n>` containing one JSON artifact per run. Each artifact records the interleaved conversation, tool calls/results, and verifier snapshots in execution order.
- The CLI uses LangChain's tool binding APIs; tool responses and final assistant messages are echoed to the terminal to aid debugging.
- If you add new scenarios, simply update the JSON file—no code changes are required.
- Run artifacts include the gym name and URL used for the run, making it easy to track which environment was tested.
