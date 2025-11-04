# Jira MCP Benchmark

A comprehensive benchmarking harness for testing LLM agents against a local Jira MCP (Model Context Protocol) server. This tool enables systematic evaluation of different LLM providers (OpenAI, Anthropic, Google, xAI) performing project management tasks through MCP tools.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Running Benchmarks](#running-benchmarks)
  - [Viewing Results](#viewing-results)
  - [Command Reference](#command-reference)
- [Benchmark Format](#benchmark-format)
- [Supported Models](#supported-models)
- [Output & Results](#output--results)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Overview

This benchmark harness:
- Connects to a local MCP server (Jira implementation) via HTTP transport
- Executes predefined scenarios with multiple prompts and verification steps
- Supports parallel runs with isolated database instances
- Provides both plain console and rich TUI (Textual) interfaces
- Records detailed artifacts including conversation logs and verifier results
- Enables cross-model comparison and replay of previous sessions

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    Benchmark Harness (CLI)             │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   OpenAI     │  │  Anthropic   │  │   Google     │  │
│  │   Provider   │  │   Provider   │  │   Provider   │  │
│  └──────┬───────┘  └───────┬──────┘  └──────┬───────┘  │
│         │                  │                │          │
│         └──────────────────┴────────────────┘          │
│                            │                           │
│                    ┌───────▼────────┐                  │
│                    │  LangChain     │                  │
│                    │  Agent Loop    │                  │
│                    └───────┬────────┘                  │
│                            │                           │
│                    ┌───────▼────────┐                  │
│                    │  MCP Adapter   │                  │
│                    │  (langchain-   │                  │
│                    │   mcp-adapters)│                  │
│                    └───────┬────────┘                  │
└────────────────────────────┼───────────────────────────┘
                             │
                             │ HTTP (streamable_http transport)
                             │ x-database-id header for isolation
                             │
                    ┌────────▼────────┐
                    │                 │
                    │   Jira MCP      │
                    │   Server        │
                    │                 │
                    │  Port: 8015     │
                    │  Endpoint:      │
                    │  /mcp           │
                    │                 │
                    └─────────────────┘
```

### Key Components

1. **CLI (`cli.py`)**: Main entry point, handles argument parsing, session management, and orchestration
2. **Agent (`agent.py`)**: Executes scenarios using LangChain, manages tool calls and retries
3. **MCP Loader (`mcp_loader.py`)**: Connects to MCP server and loads tools
4. **Providers (`providers.py`)**: Abstracts different LLM providers (OpenAI, Anthropic, Google, xAI)
5. **Verifier (`verifier.py`)**: Validates agent outputs against expected results
6. **Run Logging**: Captures conversation history and status updates

## Prerequisites

### 1. Jira MCP Server

**CRITICAL**: You must have a Jira MCP server running locally before using this harness.

- **Port**: `8015` (default)
- **Endpoint**: `http://localhost:8015/mcp`
- **Transport**: `streamable_http`
- **Required Header**: `x-database-id` (automatically set by harness for database isolation)

The harness expects the MCP server to:
- Accept HTTP connections on port 8015
- Provide MCP tools via the `/mcp` endpoint
- Support the `x-database-id` header for multi-tenant database isolation
- Expose a SQL runner endpoint at `http://localhost:8015/api/sql-runner` for verifiers

**Note**: The MCP server is NOT included in this repository. You must set it up separately.

### 2. Python Environment

- Python 3.10 or higher
- pip or uv package manager

### 3. API Keys

You'll need API keys for the LLM providers you want to test:
- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic**: `ANTHROPIC_API_KEY`
- **Google**: `GOOGLE_API_KEY`
- **xAI**: `XAI_API_KEY`

## Installation

### Option 1: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd agent_harness

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 2: Using uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd agent_harness

# Install with uv
uv pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: At least one provider API key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
XAI_API_KEY=...

# Optional: Google thinking budget
GOOGLE_THINKING_BUDGET=10000
```

Alternatively, export these variables in your shell:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### MCP Server Configuration

The harness uses these default settings (currently hardcoded in `cli.py`):

```python
DEFAULT_MCP_URL = "http://localhost:8015/mcp"
DEFAULT_MCP_TRANSPORT = "streamable_http"
DEFAULT_SQL_RUNNER_URL = "http://localhost:8015/api/sql-runner"
```

**To change these**: Edit the constants in `src/jira_mcp_benchmark/cli.py` if your MCP server runs on a different port or endpoint.

## Usage

### Running Benchmarks

#### Basic Usage

```bash
# Run with default model (gpt-5-high)
python -m jira_mcp_benchmark --prompt-file test_harness.json

# Run with specific model
python -m jira_mcp_benchmark --prompt-file test_harness.json --model claude-sonnet-4-5

# Run with multiple models for comparison
python -m jira_mcp_benchmark \
  --prompt-file test_harness.json \
  --model gpt-5-high \
  --model claude-sonnet-4-5 \
  --model gemini-2.5-pro
```

#### Advanced Options

```bash
# Multiple parallel runs with unique databases
python -m jira_mcp_benchmark \
  --prompt-file test_harness.json \
  --model gpt-5-high \
  --runs 3

# Run entire directory of test files
python -m jira_mcp_benchmark --prompt-file test_harness_small_10-20/

# Custom temperature and token limits
python -m jira_mcp_benchmark \
  --prompt-file test_harness.json \
  --model gpt-5-high \
  --temperature 0.5 \
  --max-output-tokens 4096

# Force plain console output (no TUI)
python -m jira_mcp_benchmark \
  --prompt-file test_harness.json \
  --model gpt-5-high \
  --ui plain

# Use custom .env file
python -m jira_mcp_benchmark \
  --prompt-file test_harness.json \
  --env-file .env.production
```

### Viewing Results

#### List Previous Sessions

```bash
# List all saved sessions
python -m jira_mcp_benchmark --view list
```

#### Interactive Session Picker

```bash
# Launch interactive session picker (TUI)
python -m jira_mcp_benchmark --view

# Use plain console picker
python -m jira_mcp_benchmark --view --ui plain
```

#### View Specific Session

```bash
# View by path
python -m jira_mcp_benchmark --view results/test_harness_1

# View with TUI replay
python -m jira_mcp_benchmark --view results/test_harness_1 --ui textual
```

### Command Reference

```
python -m jira_mcp_benchmark [OPTIONS]

Options:
  --prompt-file PATH           Benchmark JSON file or directory
  --harness-file PATH          Alternative to --prompt-file
  --model TEXT                 Model to run (can be repeated)
                              Format: [provider:]model
                              Examples: gpt-5-high, anthropic:claude-sonnet-4-5
  --temperature FLOAT          Sampling temperature (0.0-1.0) [default: 0.1]
  --max-output-tokens INTEGER  Maximum output tokens
  --runs INTEGER               Number of parallel runs [default: 1]
  --ui [auto|plain|textual]    UI mode [default: auto]
  --env-file PATH              Load environment from .env file
  --view [PATH|list]           View previous session
  --help                       Show help message
```

## Benchmark Format

Benchmark files are JSON files with the following structure:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_issue_basic",
      "name": "Create Basic Issue",
      "description": "Test creating a simple issue in a project",
      "conversation_mode": false,
      "metadata": {},
      "prompts": [
        {
          "prompt_text": "Create a new bug issue in project DEMO with summary 'Login button not working'",
          "expected_tools": ["mcp_jira-mcp-frozen_create_issue"],
          "verifier": {
            "verifier_type": "database_state",
            "name": "Issue Created",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM issues WHERE summary = 'Login button not working'",
              "expected_value": 1,
              "comparison_type": "equals"
            }
          }
        }
      ]
    }
  ]
}
```

### Verifier Types

- **database_state**: Execute SQL query against the MCP server's SQL runner endpoint and compare result
  - Supports `comparison_type`: `equals` (only comparison type currently implemented)
  - Requires `query`, `expected_value`, and `comparison_type` in `validation_config`

## Supported Models

### OpenAI
- `gpt-5-high` (default, with high reasoning effort)
- `gpt-5`
- `gpt-5-mini`
- `gpt-4o` (also accepts `gpt4o`)
- `o4-mini`

### Anthropic
- `claude-sonnet-4-5` (default, with extended thinking)
- `claude-4.5-sonnet`
- `claude-4.5-sonnet-reasoning`
- `claude-sonnet-4.5`
- `claude-sonnet-4.5-reasoning`

### Google
- `gemini-2.5-pro` (default)
- `gemini-2.5-pro-latest`
- `gemini-2.5-pro-preview`
- `gemini-2-5-pro`
- `models/gemini-2.5-pro`

### xAI
- `grok-4` (default, also accepts `grok4`)
- `grok-3-mini`
- `grok-2`

### Specifying Providers

```bash
# Auto-detect provider
--model gpt-5-high

# Explicit provider
--model openai:gpt-5-high
--model anthropic:claude-sonnet-4-5
--model google:gemini-2.5-pro
--model xai:grok-4
```

## Output & Results

### Session Directory Structure

Each run creates a session directory in `results/`:

```
results/
└── test_harness_1/
    ├── session.json              # Session metadata and summary
    ├── run_gpt-5-high.json       # Detailed run artifacts
    ├── run_claude-sonnet-4-5.json
    └── ...
```

### Artifact Contents

Each `run_*.json` file contains:
- Full conversation history (messages, tool calls, responses)
- Verifier results (pass/fail for each check)
  - Includes SQL query used for verification
  - Comparison type and expected/actual values
  - Success status and error details (if any)
- Model metadata (provider, model name, temperature)
- Timing information
- Error details (if any)

### Verifier Result Format

Example verifier result in `run_*.json`:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_issue_basic",
      "verifiers": [
        {
          "name": "Issue Created",
          "comparison": "equals",
          "expected": 1,
          "actual": 1,
          "success": true,
          "error": null,
          "sql_query": "SELECT COUNT(*) FROM issues WHERE summary = 'Login button not working'"
        }
      ]
    }
  ]
}
```

Each verifier result includes:
- `name`: Verifier display name
- `comparison`: Comparison type used (e.g., "equals")
- `expected`: Expected value from the benchmark
- `actual`: Actual value returned from the SQL query
- `success`: Boolean indicating if verification passed
- `error`: Error message if verification failed (null if successful)
- `sql_query`: The SQL query executed for verification

### Session Manifest

The `session.json` file includes:
- Timestamp and display name
- List of all runs with summaries
- Model usage statistics
- Prompt file references

## Troubleshooting

### MCP Connection Issues

**Error**: `Failed to connect to MCP server`

**Solution**:
1. Verify MCP server is running: `curl http://localhost:8015/mcp`
2. Check the port (default: 8015)
3. Ensure no firewall is blocking localhost connections

### API Key Errors

**Error**: `OPENAI_API_KEY is not set`

**Solution**:
1. Create `.env` file with API keys
2. Or export environment variables
3. Use `--env-file` to specify custom .env location

### File Descriptor Exhaustion

**Error**: `Too many open files`

**Solution**: The harness limits concurrent MCP connections to 20 (see `MAX_CONCURRENT_RUNS` in `cli.py`). If you still hit limits:
1. Reduce `--runs` parameter
2. Increase system file descriptor limit: `ulimit -n 4096`

### Model Timeout

**Error**: `Model call timed out after 10 minute(s)`

**Solution**: The harness retries automatically. If persistent:
1. Check your network connection
2. Verify API key is valid
3. Check provider status page

## Roadmap

### Planned Features

- [ ] **Proper background run support**: Run benchmarks as daemon processes with progress monitoring
- [ ] **Multiple MCP server support**: Connect to and test against multiple MCP servers simultaneously
- [ ] **Extended model support**: Add support for more providers (Cohere, Mistral, etc.)
- [ ] **Benchmark analytics**: Built-in analysis and visualization of results

### Current Limitations

- MCP server URL/port is hardcoded (requires code change to modify)
- Single MCP server connection per run
- No built-in MCP server management
- Limited to HTTP transport (no stdio or SSE support yet)



