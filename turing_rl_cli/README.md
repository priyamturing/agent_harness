# Turing RL CLI

Command-line interface for running agent benchmarks using the Turing RL SDK. Run evaluations against multiple LLM providers with MCP (Model Context Protocol) tool calling capabilities.

## Installation

### Option 1: Install Directly from GitHub (No Cloning Needed)

```bash
pip install git+https://github.com/TuringGpt/rl-gym-tooling.git@sdk_initial#subdirectory=turing_rl_cli
```

This automatically installs the SDK as a dependency.

> **Note:** Once merged to `main`, remove `@sdk_initial` from the URL above.

### Option 2: Install from Local Clone (Development)

```bash
# Clone the repository and checkout the branch
git clone https://github.com/TuringGpt/rl-gym-tooling.git
cd rl-gym-tooling
git checkout sdk_initial
cd turing_rl_cli

# Install in editable mode
pip install -e .
```

## Quick Start

```bash
# Run a single benchmark harness file
turing-rl --prompt-file test_harness.json --model gpt-5-high

# Run with multiple models in parallel
turing-rl --prompt-file test_harness.json \
  --model gpt-4o \
  --model claude-sonnet-4-5 \
  --model gemini-2.0-flash

# Run all JSON files in a directory
turing-rl --harness-file ./benchmarks --model gpt-4o
```

## Supported Models

**Important:** Model names must match the exact identifiers expected by each provider. The CLI auto-detects the provider from the model name prefix (e.g., `gpt-`, `claude-`, `gemini-`, `grok-`) or from an explicit provider prefix (`openai:`, `anthropic:`, `google:`, `xai:`).

### Model Name Reference

| Provider | Model Name Examples | Documentation Link |
| --- | --- | --- |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o1`, `o3-mini` | [OpenAI Models](https://platform.openai.com/docs/models) |
| **Anthropic** | `claude-sonnet-4-5`, `claude-opus-4`, `claude-haiku-3-5` | [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models) |
| **Google** | `gemini-2.0-flash-exp`, `gemini-2.5-pro`, `gemini-1.5-pro` | [Google AI Models](https://ai.google.dev/gemini-api/docs/models/gemini) |
| **xAI** | `grok-4`, `grok-vision-beta` | [xAI Models](https://docs.x.ai/docs/models) |
| **Qwen** | `qwen-max`, `qwen-plus`, `qwen-turbo` | [Qwen Models](https://help.aliyun.com/zh/dashscope/developer-reference/model-square) |

### Usage Examples

**OpenAI (GPT):**
```bash
turing-rl --prompt-file test.json --model gpt-4o
turing-rl --prompt-file test.json --model openai:gpt-4o-mini
```

**Anthropic (Claude):**
```bash
turing-rl --prompt-file test.json --model claude-sonnet-4-5
turing-rl --prompt-file test.json --model anthropic:claude-opus-4
```

**Google (Gemini):**
```bash
turing-rl --prompt-file test.json --model gemini-2.0-flash-exp
turing-rl --prompt-file test.json --model google:gemini-2.5-pro
```

**xAI (Grok):**
```bash
turing-rl --prompt-file test.json --model grok-4
turing-rl --prompt-file test.json --model xai:grok-vision-beta
```

**Qwen:**
```bash
turing-rl --prompt-file test.json --model qwen-max
```

### OpenRouter (Hundreds of Models)

OpenRouter provides access to hundreds of models through a unified API:

1. Get an API key from <https://openrouter.ai/keys> and set it:
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```

2. (Optional) Set attribution headers for the leaderboard:
   ```bash
   export OPENROUTER_HTTP_REFERER="https://your-app.example"
   export OPENROUTER_APP_TITLE="Your App Name"
   ```

3. Use any tool-capable model from <https://openrouter.ai/api/v1/models> with the `openrouter:` prefix:
   ```bash
   turing-rl --prompt-file test.json \
     --model openrouter:anthropic/claude-3.5-sonnet \
     --model openrouter:google/gemini-2.0-flash-exp
   ```

Reasoning tokens are automatically requested for compatible models. Override with:
- `OPENROUTER_REASONING='{"effort":"high"}'`
- `OPENROUTER_INCLUDE_REASONING=false`

See the [Reasoning Tokens guide](https://openrouter.ai/docs/use-cases/reasoning-tokens) for more options.

## Harness File Format

Harness files are JSON files defining benchmark scenarios with prompts and verifiers. Example:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_task",
      "name": "Create Task",
      "description": "Test agent's ability to create a task",
      "prompts": [
        {
          "prompt_text": "Create a task called 'Update documentation' in the TODO project",
          "expected_tools": ["create_task"],
          "verifier": {
            "verifier_type": "database_state",
            "name": "task_created",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM tasks WHERE title = 'Update documentation'",
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

## UI Modes

The CLI supports three output modes:

### Quiet Mode (Default)
Progress bar with summary only. Best for CI/CD and batch runs:
```bash
turing-rl --prompt-file test.json --model gpt-4o --ui quiet
```

### Plain Mode
Streaming output with detailed step-by-step agent execution:
```bash
turing-rl --prompt-file test.json --model gpt-4o --ui plain
```

### Textual Mode
Rich terminal UI with real-time progress for multiple concurrent runs:
```bash
turing-rl --prompt-file test.json --model gpt-4o --ui textual
```

### Auto Mode
Automatically selects best UI based on number of runs (textual for multiple, plain for single):
```bash
turing-rl --prompt-file test.json --model gpt-4o --ui auto
```

## Configuration Options

### Model Configuration
```bash
# Temperature (defaults to 0.1, or 1.0 for Claude models)
turing-rl --prompt-file test.json --model gpt-4o --temperature 0.5

# Max output tokens
turing-rl --prompt-file test.json --model gpt-4o --max-output-tokens 4096
```

### Execution Control
```bash
# Multiple runs per scenario
turing-rl --prompt-file test.json --model gpt-4o --runs 5

# Maximum agent steps per run
turing-rl --prompt-file test.json --model gpt-4o --max-steps 100

# Tool call limit per run
turing-rl --prompt-file test.json --model gpt-4o --tool-call-limit 50

# Maximum concurrent runs (default: 20)
turing-rl --prompt-file test.json --model gpt-4o --max-concurrent-runs 10
```

### MCP Server Configuration
```bash
# Use custom MCP server (defaults to local JIRA MCP at http://localhost:8015/mcp)
turing-rl --prompt-file test.json --model gpt-4o --mcp-url http://localhost:9000/mcp
```

### Environment Files
```bash
# Load environment variables from custom .env file
turing-rl --prompt-file test.json --model gpt-4o --env-file .env.production
```

## Observability & Tracing

### LangSmith Integration

Enable LangSmith tracing for detailed observability:

```bash
# Set API key
export LANGCHAIN_API_KEY="lsv2_pt_..."

# Enable with default project name
turing-rl --prompt-file test.json --model gpt-4o --langsmith

# Enable with custom project name
turing-rl --prompt-file test.json --model gpt-4o \
  --langsmith --langsmith-project "my-benchmark-project"
```

Or configure via environment variables:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="my-project"
export LANGCHAIN_API_KEY="lsv2_pt_..."
turing-rl --prompt-file test.json --model gpt-4o
```

### Langfuse Integration

Enable Langfuse tracing for open-source observability:

```bash
# Set API keys
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."

# Enable with default settings
turing-rl --prompt-file test.json --model gpt-4o --langfuse

# Enable with custom configuration
turing-rl --prompt-file test.json --model gpt-4o \
  --langfuse \
  --langfuse-base-url "https://langfuse.example.com" \
  --langfuse-environment "production" \
  --langfuse-release "v1.0.0"
```

Or configure via environment variables:
```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_TRACING_ENVIRONMENT="staging"
export LANGFUSE_RELEASE="v2.1.0"
turing-rl --prompt-file test.json --model gpt-4o
```

## Results & Output

Results are saved in the `results/` directory with the following structure:

```
results/
└── session_name/
    ├── session.json                           # Session manifest
    ├── scenario1_gpt-4o.json                 # Individual run results
    ├── scenario1_claude-sonnet-4-5.json
    └── harness_name_model_name.json          # Aggregated results
```

Each result file contains:
- Full conversation history with tool calls
- Verifier results and validation status
- Model metadata (provider, temperature, tokens used)
- Tracing URLs (LangSmith/Langfuse) if enabled
- Timing information and error details

## Complete Example

```bash
# Run comprehensive benchmark across multiple models with tracing
turing-rl \
  --harness-file ./benchmarks/jira_workflows \
  --model gpt-4o \
  --model claude-sonnet-4-5 \
  --model gemini-2.0-flash \
  --runs 3 \
  --temperature 0.1 \
  --max-steps 50 \
  --max-concurrent-runs 15 \
  --ui textual \
  --langsmith \
  --langsmith-project "jira-workflows-eval" \
  --env-file .env.production
```

## Environment Variables

Required API keys (set at least one):
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GOOGLE_API_KEY` or `GOOGLE_GENAI_API_KEY` - For Gemini models
- `XAI_API_KEY` - For Grok models
- `DASHSCOPE_API_KEY` - For Qwen models
- `OPENROUTER_API_KEY` - For OpenRouter models

Optional tracing:
- `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` - LangSmith
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` - Langfuse

OpenRouter attribution (optional but recommended):
- `OPENROUTER_HTTP_REFERER` - Your app URL
- `OPENROUTER_APP_TITLE` - Your app name

## Command Reference

```
turing-rl [OPTIONS]

Options:
  --prompt-file PATH           Path to benchmark JSON file or directory
  --harness-file PATH          Alternative to --prompt-file
  --model TEXT                 Model to run (repeatable for multiple models)
  --temperature FLOAT          Sampling temperature
  --max-output-tokens INT      Maximum output tokens
  --runs INT                   Number of parallel runs per model/scenario
  --ui [auto|plain|textual|quiet]  UI mode
  --env-file PATH              Path to .env file
  --mcp-url TEXT               MCP server URL
  --max-steps INT              Maximum agent steps per run
  --max-concurrent-runs INT    Maximum concurrent runs
  --tool-call-limit INT        Maximum tool calls per run
  --langsmith                  Enable LangSmith tracing
  --langsmith-project TEXT     LangSmith project name
  --langfuse                   Enable Langfuse tracing
  --langfuse-public-key TEXT   Langfuse public key
  --langfuse-secret-key TEXT   Langfuse secret key
  --langfuse-base-url TEXT     Langfuse base URL
  --langfuse-environment TEXT  Langfuse environment label
  --langfuse-release TEXT      Langfuse release identifier
```

## See Also

- [MCP Benchmark SDK](../turing_rl_sdk/README.md) - Core SDK for programmatic usage
- [Model Context Protocol](https://modelcontextprotocol.io) - MCP specification
