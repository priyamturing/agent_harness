# Turing RL SDK

A harness-first Python SDK for running **LLM agent benchmarks against Model Context Protocol (MCP) servers**. The SDK ships with:

- An asynchronous `TestHarness` that loads JSON scenarios, manages concurrency, and aggregates verifier results.
- Reference agents for OpenAI (`GPTAgent`), Anthropic (`ClaudeAgent`), Google (`GeminiAgent`), and xAI (`GrokAgent`) plus a `create_agent()` factory for auto-selection.
- A consistent telemetry surface (LangSmith + Langfuse), structured observers, and tooling-aware transcripts for debugging.
- Database verifiers powered by SQL runner endpoints so you can assert task outcomes deterministically.

Everything in the SDK is built on the primitives exported from `turing_rl_sdk.__init__`—you can adopt only what you need.

---

## Installation

### Option 1: Install Directly from GitHub (No Cloning Needed)

```bash
pip install git+https://github.com/TuringGpt/rl-gym-tooling.git@sdk_initial#subdirectory=turing_rl_sdk
```

> **Note:** Once merged to `main`, remove `@sdk_initial` from the URL above.

### Option 2: Install from Local Clone (Development)

```bash
# Clone the repository and checkout the branch
git clone https://github.com/TuringGpt/rl-gym-tooling.git
cd rl-gym-tooling
git checkout sdk_initial
cd turing_rl_sdk

# Install in editable mode
pip install -e .

# Or with optional extras
pip install -e ".[cli]"  # Adds companion CLI
pip install -e ".[dev]"  # Adds linting + pytest tooling
```

### Requirements

- Python 3.10+
- API keys for any LLM vendors you plan to benchmark
- A reachable MCP server (HTTP/SSE or stdio)

---

## Quick Start

### 1. Define a Harness File

Harness files are JSON documents with **single-prompt scenarios**. Each prompt can attach one or more verifiers (currently `database_state`). Example saved as `simple_task.json`:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_bug",
      "name": "Create a bug issue",
      "description": "Checks whether an agent can create a JIRA bug",
      "prompts": [
        {
          "prompt_text": "Create a bug called 'Login button not working'",
          "expected_tools": ["create_issue"],
          "verifier": {
            "verifier_type": "database_state",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Login button not working'",
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

### 2. Run the Harness

```python
import asyncio
from pathlib import Path
from turing_rl_sdk import (
    MCPConfig,
    TestHarness,
    TestHarnessConfig,
    create_agent,
)

async def main() -> None:
    harness = TestHarness(
        harness_path=Path("simple_task.json"),
        config=TestHarnessConfig(
            mcp=MCPConfig(
                name="jira",
                url="http://localhost:8015/mcp",
                transport="streamable_http",
            ),
            max_steps=50,
            tool_call_limit=100,
            runs_per_scenario=1,
            max_concurrent_runs=5,
        ),
    )
    
    results = await harness.run(
        models=["gpt-4o", "claude-sonnet-4-5"],
        agent_factory=create_agent,
    )
    
    for result in results:
        status = "PASS" if result.success else "FAIL"
        print(f"{result.model} · {result.scenario_id} · {status}")
        if result.error:
            print(f"  error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

The harness:

- Loads every scenario from a file or directory
- Spins up agent instances per model
- Establishes MCP connections (injecting `x-database-id` for isolation)
- Executes prompts with concurrency control
- Runs verifiers (SQL runners derived from the MCP URL)
- Returns a `ResultBundle` with helpers for exports and downstream analytics

### Model Names

**Important:** Model names must match the exact identifiers expected by each LLM provider. The SDK auto-detects the provider from the model name prefix (e.g., `gpt-`, `claude-`, `gemini-`, `grok-`) or from an explicit provider prefix (`openai:`, `anthropic:`, `google:`, `xai:`).

**Supported Providers:**

| Provider | Model Name Examples | Documentation Link |
| --- | --- | --- |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o1`, `o3-mini` | [OpenAI Models](https://platform.openai.com/docs/models) |
| **Anthropic** | `claude-sonnet-4-5`, `claude-opus-4`, `claude-haiku-3-5` | [Anthropic Models](https://docs.anthropic.com/en/docs/about-claude/models) |
| **Google** | `gemini-2.0-flash-exp`, `gemini-2.5-pro`, `gemini-1.5-pro` | [Google AI Models](https://ai.google.dev/gemini-api/docs/models/gemini) |
| **xAI** | `grok-4`, `grok-vision-beta` | [xAI Models](https://docs.x.ai/docs/models) |

**Usage:**

```python
# Auto-detection (recommended)
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash-exp"],
    agent_factory=create_agent,
)

# Explicit provider prefix (when auto-detection fails)
results = await harness.run(
    models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"],
    agent_factory=create_agent,
)
```

---

## Feature Overview

- **Harness orchestration** – `TestHarness` interleaves runs per model, enforces `max_steps`, `tool_call_limit`, and `runs_per_scenario`, and records execution timing + metadata.
- **Scenario loading** – `HarnessLoader`, `load_harness_file()`, and `load_harness_directory()` validate JSON and guard against multi-prompt scenarios (not supported).
- **Agent system** – A common `Agent` base class handles tool loading, loop execution, retry-safe MCP calls, and conversation logging. Provider-specific subclasses override `_build_llm()` and `get_response_parser()`.
- **Factory helpers** – `create_agent()` infers the provider from strings like `"gpt-4o"` or `"anthropic:claude-sonnet-4-5"`, while `create_traced_agent()` automatically wraps agents with tracing when enabled.
- **Runtime observers** – Implement `RunObserver` to watch `on_message`, `on_tool_call`, and `on_status` events. Attach factories with `TestHarness.add_observer_factory()` to get per-run instrumentation or UI hooks.
- **Telemetry** – `configure_langsmith()` / `configure_langfuse()` set the necessary environment flags. `with_tracing()` or `create_traced_agent()` inject LangChain callbacks, storing trace URLs on `Result`.
- **Verification pipeline** – Verifiers are described in JSON via `VerifierDefinition` and materialized just-in-time. `DatabaseVerifier` sends SQL to the `/api/sql-runner` endpoint derived from your MCP URL and supports equality and ordered comparisons.
- **Result exports** – `RunResult` exposes `get_conversation_history()` and `to_dict()`. `ResultBundle.build_model_reports()` groups runs by harness/model, producing payloads that match downstream benchmark ingestion formats.

---

## Harness JSON Reference

| Field | Description |
| --- | --- |
| `scenario_id` | Unique identifier, becomes metadata on `Task` and `RunResult`. |
| `prompts` | Exactly one entry per scenario. Multi-prompt harnesses are rejected at load time. |
| `prompt_text` | User instruction delivered as `Task.prompt`. |
| `expected_tools` | Advisory list only (stored for analytics, not enforced automatically). |
| `verifier` / `verifiers` | Either a single object or an array. Set `verifier_type` to `database_state`. |
| `metadata` | Arbitrary fields merged into task metadata (difficulty, tags, etc.). |

`database_state` verifier config:

```json
{
  "verifier_type": "database_state",
  "name": "issue_inserted",
  "validation_config": {
    "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Login button not working'",
    "expected_value": 1,
    "comparison_type": "equals"
  }
}
```

Comparison types include `equals`, `not_equal`, `greater_than`, `less_than`, `greater_than_equal`, and `less_than_equal` (case-insensitive with aliases like `eq`, `!=`, `gt`, `<=`, etc.). In Python code, you can use the `ComparisonType` enum for type safety (e.g., `ComparisonType.GREATER_THAN`). See the [harness README](src/turing_rl_sdk/harness/README.md#database-verifier-comparison-types) for the full list of supported comparisons and examples.

---

## Programmatic Patterns

### Run Harnesses Programmatically

```python
from pathlib import Path
from turing_rl_sdk import (
    MCPConfig,
    TestHarness,
    TestHarnessConfig,
    create_agent,
)

mcp = MCPConfig(name="jira", url="http://localhost:8015/mcp")
harness = TestHarness(
    harness_path=Path("benchmarks/"),
    config=TestHarnessConfig(
        mcp=mcp,
        runs_per_scenario=3,
        max_concurrent_runs=10,
        temperature=0.2,
    ),
)

# Optional observers (e.g., console, UI, tracing)
from turing_rl_sdk import RunObserver

class ConsoleObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"[{role}] {content[:80]}")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        status = "✗" if is_error else "✓"
        print(f"{status} tool {tool_name}")

    async def on_status(self, message, level="info"):
        print(f"[{level}] {message}")

harness.add_observer_factory(ConsoleObserver)
results = await harness.run(models=["gpt-4o", "gemini-2.0-flash-exp"], agent_factory=create_agent)

bundle = results  # ResultBundle
reports = bundle.build_model_reports()
```

### Direct Agent Usage (No Harness)

```python
from turing_rl_sdk import ClaudeAgent, MCPConfig, Task

agent = ClaudeAgent(
    model="claude-sonnet-4-5",
    temperature=0.1,
    tool_call_limit=50,
)

task = Task(
    prompt="Create a bug in DEMO titled 'Homepage not loading'",
    mcp=MCPConfig(name="jira", url="http://localhost:8015/mcp"),
)

result = await agent.run(task, max_steps=40)
print("Success:", result.success)

# Manual verification
from turing_rl_sdk import DatabaseVerifier

verifier = DatabaseVerifier(
    query="SELECT COUNT(*) FROM issue WHERE summary = 'Homepage not loading'",
    expected_value=1,
    comparison="equals",
    mcp_url=task.mcp.url,
    database_id=result.database_id,
)
verifier_result = await verifier.verify()
print("Verified:", verifier_result.success)
```

### Custom Agents

Subclass `Agent` (from `turing_rl_sdk.agents.core.base`) and override:

- `_build_llm()` – create or wrap your LangChain model/runnable
- `get_response()` – call the model and parse into `AgentResponse`
- `get_response_parser()` – return a provider-specific parser

Example skeleton:

```python
from turing_rl_sdk import Agent, AgentResponse
from langchain_core.messages import AIMessage

class QwenAgent(Agent):
    def __init__(self, model="qwen-plus", **kwargs):
        super().__init__(**kwargs)
        self.model = model
    
    def _build_llm(self):
        # Build ChatOpenAI / DashScope client, optionally bind tools
        ...

    async def get_response(self, messages):
        ai_message = await self._llm.ainvoke(messages)
        parsed = self.get_response_parser().parse(ai_message)
        return AgentResponse(
            content=parsed.content,
            tool_calls=parsed.tool_calls,
            reasoning=parsed.reasoning,
            done=not bool(parsed.tool_calls),
        ), ai_message
    
    def get_response_parser(self):
        from turing_rl_sdk import ResponseParser
        return ResponseParser()  # e.g., OpenAIResponseParser
```

Register this agent inside a custom factory and pass it to the harness.

### MCP Connectivity

- Use `MCPConfig` for HTTP/SSE (`url="https://..."`) or stdio (`command="npx"`, `args=["@modelcontextprotocol/server-filesystem"]`).
- `TestHarness` and `Agent` instances automatically attach the `database_id` as `x-database-id` so SQL verifiers run against the same logical tenant.
- `derive_sql_runner_url()` converts `http://host:port/mcp` into the SQL runner route (`http://host:port/api/sql-runner`).

### Tracing & Telemetry

```python
import os
from turing_rl_sdk import configure_langsmith, with_tracing, ClaudeAgent

configure_langsmith(enabled=True, api_key=os.environ["LANGCHAIN_API_KEY"], project_name="jira-benchmarks")

agent = with_tracing(ClaudeAgent(model="claude-sonnet-4-5"))
result = await agent.run(task)
print("LangSmith trace:", result.langsmith_url)
```

Set `configure_langfuse(...)` similarly to capture Langfuse traces. When tracing is enabled, `create_traced_agent()` returns a `TracingAgent` wrapper that injects callbacks automatically.

---

## Examples & Further Reading

- `examples/simple_harness_example.py` – minimal harness runner
- `examples/simple_task.json` – starter harness file
- `examples/SDK_USAGE_GUIDE.ipynb` – exploratory notebook
- `QUICKSTART.md` – narrative tutorial
- `IMPORT_GUIDE.md` – tips for importing external benchmark definitions
- `src/turing_rl_sdk/harness/README.md` – harness deep dive

---

## Development

```bash
pip install -e ".[dev]"
ruff check
pytest
```

The repository uses `src/` layout with setuptool’s package discovery. When contributing:

- Keep new code ASCII unless a file already uses Unicode.
- Follow the `RunObserver` contract for logging utilities.
- Prefer instrumented MCP interactions via `MCPClientManager` instead of bespoke clients.

---

## License

Distributed under the MIT License. See `LICENSE` in the repo root.

