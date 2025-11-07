# MCP Benchmark SDK

A Python SDK for building and running **LLM agent benchmarks** against MCP (Model Context Protocol) servers. The SDK is **harness-first**, designed to make it easy to run systematic benchmarks across multiple models, scenarios, and verifiers.

## Table of Contents

- [Why This SDK?](#why-this-sdk)
- [Installation](#installation)
- [Quick Reference](#quick-reference)
- [Quick Start - The Simplest Harness](#quick-start---the-simplest-harness)
- [Core Concepts](#core-concepts)
- [Usage Patterns](#usage-patterns)
  - [1. Run Tests with Harness (Recommended)](#1-run-tests-with-harness-recommended)
  - [2. Use Agents Without Harness](#2-use-agents-without-harness)
  - [3. Create Custom Agents](#3-create-custom-agents)
  - [4. Create Custom Verifiers](#4-create-custom-verifiers)
- [Harness File Format](#harness-file-format)
- [Built-in Agents](#built-in-agents)
- [Verifiers](#verifiers)
- [API Reference](#api-reference)
- [Examples](#examples)

---

## Why This SDK?

The **MCP Benchmark SDK** is built around the **TestHarness** - a simple, powerful way to:

- ✅ **Systematically benchmark** LLM agents against MCP servers
- ✅ **Run multiple scenarios** with consistent evaluation criteria
- ✅ **Compare models** (GPT, Claude, Gemini, Grok, custom models)
- ✅ **Verify results** using SQL queries against the database state
- ✅ **Track execution** with built-in observers and LangSmith integration
- ✅ **Run at scale** with concurrent execution and automatic retries

The harness orchestrates everything: agent creation, MCP connections, task execution, verification, and result collection.

---

## Installation

```bash
pip install mcp-benchmark-sdk
```

**Requirements:**
- Python 3.10+
- API keys for LLM providers (OpenAI, Anthropic, Google, xAI, etc.)

### Package structure

The SDK is now composed of two installable libraries:

1. `mcp-benchmark-agents` – core agent runtime (LLM adapters, MCP clients, parsers, telemetry, verifiers).
2. `mcp-benchmark-harness` – scenario loader + benchmark orchestrator that depends on the agent library.

Installing `mcp-benchmark-sdk` brings both packages along for a batteries-included experience, but you can
also install either library individually when you only need part of the stack.

---

## Quick Reference

**Simplest possible usage:**

```python
from pathlib import Path
from mcp_benchmark_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

harness = TestHarness(
    harness_path=Path("task.json"),
    config=TestHarnessConfig(mcp=MCPConfig(name="jira", url="http://localhost:8015/mcp", transport="streamable_http"))
)

results = await harness.run(models=["gpt-4o"], agent_factory=create_agent)
print(f"Pass rate: {sum(r.success for r in results)}/{len(results)}")
```

**That's the SDK!** 🚀

The harness handles everything else - you just define scenarios and let it run.

---

## Quick Start - The Simplest Harness

Let's start with the absolute simplest way to use the SDK: **create and run a harness**.

### Step 1: Create a Harness File

Create a file called `simple_task.json`:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_bug",
      "name": "Create a bug issue",
      "description": "Test if agent can create a bug issue",
      "prompts": [
        {
          "prompt_text": "Create a bug issue in project DEMO with summary 'Login button not working' and description 'Users report the login button is unresponsive'",
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
      ],
      "metadata": {
        "difficulty": "easy"
      },
      "conversation_mode": false
    }
  ]
}
```

### Step 2: Run the Harness

```python
import asyncio
from pathlib import Path
from mcp_benchmark_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

async def main():
    # Configure MCP server connection
    mcp_config = MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http"
    )
    
    # Create harness
    harness = TestHarness(
        harness_path=Path("simple_task.json"),
        config=TestHarnessConfig(
            mcp=mcp_config,
            max_steps=50,
            tool_call_limit=100,
            runs_per_scenario=1,
        )
    )
    
    # Run benchmarks
    results = await harness.run(
        models=["gpt-4o"],
        agent_factory=create_agent,
    )
    
    # Print results
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"{result.model} - {result.scenario_id}: {status}")
        if result.error:
            print(f"  Error: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())
```

**That's it!** The harness handles everything:
- Loading scenarios from JSON
- Creating agents
- Connecting to MCP servers
- Running tasks
- Verifying results
- Collecting metrics

---

## Core Concepts

### 1. **TestHarness** - The Orchestrator

The `TestHarness` is the main entry point. It:
- Loads scenarios from JSON files or directories
- Creates agents for each model
- Executes tasks with proper isolation (separate database IDs)
- Runs verifiers to check results
- Collects and aggregates results

### 2. **Scenarios** - Test Definitions

A **scenario** defines:
- A unique ID and name
- One or more prompts (for multi-turn conversations)
- Expected tools the agent should use
- Verifiers to validate the outcome
- Metadata (difficulty, tags, etc.)

### 3. **Agents** - LLM Wrappers

**Agents** wrap LLM APIs and handle:
- Tool calling with MCP servers
- Message formatting
- Response parsing
- Retry logic
- Conversation management

Built-in agents: `ClaudeAgent`, `GPTAgent`, `GeminiAgent`, `GrokAgent`

### 4. **Verifiers** - Result Validation

**Verifiers** check if the agent succeeded by:
- Running SQL queries against the database
- Comparing actual vs expected values
- Supporting multiple comparison types (equals, greater_than, etc.)

---

## Usage Patterns

### 1. Run Tests with Harness (Recommended)

**Best for:** Systematic benchmarking, comparing models, running at scale

```python
from pathlib import Path
from mcp_benchmark_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

# Setup
mcp_config = MCPConfig(name="jira", url="http://localhost:8015/mcp", transport="streamable_http")
harness = TestHarness(
    harness_path=Path("benchmarks/"),  # Load all JSON files from directory
    config=TestHarnessConfig(
        mcp=mcp_config,
        max_steps=1000,
        runs_per_scenario=3,  # Run each scenario 3 times
        max_concurrent_runs=10,
    )
)

# Run across multiple models
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash-exp"],
    agent_factory=create_agent,
)

# Analyze results
successful = sum(1 for r in results if r.success)
print(f"Pass rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")

# Export to JSON for analysis
import json
with open("results.json", "w") as f:
    json.dump([r.to_dict() for r in results], f, indent=2)
```

**With Observers** (for progress tracking):

```python
from mcp_benchmark_sdk import RunObserver

class ConsoleObserver(RunObserver):
    def __init__(self, label: str):
        self.label = label
    
    async def on_message(self, role: str, content: str, metadata=None):
        print(f"[{self.label}] {role}: {content[:100]}")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        status = "✗" if is_error else "✓"
        print(f"[{self.label}] Tool {status}: {tool_name}")
    
    async def on_status(self, message: str, level: str = "info"):
        print(f"[{self.label}] {message}")

# Add observer factory
harness.add_observer_factory(lambda: ConsoleObserver("benchmark"))
```

---

### 2. Use Agents Without Harness

**Best for:** One-off tasks, interactive testing, custom workflows

```python
from mcp_benchmark_sdk import ClaudeAgent, Task, MCPConfig

# Create agent
agent = ClaudeAgent(
    model="claude-sonnet-4-5",
    temperature=0.1,
    tool_call_limit=100,
)

# Define task
task = Task(
    prompt="Create a bug issue in project DEMO titled 'Homepage not loading'",
    mcp=MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http"
    )
)

# Run task
result = await agent.run(task, max_steps=50)

print(f"Success: {result.success}")
print(f"Steps: {result.metadata.get('steps')}")

# Access conversation history
conversation = result.get_conversation_history()
for entry in conversation:
    if entry["type"] == "message":
        print(f"{entry['role']}: {entry['content']}")
    elif entry["type"] == "tool_call":
        print(f"Tool: {entry['tool']} with {entry['args']}")
```

**Manual Verification:**

```python
from mcp_benchmark_sdk import DatabaseVerifier

# Create verifier
verifier = DatabaseVerifier(
    query="SELECT COUNT(*) FROM issue WHERE summary = 'Homepage not loading'",
    expected_value=1,
    mcp_url="http://localhost:8015/mcp",
    database_id=result.database_id,  # Use same database as task
    comparison="equals"
)

# Run verification
verifier_result = await verifier.verify()
print(f"Verified: {verifier_result.success}")
print(f"Expected: {verifier_result.expected_value}, Got: {verifier_result.actual_value}")
```

---

### 3. Create Custom Agents

**Best for:** Integrating new LLM providers, custom model configurations

Here's how to integrate a custom model (e.g., Qwen from Alibaba Cloud):

```python
from mcp_benchmark_sdk import Agent, AgentResponse
from mcp_benchmark_sdk.parsers import OpenAIResponseParser, ResponseParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
import os

class QwenAgent(Agent):
    """Custom agent for Alibaba Cloud Qwen models."""
    
    def __init__(
        self,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        max_output_tokens: int | None = None,
        tool_call_limit: int = 1000,
        system_prompt: str | None = None,
    ):
        super().__init__(system_prompt=system_prompt, tool_call_limit=tool_call_limit)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    def _build_llm(self):
        """Build LLM client (called during initialization)."""
        config = {
            "model": self.model,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        if self.max_output_tokens:
            config["max_completion_tokens"] = self.max_output_tokens
        
        llm = ChatOpenAI(**config)
        return llm.bind_tools(self._tools) if self._tools else llm
    
    async def get_response(self, messages: list[BaseMessage]) -> tuple[AgentResponse, AIMessage]:
        """Get model response."""
        ai_message = await self._llm.ainvoke(messages)
        
        parser = self.get_response_parser()
        parsed = parser.parse(ai_message)
        
        return AgentResponse(
            content=parsed.content,
            tool_calls=parsed.tool_calls,
            done=not bool(parsed.tool_calls),
        ), ai_message
    
    def get_response_parser(self) -> ResponseParser:
        """Get response parser."""
        return OpenAIResponseParser()

# Use custom agent
agent = QwenAgent(model="qwen-plus", temperature=0.0)
result = await agent.run(task)
```

**Using Custom Agents with Harness:**

```python
def custom_agent_factory(model: str, **kwargs):
    """Factory function for creating custom agents."""
    if model.startswith("qwen"):
        return QwenAgent(model=model, **kwargs)
    else:
        # Fall back to built-in agents
        return create_agent(model, **kwargs)

# Use with harness
results = await harness.run(
    models=["qwen-plus", "gpt-4o"],
    agent_factory=custom_agent_factory,
)
```

---

### 4. Create Custom Verifiers

**Best for:** Complex validation logic, custom result checks

```python
from mcp_benchmark_sdk import Verifier, VerifierResult
import httpx

class APIResponseVerifier(Verifier):
    """Verify that an API endpoint returns expected data."""
    
    def __init__(self, endpoint: str, expected_field: str, expected_value: any, name: str | None = None):
        super().__init__(name or "APIResponseVerifier")
        self.endpoint = endpoint
        self.expected_field = expected_field
        self.expected_value = expected_value
    
    async def verify(self) -> VerifierResult:
        """Execute verification."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.endpoint)
                response.raise_for_status()
                data = response.json()
                
                actual_value = data.get(self.expected_field)
                success = actual_value == self.expected_value
                
                return VerifierResult(
                    name=self.name,
                    success=success,
                    expected_value=self.expected_value,
                    actual_value=actual_value,
                    comparison_type="equals",
                    error=None if success else "Value mismatch",
                )
        except Exception as exc:
            return VerifierResult(
                name=self.name,
                success=False,
                expected_value=self.expected_value,
                actual_value=None,
                comparison_type="equals",
                error=str(exc),
            )

# Use custom verifier
verifier = APIResponseVerifier(
    endpoint="http://localhost:8015/api/issues/DEMO-123",
    expected_field="status",
    expected_value="Open"
)
result = await verifier.verify()
```

**Using Custom Verifiers with Harness:**

For custom verifiers, you'll need to extend the harness loader to recognize your custom verifier types. The built-in loader supports `database_state` verifiers out of the box.

---

## Harness File Format

A harness file is JSON with this structure:

```json
{
  "system_prompt": "Optional system prompt for all scenarios",
  "scenarios": [
    {
      "scenario_id": "unique_id",
      "name": "Human-readable name",
      "description": "What this scenario tests",
      "prompts": [
        {
          "prompt_text": "The task for the agent",
          "expected_tools": ["tool1", "tool2"],
          "verifier": {
            "verifier_type": "database_state",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM table WHERE condition",
              "expected_value": 1,
              "comparison_type": "equals"
            },
            "name": "Optional verifier name"
          }
        }
      ],
      "metadata": {
        "difficulty": "easy|medium|hard",
        "tags": ["category1", "category2"]
      },
      "conversation_mode": false
    }
  ]
}
```

**Fields:**

- `scenario_id`: Unique identifier
- `name`: Human-readable name
- `description`: What the scenario tests
- `prompts`: Array of prompts (use multiple for multi-turn conversations)
- `verifier`: Can be a single object or array of verifiers
- `conversation_mode`: If `true`, concatenates all prompts; if `false`, uses only the first
- `metadata`: Arbitrary metadata for analysis

**Verifier Types:**

Currently supported:
- `database_state`: Run SQL queries to verify database state

**Comparison Types:**

- `equals`, `eq`, `==`: Exact equality
- `greater_than`, `gt`, `>`: Greater than
- `less_than`, `lt`, `<`: Less than
- `greater_than_equal`, `gte`, `>=`: Greater than or equal
- `less_than_equal`, `lte`, `<=`: Less than or equal

---

## Built-in Agents

### ClaudeAgent

```python
from mcp_benchmark_sdk import ClaudeAgent

agent = ClaudeAgent(
    model="claude-sonnet-4-5",
    temperature=0.1,
    max_output_tokens=4096,
    tool_call_limit=1000,
    system_prompt="Optional system prompt",
)
```

**Requires:** `ANTHROPIC_API_KEY` environment variable

**Supported models:** `claude-3-5-sonnet-20241022`, `claude-sonnet-4-5`, `claude-opus-4-20250514`, etc.

### GPTAgent

```python
from mcp_benchmark_sdk import GPTAgent

agent = GPTAgent(
    model="gpt-4o",
    temperature=0.1,
    max_output_tokens=4096,
    tool_call_limit=1000,
    system_prompt="Optional system prompt",
)
```

**Requires:** `OPENAI_API_KEY` environment variable

**Supported models:** `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, etc.

### GeminiAgent

```python
from mcp_benchmark_sdk import GeminiAgent

agent = GeminiAgent(
    model="gemini-2.0-flash-exp",
    temperature=0.1,
    max_output_tokens=4096,
    tool_call_limit=1000,
    system_prompt="Optional system prompt",
)
```

**Requires:** `GOOGLE_API_KEY` environment variable

**Supported models:** `gemini-2.0-flash-exp`, `gemini-1.5-pro`, `gemini-1.5-flash`, etc.

### GrokAgent

```python
from mcp_benchmark_sdk import GrokAgent

agent = GrokAgent(
    model="grok-beta",
    temperature=0.1,
    max_output_tokens=4096,
    tool_call_limit=1000,
    system_prompt="Optional system prompt",
)
```

**Requires:** `XAI_API_KEY` environment variable

**Supported models:** `grok-beta`, `grok-vision-beta`, etc.

---

## Verifiers

### DatabaseVerifier

Runs SQL queries against the MCP server's database.

```python
from mcp_benchmark_sdk import DatabaseVerifier

verifier = DatabaseVerifier(
    query="SELECT COUNT(*) FROM issue WHERE status = 'Open'",
    expected_value=5,
    mcp_url="http://localhost:8015/mcp",
    database_id="db123",  # From task execution
    comparison="greater_than_equal",
    name="Check open issues count"
)

result = await verifier.verify()
print(f"Success: {result.success}")
print(f"Expected: {result.expected_value}, Got: {result.actual_value}")
```

**Notes:**
- Queries must return exactly **one column** (use aggregates like `COUNT(*)`)
- The SQL runner endpoint is auto-derived from MCP URL
- Uses `x-database-id` header for isolation

---

## API Reference

### TestHarness

```python
harness = TestHarness(
    harness_path: Path,           # Path to JSON file or directory
    config: TestHarnessConfig,    # Configuration
)

# Add observers
harness.add_observer_factory(factory: Callable[[], RunObserver])

# Run benchmarks
results: list[RunResult] = await harness.run(
    models: list[str],                           # List of model identifiers
    agent_factory: Callable[..., Agent],         # Factory to create agents
    observer_config: dict[str, Any] | None = None,  # Optional observer config
)
```

### TestHarnessConfig

```python
config = TestHarnessConfig(
    mcp: MCPConfig,                  # MCP server configuration
    max_steps: int = 1000,           # Max agent turns per scenario
    tool_call_limit: int = 1000,     # Max tool calls per scenario
    temperature: float = 0.1,        # LLM temperature
    max_output_tokens: int | None = None,  # Max output tokens
    max_concurrent_runs: int = 20,   # Max parallel executions
    runs_per_scenario: int = 1,      # Repetitions per scenario
)
```

### Agent.run()

```python
result: Result = await agent.run(
    task: Task,                      # Task to execute
    max_steps: int = 1000,           # Max agent turns
    run_context: RunContext | None = None,  # Optional runtime context
)
```

### Task

```python
task = Task(
    prompt: str,                     # User prompt
    mcp: MCPConfig,                  # MCP server config
    max_steps: int = 1000,           # Max steps
    metadata: dict[str, Any] = {},   # Arbitrary metadata
    database_id: str | None = None,  # Database isolation ID
    conversation_mode: bool = False, # Multi-turn conversation
)
```

### Result

```python
result.success: bool                 # Whether task succeeded
result.error: str | None             # Error message if failed
result.messages: list[BaseMessage]   # Raw LangChain messages
result.metadata: dict[str, Any]      # Metadata (steps, etc.)
result.database_id: str              # Database ID used
result.reasoning_traces: list[str]   # Reasoning from model (if available)

# Get clean conversation history
conversation = result.get_conversation_history()
# Returns: [{"type": "message", "role": "user", "content": "..."}, ...]
```

### RunResult

```python
result.model: str                    # Model identifier
result.scenario_id: str              # Scenario ID
result.scenario_name: str            # Scenario name
result.run_number: int               # Run number (for multiple runs)
result.success: bool                 # Overall success (agent + verifiers)
result.error: str | None             # Error message
result.result: Result                # Agent result
result.verifier_results: list[VerifierResult]  # Verifier results

# Get conversation history
conversation = result.get_conversation_history()

# Export to dict (for JSON serialization)
data = result.to_dict()
```

---

## Examples

See the `examples/` directory for complete examples:

- **`simple_harness_example.py`** - Basic harness usage with console observer
- **`sdk_usage_showcase.ipynb`** - Interactive notebook with all patterns
- **`save_artifacts_example.py`** - How to export results to JSON

---

## Advanced Features

### LangSmith Integration

The SDK supports automatic LangSmith tracing:

```python
from mcp_benchmark_sdk import create_traced_agent, configure_langsmith

# Configure LangSmith (optional, uses env vars by default)
configure_langsmith(
    api_key="your-api-key",
    project="my-benchmarks",
    enabled=True,
)

# Create agent with tracing
agent = create_traced_agent("gpt-4o")

# All runs are automatically traced
result = await agent.run(task)

# Get trace URL
from mcp_benchmark_sdk import get_trace_url
print(f"View trace: {get_trace_url()}")
```

**Environment Variables:**
- `LANGCHAIN_API_KEY`: LangSmith API key
- `LANGCHAIN_PROJECT`: Project name
- `LANGCHAIN_TRACING_V2=true`: Enable tracing

### Custom System Prompts

You can override the default system prompt per harness file:

```json
{
  "system_prompt": "You are a project management assistant. You must complete tasks efficiently.",
  "scenarios": [...]
}
```

Or per agent:

```python
agent = ClaudeAgent(
    model="claude-sonnet-4-5",
    system_prompt="Custom instructions for this agent",
)
```

### Multi-turn Conversations

Set `conversation_mode: true` in your scenario to enable multi-turn:

```json
{
  "scenario_id": "multi_turn",
  "conversation_mode": true,
  "prompts": [
    {"prompt_text": "Create a bug issue in DEMO"},
    {"prompt_text": "Now assign it to John Smith"},
    {"prompt_text": "Add a comment: 'This is urgent'"}
  ]
}
```

The agent will process all prompts in sequence within a single conversation.

---

## Contributing

Contributions welcome! Please see `CONTRIBUTING.md` for guidelines.

---

## License

MIT License - see `LICENSE` file for details.

---

## Support

- **Documentation:** See `examples/sdk_usage_showcase.ipynb` for interactive examples
- **Issues:** https://github.com/your-repo/issues
- **Discussions:** https://github.com/your-repo/discussions
