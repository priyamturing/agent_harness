# Test Harness

The test harness provides a simple, HUD-like experience for running benchmarks against agents.

## Quick Start

```python
from pathlib import Path
from turing_rl_sdk import (
    TestHarness,
    TestHarnessConfig,
    MCPConfig,
    create_agent,
)

# Configure your MCP server
mcp_config = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http",
)

# Create harness with your benchmark file
harness = TestHarness(
    harness_path=Path("benchmarks/task1.json"),
    config=TestHarnessConfig(
        mcp=mcp_config,
        max_steps=1000,
        tool_call_limit=1000,
        runs_per_scenario=3,  # Run each scenario 3 times
    )
)

# Run benchmarks
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.5-pro"],
    agent_factory=create_agent,
)

# Analyze results
for result in results:
    print(f"{result.model} - {result.scenario_id}: {'✓' if result.success else '✗'}")
    
    # Access conversation history
    conversation = result.get_conversation_history()
    print(f"  Total messages: {len(conversation)}")
    
    # Export to dict for saving
    result_dict = result.to_dict()
    # result_dict contains: conversation, verifier_results, reasoning_traces, etc.
```

## Model Names

**Important:** Model names must match the exact identifiers expected by each LLM provider. The SDK supports:

- **OpenAI** - `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o1`, `o3-mini` ([Docs](https://platform.openai.com/docs/models))
- **Anthropic** - `claude-sonnet-4-5`, `claude-opus-4`, `claude-haiku-3-5` ([Docs](https://docs.anthropic.com/en/docs/about-claude/models))
- **Google** - `gemini-2.0-flash-exp`, `gemini-2.5-pro`, `gemini-1.5-pro` ([Docs](https://ai.google.dev/gemini-api/docs/models/gemini))
- **xAI** - `grok-2-latest`, `grok-vision-beta` ([Docs](https://docs.x.ai/docs/models))

The SDK auto-detects providers from model name prefixes. Use explicit prefixes if needed:

```python
results = await harness.run(
    models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"],
    agent_factory=create_agent,
)
```

## Using with Custom Observers

You can attach observers to monitor progress:

```python
from turing_rl_sdk import RunObserver

class MyProgressObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"[{role}] {content[:100]}...")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        print(f"Tool: {tool_name} {'✗' if is_error else '✓'}")
    
    async def on_status(self, message, level="info"):
        print(f"[{level}] {message}")

# Add observer factory
harness.add_observer_factory(lambda: MyProgressObserver())

# Now run - each task will get its own observer instance
results = await harness.run(
    models=["gpt-4o"],
    agent_factory=create_agent,
)
```

## Loading from Directory

```python
# Load all JSON harness files from a directory
harness = TestHarness(
    harness_path=Path("benchmarks/"),  # Directory with multiple .json files
    config=config,
)

# All scenarios from all files will be run
results = await harness.run(models=["gpt-4o"], agent_factory=create_agent)
```

## Custom Agent Factory

You can provide your own agent factory to support custom models:

```python
from turing_rl_sdk import Agent, create_agent

def my_agent_factory(model: str, **kwargs) -> Agent:
    if model.startswith("my-custom-model"):
        return MyCustomAgent(model=model, **kwargs)
    
    # Fall back to SDK's create_agent for standard models
    return create_agent(model, **kwargs)

results = await harness.run(
    models=["my-custom-model-1", "gpt-4o"],
    agent_factory=my_agent_factory,
)
```

## Harness File Format

JSON harness files follow this structure:

```json
{
  "scenarios": [
    {
      "scenario_id": "create_issue",
      "name": "Create JIRA Issue",
      "description": "Test creating a new issue",
      "prompts": [
        {
          "prompt_text": "Create a new issue titled 'Test Issue'",
          "expected_tools": ["create_issue"],
          "verifier": {
            "verifier_type": "database_state",
            "name": "issue_created",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM issues WHERE title = 'Test Issue'",
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

### Database Verifier Comparison Types

The `database_state` verifier supports the following comparison types:

- **equals** (aliases: `eq`, `==`, `equal`) - Exact equality, works with None
- **not_equal** (aliases: `not_equals`, `neq`, `ne`, `!=`) - Inequality, works with None
- **greater_than** (aliases: `gt`, `>`) - Actual > expected
- **less_than** (aliases: `lt`, `<`) - Actual < expected
- **greater_than_equal** (aliases: `greater_than_or_equal`, `gte`, `>=`) - Actual >= expected
- **less_than_equal** (aliases: `less_than_or_equal`, `lte`, `<=`) - Actual <= expected

#### JSON Examples

```json
{
  "verifier_type": "database_state",
  "validation_config": {
    "query": "SELECT COUNT(*) FROM deleted_items",
    "expected_value": 0,
    "comparison_type": "not_equal"
  }
}
```

```json
{
  "verifier_type": "database_state",
  "validation_config": {
    "query": "SELECT priority FROM tasks WHERE id = 123",
    "expected_value": 5,
    "comparison_type": "greater_than"
  }
}
```

#### Python API with Type Safety

For Python code, you can use the `ComparisonType` enum for type-safe comparisons:

```python
from turing_rl_sdk import DatabaseVerifier, ComparisonType

verifier = DatabaseVerifier(
    query="SELECT COUNT(*) FROM users WHERE status = 'active'",
    expected_value=10,
    comparison=ComparisonType.GREATER_THAN,  # Type-safe enum
    mcp_url="http://localhost:8015/mcp",
    database_id="test-db-123",
)

result = await verifier.verify()
```

String comparisons (including aliases) are still fully supported:

```python
verifier = DatabaseVerifier(
    query="SELECT status FROM job WHERE id = 1",
    expected_value="failed",
    comparison="!=",  # Alias for not_equal
    mcp_url="http://localhost:8015/mcp",
    database_id="test-db-123",
)
```

## Accessing Conversation History

Each `RunResult` provides access to the full conversation history:

```python
results = await harness.run(models=["gpt-4o"], agent_factory=create_agent)

for result in results:
    # Get conversation as list of dicts
    conversation = result.get_conversation_history()
    for msg in conversation:
        print(f"{msg['role']}: {msg['content'][:100]}...")
        if 'tool_calls' in msg:
            print(f"  Tool calls: {[tc['name'] for tc in msg['tool_calls']]}")
    
    # Export full result to dict for saving
    import json
    result_dict = result.to_dict()
    with open(f"result_{result.model}_{result.scenario_id}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
```

The `to_dict()` method provides a complete serializable representation including:
- Full conversation history
- Verifier results
- Reasoning traces
- Metadata (steps, database_id, etc.)

## Advanced Configuration

```python
config = TestHarnessConfig(
    mcp=mcp_config,
    max_steps=1000,              # Max agent steps per run
    tool_call_limit=1000,        # Max tool calls per run
    temperature=0.1,             # Model temperature
    max_output_tokens=4096,      # Max output tokens
    max_concurrent_runs=20,      # Max parallel runs
    runs_per_scenario=1,         # Runs per scenario
)
```
