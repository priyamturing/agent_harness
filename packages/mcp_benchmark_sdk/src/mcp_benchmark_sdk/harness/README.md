# Test Harness

The test harness provides a simple, HUD-like experience for running benchmarks against agents.

## Quick Start

```python
from pathlib import Path
from mcp_benchmark_sdk import (
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
        mcps=[mcp_config],
        sql_runner_url="http://localhost:8015/api/sql-runner",
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

## Using with Custom Observers

You can attach observers to monitor progress:

```python
from mcp_benchmark_sdk import RunObserver

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
from mcp_benchmark_sdk import Agent, create_agent

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
    mcps=[mcp_config],
    sql_runner_url="http://localhost:8015/api/sql-runner",
    max_steps=1000,              # Max agent steps per run
    tool_call_limit=1000,        # Max tool calls per run
    temperature=0.1,             # Model temperature
    max_output_tokens=4096,      # Max output tokens
    max_concurrent_runs=20,      # Max parallel runs
    runs_per_scenario=1,         # Runs per scenario
)
```
