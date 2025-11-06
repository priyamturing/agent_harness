# Conversation History Support in Test Harness

## Overview

The SDK's `RunResult` now provides full access to conversation history, allowing you to export complete artifacts with all messages, tool calls, verifier results, and reasoning traces.

## Features

### 1. Get Conversation History

Extract the full conversation as a list of dictionaries:

```python
results = await harness.run(models=["gpt-4o"], agent_factory=create_agent)

for result in results:
    conversation = result.get_conversation_history()
    
    for msg in conversation:
        print(f"{msg['role']}: {msg['content']}")
        
        # Access tool calls if present
        if 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                print(f"  → Tool: {tc['name']}")
                print(f"    Args: {tc['args']}")
```

### 2. Export Complete Artifacts

Use `to_dict()` to get a complete, JSON-serializable representation:

```python
import json

for result in results:
    result_dict = result.to_dict()
    
    # Save to file
    with open(f"result_{result.model}_{result.scenario_id}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
```

### 3. Artifact Contents

The `to_dict()` method includes:

```python
{
    "model": "gpt-4o",
    "scenario_id": "create_issue",
    "scenario_name": "Create JIRA Issue",
    "run_number": 1,
    "success": true,
    "error": null,
    "metadata": {...},
    
    # Full conversation history
    "conversation": [
        {
            "role": "system",
            "content": "You are an autonomous agent..."
        },
        {
            "role": "human",
            "content": "Create a new issue..."
        },
        {
            "role": "ai",
            "content": "I'll create the issue...",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "create_issue",
                    "args": {"title": "...", "description": "..."}
                }
            ]
        },
        {
            "role": "tool",
            "content": "{\"id\": \"PROJ-123\", ...}",
            "tool_call_id": "call_123",
            "name": "create_issue"
        }
    ],
    
    # Verifier results
    "verifier_results": [
        {
            "name": "issue_created",
            "success": true,
            "message": "Verification passed",
            "expected_value": 1,
            "actual_value": 1,
            "comparison": "equals",
            "error": null
        }
    ],
    
    # Reasoning traces
    "reasoning_traces": [
        "I need to create an issue with the given title...",
        "The issue was successfully created..."
    ],
    
    # Execution metadata
    "steps": 3,
    "database_id": "abc-123-def"
}
```

## Message Structure

Each message in the conversation includes:

**Common fields:**
- `role`: Message type (system, human, ai, tool)
- `content`: Message content (string or structured)

**AI Messages:**
- `tool_calls`: List of tool invocations
  - `id`: Tool call identifier
  - `name`: Tool name
  - `args`: Tool arguments dict

**Tool Messages:**
- `tool_call_id`: Links to the tool call
- `name`: Tool name

**Additional fields:**
- `additional_kwargs`: Extra metadata from LLM providers
- Custom fields as needed

## Examples

### Save All Results

```python
from pathlib import Path
import json

artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

for result in results:
    filename = f"{result.model}_{result.scenario_id}_run{result.run_number}.json"
    filepath = artifacts_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    
    print(f"Saved {filename}")
```

### Analyze Conversation

```python
for result in results:
    conversation = result.get_conversation_history()
    
    # Count tool calls
    tool_calls = sum(
        len(msg.get('tool_calls', [])) 
        for msg in conversation 
        if msg['role'] == 'ai'
    )
    
    # Find errors
    errors = [
        msg for msg in conversation 
        if msg['role'] == 'tool' and 'error' in msg.get('content', '').lower()
    ]
    
    print(f"{result.scenario_id}:")
    print(f"  Total messages: {len(conversation)}")
    print(f"  Tool calls: {tool_calls}")
    print(f"  Errors: {len(errors)}")
```

### Extract Tool Usage Stats

```python
from collections import Counter

tool_usage = Counter()

for result in results:
    conversation = result.get_conversation_history()
    
    for msg in conversation:
        if msg['role'] == 'ai' and 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                tool_usage[tc['name']] += 1

print("Tool usage:")
for tool, count in tool_usage.most_common():
    print(f"  {tool}: {count}")
```

## Complete Example

See `packages/mcp_benchmark_sdk/examples/save_artifacts_example.py` for a full working example.

## CLI Integration

The CLI automatically benefits from this feature - the SDK's `RunResult` is used internally, so conversation history is preserved in CLI artifacts as well.

## Benefits

1. **Complete Audit Trail**: Full conversation history for debugging and analysis
2. **Reproducibility**: All data needed to understand what happened in each run
3. **Tool Usage Analysis**: Track which tools were called and with what arguments
4. **Error Investigation**: See exact error messages and context
5. **Reasoning Traces**: Access model's thinking process (when available)
6. **Verifier Context**: Understand why verifiers passed or failed
7. **JSON Serializable**: Easy to export, store, and process

## Notes

- Conversation history includes ALL messages (system, human, AI, tool)
- Tool calls and results are linked via `tool_call_id`
- Content is preserved as-is (strings, structured data, etc.)
- Additional LLM provider metadata is included when available
- Reasoning traces are captured separately (when supported by the model)
