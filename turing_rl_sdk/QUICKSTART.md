# MCP Benchmark SDK - Quick Start

**The SDK is harness-first.** The harness orchestrates everything - you define scenarios and let it run.

## 30-Second Start

```python
from pathlib import Path
from turing_rl_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

# Configure MCP
mcp_config = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http"
)

# Create harness from JSON file
harness = TestHarness(
    harness_path=Path("task.json"),
    config=TestHarnessConfig(mcp=mcp_config)
)

# Run benchmarks
results = await harness.run(
    models=["gpt-4o"],
    agent_factory=create_agent,
)

# Check results
for r in results:
    print(f"{r.scenario_id}: {'✓ PASS' if r.success else '✗ FAIL'}")
```

**That's it!** The harness handles agent creation, MCP connections, task execution, verification, and metrics.

## What You Need

1. **A harness file** (`task.json`):

```json
{
  "scenarios": [
    {
      "scenario_id": "create_bug",
      "name": "Create Bug Issue",
      "prompts": [
        {
          "prompt_text": "Create a bug in DEMO project with summary 'Login broken'",
          "expected_tools": ["create_issue"],
          "verifier": {
            "verifier_type": "database_state",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Login broken'",
              "expected_value": 1,
              "comparison_type": "equals"
            }
          }
        }
      ],
      "conversation_mode": false
    }
  ]
}
```

2. **API keys** for LLM providers (OpenAI, Anthropic, Google, etc.)

3. **Running MCP server** (e.g., Jira MCP server on localhost:8015)

## Model Names

**Important:** Model names must match the exact identifiers expected by each provider. The SDK auto-detects the provider from the model name prefix.

**Supported Providers:**

- **OpenAI** (`gpt-4o`, `gpt-4o-mini`, `gpt-5`, `o1`, `o3-mini`) - [Docs](https://platform.openai.com/docs/models)
- **Anthropic** (`claude-sonnet-4-5`, `claude-opus-4`, `claude-haiku-3-5`) - [Docs](https://docs.anthropic.com/en/docs/about-claude/models)
- **Google** (`gemini-2.0-flash-exp`, `gemini-2.5-pro`, `gemini-1.5-pro`) - [Docs](https://ai.google.dev/gemini-api/docs/models/gemini)
- **xAI** (`grok-2-latest`, `grok-vision-beta`) - [Docs](https://docs.x.ai/docs/models)

Use explicit prefix (`openai:`, `anthropic:`, `google:`, `xai:`) if auto-detection fails:

```python
models=["openai:gpt-4o", "anthropic:claude-sonnet-4-5"]
```

## Usage Patterns

### Pattern 1: Simplest Harness (⭐ Start here!)

```python
harness = TestHarness(
    harness_path=Path("task.json"),
    config=TestHarnessConfig(mcp=mcp_config)
)

results = await harness.run(
    models=["gpt-4o"],
    agent_factory=create_agent,
)
```

### Pattern 2: Direct Agent Usage

```python
from turing_rl_sdk import ClaudeAgent, Task

agent = ClaudeAgent(model="claude-sonnet-4-5")
task = Task(prompt="Create a bug in DEMO", mcp=mcp_config)
result = await agent.run(task)
```

### Pattern 3: Custom Agent

```python
from turing_rl_sdk import Agent, AgentResponse

class MyAgent(Agent):
    def _build_llm(self):
        # Build your LLM client
        pass
    
    async def get_response(self, messages):
        # Get model response
        pass
    
    def get_response_parser(self):
        # Return parser
        pass

# Use with harness
def my_factory(model, **kwargs):
    if model == "my-model":
        return MyAgent(**kwargs)
    return create_agent(model, **kwargs)

results = await harness.run(models=["my-model"], agent_factory=my_factory)
```

### Pattern 4: Custom Verifier

```python
from turing_rl_sdk import Verifier, VerifierResult

class MyVerifier(Verifier):
    async def verify(self) -> VerifierResult:
        # Check your conditions
        return VerifierResult(
            name=self.name,
            success=True,
            expected_value=None,
            actual_value=None,
            comparison_type="custom"
        )

# Use directly
verifier = MyVerifier()
result = await verifier.verify()
```

### Pattern 5: Add Observers

```python
from turing_rl_sdk import RunObserver

class MyObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"{role}: {content[:50]}")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        print(f"Tool: {tool_name} ({'✗' if is_error else '✓'})")

harness.add_observer_factory(lambda: MyObserver())
```

### Pattern 6: Compare Models

```python
results = await harness.run(
    models=["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash-exp"],
    agent_factory=create_agent,
)

# Aggregate and compare
for model in ["gpt-4o", "claude-sonnet-4-5", "gemini-2.0-flash-exp"]:
    model_results = [r for r in results if r.model == model]
    pass_rate = sum(r.success for r in model_results) / len(model_results) * 100
    print(f"{model}: {pass_rate:.1f}% pass rate")
```

## Key Concepts

- **TestHarness**: Orchestrates everything (agent creation, MCP, verification)
- **Scenarios**: Test definitions loaded from JSON
- **Agents**: LLM wrappers (built-in: Claude, GPT, Gemini, Grok)
- **Verifiers**: Result validation (built-in: database queries)
- **Observers**: Real-time progress tracking

## Documentation

- **README.md** - Complete API reference with all details
- **SDK_USAGE_GUIDE.ipynb** - Interactive notebook with all patterns
- **simple_harness_example.py** - Basic example script
- **9_tasks/** - Real benchmark scenarios

## Next Steps

1. Start with **Pattern 1** (simplest harness)
2. Create your own harness files
3. Add custom agents for your LLM providers
4. Build custom verifiers for your validation logic
5. Run large-scale benchmarks!

---

**Remember:** The SDK is **harness-first**. Start with the harness, not the agents.

Happy benchmarking! 🚀

