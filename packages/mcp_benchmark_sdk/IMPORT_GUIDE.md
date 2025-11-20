# MCP Benchmark SDK - Import Guide

This guide explains the SDK's module structure and import patterns.

## Two Import Styles

The SDK supports **two import styles** - choose based on your needs:

### 1. Top-Level Imports (Convenience) ✅ Recommended for Quick Scripts

```python
from mcp_benchmark_sdk import (
    ClaudeAgent,
    GPTAgent,
    Task,
    Result,
    MCPConfig,
    TestHarness,
    TestHarnessConfig,
    RunContext,
    RunObserver,
)
```

**Best for:**
- Quick scripts and prototypes
- When you want minimal boilerplate
- When you don't care about internal structure

### 2. Explicit Module Imports (Educational) ✅ Recommended for Learning

```python
# Agent implementations
from mcp_benchmark_sdk.agents.core import ClaudeAgent, GPTAgent, GeminiAgent, GrokAgent

# Task and Result structures
from mcp_benchmark_sdk.agents.tasks import Task, Result, AgentResponse

# MCP configuration
from mcp_benchmark_sdk.agents.mcp import MCPConfig, MCPClientManager

# Runtime components
from mcp_benchmark_sdk.agents.runtime import RunContext, RunObserver

# Test harness
from mcp_benchmark_sdk.harness import TestHarness, TestHarnessConfig, create_agent

# Verifiers
from mcp_benchmark_sdk.harness.verifiers import Verifier, DatabaseVerifier, VerifierResult
```

**Best for:**
- Understanding the SDK architecture
- Learning where components come from
- Teaching and documentation
- IDE autocomplete hints

## SDK Module Structure

```
mcp_benchmark_sdk/
├── agents/                    # Agent system
│   ├── core/                  # Agent implementations
│   │   ├── base.py           # → Agent (base class)
│   │   ├── claude.py         # → ClaudeAgent
│   │   ├── gpt.py            # → GPTAgent
│   │   ├── gemini.py         # → GeminiAgent
│   │   └── grok.py           # → GrokAgent
│   │
│   ├── tasks/                 # Task & Result structures
│   │   ├── task.py           # → Task
│   │   ├── result.py         # → Result, AgentResponse
│   │   └── __init__.py       # → ToolCall, ToolResult
│   │
│   ├── mcp/                   # MCP integration
│   │   ├── config.py         # → MCPConfig
│   │   ├── loader.py         # → MCPClientManager
│   │   └── tool_fixer.py     # → Tool schema fixing
│   │
│   ├── runtime/               # Execution context
│   │   ├── context.py        # → RunContext
│   │   └── events.py         # → RunObserver
│   │
│   ├── parsers/               # Response parsers
│   │   ├── base.py           # → ResponseParser, ParsedResponse
│   │   ├── anthropic.py      # → AnthropicResponseParser
│   │   ├── openai.py         # → OpenAIResponseParser
│   │   ├── google.py         # → GoogleResponseParser
│   │   └── xai.py            # → XAIResponseParser
│   │
│   ├── telemetry.py           # Tracing integration
│   │                         # → configure_langfuse, configure_langsmith, with_tracing
│   │
│   └── utils/                 # Utilities
│       ├── retry.py          # → retry_with_backoff
│       └── mcp.py            # → derive_sql_runner_url
│
└── harness/                   # Test harness system
    ├── orchestrator.py        # → TestHarness, TestHarnessConfig, RunResult
    ├── loader.py              # → load_harness_file, scenario_to_task
    ├── agent_factory.py       # → create_agent, create_traced_agent
    ├── scenario.py            # → Scenario, ScenarioPrompt, VerifierDefinition
    │
    └── verifiers/             # Verification system
        ├── base.py            # → Verifier, VerifierResult
        └── database.py        # → DatabaseVerifier
```

## Import Examples by Use Case

### Use Case 1: Run a Simple Agent Task

**Top-level (quick):**
```python
from mcp_benchmark_sdk import ClaudeAgent, Task, MCPConfig

agent = ClaudeAgent()
task = Task(prompt="...", mcp=MCPConfig(name="jira", url="..."))
result = await agent.run(task)
```

**Explicit (educational):**
```python
from mcp_benchmark_sdk.agents.core import ClaudeAgent
from mcp_benchmark_sdk.agents.tasks import Task
from mcp_benchmark_sdk.agents.mcp import MCPConfig

agent = ClaudeAgent()
task = Task(prompt="...", mcp=MCPConfig(name="jira", url="..."))
result = await agent.run(task)
```

### Use Case 2: Run Test Harness

**Top-level (quick):**
```python
from mcp_benchmark_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

harness = TestHarness(path, config=TestHarnessConfig(mcp=MCPConfig(...)))
results = await harness.run(models=["gpt-5-high"], agent_factory=create_agent)
```

**Explicit (educational):**
```python
from mcp_benchmark_sdk.harness import TestHarness, TestHarnessConfig, create_agent
from mcp_benchmark_sdk.agents.mcp import MCPConfig

harness = TestHarness(path, config=TestHarnessConfig(mcp=MCPConfig(...)))
results = await harness.run(models=["gpt-5-high"], agent_factory=create_agent)
```

### Use Case 3: Custom Agent with Observers

**Top-level (quick):**
```python
from mcp_benchmark_sdk import Agent, AgentResponse, RunContext, RunObserver

class MyAgent(Agent):
    async def get_response(self, messages):
        return AgentResponse(...)

class MyObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"{role}: {content}")

async with RunContext() as ctx:
    ctx.add_observer(MyObserver())
    result = await agent.run(task, run_context=ctx)
```

**Explicit (educational):**
```python
from mcp_benchmark_sdk.agents.core.base import Agent
from mcp_benchmark_sdk.agents.tasks import AgentResponse
from mcp_benchmark_sdk.agents.runtime import RunContext, RunObserver

class MyAgent(Agent):
    async def get_response(self, messages):
        return AgentResponse(...)

class MyObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"{role}: {content}")

async with RunContext() as ctx:
    ctx.add_observer(MyObserver())
    result = await agent.run(task, run_context=ctx)
```

### Use Case 4: Custom Verifier

**Top-level (quick):**
```python
from mcp_benchmark_sdk import Verifier, VerifierResult

class MyVerifier(Verifier):
    async def verify(self) -> VerifierResult:
        return VerifierResult(name="check", success=True, ...)
```

**Explicit (educational):**
```python
from mcp_benchmark_sdk.harness.verifiers import Verifier, VerifierResult

class MyVerifier(Verifier):
    async def verify(self) -> VerifierResult:
        return VerifierResult(name="check", success=True, ...)
```

## Why Two Styles?

### Top-Level Imports: For Productivity
- **Less typing** - shorter import statements
- **Familiar pattern** - like `from flask import Flask`
- **Auto-complete friendly** - IDEs suggest from one namespace
- **Quick prototyping** - less boilerplate

### Explicit Module Imports: For Understanding
- **Clear separation** - see where each component lives
- **Educational** - learn the architecture
- **Self-documenting** - imports tell a story
- **Debugging** - easier to trace through modules

## When to Use Which?

| Scenario | Recommended Style | Reason |
|----------|------------------|--------|
| Production code | Top-level | Cleaner, more maintainable |
| Tutorials/docs | Explicit | Shows architecture |
| Learning SDK | Explicit | Understand structure |
| Quick scripts | Top-level | Less boilerplate |
| Teaching | Explicit | Educational |
| Notebooks | Explicit | Shows what's where |

## Complete Import Reference

### Top-Level Exports (`from mcp_benchmark_sdk import ...`)

**Agents:**
- `Agent` - Base agent class
- `ClaudeAgent` - Anthropic Claude
- `GPTAgent` - OpenAI GPT
- `GeminiAgent` - Google Gemini
- `GrokAgent` - xAI Grok

**Tasks & Results:**
- `Task` - Task definition
- `Result` - Execution result
- `AgentResponse` - Agent response structure
- `ToolCall` - Tool call structure
- `ToolResult` - Tool execution result

**MCP:**
- `MCPConfig` - MCP server configuration
- `MCPClientManager` - MCP connection manager

**Runtime:**
- `RunContext` - Execution context
- `RunObserver` - Observer interface

**Harness:**
- `TestHarness` - Main orchestrator
- `TestHarnessConfig` - Harness configuration
- `RunResult` - Test run result
- `create_agent` - Agent factory
- `create_traced_agent` - Traced agent factory

**Scenarios:**
- `Scenario` - Scenario definition
- `ScenarioPrompt` - Prompt specification
- `VerifierDefinition` - Verifier config

**Verifiers:**
- `Verifier` - Base verifier
- `DatabaseVerifier` - SQL verifier
- `VerifierResult` - Verification result

**Parsers:**
- `ResponseParser` - Base parser interface
- `ParsedResponse` - Parsed response structure

**Telemetry:**
- `configure_langsmith` - Configure tracing
- `with_tracing` - Wrap with tracing
- `TracingAgent` - Tracing wrapper
- `is_tracing_enabled` - Check tracing status
- `get_trace_url` - Get LangSmith URL

**Utils:**
- `derive_sql_runner_url` - Derive SQL endpoint

**Constants:**
- All default values (DEFAULT_TOOL_CALL_LIMIT, etc.)

### Module-Level Exports

See the module structure diagram above for the complete mapping.

## IDE Configuration

For best autocomplete:

**PyCharm / IntelliJ:**
- Mark `src/` as Sources Root
- Enable type checking

**VSCode:**
```json
{
  "python.analysis.extraPaths": [
    "${workspaceFolder}/packages/mcp_benchmark_sdk/src"
  ],
  "python.analysis.typeCheckingMode": "basic"
}
```

## Migration Guide

If you want to switch from one style to the other:

**Top-level → Explicit:**
```bash
# Replace imports
sed -i 's/from mcp_benchmark_sdk import ClaudeAgent/from mcp_benchmark_sdk.agents.core import ClaudeAgent/g' *.py
```

**Explicit → Top-level:**
```bash
# Replace imports (reverse)
sed -i 's/from mcp_benchmark_sdk.agents.core import \(.*\)/from mcp_benchmark_sdk import \1/g' *.py
```

## Best Practices

1. **Be consistent** - Pick one style per file/project
2. **Document why** - Comment which style and why
3. **Group imports** - Organize by module when using explicit style
4. **Use __all__** - In your own modules, export clearly
5. **Type hints** - Add types regardless of import style

---

**See Also:**
- [SDK README](README.md) - Complete documentation
- [Quick Start](QUICKSTART.md) - Get started fast
- [Examples](examples/) - Working code samples














