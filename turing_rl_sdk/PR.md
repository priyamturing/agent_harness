# Turing RL SDK - Initial Release

## Ticket

https://turingservices.atlassian.net/browse/RG-21?atlOrigin=eyJpIjoiNTYwNTQ5ODA1ODhhNDQ4ODllODIxODYzMGViN2U1ZWQiLCJwIjoiaiJ9

## Overview

This PR introduces the **Turing RL SDK**, a comprehensive Python framework for building and running LLM agent benchmarks against MCP (Model Context Protocol) servers. The SDK provides a **harness-first** design to systematically evaluate LLM agents across multiple models, scenarios, and verification criteria.

### Key Features

- ✅ **Harness-First Design** - TestHarness orchestrates everything from a single entry point
- ✅ **Multi-Model Support** - Built-in support for GPT, Claude, Gemini, and Grok with extensible architecture
- ✅ **Concurrent Execution** - Parallel test runs with database isolation
- ✅ **Observable Execution** - Real-time monitoring via observer pattern
- ✅ **SQL Verification** - Database state validation with multiple comparison operators
- ✅ **Conversation Export** - Complete conversation history with tool calls and results
- ✅ **LangSmith & Langfuse Integration** - Distributed tracing and observability
- ✅ **Custom Agents** - Easy integration of new LLM providers
- ✅ **Type-Safe** - Full type annotations with Pydantic validation

## Stats

- **46 files changed**
- **7,849 lines added**
- **New Package**: `turing-rl-sdk`
- **2 Packages Total**: SDK (core) 

---

## Changes

### Core SDK Structure

The SDK is organized into modular, loosely-coupled components:

```
turing_rl_sdk/
├── agents/                # LLM agent system
│   ├── core/             # Agent implementations (Claude, GPT, Gemini, Grok)
│   ├── tasks/            # Task definitions and result handling
│   ├── mcp/              # MCP server integration and configuration
│   ├── parsers/          # Response parsing for different providers
│   ├── runtime/          # Execution context and event handling
│   ├── utils/            # Retry logic and helper utilities
│   ├── telemetry.py      # LangSmith/Langfuse tracing integration
│   └── constants.py      # Global constants and defaults
│
└── harness/              # Test orchestration system
    ├── orchestrator.py   # TestHarness main class
    ├── loader.py         # Scenario loading from JSON
    ├── scenario.py       # Scenario data models
    ├── agent_factory.py  # Agent creation utilities
    └── verifiers/        # Result verification system
        ├── base.py       # Verifier interface
        └── database.py   # SQL-based verification
```

### Key Components

#### 1. Test Harness System

Orchestration engine that manages the complete benchmark lifecycle:

**Features:**
- Load test scenarios from JSON files or entire directories
- Create and manage agents for multiple LLM models
- Execute tasks with automatic database isolation (`x-database-id` header)
- Run verifiers to validate outcomes against expected results
- Collect comprehensive metrics and aggregate results
- Support concurrent execution with configurable parallelism (default: 20 concurrent runs)
- Real-time progress tracking via observers

**Usage:**

```python
from pathlib import Path
from turing_rl_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

harness = TestHarness(
    harness_path=Path("task.json"),
    config=TestHarnessConfig(
        mcp=MCPConfig(
            name="jira", 
            url="http://localhost:8015/mcp",
            transport="streamable_http"
        ),
        max_concurrent_runs=10,
        runs_per_scenario=3,
        max_steps=1000,
        tool_call_limit=1000,
    )
)

results = await harness.run(
    models=["gpt-5", "claude-sonnet-4-5", "gemini-2.5-pro"],
    agent_factory=create_agent
)

# Analyze results
for result in results:
    print(f"{result.model}: {'✓' if result.success else '✗'}")
    conversation = result.get_conversation_history()  # Full conversation export
```

#### 2. Multi-Model Agent Support

Built-in implementations for major LLM providers with reasoning capabilities:

| Provider | Agent Class | Default Model | Features |
|----------|-------------|---------------|----------|
| **OpenAI** | `GPTAgent` | `gpt-5` | Tool calling, reasoning support |
| **Anthropic** | `ClaudeAgent` | `claude-sonnet-4-5` | Extended thinking, 42k token budget |
| **Google** | `GeminiAgent` | `gemini-2.5-pro` | Thinking budget, thought inclusion |
| **xAI** | `GrokAgent` | `grok-4` | Reasoning support |

**Agent Features:**
- ✅ Unified interface via abstract `Agent` base class
- ✅ Automatic retry logic with exponential backoff (LangChain: 3 retries, SDK wrapper: 2 attempts)
- ✅ Tool calling with MCP servers via langchain-mcp-adapters
- ✅ Conversation history management (full message list)
- ✅ Response parsing for each provider's format
- ✅ Custom system prompt support (per agent or per scenario)
- ✅ Tool call limiting (default: 1000 calls)
- ✅ Reasoning/thinking trace extraction

**Usage:**

```python
from turing_rl_sdk import ClaudeAgent, GPTAgent, GeminiAgent, GrokAgent

# Use defaults (with reasoning enabled)
agent = ClaudeAgent()  # claude-sonnet-4-5 with thinking (temperature=1.0 required)

# Or customize
agent = GPTAgent(
    model="gpt-5",
    temperature=0.1,
    max_output_tokens=4096
)
```

#### 3. Verification System

**DatabaseVerifier** provides robust SQL-based validation:

**Features:**
- Execute SQL queries against MCP server's SQL runner endpoint
- Compare results with expected values using multiple operators
- Database isolation via unique database IDs
- Detailed error reporting with query context
- Automatic SQL runner URL derivation from MCP URL

**Comparison Types:**
- `equals`, `equal`, `eq`, `==` - Exact equality
- `greater_than`, `gt`, `>` - Numeric greater than
- `less_than`, `lt`, `<` - Numeric less than
- `greater_than_equal`, `greater_than_or_equal`, `greater_than_or_equal_to`, `greater_or_equal`, `gte`, `>=` - Greater than or equal
- `less_than_equal`, `less_than_or_equal`, `less_than_or_equal_to`, `less_or_equal`, `lte`, `<=` - Less than or equal

**Extensibility:**
Create custom verifiers by extending the `Verifier` base class for:
- HTTP API validation
- File system checks
- External service verification
- Complex multi-step validation

**Usage:**

```python
from turing_rl_sdk import DatabaseVerifier

verifier = DatabaseVerifier(
    query="SELECT COUNT(*) FROM issue WHERE status = 'Open'",
    expected_value=5,
    mcp_url="http://localhost:8015/mcp",
    database_id=result.database_id,
    comparison="greater_than_equal",
    name="Check open issues"
)

verifier_result = await verifier.verify()
print(f"Verification: {verifier_result.success}")
```

#### 4. Observable Execution

**Observer Pattern** for comprehensive monitoring:

**Features:**
- Monitor all messages (user, assistant, tool)
- Track tool calls with arguments and results
- Capture status updates and errors
- Real-time progress tracking
- Attach multiple observers per run

**Built-in Observer Support:**
- `RunObserver` base interface with three methods:
  - `on_message(role, content, metadata)` - Message events
  - `on_tool_call(tool_name, arguments, result, is_error)` - Tool execution events
  - `on_status(message, level)` - Status updates

**Usage:**

```python
from turing_rl_sdk import RunObserver, RunContext

class ProgressObserver(RunObserver):
    async def on_message(self, role, content, metadata=None):
        print(f"[{role}] {content[:100]}...")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        status = "❌" if is_error else "✅"
        print(f"{status} {tool_name}")
    
    async def on_status(self, message, level="info"):
        print(f"[{level}] {message}")

# Attach to harness
harness.add_observer_factory(lambda: ProgressObserver())

# Or use with agent directly
async with RunContext() as ctx:
    ctx.add_observer(ProgressObserver())
    result = await agent.run(task, run_context=ctx)
```

#### 5. Telemetry & Tracing

**Dual Tracing Support** for comprehensive observability:

**LangSmith Integration:**
- Automatic trace wrapping with `with_tracing(agent)`
- Trace URL capture in results
- Environment-based configuration
- Project and tag support

**Langfuse Integration:**
- Distributed tracing for production deployments
- Performance monitoring
- Cost tracking
- User session management

**Usage:**

```python
from turing_rl_sdk import configure_langsmith, with_tracing, create_traced_agent

# Option 1: Configure via environment
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-key
# LANGCHAIN_PROJECT=my-benchmarks

# Option 2: Configure programmatically
configure_langsmith(api_key="...", project="benchmarks", enabled=True)

# Create traced agent
agent = create_traced_agent("gpt-5")
result = await agent.run(task)
print(f"Trace: {result.langsmith_url}")

# Or wrap existing agent
agent = with_tracing(ClaudeAgent())
```

**Telemetry Captured:**
- Token usage (prompt + completion)
- Tool call statistics
- Execution time metrics
- Error rates and types
- Reasoning traces (when available)
- Complete conversation history

---

## 📁 File Structure

### Core SDK Files (`src/turing_rl_sdk/`)

#### Agent System (`agents/`)
- **`core/base.py`** (673 lines) - Abstract agent class with multi-level API (high/mid/low)
- **`core/claude.py`** (229 lines) - Anthropic Claude with extended thinking
- **`core/gpt.py`** (197 lines) - OpenAI GPT with reasoning effort
- **`core/gemini.py`** (225 lines) - Google Gemini with thinking budget
- **`core/grok.py`** (174 lines) - xAI Grok with reasoning support
- **`constants.py`** (64 lines) - Global constants and defaults

#### Task Management (`tasks/`)
- **`task.py`** (63 lines) - Task definition and configuration
- **`result.py`** (372 lines) - Result models with conversation history export

#### MCP Integration (`mcp/`)
- **`loader.py`** (143 lines) - MCP client manager, tool loading and aggregation
- **`config.py`** (56 lines) - MCP configuration with validation
- **`tool_fixer.py`** (119 lines) - Fix tool schemas for reserved keywords

#### Response Parsers (`parsers/`)
- **`base.py`** (39 lines) - Parser interface and common structures
- **`anthropic.py`** (98 lines) - Parse Claude thinking blocks and tool calls
- **`openai.py`** (37 lines) - Parse GPT tool calls and reasoning
- **`google.py`** (72 lines) - Parse Gemini complex content structure
- **`xai.py`** (37 lines) - Parse Grok responses with reasoning
- **`_common.py`** (88 lines) - Shared parsing utilities

#### Runtime System (`runtime/`)
- **`context.py`** (115 lines) - Execution context with database isolation
- **`events.py`** (90 lines) - Observer pattern implementation

#### Utilities (`utils/`)
- **`retry.py`** (180 lines) - Exponential backoff with jitter and timeout
- **`mcp.py`** (58 lines) - MCP helper utilities (URL derivation)

#### Telemetry (`telemetry.py`)
- **768 lines** - LangSmith/Langfuse integration, token tracking, trace management

#### Harness System (`harness/`)
- **`orchestrator.py`** (663 lines) - TestHarness class, concurrent execution, result aggregation
- **`loader.py`** (320 lines) - Load scenarios from JSON with validation
- **`scenario.py`** (120 lines) - Scenario data models (frozen dataclasses)
- **`agent_factory.py`** (201 lines) - Agent creation with auto-detection
- **`verifiers/base.py`** (47 lines) - Verifier interface
- **`verifiers/database.py`** (264 lines) - SQL-based verification with comparison logic

---

### 📚 Documentation & Examples

#### Documentation Files
- **`README.md`** (364 lines) - Complete SDK reference with:
  - Installation instructions for SDK and CLI
  - Quick start guide (30-second start)
  - Core concepts explanation
  - 4 usage patterns (harness, direct agents, custom agents, custom verifiers)
  - Complete API reference
  - Advanced features (tracing, observers, system prompts)
  - Troubleshooting guide
  
- **`harness/README.md`** (184 lines) - Test harness specific documentation




---

##  Technical Implementation

### Architecture Decisions

1. **Harness-First Design** - Built around TestHarness as the primary API, with direct agent usage as an alternative for custom workflows.

2. **Async-First** - All I/O operations are async from the ground up for optimal performance and concurrency control.

3. **Provider Abstraction** - Uses LangChain as the LLM abstraction layer while providing provider-specific optimizations (thinking modes, reasoning, etc.).

4. **Type Safety** - Extensive use of Pydantic v2 for data validation, type safety, and runtime checking.

5. **Extensibility** - Clean abstract interfaces (`Agent`, `Verifier`, `RunObserver`) with clear extension points.

6. **Database Isolation** - Unique database IDs per run prevent test interference in concurrent execution.

7. **Error Handling** - Comprehensive retry logic with exponential backoff, jitter, and configurable timeouts.

8. **Observable Execution** - Observer pattern decouples monitoring from execution logic.

### Multi-Level API

The SDK supports three levels of abstraction:

```python
# HIGH-LEVEL: Complete automation
agent = ClaudeAgent()
result = await agent.run(task)

# MID-LEVEL: Custom LLM integration
class MyAgent(Agent):
    async def get_response(self, messages):
        # Override to use custom LLM
        return AgentResponse(...)

# LOW-LEVEL: Manual control
await agent.initialize(task, run_context)
messages = agent.get_initial_messages()
# ... custom execution loop ...
await agent.cleanup()
```

---

##  Usage Examples

### 1. Harness-Based Testing (Primary Pattern)

```python
from pathlib import Path
from turing_rl_sdk import TestHarness, TestHarnessConfig, MCPConfig, create_agent

# Configure MCP server
mcp_config = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http"
)

# Create harness
harness = TestHarness(
    harness_path=Path("benchmarks/"),  # Directory with multiple .json files
    config=TestHarnessConfig(
        mcp=mcp_config,
        max_steps=1000,
        tool_call_limit=1000,
        runs_per_scenario=3,
        max_concurrent_runs=10,
    )
)

# Run across multiple models
results = await harness.run(
    models=["gpt-5", "claude-sonnet-4-5", "gemini-2.5-pro"],
    agent_factory=create_agent
)

# Analyze results
successful = sum(1 for r in results if r.success)
print(f"Pass rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")

# Export complete results with conversation history
import json
for result in results:
    result_dict = result.to_dict()  # Includes conversation, verifiers, reasoning
    with open(f"result_{result.model}_{result.scenario_id}.json", "w") as f:
        json.dump(result_dict, f, indent=2)
```

### 2. Direct Agent Usage (Custom Workflows)

```python
from turing_rl_sdk import ClaudeAgent, Task, MCPConfig, RunContext

# Create agent with defaults
agent = ClaudeAgent(
    model="claude-sonnet-4-5",  # Default model
    temperature=1.0,             # Required for thinking mode
    enable_thinking=True,        # Extended thinking enabled
)

# Define task
task = Task(
    prompt="Create a high-priority bug issue in project DEMO titled 'Login page crashes'",
    mcp=MCPConfig(
        name="jira",
        url="http://localhost:8015/mcp",
        transport="streamable_http"
    )
)

# Run with context
async with RunContext() as ctx:
    result = await agent.run(task, max_steps=50, run_context=ctx)
    
    print(f"Success: {result.success}")
    print(f"Database ID: {result.database_id}")
    
    # Access conversation history
    for msg in result.messages:
        print(f"{msg.type}: {msg.content[:100]}...")
```

### 3. Custom Agent Integration

Example: Integrating Alibaba Cloud Qwen models

```python
from turing_rl_sdk.agents.core.base import Agent
from turing_rl_sdk.agents.tasks import AgentResponse
from turing_rl_sdk.agents.parsers import OpenAIResponseParser
from langchain_openai import ChatOpenAI

class QwenAgent(Agent):
    """Custom agent for Alibaba Cloud Qwen models."""
    
    def __init__(self, model: str = "qwen-plus", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = os.environ["DASHSCOPE_API_KEY"]
        self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    
    def _build_llm(self):
        llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key
        )
        return llm.bind_tools(self._tools) if self._tools else llm
    
    async def get_response(self, messages):
        ai_message = await self._llm.ainvoke(messages)
        parser = self.get_response_parser()
        parsed = parser.parse(ai_message)
        
        return AgentResponse(
            content=parsed.content,
            tool_calls=parsed.tool_calls,
            done=not bool(parsed.tool_calls)
        ), ai_message
    
    def get_response_parser(self):
        return OpenAIResponseParser()

# Use with harness
def custom_factory(model, **kwargs):
    if model.startswith("qwen"):
        return QwenAgent(model=model, **kwargs)
    return create_agent(model, **kwargs)

results = await harness.run(
    models=["qwen-plus", "gpt-5"],
    agent_factory=custom_factory
)
```

### 4. Custom Verifiers

```python
from turing_rl_sdk.harness.verifiers import Verifier, VerifierResult
import httpx

class APIResponseVerifier(Verifier):
    """Verify API endpoint returns expected data."""
    
    def __init__(self, endpoint: str, expected_status: int):
        super().__init__(name="api_response_check")
        self.endpoint = endpoint
        self.expected_status = expected_status
    
    async def verify(self) -> VerifierResult:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.endpoint)
                success = response.status_code == self.expected_status
                
                return VerifierResult(
                    name=self.name,
                    success=success,
                    expected_value=self.expected_status,
                    actual_value=response.status_code,
                    comparison_type="equals",
                    error=None if success else "Status mismatch"
                )
        except Exception as e:
            return VerifierResult(
                name=self.name,
                success=False,
                expected_value=self.expected_status,
                actual_value=None,
                comparison_type="equals",
                error=str(e)
            )
```

### 5. Observable Execution with Custom Observers

```python
from turing_rl_sdk.agents.runtime import RunObserver

class DetailedObserver(RunObserver):
    def __init__(self):
        self.messages = []
        self.tool_calls = []
        self.errors = []
    
    async def on_message(self, role, content, metadata=None):
        self.messages.append({"role": role, "content": content})
        print(f"[{role.upper()}] {content[:80]}...")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error=False):
        self.tool_calls.append({
            "tool": tool_name,
            "args": arguments,
            "success": not is_error
        })
        if is_error:
            self.errors.append(result)
        print(f"{'❌' if is_error else '✅'} {tool_name}")
    
    async def on_status(self, message, level="info"):
        print(f"[{level.upper()}] {message}")

# Use with harness
harness.add_observer_factory(lambda: DetailedObserver())
```

---

## Testing Capabilities

### Scenario Definition Format

JSON-based scenario definitions for reproducible testing:

```json
{
  "system_prompt": "Optional global system prompt",
  "scenarios": [
    {
      "scenario_id": "create_bug_issue",
      "name": "Create Bug Issue",
      "description": "Test if agent can create a bug issue with correct fields",
      "prompts": [
        {
          "prompt_text": "Create a high-priority bug issue in project DEMO with summary 'Login page crashes on mobile' and description 'Users report crashes when accessing login page from mobile devices'",
          "expected_tools": ["create_issue"],
          "verifier": {
            "verifier_type": "database_state",
            "name": "Bug Issue Created",
            "validation_config": {
              "query": "SELECT COUNT(*) FROM issue WHERE summary = 'Login page crashes on mobile' AND type = 'Bug' AND priority = 'High'",
              "expected_value": 1,
              "comparison_type": "equals"
            }
          }
        }
      ],
      "metadata": {
        "difficulty": "medium",
        "tags": ["issue-creation", "bug-tracking"],
        "category": "core-functionality"
      },
      "conversation_mode": false
    }
  ]
}
```

**Important:** Each scenario must contain **exactly one prompt**. Multi-prompt/multi-turn conversations are not currently supported.

### Verification Types

**Database State Verifier:**
- Execute SQL queries against isolated test databases
- Support for all comparison operators
- Automatic query parameterization
- Detailed error reporting with query context

**Custom Verifiers:**
Extend `Verifier` class for:
- HTTP API validation
- File system state checks
- External service verification
- Complex multi-step validation

### Comparison Operators

| Operator | Aliases | Description |
|----------|---------|-------------|
| `equals` | `equal`, `eq`, `==` | Exact equality |
| `greater_than` | `gt`, `>` | Numeric greater than |
| `less_than` | `lt`, `<` | Numeric less than |
| `greater_than_equal` | `greater_than_or_equal`, `greater_than_or_equal_to`, `greater_or_equal`, `gte`, `>=` | Greater or equal |
| `less_than_equal` | `less_than_or_equal`, `less_than_or_equal_to`, `less_or_equal`, `lte`, `<=` | Less or equal |

---

## Results & Metrics

### RunResult Model

Each test run produces a comprehensive `RunResult`:

```python
@dataclass
class RunResult:
    model: str                          # Model identifier (e.g., "gpt-5")
    scenario_id: str                    # Scenario unique ID
    scenario_name: str                  # Human-readable name
    run_number: int                     # Run iteration number
    success: bool                       # Overall success (agent + all verifiers)
    result: Optional[Result]            # Agent execution result
    verifier_results: list[VerifierResult]  # All verifier outcomes
    error: Optional[str]                # Error message if failed
    metadata: dict[str, Any]            # Additional metadata
    
    # Methods
    def get_conversation_history() -> list[dict]  # Full conversation export
    def to_dict() -> dict                         # Complete serialization
```

### Conversation History Export

```python
# Get structured conversation history
conversation = result.get_conversation_history()

# Returns list of entries:
[
    {
        "type": "message",
        "role": "user",
        "content": "Create a bug issue..."
    },
    {
        "type": "message", 
        "role": "assistant",
        "content": "I'll create the bug issue...",
        "tool_calls": [{"id": "...", "name": "create_issue", "arguments": {...}}]
    },
    {
        "type": "tool_result",
        "tool_call_id": "...",
        "tool": "create_issue",
        "result": {...},
        "is_error": false
    },
    # ... more entries
]
```

### Export Capabilities

```python
# Export complete result with all data
result_dict = result.to_dict()

# Includes:
# - Full conversation history
# - Verifier results with SQL queries
# - Reasoning traces (if available)
# - Token usage statistics
# - Execution metadata (steps, database_id, etc.)
# - Error details (if any)

# Save to JSON
import json
with open("result.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

---

## Current Model Support

### Default Models (Optimized for Reasoning)

| Provider | Default Model | Temperature | Special Features |
|----------|---------------|-------------|------------------|
| **OpenAI** | `gpt-5` | 0.1 | Tool calling, reasoning support |
| **Anthropic** | `claude-sonnet-4-5` | 1.0 | Extended thinking, 42k token budget |
| **Google** | `gemini-2.5-pro` | 0.1 | Thinking budget, thought inclusion |
| **xAI** | `grok-4` | 0.1 | Reasoning support |

### Model Auto-Detection

```python
# SDK automatically detects provider from model name
agent = create_agent("gpt-5")             # → GPTAgent
agent = create_agent("claude-sonnet-4-5") # → ClaudeAgent
agent = create_agent("gemini-2.5-pro")    # → GeminiAgent
agent = create_agent("grok-4")            # → GrokAgent

# Or use explicit provider prefix
agent = create_agent("openai:gpt-5")
agent = create_agent("anthropic:claude-sonnet-4-5")
```

### Supported Models

**OpenAI:** `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-pro`, `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, `o4-mini`  
**Anthropic:** `claude-sonnet-4-5`, `claude-3-5-sonnet-20241022`, `claude-opus-4-20250514`  
**Google:** `gemini-2.5-pro`, `gemini-2.0-flash-exp`, `gemini-1.5-pro`  
**xAI:** `grok-4`, `grok-3-mini`, `grok-2`, `grok-beta`

**Note:** Claude's extended thinking mode requires `temperature=1.0`. The ClaudeAgent will automatically override any other temperature value and emit a warning when `enable_thinking=True` (default).

---

## 🔍 Import Patterns

The SDK supports **two import styles** for different use cases:

### Top-Level Imports (Production/Quick Use)

```python
from turing_rl_sdk import (
    TestHarness,
    TestHarnessConfig,
    ClaudeAgent,
    GPTAgent,
    Task,
    Result,
    MCPConfig,
    RunContext,
    RunObserver,
)
```

### Explicit Module Imports (Educational/Learning)

```python
# Shows logical component separation
from turing_rl_sdk.agents.core import ClaudeAgent, GPTAgent
from turing_rl_sdk.agents.tasks import Task, Result
from turing_rl_sdk.agents.mcp import MCPConfig
from turing_rl_sdk.agents.runtime import RunContext, RunObserver
from turing_rl_sdk.harness import TestHarness, TestHarnessConfig
from turing_rl_sdk.harness.verifiers import DatabaseVerifier
```
