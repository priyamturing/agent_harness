# MCP Benchmark SDK - Architecture

## Overview

The MCP Benchmark SDK is designed to benchmark LLM agents against MCP (Model Context Protocol) servers. It provides a flexible, multi-level API for running and evaluating agent tasks.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Orchestration Layer (CLI)"
        CLI[CLI/Orchestrator]
    end

    subgraph "User Layer"
        USER[User Code]
        TASK[Task Definition]
    end

    subgraph "Agent Layer"
        AGENT[Agent Base Class]
        CLAUDE[ClaudeAgent]
        GPT[GPTAgent]
        GEMINI[GeminiAgent]
        GROK[GrokAgent]
    end

    subgraph "Runtime Layer"
        CONTEXT[RunContext]
        OBSERVER[RunObserver]
        EVENTS[Event System]
    end

    subgraph "MCP Layer"
        MCP_CONFIG[MCPConfig]
        MCP_MANAGER[MCPClientManager]
        MCP_LOADER[MCP Loader]
        TOOLS[LangChain Tools]
    end

    subgraph "Parser Layer"
        PARSER_BASE[ResponseParser Base]
        ANTHROPIC_PARSER[Anthropic Parser]
        OPENAI_PARSER[OpenAI Parser]
        GOOGLE_PARSER[Google Parser]
        XAI_PARSER[XAI Parser]
    end

    subgraph "Verifier Layer (SDK Utilities)"
        VERIFIER[Verifier Base]
        DB_VERIFIER[DatabaseVerifier]
        EXECUTOR[Verifier Executor]
    end

    subgraph "Task Management"
        SCENARIO[Scenario]
        RESULT[Result]
        AGENT_RESPONSE[AgentResponse]
    end

    subgraph "Utils"
        RETRY[Retry Logic]
        COMMON[Common Utils]
    end

    USER --> TASK
    USER --> AGENT
    CLI --> AGENT
    CLI --> EXECUTOR
    
    AGENT --> CLAUDE
    AGENT --> GPT
    AGENT --> GEMINI
    AGENT --> GROK
    
    CLAUDE --> ANTHROPIC_PARSER
    GPT --> OPENAI_PARSER
    GEMINI --> GOOGLE_PARSER
    GROK --> XAI_PARSER
    
    AGENT --> CONTEXT
    AGENT --> MCP_MANAGER
    
    CONTEXT --> OBSERVER
    CONTEXT --> EVENTS
    
    MCP_CONFIG --> MCP_MANAGER
    MCP_MANAGER --> MCP_LOADER
    MCP_LOADER --> TOOLS
    
    TASK --> SCENARIO
    TASK --> MCP_CONFIG
    
    VERIFIER --> DB_VERIFIER
    EXECUTOR --> VERIFIER
    
    AGENT --> RESULT
    RESULT --> AGENT_RESPONSE
    
    AGENT --> RETRY
    PARSER_BASE --> ANTHROPIC_PARSER
    PARSER_BASE --> OPENAI_PARSER
    PARSER_BASE --> GOOGLE_PARSER
    PARSER_BASE --> XAI_PARSER
```

## Component Details

### 1. Agent Layer
**Location**: `agents/`

The core abstraction for LLM agents with a multi-level API:

- **Agent (Base)**: Abstract base class providing three levels of usage:
  - **High-level**: `agent.run(task)` - Complete automation
  - **Mid-level**: Override `get_response()`, `get_model_config()` - Custom LLM integration
  - **Low-level**: Manual loop control via primitives

- **Concrete Agents**:
  - `ClaudeAgent` - Anthropic Claude models
  - `GPTAgent` - OpenAI GPT models
  - `GeminiAgent` - Google Gemini models
  - `GrokAgent` - xAI Grok models

**Key Responsibilities**:
- Task execution orchestration
- LLM interaction
- Tool call management
- Message history tracking
- Result generation (without verification)

### 2. Task Management
**Location**: `tasks/`

Defines the benchmark task structure:

- **Task**: User-facing task definition with:
  - `prompt`: The task instruction
  - `mcps`: List of MCP configurations
  - `max_steps`: Maximum agent turns
  - `metadata`: Additional context
  - `database_id`: Isolated database instance
  - Note: Verifiers are no longer part of Task - they are managed separately by orchestrators (e.g., CLI)

- **Result**: Task execution outcome with:
  - Success/failure status (from agent execution)
  - Final message history
  - Verifier results (can be enriched post-execution by orchestrators)
  - Metadata

- **Scenario**: Advanced task with multiple prompts for conversation mode

- **AgentResponse**: Structured agent turn with messages and tool calls

### 3. Runtime Layer
**Location**: `runtime/`

Manages execution context and observability:

- **RunContext**: Centralized runtime state
  - Unique database ID per run
  - SQL runner URL for verifiers
  - Shared HTTP client
  - Event observers
  - Async context manager support

- **RunObserver**: Event observation interface
  - `on_message()` - Message events
  - `on_tool_call()` - Tool execution events
  - `on_verifier_update()` - Verification events
  - `on_status()` - Status updates

**Key Features**:
- Resource lifecycle management
- Event broadcasting
- Shared client pooling

### 4. MCP Layer
**Location**: `mcp/`

Handles Model Context Protocol integration:

- **MCPConfig**: MCP server configuration
  - Command and arguments
  - Environment variables
  - Server metadata

- **MCPClientManager**: MCP client lifecycle
  - Session management
  - Connection handling
  - Tool discovery

- **MCP Loader**: Converts MCP tools to LangChain tools
  - Tool schema conversion
  - Parameter validation
  - Error handling

**Integration Flow**:
1. Load MCP config
2. Start MCP server process
3. Establish client session
4. List available tools
5. Convert to LangChain format
6. Provide to agent

### 5. Parser Layer
**Location**: `parsers/`

Handles LLM response parsing across providers:

- **ResponseParser (Base)**: Abstract parser interface
  - `parse()` - Extract structured response
  - Provider-agnostic output format

- **Provider-Specific Parsers**:
  - `AnthropicParser` - Claude response format
  - `OpenAIParser` - GPT response format
  - `GoogleParser` - Gemini response format
  - `XAIParser` - Grok response format

**ParsedResponse Format**:
- Text content
- Tool calls with arguments
- Stop reason
- Usage metrics

### 6. Verifier Layer (SDK Utilities)
**Location**: `verifiers/`

**Important**: The verifier layer is a set of SDK utilities that can be used independently by orchestrators (e.g., CLI).
Agents have zero knowledge of verifiers - verification is orchestrated externally.

- **Verifier (Base)**: Abstract verifier interface
  - `verify()` - Returns pass/fail result
  - `description` - Human-readable check
  - Pure utility class, not coupled to agents

- **DatabaseVerifier**: SQL-based verification
  - Executes queries against test database
  - Compares results to expected values
  - Supports complex assertions
  - Can be used by any orchestrator

- **Verifier Executor**: Batch execution utility
  - `execute_verifiers()` - Standalone function
  - Runs multiple verifiers
  - Manages HTTP client lifecycle
  - Aggregates results
  - No agent dependencies

**VerificationContext**:
- SQL runner URL
- Database ID (isolation)
- HTTP client
- Used by verifiers, independent of agent execution

### 7. Utils
**Location**: `utils/`

Shared utilities:

- **Retry Logic**: Exponential backoff with jitter
  - Configurable max attempts
  - Custom retry conditions
  - Provider-specific error handling

## Execution Flow

### SDK-Level (Decoupled from Verification)

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant RunContext
    participant MCPManager
    participant LLM
    participant Parser
    participant Tools

    User->>Agent: run(task)
    Agent->>RunContext: Create/receive context
    Agent->>MCPManager: Initialize MCP servers
    MCPManager->>Tools: Load & convert tools
    
    loop Until complete or max_steps
        Agent->>LLM: Send messages + tools
        LLM->>Parser: Raw response
        Parser->>Agent: ParsedResponse
        
        alt Has tool calls
            Agent->>Tools: Execute tool calls
            Tools->>Agent: Tool results
            Agent->>RunContext: Notify observers
        else No tool calls
            Agent->>Agent: Break loop (done)
        end
    end
    
    Agent->>User: Return Result (no verification)
    Agent->>RunContext: Cleanup resources
```

### CLI-Level (Orchestrated Verification)

```mermaid
sequenceDiagram
    participant CLI
    participant Agent
    participant VerifierRunner
    participant Verifier
    participant RunContext

    CLI->>Agent: run(task)
    Agent->>CLI: Return Result (agent success only)
    
    CLI->>VerifierRunner: run_verifiers(verifiers, context)
    VerifierRunner->>Verifier: Execute each verifier
    Verifier->>VerifierRunner: Verification results
    VerifierRunner->>RunContext: Notify observers
    VerifierRunner->>CLI: Return verifier results
    
    CLI->>CLI: Merge agent + verifier results
    CLI->>CLI: Determine final success
```

## Key Design Principles

### 1. **Multi-Level API**
- High-level: Simple `agent.run()` for common cases
- Mid-level: Override methods for customization
- Low-level: Full control over execution loop

### 2. **Provider Abstraction**
- Unified interface across LLM providers
- Provider-specific parsers handle differences
- Easy to add new providers

### 3. **Observable Execution**
- Event-driven observer pattern
- Non-intrusive logging/monitoring
- Real-time progress tracking

### 4. **Resource Management**
- Context managers for cleanup
- Ownership tracking
- HTTP client pooling

### 5. **Isolation**
- Database-per-run via unique IDs
- Independent verifier execution
- Clean state between runs

### 6. **Extensibility**
- Abstract base classes
- Plugin-style verifiers
- Custom tool support

### 7. **Decoupled Verification**
- Agents have zero knowledge of verifiers
- Verification is orchestrated externally (e.g., by CLI)
- Verifiers are SDK utilities, not agent responsibilities
- Enables any agent to work with any verifier
- CLI controls when and how verification happens

## Usage Patterns

### Basic Usage - SDK Only (Agent Execution)
```python
from mcp_benchmark_sdk import Task, ClaudeAgent, MCPConfig

# Task no longer has verifiers - pure execution config
task = Task(
    prompt="Create a new Jira issue",
    mcps=[MCPConfig(command="jira-mcp", args=[])],
)

agent = ClaudeAgent()
result = await agent.run(task)
# result.success is based ONLY on agent completion, not verification
print(result.success)
```

### Advanced Usage - With Verification (Orchestrator Pattern)
```python
from mcp_benchmark_sdk import Task, ClaudeAgent, MCPConfig, RunContext
from mcp_benchmark_sdk.verifiers import DatabaseVerifier, execute_verifiers

# Define task (no verifiers)
task = Task(
    prompt="Create a new Jira issue",
    mcps=[MCPConfig(command="jira-mcp", args=[])],
)

# Define verifiers separately
verifiers = [
    DatabaseVerifier(query="SELECT COUNT(*) FROM issues", expected_value=1)
]

# Run agent
agent = ClaudeAgent()
async with RunContext(sql_runner_url="http://localhost:8080") as ctx:
    result = await agent.run(task, run_context=ctx)
    
    # Orchestrate verification separately
    verifier_results = await execute_verifiers(
        verifiers,
        ctx.sql_runner_url,
        ctx.database_id,
        ctx.get_http_client(),
    )
    
    # Merge results
    result.verifier_results = verifier_results
    final_success = result.success and all(v.success for v in verifier_results)
    print(final_success)
```

### With Observability (Mid-Level)
```python
from mcp_benchmark_sdk import RunContext, RunObserver

class MyObserver(RunObserver):
    async def on_message(self, role, content, metadata):
        print(f"{role}: {content}")
    
    async def on_tool_call(self, tool_name, arguments, result, is_error):
        print(f"Tool: {tool_name}({arguments}) -> {result}")
    
    async def on_verifier_update(self, results):
        print(f"Verifiers: {results}")

# Agent execution with observation
async with RunContext() as ctx:
    ctx.add_observer(MyObserver())
    result = await agent.run(task, run_context=ctx)
    
    # Verification can also be observed
    verifier_results = await execute_verifiers(verifiers, ctx.sql_runner_url, ctx.database_id, ctx.get_http_client())
    await ctx.notify_verifier_update(verifier_results)
```

### Custom Agent (Low-Level)
```python
from mcp_benchmark_sdk import Agent

class CustomAgent(Agent):
    def get_model_config(self):
        return custom_llm, custom_parser
    
    async def get_response(self, messages, tools):
        # Custom LLM call logic
        return parsed_response

agent = CustomAgent()
result = await agent.run(task)
```

## File Structure

```
mcp_benchmark_sdk/
├── __init__.py           # Public API exports
├── agents/               # Agent implementations
│   ├── base.py          # Abstract Agent class
│   ├── claude.py        # Claude agent
│   ├── gpt.py           # GPT agent
│   ├── gemini.py        # Gemini agent
│   └── grok.py          # Grok agent
├── tasks/                # Task definitions
│   ├── task.py          # Task dataclass
│   ├── result.py        # Result types
│   └── scenario.py      # Multi-turn scenarios
├── runtime/              # Execution context
│   ├── context.py       # RunContext
│   └── events.py        # RunObserver
├── mcp/                  # MCP integration
│   ├── config.py        # MCPConfig
│   ├── loader.py        # Tool loading
│   └── tool_fixer.py    # Tool schema fixes
├── parsers/              # Response parsers
│   ├── base.py          # Parser interface
│   ├── anthropic.py     # Claude parser
│   ├── openai.py        # GPT parser
│   ├── google.py        # Gemini parser
│   └── xai.py           # Grok parser
├── verifiers/            # Verification system
│   ├── base.py          # Verifier interface
│   ├── database.py      # SQL verifiers
│   └── executor.py      # Batch execution
└── utils/                # Utilities
    └── retry.py         # Retry logic
```

## Extension Points

### Adding a New LLM Provider

1. Create agent class in `agents/`:
```python
class MyAgent(Agent):
    def get_model_config(self):
        model = MyLLM()
        parser = MyParser()
        return model, parser
```

2. Create parser in `parsers/`:
```python
class MyParser(ResponseParser):
    def parse(self, response) -> ParsedResponse:
        # Extract content, tool calls, etc.
        return ParsedResponse(...)
```

3. Export in `__init__.py`

### Adding a New Verifier Type

Verifiers are SDK utilities that can be used by any orchestrator (CLI, custom code, etc.).
They have no knowledge of agents.

1. Create verifier in `verifiers/`:
```python
class MyVerifier(Verifier):
    async def verify(self, context: VerificationContext) -> VerifierResult:
        # Custom verification logic
        # No agent dependencies - pure utility
        return VerifierResult(...)
```

2. Export in `__init__.py`

3. Use in orchestrator (e.g., CLI):
```python
verifier = MyVerifier(...)
results = await execute_verifiers([verifier], sql_url, db_id, http_client)
```

### Adding Custom Observers

1. Implement RunObserver interface:
```python
class MyObserver(RunObserver):
    async def on_message(self, role, content, metadata):
        # Custom logging/monitoring
        pass
```

2. Register with RunContext:
```python
ctx.add_observer(MyObserver())
```

## Dependencies

- **LangChain**: LLM abstraction and tool framework
- **httpx**: Async HTTP client
- **anthropic**: Claude API
- **openai**: GPT API
- **google-generativeai**: Gemini API
- **MCP SDK**: Model Context Protocol client

## Testing Strategy

- Unit tests for individual components
- Integration tests for agent execution
- Provider-specific tests for parsers
- End-to-end benchmark tests
- Mock MCP servers for testing

