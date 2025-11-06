# Test Harness Design

## Architecture

The test harness provides a high-level API for running benchmarks against LLM agents with MCP servers, inspired by HUD's simple developer experience.

### Core Components

```
harness/
├── scenario.py         # Data structures for test scenarios
├── loader.py           # Load scenarios from JSON files
├── agent_factory.py    # Create agents from model strings
└── orchestrator.py     # TestHarness + VerifierRunner
```

### Data Flow

```
JSON Harness File(s)
    ↓
HarnessLoader / load_harness_file()
    ↓
List[Scenario]
    ↓
scenario_to_task()
    ↓
Task + VerifierDefinitions
    ↓
TestHarness.run()
    ↓
    ├─→ Agent.run(task)
    │   ├─→ RunContext (with observers)
    │   └─→ Result
    │
    └─→ VerifierRunner.run_verifiers()
        └─→ List[VerifierResult]
    ↓
RunResult
```

## Key Design Decisions

### 1. Separation of Concerns

**Problem**: Original implementation had test harness logic mixed with CLI logic.

**Solution**: 
- SDK provides core orchestration (`TestHarness`, `VerifierRunner`)
- CLI adds session management, UI, result formatting
- Both can be used independently

### 2. Observer Pattern for UI

**Problem**: Different UIs (console, textual, web) need different progress tracking.

**Solution**:
- `TestHarness` accepts observer factories
- Each run gets its own observer instance
- Observers receive events via `RunContext`
- CLI can inject custom observers without modifying SDK

```python
# SDK usage with observer
def make_observer():
    return MyProgressBar()

harness.add_observer_factory(make_observer)
```

### 3. Agent Factory Pattern

**Problem**: Different users need different agent implementations.

**Solution**:
- SDK provides `create_agent()` for standard models
- Users can provide custom factory functions
- CLI extends with `create_agent_from_string()` for Qwen support

```python
# Custom agent factory
def my_factory(model: str, **kwargs) -> Agent:
    if model.startswith("my-model"):
        return MyCustomAgent(model, **kwargs)
    return create_agent(model, **kwargs)

results = await harness.run(
    models=["my-model-v1", "gpt-4o"],
    agent_factory=my_factory,
)
```

### 4. Verifier Definitions vs. Instances

**Problem**: Verifiers need runtime context (database_id, http_client) not available at load time.

**Solution**:
- Harness loader returns `VerifierDefinition` (data only)
- `VerifierRunner` creates instances during execution
- HTTP client shared across all verifiers (efficiency)

### 5. Concurrent Execution

**Problem**: Running hundreds of tests sequentially is slow.

**Solution**:
- `TestHarness` uses `asyncio.gather()` for parallelism
- Semaphore limits concurrent runs (prevents resource exhaustion)
- Shared HTTP client for all verifiers (connection pooling)

```python
config = TestHarnessConfig(
    max_concurrent_runs=20,  # At most 20 parallel runs
    runs_per_scenario=3,     # Each scenario runs 3 times
)
```

### 6. Configuration Object

**Problem**: Too many parameters to TestHarness constructor.

**Solution**:
- `TestHarnessConfig` dataclass groups related settings
- Clear defaults for common use cases
- Easy to extend without breaking existing code

```python
config = TestHarnessConfig(
    mcps=[mcp_config],
    sql_runner_url="http://localhost:8015/api/sql-runner",
    max_steps=1000,
    tool_call_limit=1000,
    temperature=0.1,
    runs_per_scenario=1,
    max_concurrent_runs=20,
)
```

### 7. Result Aggregation

**Problem**: Need to track results across many runs.

**Solution**:
- `RunResult` contains all information about a single run
- `TestHarness.run()` returns `List[RunResult]`
- Users can filter, aggregate, export as needed

```python
results = await harness.run(models=["gpt-4o"], agent_factory=create_agent)

# Filter successful runs
successful = [r for r in results if r.success]

# Group by model
by_model = {}
for r in results:
    by_model.setdefault(r.model, []).append(r)
```

## Extensibility

### Adding New Verifier Types

1. Implement in `mcp_benchmark_sdk/verifiers/`
2. Update `create_verifier_from_definition()` in `loader.py`
3. Document in harness JSON schema

### Adding New Agent Types

1. Implement in `mcp_benchmark_sdk/agents/`
2. Update `create_agent()` in `agent_factory.py`
3. Or provide custom factory to `TestHarness.run()`

### Adding New Observers

1. Implement `RunObserver` interface
2. Pass factory to `harness.add_observer_factory()`
3. Observer receives all events during execution

## Performance Considerations

### Memory

- Observers use bounded queues (maxsize=1000) for textual UI
- HTTP clients are shared, not per-run
- Verifiers are created once, reused across verifications

### Concurrency

- Semaphore prevents resource exhaustion
- Default max_concurrent_runs=20 (tunable)
- Each run has its own agent instance (no sharing)

### I/O

- Single HTTP client for all verifiers (connection pooling)
- Async I/O throughout (no blocking)
- MCP connections managed by SDK's `MCPClientManager`

## Testing Strategy

### Unit Tests

- Test each component in isolation
- Mock MCP servers, verifiers, agents
- Validate data transformations

### Integration Tests

- Test full harness execution
- Use noop MCP server (no-op tools)
- Verify results, verifier execution, observer events

### CLI Tests

- Test CLI remains backward compatible
- Verify session management, UI modes
- Check result formatting, manifest generation

## Future Enhancements

1. **Result Exporters**: JSON, HTML, CSV exporters
2. **Harness Validation**: Validate JSON before running
3. **Streaming Results**: Yield results as they complete
4. **Result Caching**: Cache results to avoid re-runs
5. **Harness Composition**: Combine multiple harnesses
6. **Custom Metrics**: User-defined success criteria
7. **Distributed Execution**: Run across multiple machines
8. **Web UI**: Real-time progress dashboard
