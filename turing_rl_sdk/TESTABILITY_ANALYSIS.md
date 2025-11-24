# Testability Analysis: Direct Instantiation Issues

This document identifies all areas in the codebase where direct object instantiation prevents effective unit testing with mocks/stubs.

## Executive Summary

**Total Issues Found: 15 areas**
- **Critical (High Impact)**: 7 areas
- **Medium Impact**: 5 areas  
- **Low Impact**: 3 areas

## Severity Levels

- **CRITICAL**: Core functionality that requires external dependencies (network, MCP servers, LLM APIs)
- **MEDIUM**: Infrastructure/utility components that are harder to test but less frequently changed
- **LOW**: Stateless/simple objects where direct instantiation is less problematic

---

## Critical Issues (Must Fix for Testability)

### 1. Agent.initialize() - MCPClientManager Creation
**File**: `src/turing_rl_sdk/agents/core/base.py`  
**Line**: 210

```python
# Current problematic code:
async def initialize(self, task: Task, run_context: RunContext) -> None:
    self._task = task
    self._mcp_manager = MCPClientManager()  # ❌ Hard-coded instantiation
    await self._mcp_manager.connect(task.mcp, run_context.database_id)
```

**Problem**: Cannot test agent initialization without connecting to real MCP servers.

**Impact**: 
- Cannot test agent initialization in isolation
- Tests require real MCP server running
- Cannot verify initialization logic handles errors correctly

**Recommended Fix**:
```python
def __init__(
    self,
    system_prompt: Optional[str] = None,
    tool_call_limit: Optional[int] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    mcp_manager_factory: Optional[Callable[[], MCPClientManager]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._mcp_manager_factory = mcp_manager_factory or MCPClientManager
    
async def initialize(self, task: Task, run_context: RunContext) -> None:
    self._task = task
    self._mcp_manager = self._mcp_manager_factory()  # ✅ Use factory
    await self._mcp_manager.connect(task.mcp, run_context.database_id)
```

**Test Example**:
```python
async def test_initialize():
    # Create a mock MCP manager
    mock_manager = Mock(spec=MCPClientManager)
    mock_manager.connect = AsyncMock()
    mock_manager.get_all_tools = Mock(return_value=[])
    
    # Inject mock via factory
    agent = ClaudeAgent(mcp_manager_factory=lambda: mock_manager)
    
    # Test initialization
    task = Task(prompt="test", mcp=MCPConfig(...))
    await agent.initialize(task, RunContext())
    
    # Verify behavior
    mock_manager.connect.assert_called_once()
```

---

### 2. Agent.run() - RunContext Creation
**File**: `src/turing_rl_sdk/agents/core/base.py`  
**Line**: 130

```python
# Current problematic code:
async def run(
    self,
    task: Task,
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    run_context: Optional[RunContext] = None,
) -> Result:
    if run_context is None:
        run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)  # ❌ Hard-coded
        owns_context = True
```

**Problem**: Cannot test run() behavior when RunContext is auto-created.

**Impact**:
- Cannot inject mock observers
- Cannot verify correct context setup
- Cannot test context ownership logic

**Recommended Fix**:
```python
def __init__(
    self,
    # ... existing params ...
    run_context_factory: Optional[Callable[[Optional[str]], RunContext]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._run_context_factory = run_context_factory or (lambda db_id: RunContext(database_id=db_id) if db_id else RunContext())

async def run(
    self,
    task: Task,
    max_steps: int = DEFAULT_MAX_STEPS,
    *,
    run_context: Optional[RunContext] = None,
) -> Result:
    if run_context is None:
        run_context = self._run_context_factory(task.database_id)  # ✅ Use factory
        owns_context = True
```

---

### 3. Agent._build_llm() - LLM Client Creation
**File**: Multiple agent implementations
- `src/turing_rl_sdk/agents/core/claude.py:171`
- `src/turing_rl_sdk/agents/core/gpt.py:140`
- `src/turing_rl_sdk/agents/core/gemini.py:165`

```python
# Current problematic code (Claude example):
def _build_llm(self) -> Union[BaseChatModel, Runnable]:
    config: dict[str, Any] = {
        "model": self.model,
        "temperature": self.temperature,
        # ... more config ...
    }
    llm = ChatAnthropic(**config)  # ❌ Hard-coded instantiation
    return llm.bind_tools(self._tools) if self._tools else llm
```

**Problem**: Cannot test agent LLM interactions without making real API calls.

**Impact**:
- Cannot test get_response() without API keys
- Cannot test retry logic
- Cannot test error handling
- Tests are slow and expensive

**Recommended Fix**:
```python
def __init__(
    self,
    # ... existing params ...
    llm_factory: Optional[Callable[[dict[str, Any]], BaseChatModel]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._llm_factory = llm_factory or ChatAnthropic

def _build_llm(self) -> Union[BaseChatModel, Runnable]:
    config: dict[str, Any] = {
        "model": self.model,
        "temperature": self.temperature,
        # ... more config ...
    }
    llm = self._llm_factory(**config)  # ✅ Use factory
    return llm.bind_tools(self._tools) if self._tools else llm
```

**Test Example**:
```python
async def test_get_response():
    # Create mock LLM
    mock_llm = Mock(spec=BaseChatModel)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="test response"))
    
    # Inject mock via factory
    agent = ClaudeAgent(
        llm_factory=lambda **kwargs: mock_llm
    )
    
    # Test response generation
    await agent.initialize(task, run_context)
    response, ai_msg = await agent.get_response([HumanMessage(content="test")])
    
    # Verify behavior
    assert response.content == "test response"
    mock_llm.ainvoke.assert_called_once()
```

---

### 4. MCPClientManager.connect() - MultiServerMCPClient Creation
**File**: `src/turing_rl_sdk/agents/mcp/loader.py`  
**Line**: 70

```python
# Current problematic code:
async def connect(
    self,
    config: MCPConfig,
    database_id: Optional[str] = None,
) -> None:
    # ... setup connection config ...
    
    async def _attempt_connect() -> None:
        self._client = MultiServerMCPClient(server_configs)  # ❌ Hard-coded instantiation
        raw_tools = await self._client.get_tools()
        self._tools = fix_tool_schemas(raw_tools)
```

**Problem**: Cannot test MCP manager without real MCP servers.

**Impact**:
- Cannot test connection logic
- Cannot test error handling (retries, timeouts)
- Cannot test tool loading

**Recommended Fix**:
```python
def __init__(
    self,
    client_factory: Optional[Callable[[dict], MultiServerMCPClient]] = None,  # ✅ Inject factory
):
    self._client: Optional[MultiServerMCPClient] = None
    self._config: Optional[MCPConfig] = None
    self._tools: list[BaseTool] = []
    self._client_factory = client_factory or MultiServerMCPClient

async def connect(
    self,
    config: MCPConfig,
    database_id: Optional[str] = None,
) -> None:
    # ... setup connection config ...
    
    async def _attempt_connect() -> None:
        self._client = self._client_factory(server_configs)  # ✅ Use factory
        raw_tools = await self._client.get_tools()
        self._tools = fix_tool_schemas(raw_tools)
```

---

### 5. TestHarness.run() - httpx.AsyncClient Creation
**File**: `src/turing_rl_sdk/harness/orchestrator.py`  
**Line**: 403

```python
# Current problematic code:
async def run(
    self,
    models: list[str],
    agent_factory: Callable[..., Agent],
    observer_config: Optional[dict[str, Any]] = None,
) -> ResultBundle:
    run_configs = self._build_run_configs(models)
    
    async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:  # ❌ Hard-coded
        # ... test execution ...
```

**Problem**: Cannot test harness without making real HTTP calls.

**Impact**:
- Cannot test verifier execution in isolation
- Cannot test HTTP error handling
- Tests require real servers

**Recommended Fix**:
```python
def __init__(
    self,
    harness_path: Path,
    config: TestHarnessConfig,
    http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,  # ✅ Inject factory
):
    self.harness_path = harness_path
    self.config = config
    self.scenarios: list[Scenario] = []
    self.file_map: dict[str, list[Scenario]] = {}
    self.observer_factories: list[Callable[[], RunObserver]] = []
    self._http_client_factory = http_client_factory
    
    self._load_scenarios()

async def run(
    self,
    models: list[str],
    agent_factory: Callable[..., Agent],
    observer_config: Optional[dict[str, Any]] = None,
) -> ResultBundle:
    run_configs = self._build_run_configs(models)
    
    # Use injected factory or default
    if self._http_client_factory:
        async with self._http_client_factory() as http_client:
            # ... test execution ...
    else:
        async with httpx.AsyncClient(timeout=HTTP_CLIENT_TIMEOUT_SECONDS) as http_client:
            # ... test execution ...
```

---

### 6. TestHarness._run_single() - RunContext and VerifierRunner Creation
**File**: `src/turing_rl_sdk/harness/orchestrator.py`  
**Lines**: 516, 528

```python
# Current problematic code:
async def _run_single(
    self,
    run_config: dict[str, Any],
    agent_factory: Callable[..., Agent],
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    observer_config: Optional[dict[str, Any]],
) -> RunResult:
    async with semaphore:
        # ...
        run_context = RunContext()  # ❌ Hard-coded instantiation
        
        # ... add observers ...
        
        verifier_runner = VerifierRunner(  # ❌ Hard-coded instantiation
            verifier_defs,
            run_context,
            http_client=http_client,
            mcp_url=mcp_url,
        )
```

**Problem**: Cannot test single run execution with mock contexts/verifiers.

**Impact**:
- Cannot test run lifecycle
- Cannot test observer integration
- Cannot test verifier integration

**Recommended Fix**:
```python
def __init__(
    self,
    harness_path: Path,
    config: TestHarnessConfig,
    run_context_factory: Optional[Callable[[], RunContext]] = None,  # ✅ Inject factory
    verifier_runner_factory: Optional[Callable[..., VerifierRunner]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._run_context_factory = run_context_factory or RunContext
    self._verifier_runner_factory = verifier_runner_factory or VerifierRunner

async def _run_single(
    self,
    run_config: dict[str, Any],
    agent_factory: Callable[..., Agent],
    http_client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    observer_config: Optional[dict[str, Any]],
) -> RunResult:
    async with semaphore:
        # ...
        run_context = self._run_context_factory()  # ✅ Use factory
        
        # ... add observers ...
        
        verifier_runner = self._verifier_runner_factory(  # ✅ Use factory
            verifier_defs,
            run_context,
            http_client=http_client,
            mcp_url=mcp_url,
        )
```

---

### 7. DatabaseVerifier.verify() - httpx.AsyncClient Creation
**File**: `src/turing_rl_sdk/harness/verifiers/database.py`  
**Line**: 166

```python
# Current problematic code:
async def verify(self) -> VerifierResult:
    http_client = self._http_client
    
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=DATABASE_VERIFIER_TIMEOUT_SECONDS)  # ❌ Hard-coded
    
    try:
        response = await http_client.post(
            self.sql_runner_url,
            # ... make HTTP call ...
        )
```

**Problem**: Cannot test verifier without making real HTTP calls.

**Impact**:
- Cannot test SQL query execution logic
- Cannot test comparison logic with different responses
- Cannot test error handling

**Recommended Fix**:
```python
def __init__(
    self,
    query: str,
    expected_value: Any,
    mcp_url: str,
    database_id: str,
    comparison: Union[ComparisonType, str] = "equals",
    name: Optional[str] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._http_client = http_client
    self._http_client_factory = http_client_factory or (lambda: httpx.AsyncClient(timeout=DATABASE_VERIFIER_TIMEOUT_SECONDS))
    self._owns_client = http_client is None

async def verify(self) -> VerifierResult:
    http_client = self._http_client
    
    if http_client is None:
        http_client = self._http_client_factory()  # ✅ Use factory
    
    try:
        response = await http_client.post(
            self.sql_runner_url,
            # ... make HTTP call ...
        )
```

---

## Medium Impact Issues

### 8. Agent.get_response_parser() - Parser Creation
**Files**: All agent implementations
- `src/turing_rl_sdk/agents/core/claude.py:228`
- `src/turing_rl_sdk/agents/core/gpt.py:197`
- `src/turing_rl_sdk/agents/core/gemini.py:224`

```python
# Current problematic code:
def get_response_parser(self) -> ResponseParser:
    return AnthropicResponseParser()  # ❌ Hard-coded instantiation
```

**Problem**: Cannot test with custom parsers or verify parser behavior.

**Impact**: Medium - parsers are relatively stateless, but still prevents testing parser integration.

**Recommended Fix**:
```python
def __init__(
    self,
    # ... existing params ...
    parser_factory: Optional[Callable[[], ResponseParser]] = None,  # ✅ Inject factory
):
    # ... existing code ...
    self._parser_factory = parser_factory or AnthropicResponseParser

def get_response_parser(self) -> ResponseParser:
    return self._parser_factory()  # ✅ Use factory
```

---

### 9. TracingAgent.__init__() - Callback Handler Creation
**File**: `src/turing_rl_sdk/agents/telemetry.py`  
**Line**: 518

```python
# Current problematic code:
def __init__(self, agent: "Agent"):
    self._agent = agent
    self._langfuse_handler: Optional[Any] = None
    
    if is_langfuse_enabled() and LangchainCallbackHandler is not None:
        try:
            self._langfuse_handler = LangchainCallbackHandler()  # ❌ Hard-coded instantiation
        except Exception:
            self._langfuse_handler = None
```

**Problem**: Cannot test tracing behavior without real Langfuse configuration.

**Impact**: Medium - prevents testing tracing integration logic.

**Recommended Fix**:
```python
def __init__(
    self,
    agent: "Agent",
    langfuse_handler_factory: Optional[Callable[[], Any]] = None,  # ✅ Inject factory
):
    self._agent = agent
    self._langfuse_handler: Optional[Any] = None
    self._langfuse_handler_factory = langfuse_handler_factory
    
    if is_langfuse_enabled() and LangchainCallbackHandler is not None:
        try:
            factory = self._langfuse_handler_factory or LangchainCallbackHandler
            self._langfuse_handler = factory()  # ✅ Use factory
        except Exception:
            self._langfuse_handler = None
```

---

### 10-12. Telemetry Functions - Client/Handler Creation
**File**: `src/turing_rl_sdk/agents/telemetry.py`

Multiple functions create clients/handlers directly:
- Line 172: `LangchainCallbackHandler()` in `_get_or_create_langfuse_handler()`
- Line 279: `Client()` in `_get_current_trace_url()`
- Line 394: `Client(api_key=api_key)` in `get_langsmith_client()`

**Problem**: Cannot test telemetry functions without real API credentials.

**Impact**: Medium - these are utility functions that could be wrapped or dependency-injected at module level.

**Recommended Fix**: Use module-level factories that can be monkey-patched in tests:

```python
# At module level
_CLIENT_FACTORY: Callable[[Optional[str]], Client] = lambda api_key=None: Client(api_key=api_key) if api_key else Client()
_LANGFUSE_HANDLER_FACTORY: Callable[[], Any] = LangchainCallbackHandler

def set_client_factory(factory: Callable[[Optional[str]], Client]) -> None:
    """For testing: inject custom client factory"""
    global _CLIENT_FACTORY
    _CLIENT_FACTORY = factory

def set_langfuse_handler_factory(factory: Callable[[], Any]) -> None:
    """For testing: inject custom handler factory"""
    global _LANGFUSE_HANDLER_FACTORY
    _LANGFUSE_HANDLER_FACTORY = factory

# Then use factories in functions:
def get_langsmith_client() -> Optional[Client]:
    # ...
    return _CLIENT_FACTORY(api_key)  # ✅ Use factory
```

---

### 13. agent_factory.create_agent() - Agent Creation
**File**: `src/turing_rl_sdk/harness/agent_factory.py`  
**Lines**: 62-151

```python
# Current problematic code:
def create_agent(
    model: str,
    temperature: float = 0.1,
    # ... params ...
) -> Agent:
    # ... provider detection ...
    
    if provider_hint == "openai":
        return GPTAgent(  # ❌ Hard-coded instantiation
            model=model_name,
            temperature=temperature,
            # ...
        )
    elif provider_hint == "anthropic":
        return ClaudeAgent(  # ❌ Hard-coded instantiation
            model=model_name,
            temperature=temperature,
            # ...
        )
```

**Problem**: Factory itself creates concrete instances. This is expected for a factory, but limits composability.

**Impact**: Medium - this IS a factory function, so direct instantiation is somewhat expected. However, you can't easily inject mock agents for testing code that uses this factory.

**Recommended Fix**: Create an `AgentFactory` class that can be injected:

```python
class AgentFactory:
    """Factory for creating agents with injectable constructors."""
    
    def __init__(
        self,
        gpt_agent_class: type[Agent] = GPTAgent,
        claude_agent_class: type[Agent] = ClaudeAgent,
        gemini_agent_class: type[Agent] = GeminiAgent,
        grok_agent_class: type[Agent] = GrokAgent,
    ):
        self.gpt_agent_class = gpt_agent_class
        self.claude_agent_class = claude_agent_class
        self.gemini_agent_class = gemini_agent_class
        self.grok_agent_class = grok_agent_class
    
    def create_agent(
        self,
        model: str,
        temperature: float = 0.1,
        # ... params ...
    ) -> Agent:
        # ... provider detection ...
        
        if provider_hint == "openai":
            return self.gpt_agent_class(  # ✅ Use injected class
                model=model_name,
                temperature=temperature,
                # ...
            )
        elif provider_hint == "anthropic":
            return self.claude_agent_class(  # ✅ Use injected class
                model=model_name,
                temperature=temperature,
                # ...
            )

# Keep module-level function for backward compatibility
_default_factory = AgentFactory()

def create_agent(*args, **kwargs) -> Agent:
    return _default_factory.create_agent(*args, **kwargs)
```

---

## Low Impact Issues

### 14. VerifierRunner Creation in loader
**File**: `src/turing_rl_sdk/harness/loader.py`  
**Line**: 254

The `create_verifier_from_definition()` function creates DatabaseVerifier directly. This is a factory function, so it's expected, but could be improved for testability.

**Impact**: Low - factory functions are expected to create instances.

---

### 15. Response Parser Instantiation
**Files**: Various parser files

Parsers are stateless utility classes, so direct instantiation is less problematic. However, for consistency and testability, they should still use the factory pattern.

**Impact**: Low - stateless objects are easy to mock.

---

## Summary of Recommended Changes

### Immediate Priority (Critical Issues)

1. **Add factory injection to Agent base class**:
   - `mcp_manager_factory`
   - `run_context_factory`
   - All concrete agents need `llm_factory`

2. **Add factory injection to MCPClientManager**:
   - `client_factory`

3. **Add factory injection to TestHarness**:
   - `http_client_factory`
   - `run_context_factory`
   - `verifier_runner_factory`

4. **Add factory injection to DatabaseVerifier**:
   - `http_client_factory` (when no client provided)

### Medium Priority

5. **Add parser factory injection to agents**
6. **Add handler factory injection to TracingAgent**
7. **Make telemetry functions use module-level injectable factories**
8. **Convert agent_factory to class-based factory**

### Long-term Improvements

9. **Create singleton factory pattern**:
   ```python
   class Dependencies:
       """Singleton container for injectable factories."""
       
       mcp_client_factory: Callable[[], MCPClientManager] = MCPClientManager
       http_client_factory: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient()
       run_context_factory: Callable[[], RunContext] = RunContext
       # ... etc
   
   # Usage:
   dependencies = Dependencies()
   
   # In tests:
   dependencies.mcp_client_factory = lambda: MockMCPManager()
   ```

---

## Testing Strategy After Refactoring

Once factories are injected, you can write tests like:

```python
# Example: Testing agent initialization
async def test_agent_initialize_handles_mcp_error():
    # Arrange: Create mock that raises connection error
    mock_manager = Mock(spec=MCPClientManager)
    mock_manager.connect = AsyncMock(side_effect=ConnectionError("Server down"))
    
    agent = ClaudeAgent(
        mcp_manager_factory=lambda: mock_manager
    )
    
    task = Task(prompt="test", mcp=MCPConfig(...))
    run_context = RunContext()
    
    # Act & Assert
    with pytest.raises(ConnectionError, match="Server down"):
        await agent.initialize(task, run_context)

# Example: Testing harness with mock agents
async def test_harness_run():
    # Arrange: Create mock agent that succeeds immediately
    mock_agent = Mock(spec=Agent)
    mock_agent.run = AsyncMock(return_value=Result(success=True, messages=[]))
    
    harness = TestHarness(
        harness_path=Path("test.json"),
        config=TestHarnessConfig(mcp=MCPConfig(...)),
        http_client_factory=lambda: Mock(spec=httpx.AsyncClient)
    )
    
    # Act
    results = await harness.run(
        models=["test-model"],
        agent_factory=lambda *args, **kwargs: mock_agent
    )
    
    # Assert
    assert len(results) > 0
    assert results[0].success
```

---

## Conclusion

The codebase has **15 areas** where direct instantiation prevents effective unit testing. The most critical issues are:

1. Agent dependencies (MCP manager, LLM clients, RunContext)
2. TestHarness dependencies (HTTP clients, contexts, verifiers)
3. DatabaseVerifier HTTP client creation

**Recommended Approach**:
1. Start with Agent base class and concrete implementations (issues #1-3)
2. Then fix MCPClientManager (issue #4)
3. Then fix TestHarness (issues #5-6)
4. Then DatabaseVerifier (issue #7)
5. Finally address medium/low priority issues

**Benefits After Refactoring**:
- Fast unit tests without external dependencies
- Ability to test error handling and edge cases
- Better code organization and separation of concerns
- Easier to mock/stub for different test scenarios

