# Dependency Injection Pattern Guide

## Quick Reference: How to Fix Direct Instantiation

### The Problem Pattern ❌

```python
class MyClass:
    def __init__(self):
        pass
    
    def some_method(self):
        # Direct instantiation - NOT TESTABLE
        dependency = SomeDependency()
        return dependency.do_something()
```

**Why it's bad**:
- Cannot inject mock for testing
- Tightly coupled to concrete implementation
- Cannot test error handling without side effects

### The Solution Pattern ✅

```python
class MyClass:
    def __init__(
        self,
        dependency_factory: Optional[Callable[[], SomeDependency]] = None
    ):
        # Store factory, use default if not provided
        self._dependency_factory = dependency_factory or SomeDependency
    
    def some_method(self):
        # Use factory to create instance - TESTABLE!
        dependency = self._dependency_factory()
        return dependency.do_something()
```

**Test example**:
```python
def test_some_method():
    # Create mock
    mock_dep = Mock(spec=SomeDependency)
    mock_dep.do_something = Mock(return_value="test result")
    
    # Inject mock via factory
    obj = MyClass(dependency_factory=lambda: mock_dep)
    
    # Test behavior
    result = obj.some_method()
    
    # Verify
    assert result == "test result"
    mock_dep.do_something.assert_called_once()
```

---

## Pattern Variations

### 1. Factory with Parameters

When the dependency needs parameters at creation time:

```python
class MyClass:
    def __init__(
        self,
        dependency_factory: Optional[Callable[[str, int], SomeDependency]] = None
    ):
        self._dependency_factory = dependency_factory or SomeDependency
    
    def some_method(self, name: str, count: int):
        dependency = self._dependency_factory(name, count)
        return dependency.do_something()
```

### 2. Singleton Pattern

When you want to reuse the same instance:

```python
class MyClass:
    def __init__(
        self,
        dependency_factory: Optional[Callable[[], SomeDependency]] = None
    ):
        self._dependency_factory = dependency_factory or SomeDependency
        self._dependency_instance: Optional[SomeDependency] = None
    
    def _get_dependency(self) -> SomeDependency:
        if self._dependency_instance is None:
            self._dependency_instance = self._dependency_factory()
        return self._dependency_instance
    
    def some_method(self):
        dependency = self._get_dependency()
        return dependency.do_something()
```

### 3. Direct Instance Injection (Simpler Alternative)

When the dependency is created once and reused:

```python
class MyClass:
    def __init__(
        self,
        dependency: Optional[SomeDependency] = None
    ):
        # Allow direct instance injection OR create default
        self._dependency = dependency or SomeDependency()
    
    def some_method(self):
        return self._dependency.do_something()
```

**Test example**:
```python
def test_some_method():
    mock_dep = Mock(spec=SomeDependency)
    mock_dep.do_something = Mock(return_value="test")
    
    # Inject instance directly
    obj = MyClass(dependency=mock_dep)
    
    result = obj.some_method()
    assert result == "test"
```

---

## When to Use Each Pattern

| Pattern | Use When | Example |
|---------|----------|---------|
| **Factory** | Dependency created fresh each time | HTTP clients, connections |
| **Singleton** | Expensive to create, reusable | Database connections, managers |
| **Direct Instance** | Simpler, dependency created once | Parsers, utilities, configurations |

---

## Real Examples from Your Codebase

### Example 1: Agent with MCP Manager

**Before** ❌:
```python
class Agent:
    async def initialize(self, task: Task, run_context: RunContext) -> None:
        self._mcp_manager = MCPClientManager()  # Hard-coded!
        await self._mcp_manager.connect(task.mcp, run_context.database_id)
```

**After** ✅:
```python
class Agent:
    def __init__(
        self,
        # ... existing params ...
        mcp_manager_factory: Optional[Callable[[], MCPClientManager]] = None,
    ):
        # ... existing code ...
        self._mcp_manager_factory = mcp_manager_factory or MCPClientManager
    
    async def initialize(self, task: Task, run_context: RunContext) -> None:
        self._mcp_manager = self._mcp_manager_factory()
        await self._mcp_manager.connect(task.mcp, run_context.database_id)
```

**Test**:
```python
async def test_initialize_handles_connection_error():
    # Arrange
    mock_manager = Mock(spec=MCPClientManager)
    mock_manager.connect = AsyncMock(side_effect=ConnectionError("Failed"))
    
    agent = ClaudeAgent(mcp_manager_factory=lambda: mock_manager)
    
    # Act & Assert
    with pytest.raises(ConnectionError):
        await agent.initialize(task, run_context)
```

---

### Example 2: TestHarness with HTTP Client

**Before** ❌:
```python
class TestHarness:
    async def run(self, models: list[str], agent_factory) -> ResultBundle:
        async with httpx.AsyncClient(timeout=60) as http_client:  # Hard-coded!
            # ... use http_client ...
```

**After** ✅:
```python
class TestHarness:
    def __init__(
        self,
        harness_path: Path,
        config: TestHarnessConfig,
        http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ):
        # ... existing code ...
        self._http_client_factory = http_client_factory
    
    async def run(self, models: list[str], agent_factory) -> ResultBundle:
        # Use factory if provided, otherwise default
        if self._http_client_factory:
            async with self._http_client_factory() as http_client:
                # ... use http_client ...
        else:
            async with httpx.AsyncClient(timeout=60) as http_client:
                # ... use http_client ...
```

**Test**:
```python
async def test_run_handles_http_errors():
    # Arrange
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Network error"))
    
    # Create async context manager mock
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    
    harness = TestHarness(
        harness_path=Path("test.json"),
        config=config,
        http_client_factory=lambda: mock_client
    )
    
    # Act
    results = await harness.run(models=["test"], agent_factory=mock_factory)
    
    # Assert - verify error handling
    assert not results[0].success
```

---

### Example 3: ClaudeAgent with LLM Client

**Before** ❌:
```python
class ClaudeAgent(Agent):
    def _build_llm(self) -> BaseChatModel:
        config = {
            "model": self.model,
            "temperature": self.temperature,
        }
        llm = ChatAnthropic(**config)  # Hard-coded!
        return llm.bind_tools(self._tools)
```

**After** ✅:
```python
class ClaudeAgent(Agent):
    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        # ... existing params ...
        llm_factory: Optional[Callable[[dict], BaseChatModel]] = None,
    ):
        super().__init__(...)
        # ... existing code ...
        self._llm_factory = llm_factory or ChatAnthropic
    
    def _build_llm(self) -> BaseChatModel:
        config = {
            "model": self.model,
            "temperature": self.temperature,
        }
        llm = self._llm_factory(**config)
        return llm.bind_tools(self._tools) if self._tools else llm
```

**Test**:
```python
async def test_get_response():
    # Arrange
    mock_llm = Mock(spec=ChatAnthropic)
    mock_llm.bind_tools = Mock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content="Test response",
        tool_calls=[]
    ))
    
    agent = ClaudeAgent(
        llm_factory=lambda **kwargs: mock_llm
    )
    await agent.initialize(task, run_context)
    
    # Act
    response, ai_msg = await agent.get_response([HumanMessage(content="test")])
    
    # Assert
    assert response.content == "Test response"
    mock_llm.ainvoke.assert_called_once()
```

---

## Global Dependency Container (Advanced Pattern)

For projects with many dependencies, consider a dependency container:

```python
@dataclass
class Dependencies:
    """Global dependency container for the SDK."""
    
    mcp_manager_factory: Callable[[], MCPClientManager] = MCPClientManager
    http_client_factory: Callable[[], httpx.AsyncClient] = lambda: httpx.AsyncClient()
    run_context_factory: Callable[[Optional[str]], RunContext] = lambda db_id=None: RunContext(database_id=db_id) if db_id else RunContext()
    llm_factory_anthropic: Callable[[dict], BaseChatModel] = ChatAnthropic
    llm_factory_openai: Callable[[dict], BaseChatModel] = ChatOpenAI
    llm_factory_google: Callable[[dict], BaseChatModel] = ChatGoogleGenerativeAI

# Global instance (can be overridden in tests)
dependencies = Dependencies()

# Usage in classes:
class Agent:
    def __init__(self, deps: Dependencies = dependencies):
        self._deps = deps
    
    async def initialize(self, task: Task, run_context: RunContext) -> None:
        self._mcp_manager = self._deps.mcp_manager_factory()
        # ...

# In tests:
def test_agent():
    test_deps = Dependencies(
        mcp_manager_factory=lambda: MockMCPManager()
    )
    agent = Agent(deps=test_deps)
    # ...
```

---

## Testing Utilities

Create helper functions for common test scenarios:

```python
# test_utils.py

def create_mock_mcp_manager(tools: list[BaseTool] = None) -> Mock:
    """Create a mock MCPClientManager with standard behavior."""
    mock = Mock(spec=MCPClientManager)
    mock.connect = AsyncMock()
    mock.get_all_tools = Mock(return_value=tools or [])
    mock.cleanup = AsyncMock()
    mock.is_connected = True
    return mock

def create_mock_llm(response: str = "test", tool_calls: list = None) -> Mock:
    """Create a mock LLM that returns a predictable response."""
    mock = Mock(spec=BaseChatModel)
    mock.ainvoke = AsyncMock(return_value=AIMessage(
        content=response,
        tool_calls=tool_calls or []
    ))
    mock.bind_tools = Mock(return_value=mock)
    return mock

def create_mock_http_client(responses: dict[str, Any] = None) -> AsyncMock:
    """Create a mock HTTP client with configurable responses."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock()
    
    if responses:
        mock.post = AsyncMock(side_effect=lambda url, **kwargs: 
            Mock(json=lambda: responses.get(url, {}))
        )
    else:
        mock.post = AsyncMock(return_value=Mock(json=lambda: {}))
    
    return mock

# Usage in tests:
async def test_agent_with_helpers():
    agent = ClaudeAgent(
        mcp_manager_factory=lambda: create_mock_mcp_manager(),
        llm_factory=lambda **kwargs: create_mock_llm("success")
    )
    
    result = await agent.run(task)
    assert result.success
```

---

## Checklist for Refactoring

When converting a class to use dependency injection:

- [ ] Identify all direct instantiations (search for `= SomeClass()`)
- [ ] Add factory parameter to `__init__` with `Optional[Callable[...]]` type
- [ ] Set default factory: `self._factory = factory or DefaultClass`
- [ ] Replace direct instantiation with factory call
- [ ] Update docstring to document new parameter
- [ ] Write unit tests using mock injection
- [ ] Verify existing code still works (factories default to original behavior)

---

## Benefits Summary

✅ **Fast Tests**: No external dependencies, runs in milliseconds  
✅ **Reliable Tests**: No flaky network/API failures  
✅ **Comprehensive Tests**: Can test error paths easily  
✅ **Better Design**: Loosely coupled, easier to change  
✅ **No Breaking Changes**: Default factories maintain backward compatibility  

---

## Next Steps

1. Start with high-impact classes (Agent, TestHarness)
2. Add factory injection parameters
3. Write comprehensive unit tests
4. Gradually refactor remaining classes
5. Consider dependency container for complex scenarios

