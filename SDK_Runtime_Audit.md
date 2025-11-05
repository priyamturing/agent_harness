### SDK Runtime and Patterns Audit (mcp_benchmark_sdk)

Below are concrete findings across the SDK with exact locations, the impact at runtime, and recommended fixes. Code blocks referencing existing files use code-reference formatting; proposed snippets are shown as regular code blocks.

---

#### 1) Off-by-one recursion limit in reasoning parser

```37:43:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/parsers/_common.py
    if max_depth < 0:
        raise RecursionError(
            f"Reasoning structure nesting depth exceeded maximum limit of 10 levels. "
            f"This may indicate malformed or malicious data. "
            f"The current task will be marked as failed."
        )
```

- **Impact**: The message promises a 10-level limit, but `max_depth` is decremented before checking `< 0`, effectively allowing 11 levels. This can permit deeper nesting than expected or produce confusing error messages in boundary cases.
- **Fix**: Either check `<= 0` or initialize with one fewer level. Example:
```python
# Option A
if max_depth <= 0:
    raise RecursionError("… limit of 10 levels …")

# Option B (keep check, adjust default)
def collect_reasoning_chunks(reasoning_block, chunks, max_depth=9):
    ...
```

---

#### 2) Tool-call limit is reported as “Max steps reached”

```446:456:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
            if response.tool_calls:
                if remaining_tool_calls is not None:
                    if remaining_tool_calls < len(response.tool_calls):
                        await run_context.notify_status("Tool call limit reached", "warning")
                        break
                    remaining_tool_calls -= len(response.tool_calls)

                tool_results = await self.call_tools(response.tool_calls)
```
```460:472:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
        # Max steps reached
        await run_context.notify_status("Max steps reached", "warning")
        verifier_results = await self.run_verifiers()

        return Result(
            success=False,
            messages=messages,
            verifier_results=verifier_results,
            metadata={"steps": max_steps, "reason": "max_steps_reached"},
            database_id=run_context.database_id,
            reasoning_traces=all_reasoning,
            error="Maximum steps reached",
        )
```

- **Impact**: When the tool-call budget is hit, the loop `break`s and the final result incorrectly states "Max steps reached." This obscures the real reason and can mislead debugging and automation.
- **Fix**: Return immediately with a distinct reason and message, e.g. `"tool_call_limit_reached"`, and avoid the generic max-steps footer path.

---

#### 3) Internal RunContext cleanup is not guaranteed

```74:83:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
        if run_context is None:
            # Only pass database_id if provided; otherwise let RunContext auto-generate UUID
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)

        await self.initialize(task, run_context)
        try:
            result = await self._execute_loop(max_steps, run_context)
            return result
        finally:
            await self.cleanup()
```
```83:87:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/runtime/context.py
    async def cleanup(self) -> None:
        """Cleanup resources (HTTP client, etc.)."""
        if self.http_client is not None:
            await self.http_client.aclose()
            self.http_client = None
```

- **Impact**: If the agent creates the `RunContext` internally and any code eventually calls `get_http_client()`, the HTTP client may stay open since `RunContext.cleanup()` is never invoked here. In longer-lived processes this risks connection/resource leaks.
- **Fix**: When the agent owns the context (created internally), ensure it is cleaned up in the same `finally` block:
```python
created_context = run_context is None
run_context = run_context or RunContext(...)
try:
    ...
finally:
    await self.cleanup()
    if created_context:
        await run_context.cleanup()
```

---

#### 4) Observer failures can abort the agent run

```70:74:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/runtime/context.py
    async def notify_status(self, message: str, level: str = "info") -> None:
        """Notify observers of status updates."""
        for observer in self.observers:
            await observer.on_status(message, level)
```

- **Impact**: If any observer raises, the exception propagates and may interrupt the agent loop (e.g., during status or message notifications). Observability should not break core execution.
- **Fix**: Wrap individual observer calls with try/except and log or collect observer errors, so one faulty observer cannot derail execution.

---

#### 5) Retry helper type signature invites misuse

```93:99:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/utils/retry.py
async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 2,
    timeout_seconds: float | None = None,
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
```
```116:117:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/utils/retry.py
                return await asyncio.wait_for(func(), timeout=timeout_seconds)
            return await func()
```

- **Impact**: The type says any callable, but the body always awaits it. A sync callable will fail at runtime. This creates a footgun for SDK users calling this API outside agents.
- **Fix**: Tighten the type to `Callable[[], Awaitable[Any]]` (or accept both sync/async by normalizing with `anyio.to_thread` for sync), and add a runtime assertion if needed.

---

#### 6) Pydantic schema reconstruction in tool fixer may drop constraints

```54:66:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/mcp/tool_fixer.py
        # Build new schema with renamed fields
        new_fields = {}
        field_mapping = {}  # old_name -> new_name

        for field_name, field_info in schema_fields.items():
            if field_name in RESERVED_KEYWORDS:
                # Rename reserved keyword
                new_name = f"{field_name}_"
                field_mapping[field_name] = new_name
                new_fields[new_name] = (field_info.annotation, field_info)
            else:
                new_fields[field_name] = (field_info.annotation, field_info)
```

- **Impact**: Passing the original `FieldInfo` as the default often works in pydantic v2, but certain metadata (aliases, validators) or non-default constraints may not transpose cleanly when renaming. In edge cases, validation or serialization differences can surface at runtime for renamed fields.
- **Fix**: Rebuild `FieldInfo` explicitly via `Field(default=..., description=..., ge=..., le=..., pattern=..., alias=...)` to ensure constraints survive the rename, or copy `field_info` properties explicitly into a new `Field` object. Add tests for tools with constrained/aliased fields.

---

#### 7) OpenAI config may pass unsupported extras to ChatOpenAI

```101:109:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/gpt.py
        # Use Responses API for all models (most OpenAI models support it)
        config["output_version"] = "responses/v1"
        config["use_responses_api"] = True

        config.update(self.extra_kwargs)

        llm = ChatOpenAI(**config)
```

- **Impact**: Depending on the installed `langchain_openai` version, `ChatOpenAI` may not accept `use_responses_api` or `output_version`, resulting in a model construction error. This is environment/version-sensitive.
- **Fix**: Guard extras based on version capability (feature-detect by trying to construct with/without and falling back), or use the latest compatible `langchain_openai` and document the requirement.

---

#### 8) Ambiguous value selection in SQL result extraction

```22:29:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/verifiers/database.py
    data = response_json.get("data")
    if isinstance(data, list) and data:
        first_entry = data[0]
        if isinstance(first_entry, dict):
            for value in first_entry.values():
                return value
```

- **Impact**: When a row has multiple columns, returning the first dictionary value is order-dependent and may not match the intended column. This can cause false positives/negatives.
- **Fix**: Prefer a named column (if provided), or match the first column listed in `columns` from the payload when available. Alternatively, require the query to return a single column and validate that.

---

#### 9) Tool output serialization can fail on nested non-JSON types

```485:499:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
    def _serialize_tool_output(self, result: object) -> object:
        """Serialize tool output to consistent format."""
        if isinstance(result, str):
            stripped = result.strip()
            if stripped:
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    return result
            return result

        if isinstance(result, (int, float, bool)) or result is None:
            return result

        return result
```

- **Impact**: Returning arbitrary objects (e.g., pydantic models, datetime) inside dicts/lists will later fail `json.dumps` in `call_tools`, converting the whole tool call into an error. This is correct but lossy: any serializable subset is lost.
- **Fix**: Make serialization stricter/safer by normalizing common non-JSON types (e.g., datetimes to ISO, pydantic models via `.model_dump()`), or document that MCP servers MUST return pure JSON-compatible values only (and enforce with clearer error including the path to the bad value).

---

#### 10) External tool-name prefix assumption in tool filtering

```86:93:/workspace/packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/mcp/loader.py
        if server_name is None:
            return self._tools

        # Filter tools by server name (tools have server prefix in name)
        return [
            tool for tool in self._tools
            if tool.name.startswith(f"mcp_{server_name}_")
        ]
```

- **Impact**: This relies on an external naming convention from `MultiServerMCPClient`. If that changes, filtering returns an empty set, breaking server-specific tool access while the general case still appears fine.
- **Fix**: If possible, track tool→server mapping from the client API rather than inferring from names; otherwise, assert/validate at connect time and raise a clear error if the expected prefixing is absent.

---

#### 11) Environment-variable prechecks raise EnvironmentError

- Files: `agents/gpt.py`, `agents/claude.py`, `agents/gemini.py`, `agents/grok.py`.
- **Impact**: Raising `EnvironmentError` is fine, but some users expect `RuntimeError` or a custom SDK error; consistency across agents is good but consider documenting this behavior. No change strictly required.
- **Fix**: Optional—use a dedicated `MissingApiKeyError` or similar for clearer catches.

---

#### 12) Minor: Reasoning collector message may mislead operators

- The recursion error string says the task "will be marked as failed." That is a policy decision better handled by the caller. The parser should raise a neutral error and let the agent decide.
- **Fix**: Remove policy language from the exception message and let caller compose user-facing text.

---

### Overall flow assessment

- Core loop, tool invocation, and verification wiring appear correct. Failures in tool execution and parsing are captured and surfaced without crashing the loop.
- Resource handling is generally safe; the notable gap is cleaning up an internally-created `RunContext` (see Finding 3).
- Parser coverage for provider-specific formats is reasonable; recursion limits and JSON formatting guardrails are good, with only the off-by-one to address.
- Retry logic is pragmatic and scoped; improving typing will prevent accidental sync callables.

### Suggested prioritization

1. Fix tool-call limit reporting (Finding 2).
2. Guarantee internal `RunContext` cleanup (Finding 3).
3. Correct recursion limit boundary and message (Finding 1, 12).
4. Harden observer isolation (Finding 4).
5. Tighten retry typing and consider dual sync/async support (Finding 5).
6. Clarify/guard ChatOpenAI extras and tool filtering assumptions (Findings 7, 10).
7. Improve tool output serialization ergonomics (Finding 9).
8. Make SQL scalar selection deterministic (Finding 8).

