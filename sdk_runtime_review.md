# SDK Runtime Review Findings

## Issue 1: Auto-created `RunContext` is never cleaned up

- **Location**:
```74:83:packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
        if run_context is None:
            run_context = RunContext() if task.database_id is None else RunContext(database_id=task.database_id)

        await self.initialize(task, run_context)
        try:
            result = await self._execute_loop(max_steps, run_context)
            return result
        finally:
            await self.cleanup()
```
- **Impact**: When callers rely on the high-level `agent.run()` helper and do not pass a `RunContext`, the SDK instantiates one but never closes it. Any resources the context owns (notably the shared `httpx.AsyncClient`) are leaked, leading to open-connection warnings and exhausting file descriptors during long benchmark runs.
- **Suggested fix**: Track whether the agent created the `RunContext` and, in that case, call `await run_context.cleanup()` (and optionally close observers) inside the `finally` block after agent cleanup.

## Issue 2: Plain-string tool outputs are double JSON encoded

- **Location**:
```192:213:packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
                serialized = self._serialize_tool_output(output)
                try:
                    content_str = json.dumps(serialized, ensure_ascii=False)
                except TypeError as e:
                    raise TypeError(
                        f"Tool '{tc.name}' returned non-JSON-serializable data. "
                        f"MCP servers must return JSON-compatible types. "
                        f"Received type: {type(serialized).__name__}. "
                        f"Original error: {e}"
                    ) from e

                results.append(
                    ToolResult(
                        content=content_str,
                        tool_call_id=tool_call_id,
                        is_error=False,
                        structured_content=serialized if isinstance(serialized, dict) else None,
                    )
                )
```
- **Impact**: If a tool returns plain text like `"ok"`, the code stores `content` as the JSON literal `"\"ok\""`. The agent then feeds that quoted payload back to the LLM, which sees extra quotes/escape characters and frequently responds with formatting corrections instead of progressing on the task.
- **Suggested fix**: Only JSON-encode non-string outputs. For strings, pass the raw value through (after optional trimming) so the LLM receives human-readable text.

## Issue 3: Tool-call limit exits are misreported as max-step failures

- **Location**:
```446:472:packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/agents/base.py
            if response.tool_calls:
                if remaining_tool_calls is not None:
                    if remaining_tool_calls < len(response.tool_calls):
                        await run_context.notify_status("Tool call limit reached", "warning")
                        break
                    remaining_tool_calls -= len(response.tool_calls)

                tool_results = await self.call_tools(response.tool_calls)
                tool_messages = self.format_tool_results(response.tool_calls, tool_results)
                messages.extend(tool_messages)

                await self.run_verifiers()

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
- **Impact**: When the budget is exhausted, the loop breaks and the method falls through to the "Max steps reached" report. Downstream consumers cannot distinguish step-limit failures from tool-budget enforcement, leading to wrong remediation steps and false negatives in metrics.
- **Suggested fix**: Return immediately when the tool-call limit is hit, with metadata/error that reflects the limit breach (e.g., `reason="tool_call_limit"`).

## Issue 4: `FixedTool` wrappers drop critical tool metadata/configuration

- **Location**:
```74:114:packages/mcp_benchmark_sdk/src/mcp_benchmark_sdk/mcp/tool_fixer.py
        class FixedTool(BaseTool):
            name: str = tool.name
            description: str = tool.description
            args_schema: type[BaseModel] = new_schema
            _field_mapping: dict[str, str] = field_mapping
            _original_tool: BaseTool = tool

            async def _arun(self, **kwargs: Any) -> Any:
                # ... existing code ...
                if hasattr(self._original_tool, "ainvoke"):
                    return await self._original_tool.ainvoke(original_kwargs)
                else:
                    return self._original_tool.invoke(original_kwargs)

            def _run(self, **kwargs: Any) -> Any:
                # ... existing code ...
                return self._original_tool.invoke(original_kwargs)

        fixed_tool = FixedTool()
        fixed_tools.append(fixed_tool)
```
- **Impact**: The wrapper only replays `name`, `description`, and `args_schema`. Flags such as `return_direct`, `tags`, `metadata`, custom callback handlers, or auth settings on the original `BaseTool` are lost. As a result, tools that should short-circuit the agent or emit rich telemetry silently revert to defaults, changing behavior in subtle ways.
- **Suggested fix**: Copy the full set of relevant attributes (`return_direct`, `callbacks`, `tags`, `metadata`, etc.) when constructing the wrapper, and forward `invoke/ainvoke` with the runnable `config` so callback chains remain intact.

