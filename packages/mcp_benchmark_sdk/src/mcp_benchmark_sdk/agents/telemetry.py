"""Tracing utilities for LangSmith and Langfuse.

This module centralizes telemetry helpers so SDK users can enable
observability with either LangSmith or Langfuse using the same API surface.

LangSmith provides automatic tracing for LangChain applications. See
https://docs.langchain.com/langsmith/observability-quickstart

Langfuse provides an open-source tracing backend with LangChain callback
integration. See https://langfuse.com/docs
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

from langsmith import Client, get_current_run_tree, traceable

try:  # Optional dependency exposed by the SDK
    from langfuse import get_client as _get_default_langfuse_client
    from langfuse.langchain import CallbackHandler as LangchainCallbackHandler
except Exception:  # pragma: no cover - missing optional dependency
    _get_default_langfuse_client = None
    LangchainCallbackHandler = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .core import Agent
    from .tasks import Task, Result
    from .runtime import RunContext
    from langfuse import Langfuse  # pragma: nocover

# TypeVar for preserving agent type through tracing wrapper
AgentT = TypeVar("AgentT", bound="Agent")


def _as_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


def configure_langfuse(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    *,
    base_url: Optional[str] = None,
    host: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> dict[str, str]:
    """Configure Langfuse tracing environment variables.

    The SDK uses LangChain's native Langfuse callback handler, so once the
    environment variables are configured, tracing works without further code
    changes.

    Args:
        public_key: Langfuse public API key.
        secret_key: Langfuse secret API key.
        base_url: Langfuse base URL (e.g. ``"https://cloud.langfuse.com"``).
        host: Deprecated alternative to ``base_url`` kept for compatibility.
        environment: Optional Langfuse tracing environment label.
        release: Optional release identifier associated with traces.
        enabled: Force enable/disable tracing. When ``None`` tracing will be
            enabled automatically if credentials are supplied or already present
            in the environment.

    Returns:
        dict[str, str]: Environment variables that were set or updated.

    Raises:
        ValueError: If tracing is enabled but the required API keys are missing.
    """

    env_updates: dict[str, str] = {}

    if enabled is None:
        env_flag = os.environ.get("LANGFUSE_TRACING_ENABLED")
        if env_flag is not None:
            enabled = _as_bool(env_flag)
        else:
            enabled = bool(
                (public_key or secret_key)
                or (
                    os.environ.get("LANGFUSE_PUBLIC_KEY")
                    and os.environ.get("LANGFUSE_SECRET_KEY")
                )
            )

    if enabled:
        env_updates["LANGFUSE_TRACING_ENABLED"] = "true"

        resolved_public = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        resolved_secret = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")

        if not resolved_public or not resolved_secret:
            raise ValueError(
                "Langfuse public and secret keys are required when enabling tracing."
            )

        env_updates["LANGFUSE_PUBLIC_KEY"] = resolved_public
        env_updates["LANGFUSE_SECRET_KEY"] = resolved_secret

        if base_url:
            env_updates["LANGFUSE_BASE_URL"] = base_url.rstrip("/")
        elif host:
            env_updates["LANGFUSE_HOST"] = host.rstrip("/")

        if environment:
            env_updates["LANGFUSE_TRACING_ENVIRONMENT"] = environment

        if release:
            env_updates["LANGFUSE_RELEASE"] = release
    else:
        env_updates["LANGFUSE_TRACING_ENABLED"] = "false"

    for key, value in env_updates.items():
        os.environ[key] = value

    return env_updates


def is_langfuse_enabled() -> bool:
    """Check whether Langfuse tracing is active and properly configured."""

    if LangchainCallbackHandler is None or _get_default_langfuse_client is None:
        return False

    env_flag = os.environ.get("LANGFUSE_TRACING_ENABLED")
    if env_flag is None:
        enabled = bool(
            os.environ.get("LANGFUSE_PUBLIC_KEY")
            and os.environ.get("LANGFUSE_SECRET_KEY")
        )
    else:
        enabled = _as_bool(env_flag)

    if not enabled:
        return False

    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def get_langfuse_client() -> Optional["Langfuse"]:
    """Return a Langfuse client if tracing is enabled."""

    if _get_default_langfuse_client is None or not is_langfuse_enabled():
        return None

    try:
        return _get_default_langfuse_client()
    except Exception:
        return None


def _get_or_create_langfuse_handler(agent_or_wrapper: Union["Agent", "TracingAgent"]) -> Optional[Any]:
    """Get Langfuse handler from TracingAgent wrapper or create for bare agent (backward compat)."""
    if not is_langfuse_enabled() or LangchainCallbackHandler is None:
        return None

    if isinstance(agent_or_wrapper, TracingAgent):
        return agent_or_wrapper._langfuse_handler

    handler = getattr(agent_or_wrapper, _LANGFUSE_HANDLER_ATTR, None)
    if handler is None:
        handler = LangchainCallbackHandler()
        setattr(agent_or_wrapper, _LANGFUSE_HANDLER_ATTR, handler)
    return handler


def _merge_callbacks(existing: Any, handler: Any) -> list[Any]:
    if existing is None:
        return [handler]
    if isinstance(existing, list):
        if handler not in existing:
            return [*existing, handler]
        return existing
    if isinstance(existing, tuple):
        merged = list(existing)
        if handler not in merged:
            merged.append(handler)
        return merged
    return [existing, handler]


def maybe_attach_langfuse_callback(agent_or_wrapper: Union["Agent", "TracingAgent"], config: dict[str, Any]) -> None:
    """Inject Langfuse callback into an LLM configuration dict if enabled.
    
    Note: This function is kept for backward compatibility but should not be needed
    when using TracingAgent wrapper, as the wrapper handles callback injection
    via context managers.
    """

    handler = _get_or_create_langfuse_handler(agent_or_wrapper)
    if not handler:
        return

    callbacks = config.get("callbacks")
    config["callbacks"] = _merge_callbacks(callbacks, handler)


def get_langfuse_trace_url(agent_or_wrapper: Union["Agent", "TracingAgent"]) -> Optional[str]:
    """Return the Langfuse trace URL for the most recent run if available."""

    if not is_langfuse_enabled():
        return None

    handler = _get_or_create_langfuse_handler(agent_or_wrapper)
    if not handler:
        return None

    trace_id = getattr(handler, "last_trace_id", None)
    if not trace_id:
        return None

    client = get_langfuse_client()
    if not client:
        return None

    try:
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None

_TRUTHY = {"1", "true", "yes", "on"}
_LANGFUSE_HANDLER_ATTR = "_langfuse_callback_handler"


def _clean_trace_segment(value: Optional[str], *, fallback: str) -> str:
    """Normalize trace name segments so they are concise and URL-friendly."""
    text = (value or fallback).strip()
    if not text:
        text = fallback
    text = re.sub(r"\s+", "_", text)
    text = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in text)
    text = text.strip("-_") or fallback
    return text[:120]


def _build_trace_name(
    model_name: Optional[str],
    scenario_name: Optional[str],
    run_number: Optional[Union[int, str]],
) -> str:
    """Return <model>-<scenario>-<run_number> using sanitized segments."""
    model_segment = _clean_trace_segment(model_name, fallback="model")
    scenario_segment = _clean_trace_segment(scenario_name, fallback="scenario")
    if isinstance(run_number, str):
        run_text = run_number.strip() or "1"
    elif isinstance(run_number, int):
        run_text = str(run_number)
    else:
        run_text = "1"
    run_segment = _clean_trace_segment(run_text, fallback="1")
    return f"{model_segment}-{scenario_segment}-{run_segment}"


def _get_current_trace_url() -> Optional[str]:
    """Get LangSmith trace URL for the current run using run tree.
    
    Returns:
        Optional[str]: LangSmith trace URL (shareable link) or None if:
            - No active run tree context
            - Run hasn't been persisted to LangSmith yet
            - Tracing is disabled
    """
    try:
        run_tree = get_current_run_tree()
        if not run_tree or not run_tree.id:
            return None
        
        try:
            client = Client()
            run = client.read_run(str(run_tree.id))
            
            if run and hasattr(run, "url") and run.url:
                return run.url
        except Exception:
            pass
        
        if hasattr(run_tree, "trace_url") and run_tree.trace_url:
            return run_tree.trace_url
        
        return None
    except Exception:
        return None


def configure_langsmith(
    project_name: Optional[str] = None,
    api_key: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> dict[str, str]:
    """Configure LangSmith tracing for observability.
    
    LangSmith provides:
    - Full trace visibility of LLM calls and tool executions
    - Automatic capture of prompts, responses, and tool calls
    - Performance metrics and debugging
    - Conversation history visualization in UI
    
    Since we use LangChain, tracing is AUTOMATIC once environment
    variables are set. No code changes needed!
    
    Args:
        project_name (Optional[str]): LangSmith project name shown in UI. 
            Defaults to "MCP_BENCHMARK" if not set.
        api_key (Optional[str]): LangSmith API key from smith.langchain.com. 
            If None, uses LANGCHAIN_API_KEY environment variable.
        enabled (Optional[bool]): Whether to enable tracing. If None, reads 
            from LANGCHAIN_TRACING_V2 environment variable.
        
    Returns:
        dict[str, str]: Dictionary of environment variables that were set 
            (LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT).
            
    Raises:
        ValueError: If enabled=True but no API key is available.
        
    Environment Variables (see docs):
        LANGCHAIN_TRACING_V2: Enable LangSmith tracing (true/false)
        LANGCHAIN_API_KEY: Your LangSmith API key from smith.langchain.com
        LANGCHAIN_PROJECT: Project name in LangSmith UI (optional)
        
    Example:
        >>> # Enable tracing for benchmark runs
        >>> configure_langsmith(
        ...     project_name="my-benchmark-run",
        ...     enabled=True
        ... )
        >>> # Now all LangChain calls are automatically traced!
        
        >>> # Or set via environment before running
        >>> os.environ["LANGCHAIN_TRACING_V2"] = "true"
        >>> os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
        >>> os.environ["LANGCHAIN_PROJECT"] = "mcp-benchmark"
        
    Reference:
        https://docs.langchain.com/langsmith/observability-quickstart
    """
    env_vars = {}
    
    if enabled is None:
        enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    
    if enabled:
        env_vars["LANGCHAIN_TRACING_V2"] = "true"
        
        if api_key:
            env_vars["LANGCHAIN_API_KEY"] = api_key
        elif not os.environ.get("LANGCHAIN_API_KEY"):
            raise ValueError(
                "LangSmith API key required. Set LANGCHAIN_API_KEY environment variable "
                "or pass api_key parameter. Get your key from https://smith.langchain.com"
            )
        
        if project_name:
            env_vars["LANGCHAIN_PROJECT"] = project_name
        elif not os.environ.get("LANGCHAIN_PROJECT"):
            env_vars["LANGCHAIN_PROJECT"] = "MCP_BENCHMARK"
    else:
        env_vars["LANGCHAIN_TRACING_V2"] = "false"
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars


def get_langsmith_client() -> Optional[Client]:
    """Get LangSmith client if tracing is enabled.
    
    Returns:
        LangSmith Client instance or None if tracing is disabled
        
    Example:
        >>> client = get_langsmith_client()
        >>> if client:
        ...     runs = client.list_runs(project_name="my-project")
    """
    if os.environ.get("LANGCHAIN_TRACING_V2", "").lower() != "true":
        return None
    
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return None
    
    return Client(api_key=api_key)


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is currently enabled.
    
    Returns:
        True if tracing is enabled, False otherwise
    """
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_trace_url(run_id: str, project_name: Optional[str] = None) -> Optional[str]:
    """Get the correct LangSmith trace URL for a specific run ID.
    
    Fetches the run from LangSmith API to get the proper shareable URL.
    
    Args:
        run_id (str): LangChain run ID (UUID) to look up.
        project_name (Optional[str]): Optional LangSmith project name. If None, 
            uses LANGCHAIN_PROJECT environment variable or "default".
        
    Returns:
        Optional[str]: Shareable URL to view trace in LangSmith UI (e.g., 
            "https://smith.langchain.com/public/abc-123/r"), or None if:
            - Tracing is disabled
            - Run not found
            - API error
        
    Example:
        >>> url = get_trace_url("abc-123-def")
        >>> # Returns: https://smith.langchain.com/public/abc-123-def/r
    """
    if not is_tracing_enabled():
        return None
    
    try:
        client = get_langsmith_client()
        if not client:
            return None
        
        run = client.read_run(run_id)
        if run and run.url:
            return run.url
    except Exception:
        pass
    
    project = project_name or os.environ.get("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/public/{project}/r/{run_id}"


def print_trace_summary(project_name: Optional[str] = None, limit: int = 10) -> None:
    """Print recent traces from LangSmith with clickable URLs to console.
    
    Useful for debugging and monitoring - shows recent runs with their URLs,
    status, and timestamps.
    
    Args:
        project_name (Optional[str]): LangSmith project to query. If None, 
            uses LANGCHAIN_PROJECT environment variable or "default".
        limit (int): Number of recent runs to display. Defaults to 10.
        
    Example:
        >>> print_trace_summary("mcp-benchmark", limit=5)
        Recent Traces (mcp-benchmark):
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ✓ ChatXAI - 11/6/2025 6:00:25 (10.64s)
          https://smith.langchain.com/...
        ✗ find_assignable_users - 11/6/2025 6:00:13 (0.03s)
          https://smith.langchain.com/...
    """
    if not is_tracing_enabled():
        print("LangSmith tracing is not enabled")
        return
    
    client = get_langsmith_client()
    if not client:
        print("Cannot connect to LangSmith client")
        return
    
    project = project_name or os.environ.get("LANGCHAIN_PROJECT", "default")
    
    print(f"\nRecent Traces ({project}):")
    print("━" * 80)
    
    try:
        runs = list(client.list_runs(project_name=project, limit=limit))
        
        for run in runs:
            status = "✓" if run.error is None else "✗"
            duration = f"{run.total_tokens if hasattr(run, 'total_tokens') else '?'} tokens"
            
            print(f"{status} {run.name} - {run.start_time.strftime('%m/%d/%Y %H:%M:%S') if run.start_time else '?'}")
            print(f"  {run.url}")
            print()
    except Exception as e:
        print(f"Error fetching runs: {e!r}")


class TracingAgent:
    """Wrapper that adds LangSmith and Langfuse tracing to any agent.
    
    This decorator pattern keeps the Agent class clean while providing
    full observability when needed. Handles both LangSmith and Langfuse
    tracing without requiring any changes to the agent implementation.
    
    Usage:
        agent = ClaudeAgent()
        traced_agent = with_tracing(agent)
        result = await traced_agent.run(task)
    """
    
    def __init__(self, agent: "Agent"):
        """Wrap an agent with tracing capabilities.
        
        Args:
            agent (Agent): The agent instance to wrap (ClaudeAgent, GPTAgent, etc.).
                All method calls are delegated to this wrapped agent.
        """
        self._agent = agent
        self._langfuse_handler: Optional[Any] = None
        
        if is_langfuse_enabled() and LangchainCallbackHandler is not None:
            try:
                self._langfuse_handler = LangchainCallbackHandler()
            except Exception:
                self._langfuse_handler = None
        
        self.__dict__.update({k: v for k, v in agent.__dict__.items() if not k.startswith('_')})
    
    def __getattr__(self, name: str):
        """Delegate attribute access to wrapped agent."""
        return getattr(self._agent, name)
    
    async def run(
        self,
        task: "Task",
        max_steps: int = 1000,
        *,
        run_context: Optional["RunContext"] = None,
    ) -> "Result":
        """Run agent with LangSmith tracing.

        Creates a parent trace that groups all LLM calls and tool executions.
        Metadata includes model, scenario, database_id, session_id for filtering.
        """
        langsmith_enabled = is_tracing_enabled()
        langfuse_enabled = is_langfuse_enabled()

        if not langsmith_enabled and not langfuse_enabled:
            return await self._agent.run(task, max_steps, run_context=run_context)

        model_name = self._agent.model if hasattr(self._agent, "model") else "unknown"  # type: ignore[attr-defined]

        scenario_name = "task"
        run_number: Optional[Union[int, str]] = None
        if task and hasattr(task, "metadata") and task.metadata:
            scenario_name = task.metadata.get("scenario_name", scenario_name)
            run_number = task.metadata.get("run_number")
        
        trace_name = _build_trace_name(model_name, scenario_name, run_number)
        
        db_id = run_context.database_id if run_context else (task.database_id if task and task.database_id else None)
        
        session_id = None
        thread_id = None
        if task and task.metadata:
            session_id = task.metadata.get("session_id")
            thread_id = task.metadata.get("thread_id")
            
            if not thread_id and session_id:
                thread_id = f"{session_id}_{model_name}_{scenario_name}_run{run_number or 1}"
        
        trace_metadata: dict[str, str] = {
            "model": str(model_name),
            "scenario": str(scenario_name),
            "prompt_preview": task.prompt[:100] if task and task.prompt else "N/A",
            "trace_name": trace_name,
        }
        if db_id is not None:
            trace_metadata["database_id"] = str(db_id)
        if run_number is not None:
            trace_metadata["run_number"] = str(run_number)
        if session_id:
            trace_metadata["session_id"] = session_id
        if thread_id:
            trace_metadata["thread_id"] = thread_id
        
        if langfuse_enabled and not langsmith_enabled:
            langfuse_client = get_langfuse_client()
            if langfuse_client:
                input_data = {
                    "prompt": task.prompt[:500] if task and task.prompt else "N/A",
                    "database_id": db_id,
                }
                
                if self._langfuse_handler:
                    self._agent._tracing_callbacks = [self._langfuse_handler]
                
                with langfuse_client.start_as_current_span(
                    name=trace_name,
                    input=input_data,
                    metadata=trace_metadata,
                ) as span:
                    span.update_trace(
                        name=trace_name,
                        session_id=session_id,
                        metadata=trace_metadata,
                    )
                    
                    result = await self._agent.run(task, max_steps, run_context=run_context)
                    
                    output_data = {
                        "success": result.success,
                        "steps": result.metadata.get("steps", 0),
                        "total_tokens": result.metadata.get("total_tokens", 0),
                    }
                    span.update(output=output_data)
                    
                    if not result.langfuse_url:
                        try:
                            trace_id = None
                            if hasattr(span, "trace_id"):
                                trace_id = span.trace_id  # type: ignore[attr-defined]
                            elif hasattr(span, "_trace_id"):
                                trace_id = span._trace_id  # type: ignore[attr-defined]
                            elif hasattr(span, "get_trace_id"):
                                trace_id = span.get_trace_id()  # type: ignore[attr-defined]
                            
                            if trace_id:
                                langfuse_client.flush()
                                result.langfuse_url = langfuse_client.get_trace_url(trace_id=trace_id)
                        except Exception as e:
                            pass
                    
                    return result
        
        if not langsmith_enabled:
            if self._langfuse_handler:
                self._agent._tracing_callbacks = [self._langfuse_handler]
            
            result = await self._agent.run(task, max_steps, run_context=run_context)
            self._attach_external_trace_urls(result)
            return result

        if langfuse_enabled:
            langfuse_client = get_langfuse_client()
            if langfuse_client:
                input_data = {
                    "prompt": task.prompt[:500] if task and task.prompt else "N/A",
                    "database_id": db_id,
                }
                
                if self._langfuse_handler:
                    self._agent._tracing_callbacks = [self._langfuse_handler]
                
                with langfuse_client.start_as_current_span(
                    name=trace_name,
                    input=input_data,
                    metadata=trace_metadata,
                ) as span:
                    span.update_trace(
                        name=trace_name,
                        session_id=session_id,
                        metadata=trace_metadata,
                    )
                    
                    @traceable(name=trace_name, run_type="chain", metadata=trace_metadata)
                    async def _traced_run():
                        result = await self._agent.run(task, max_steps, run_context=run_context)
                        result.langsmith_url = _get_current_trace_url()
                        
                        output_data = {
                            "success": result.success,
                            "steps": result.metadata.get("steps", 0),
                            "total_tokens": result.metadata.get("total_tokens", 0),
                        }
                        span.update(output=output_data)
                        
                        if not result.langfuse_url:
                            try:
                                trace_id = None
                                if hasattr(span, "trace_id"):
                                    trace_id = span.trace_id  # type: ignore[attr-defined]
                                elif hasattr(span, "_trace_id"):
                                    trace_id = span._trace_id  # type: ignore[attr-defined]
                                elif hasattr(span, "get_trace_id"):
                                    trace_id = span.get_trace_id()  # type: ignore[attr-defined]
                                
                                if trace_id:
                                    langfuse_client.flush()
                                    result.langfuse_url = langfuse_client.get_trace_url(trace_id=trace_id)
                            except Exception as e:
                                pass
                        
                        return result
                    
                    return await _traced_run()

        if self._langfuse_handler:
            self._agent._tracing_callbacks = [self._langfuse_handler]
        
        @traceable(name=trace_name, run_type="chain", metadata=trace_metadata)
        async def _traced_run():
            result = await self._agent.run(task, max_steps, run_context=run_context)
            result.langsmith_url = _get_current_trace_url()
            self._attach_external_trace_urls(result)
            return result
        
        return await _traced_run()
    
    async def initialize(self, *args, **kwargs):
        """Delegate to wrapped agent."""
        return await self._agent.initialize(*args, **kwargs)
    
    async def cleanup(self):
        """Delegate to wrapped agent."""
        return await self._agent.cleanup()
    
    def get_available_tools(self):
        """Delegate to wrapped agent."""
        return self._agent.get_available_tools()
    
    async def call_tools(self, *args, **kwargs):
        """Delegate to wrapped agent."""
        return await self._agent.call_tools(*args, **kwargs)

    def _attach_external_trace_urls(self, result: "Result") -> None:
        """Attach trace URLs from wrapper's handlers to result."""
        langfuse_url = get_langfuse_trace_url(self)
        if langfuse_url and not result.langfuse_url:
            result.langfuse_url = langfuse_url


def with_tracing(agent: AgentT) -> Union[AgentT, TracingAgent]:
    """Conditionally wrap an agent with LangSmith tracing.
    
    This is a convenience function that returns the agent as-is if tracing
    is disabled, or wraps it with TracingAgent if tracing is enabled. Allows
    for opt-in tracing without code changes.
    
    Args:
        agent (AgentT): Agent instance to potentially wrap. Type is preserved 
            when tracing is disabled.
        
    Returns:
        Union[AgentT, TracingAgent]: TracingAgent wrapper if tracing is enabled 
            (LANGCHAIN_TRACING_V2=true), otherwise the original agent unchanged.
        
    Example:
        >>> agent = create_agent("gpt-4o")
        >>> agent = with_tracing(agent)  # Auto-wraps if LANGCHAIN_TRACING_V2=true
        >>> result = await agent.run(task)
    """
    if isinstance(agent, TracingAgent):
        return agent

    if is_tracing_enabled() or is_langfuse_enabled():
        return TracingAgent(agent)  
    return agent


__all__ = [
    "configure_langfuse",
    "configure_langsmith",
    "get_langfuse_client",
    "get_langfuse_trace_url",
    "get_langsmith_client",
    "is_langfuse_enabled",
    "is_tracing_enabled",
    "get_trace_url",
    "print_trace_summary",
    "TracingAgent",
    "with_tracing",
]
