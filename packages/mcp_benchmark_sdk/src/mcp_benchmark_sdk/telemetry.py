"""LangSmith telemetry and tracing configuration.

LangSmith provides automatic tracing for LangChain applications.
See: https://docs.langchain.com/langsmith/observability-quickstart

Since we use LangChain agents (ChatAnthropic, ChatOpenAI, etc.),
tracing works automatically by setting environment variables.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, TypeVar, Union

from langsmith import Client, traceable, get_current_run_tree

if TYPE_CHECKING:
    from .agents import Agent
    from .tasks import Task, Result
    from .runtime import RunContext
    from .constants import DEFAULT_MAX_STEPS

# TypeVar for preserving agent type through tracing wrapper
AgentT = TypeVar("AgentT", bound="Agent")


def _get_current_trace_url() -> Optional[str]:
    """Get LangSmith trace URL for the current run.
    
    Returns:
        LangSmith trace URL (shareable link) or None if not available
    """
    try:
        run_tree = get_current_run_tree()
        if not run_tree or not run_tree.id:
            return None
        
        # Fetch the run from LangSmith to get the correct shareable URL
        try:
            client = Client()
            run = client.read_run(str(run_tree.id))
            
            if run and hasattr(run, "url") and run.url:
                return run.url
        except Exception:
            pass
        
        # Fallback: use the trace_url from run_tree
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
        project_name: LangSmith project name (defaults to "MCP_BENCHMARK")
        api_key: LangSmith API key (defaults to env var LANGCHAIN_API_KEY)
        enabled: Enable tracing (defaults to env var or False)
        
    Returns:
        Dict of environment variables that were set
        
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
    
    # Determine if tracing should be enabled
    if enabled is None:
        enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    
    if enabled:
        env_vars["LANGCHAIN_TRACING_V2"] = "true"
        
        # Set API key
        if api_key:
            env_vars["LANGCHAIN_API_KEY"] = api_key
        elif not os.environ.get("LANGCHAIN_API_KEY"):
            raise ValueError(
                "LangSmith API key required. Set LANGCHAIN_API_KEY environment variable "
                "or pass api_key parameter. Get your key from https://smith.langchain.com"
            )
        
        # Set project name
        if project_name:
            env_vars["LANGCHAIN_PROJECT"] = project_name
        elif not os.environ.get("LANGCHAIN_PROJECT"):
            env_vars["LANGCHAIN_PROJECT"] = "MCP_BENCHMARK"
    else:
        env_vars["LANGCHAIN_TRACING_V2"] = "false"
    
    # Apply to environment
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
    """Get the correct LangSmith trace URL for a specific run.
    
    Fetches the run from LangSmith API to get the proper shareable URL.
    
    Args:
        run_id: LangChain run ID
        project_name: Optional project name
        
    Returns:
        Shareable URL to view trace in LangSmith UI, or None if not found
        
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
        
        # Fetch the run to get the correct URL
        run = client.read_run(run_id)
        if run and run.url:
            return run.url
    except Exception:
        pass
    
    # Fallback format
    project = project_name or os.environ.get("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/public/{project}/r/{run_id}"


def print_trace_summary(project_name: Optional[str] = None, limit: int = 10) -> None:
    """Print recent traces from LangSmith with clickable URLs.
    
    Useful for debugging - shows recent runs with their URLs.
    
    Args:
        project_name: Project to query (defaults to current LANGCHAIN_PROJECT)
        limit: Number of recent runs to show
        
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
        print(f"Error fetching runs: {e}")


class TracingAgent:
    """Wrapper that adds LangSmith tracing to any agent.
    
    This decorator pattern keeps the Agent class clean while providing
    full LangSmith observability when needed.
    
    Usage:
        agent = ClaudeAgent()
        traced_agent = with_tracing(agent)
        result = await traced_agent.run(task)
    """
    
    def __init__(self, agent: "Agent"):
        """Wrap an agent with tracing capabilities.
        
        Args:
            agent: The agent to wrap
        """
        self._agent = agent
        # Delegate all attributes to wrapped agent
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
        if not is_tracing_enabled():
            # Tracing disabled - run agent directly
            return await self._agent.run(task, max_steps, run_context=run_context)
        
        # Build trace metadata
        model_name = self._agent.model if hasattr(self._agent, "model") else "unknown"  # type: ignore[attr-defined]
        
        # Extract scenario info from task metadata
        scenario_name = "task"
        run_number = None
        if task and hasattr(task, "metadata") and task.metadata:
            scenario_name = task.metadata.get("scenario_name", scenario_name)
            run_number = task.metadata.get("run_number")
        
        # Build readable trace name: "grok-4: scenario_name (run 2)"
        trace_name = f"{model_name}: {scenario_name}"
        if run_number is not None:
            trace_name = f"{trace_name} (run {run_number})"
        
        # Get database_id (from run_context if provided, or will be created by agent)
        db_id = run_context.database_id if run_context else (task.database_id if task and task.database_id else None)
        
        # Get session_id and thread_id for grouping (LangSmith Threads feature)
        session_id = None
        thread_id = None
        if task and task.metadata:
            session_id = task.metadata.get("session_id")
            thread_id = task.metadata.get("thread_id")
            
            # Generate thread_id if not provided (unique per model+scenario+run)
            if not thread_id and session_id:
                thread_id = f"{session_id}_{model_name}_{scenario_name}_run{run_number or 1}"
        
        # Create metadata for filtering/search
        trace_metadata = {
            "model": model_name,
            "scenario": scenario_name,
            "database_id": db_id,
            "prompt_preview": task.prompt[:100] if task and task.prompt else "N/A",
        }
        if run_number is not None:
            trace_metadata["run_number"] = run_number
        if session_id:
            trace_metadata["session_id"] = session_id
        if thread_id:
            trace_metadata["thread_id"] = thread_id
        
        # Use traceable wrapper with descriptive name
        @traceable(name=trace_name, run_type="chain", metadata=trace_metadata)
        async def _traced_run():
            result = await self._agent.run(task, max_steps, run_context=run_context)
            
            # Populate LangSmith trace URL in result
            result.langsmith_url = _get_current_trace_url()
            
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


def with_tracing(agent: AgentT) -> Union[AgentT, TracingAgent]:
    """Wrap an agent with LangSmith tracing.
    
    This is a convenience function that returns the agent as-is if tracing
    is disabled, or wraps it with TracingAgent if tracing is enabled.
    
    Args:
        agent: Agent to potentially wrap
        
    Returns:
        TracingAgent wrapper if tracing enabled, otherwise original agent
        
    Example:
        >>> agent = create_agent("gpt-4o")
        >>> agent = with_tracing(agent)  # Auto-wraps if LANGCHAIN_TRACING_V2=true
        >>> result = await agent.run(task)
    """
    if is_tracing_enabled():
        return TracingAgent(agent)  
    return agent


__all__ = [
    "configure_langsmith",
    "get_langsmith_client",
    "is_tracing_enabled",
    "get_trace_url",
    "print_trace_summary",
    "TracingAgent",
    "with_tracing",
]

