"""LangSmith telemetry and tracing configuration.

LangSmith provides automatic tracing for LangChain applications.
See: https://docs.langchain.com/langsmith/observability-quickstart

Since we use LangChain agents (ChatAnthropic, ChatOpenAI, etc.),
tracing works automatically by setting environment variables.
"""

from __future__ import annotations

import os
from typing import Optional

from langsmith import Client


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


__all__ = [
    "configure_langsmith",
    "get_langsmith_client",
    "is_tracing_enabled",
    "get_trace_url",
    "print_trace_summary",
]

