"""MCP-related utility functions."""


def derive_sql_runner_url(mcp_url: str) -> str:
    """Derive SQL runner URL from MCP server URL.
    
    The SQL runner is an endpoint on the same server as the MCP endpoint.
    For example:
    - MCP URL: http://localhost:8015/mcp
    - SQL runner: http://localhost:8015/api/sql-runner
    
    Args:
        mcp_url: MCP server URL (e.g., "http://localhost:8015/mcp")
        
    Returns:
        SQL runner endpoint URL (e.g., "http://localhost:8015/api/sql-runner")
    """
    # Remove /mcp suffix if present
    if mcp_url.endswith("/mcp"):
        base_url = mcp_url[:-4]  # Remove "/mcp"
    else:
        # If no /mcp suffix, assume the whole URL is the base
        base_url = mcp_url.rstrip("/")
    
    return f"{base_url}/api/sql-runner"

