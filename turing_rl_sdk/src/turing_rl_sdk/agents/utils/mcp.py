"""MCP-related utility functions."""

from urllib.parse import urlparse, urlunparse


def derive_sql_runner_url(mcp_url: str) -> str:
    """Derive SQL runner URL from MCP server URL.
    
    The SQL runner is an endpoint on the same server as the MCP endpoint.
    Uses proper URL parsing to handle various URL formats robustly.
    
    Examples:
        - http://localhost:8015/mcp → http://localhost:8015/api/sql-runner
        - http://localhost:8015/ → http://localhost:8015/api/sql-runner
        - http://localhost:8015/mcp/v1 → http://localhost:8015/mcp/api/sql-runner
        - http://example.com:8080/mcp → http://example.com:8080/api/sql-runner
    
    Args:
        mcp_url (str): MCP server URL with full scheme and host, for example 
            "http://localhost:8015/mcp" or "http://example.com:8080/mcp/v1".
        
    Returns:
        str: SQL runner endpoint URL derived from the MCP URL by replacing 
            the path appropriately, e.g., "http://localhost:8015/api/sql-runner".
        
    Raises:
        ValueError: If mcp_url is empty, invalid, or missing scheme/host.
    """
    if not mcp_url or not mcp_url.strip():
        raise ValueError("mcp_url cannot be empty")
    
    try:
        parsed = urlparse(mcp_url)
    except Exception as e:
        raise ValueError(f"Invalid MCP URL: {mcp_url!r} - {e!r}") from e
    
    # Validate URL has scheme and netloc (host)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(
            f"Invalid MCP URL: {mcp_url}. "
            "URL must include scheme and host (e.g., 'http://localhost:8015/mcp')"
        )
    
    # Remove trailing slashes from path
    path = parsed.path.rstrip("/")
    
    # If path ends with /mcp, remove it to get the base path
    if path.endswith("/mcp"):
        path = path[:-4]
    
    # Construct new path with /api/sql-runner
    new_path = f"{path}/api/sql-runner" if path else "/api/sql-runner"
    
    # Reconstruct URL with new path
    new_parsed = parsed._replace(path=new_path)
    return urlunparse(new_parsed)

