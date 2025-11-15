"""CLI configuration and defaults."""

from mcp_benchmark_sdk import MCPConfig

# Default JIRA MCP configuration
DEFAULT_JIRA_MCP = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http",
)

# Default SQL runner URL for verifiers
DEFAULT_SQL_RUNNER_URL = "http://localhost:8015/api/sql-runner"

# Maximum concurrent runs to prevent file descriptor exhaustion
MAX_CONCURRENT_RUNS = 20

# Results directory
RESULTS_ROOT = "results"

# Session manifest filename
SESSION_MANIFEST_FILENAME = "session.json"

