"""CLI configuration and defaults."""

from turing_rl_sdk.agents.mcp import MCPConfig

# Default JIRA MCP configuration
DEFAULT_JIRA_MCP = MCPConfig(
    name="jira",
    url="http://localhost:8015/mcp",
    transport="streamable_http",
)

# Maximum concurrent runs to prevent file descriptor exhaustion
MAX_CONCURRENT_RUNS = 20

# Results directory
RESULTS_ROOT = "results"

# Session manifest filename
SESSION_MANIFEST_FILENAME = "session.json"

