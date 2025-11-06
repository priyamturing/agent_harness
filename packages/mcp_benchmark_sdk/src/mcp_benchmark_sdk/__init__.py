"""MCP Benchmark SDK - Build and run LLM agent benchmarks against MCP servers."""

__version__ = "0.1.0"

# High-level agent classes
from .agents import Agent, ClaudeAgent, GPTAgent, GeminiAgent, GrokAgent

# Task/Result types
from .tasks import Task, Result

# MCP config
from .mcp import MCPConfig, MCPClientManager

# Verifiers
from .verifiers import Verifier, DatabaseVerifier, VerifierResult

# Runtime context and events
from .runtime import RunContext, RunObserver

# Telemetry and tracing
from .telemetry import configure_langsmith, get_langsmith_client, is_tracing_enabled, get_trace_url, print_trace_summary

# Test harness for running benchmarks
from .harness import (
    TestHarness,
    TestHarnessConfig,
    RunResult,
    HarnessLoader,
    load_harness_file,
    load_harness_directory,
    scenario_to_task,
    create_agent,
    DEFAULT_SYSTEM_PROMPT,
    Scenario,
    ScenarioPrompt,
    VerifierDefinition,
)

# For advanced users
from .parsers import ResponseParser, ParsedResponse
from .tasks import AgentResponse, ToolCall, ToolResult

# Configuration constants
from .constants import (
    DEFAULT_TOOL_CALL_LIMIT,
    DEFAULT_MAX_STEPS,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_LLM_MAX_RETRIES,
    TOOL_CALL_ID_HEX_LENGTH,
    THINKING_SAFETY_MARGIN_TOKENS,
    THINKING_DEFAULT_OUTPUT_TOKENS,
    THINKING_DEFAULT_BUDGET_TOKENS,
    REASONING_MAX_DEPTH,
    REASONING_MAX_TEXT_LENGTH,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    RETRY_DEFAULT_MAX_ATTEMPTS,
    RETRY_TRANSIENT_STATUS_CODES,
    DATABASE_VERIFIER_TIMEOUT_SECONDS,
)

__all__ = [
    # Core
    "Agent",
    "Task",
    "Result",
    "RunContext",
    "RunObserver",
    # Telemetry
    "configure_langsmith",
    "get_langsmith_client",
    "is_tracing_enabled",
    "get_trace_url",
    "print_trace_summary",
    # Agents
    "ClaudeAgent",
    "GPTAgent",
    "GeminiAgent",
    "GrokAgent",
    # Tasks
    "AgentResponse",
    "ToolCall",
    "ToolResult",
    # MCP
    "MCPConfig",
    "MCPClientManager",
    # Verifiers
    "Verifier",
    "DatabaseVerifier",
    "VerifierResult",
    # Test Harness
    "TestHarness",
    "TestHarnessConfig",
    "RunResult",
    "HarnessLoader",
    "load_harness_file",
    "load_harness_directory",
    "scenario_to_task",
    "create_agent",
    "DEFAULT_SYSTEM_PROMPT",
    "Scenario",
    "ScenarioPrompt",
    "VerifierDefinition",
    # Parsers
    "ResponseParser",
    "ParsedResponse",
    # Constants
    "DEFAULT_TOOL_CALL_LIMIT",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_LLM_TIMEOUT_SECONDS",
    "DEFAULT_LLM_MAX_RETRIES",
    "TOOL_CALL_ID_HEX_LENGTH",
    "THINKING_SAFETY_MARGIN_TOKENS",
    "THINKING_DEFAULT_OUTPUT_TOKENS",
    "THINKING_DEFAULT_BUDGET_TOKENS",
    "REASONING_MAX_DEPTH",
    "REASONING_MAX_TEXT_LENGTH",
    "RETRY_BASE_DELAY_SECONDS",
    "RETRY_MAX_DELAY_SECONDS",
    "RETRY_DEFAULT_MAX_ATTEMPTS",
    "RETRY_TRANSIENT_STATUS_CODES",
    "DATABASE_VERIFIER_TIMEOUT_SECONDS",
]

