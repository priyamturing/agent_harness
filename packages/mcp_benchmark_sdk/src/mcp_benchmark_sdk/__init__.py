"""MCP Benchmark SDK - Build and run LLM agent benchmarks against MCP servers."""

__version__ = "0.1.0"

# High-level agent classes
from .agents import Agent, ClaudeAgent, GPTAgent, GeminiAgent, GrokAgent

# Task/Result types
from .tasks import Task, Result, Scenario, ScenarioPrompt

# MCP config
from .mcp import MCPConfig, MCPClientManager

# Verifiers
from .verifiers import Verifier, DatabaseVerifier, VerifierResult

# Runtime context and events
from .runtime import RunContext, RunObserver

# For advanced users
from .parsers import ResponseParser, ParsedResponse
from .tasks import AgentResponse, ToolCall, ToolResult

__all__ = [
    # Core
    "Agent",
    "Task",
    "Result",
    "RunContext",
    "RunObserver",
    # Agents
    "ClaudeAgent",
    "GPTAgent",
    "GeminiAgent",
    "GrokAgent",
    # Tasks
    "Scenario",
    "ScenarioPrompt",
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
    # Parsers
    "ResponseParser",
    "ParsedResponse",
]

