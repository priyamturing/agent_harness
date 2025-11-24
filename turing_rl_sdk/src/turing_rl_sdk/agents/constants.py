"""Agent-specific constants and configuration values."""

import os


def _get_int_env(key: str, default: int) -> int:
    """Get integer from environment variable with fallback to default."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable with fallback to default."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_frozenset_env(key: str, default: frozenset[int]) -> frozenset[int]:
    """Get frozenset of ints from comma-separated env var."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return frozenset(int(x.strip()) for x in value.split(",") if x.strip())
    except (ValueError, AttributeError):
        return default


DEFAULT_TOOL_CALL_LIMIT = _get_int_env("TURING_RL_DEFAULT_TOOL_CALL_LIMIT", 1000)
DEFAULT_MAX_STEPS = _get_int_env("TURING_RL_DEFAULT_MAX_STEPS", 1000)

DEFAULT_LLM_TIMEOUT_SECONDS = _get_float_env("TURING_RL_DEFAULT_LLM_TIMEOUT_SECONDS", 600.0)
DEFAULT_LLM_MAX_RETRIES = _get_int_env("TURING_RL_DEFAULT_LLM_MAX_RETRIES", 3)

TOOL_CALL_ID_HEX_LENGTH = 8

THINKING_SAFETY_MARGIN_TOKENS = _get_int_env("TURING_RL_THINKING_SAFETY_MARGIN_TOKENS", 1000)
THINKING_DEFAULT_OUTPUT_TOKENS = _get_int_env("TURING_RL_THINKING_DEFAULT_OUTPUT_TOKENS", 8192)
THINKING_DEFAULT_BUDGET_TOKENS = _get_int_env("TURING_RL_THINKING_DEFAULT_BUDGET_TOKENS", 42000)

REASONING_MAX_DEPTH = _get_int_env("TURING_RL_REASONING_MAX_DEPTH", 10)
REASONING_MAX_TEXT_LENGTH = _get_int_env("TURING_RL_REASONING_MAX_TEXT_LENGTH", 1200)

RETRY_BASE_DELAY_SECONDS = _get_float_env("TURING_RL_RETRY_BASE_DELAY_SECONDS", 1.0)
RETRY_MAX_DELAY_SECONDS = _get_float_env("TURING_RL_RETRY_MAX_DELAY_SECONDS", 30.0)
RETRY_DEFAULT_MAX_ATTEMPTS = _get_int_env("TURING_RL_RETRY_DEFAULT_MAX_ATTEMPTS", 2)
RETRY_EXPONENTIAL_BASE = _get_int_env("TURING_RL_RETRY_EXPONENTIAL_BASE", 2)
RETRY_TRANSIENT_STATUS_CODES = _get_frozenset_env(
    "TURING_RL_RETRY_TRANSIENT_STATUS_CODES",
    frozenset({429, 500, 502, 503, 504})
)


