"""SDK-wide constants and configuration values."""

import os


def _get_float_env(key: str, default: float) -> float:
    """Get float from environment variable with fallback to default."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


DATABASE_VERIFIER_TIMEOUT_SECONDS = _get_float_env("TURING_RL_DATABASE_VERIFIER_TIMEOUT_SECONDS", 30.0)
HTTP_CLIENT_TIMEOUT_SECONDS = _get_float_env("TURING_RL_HTTP_CLIENT_TIMEOUT_SECONDS", 30.0)

