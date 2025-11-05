"""Retry utilities with exponential backoff."""

import asyncio
import random
from typing import Any, Callable, Optional, TypeVar

import httpx

try:
    from anthropic import errors as anthropic_errors  # type: ignore
except ImportError:
    anthropic_errors = None  # type: ignore

T = TypeVar("T")

_BASE_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 30.0
_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def compute_retry_delay(attempt: int, base_delay: float = _BASE_RETRY_DELAY_SECONDS) -> float:
    """Calculate exponential backoff with jitter.

    Args:
        attempt: Retry attempt number (1-indexed)
        base_delay: Base delay in seconds

    Returns:
        Delay in seconds
    """
    exponential = base_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0, base_delay)
    return min(_MAX_RETRY_DELAY_SECONDS, exponential + jitter)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of HTTP status code from exception."""
    for attr in ("status_code", "status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

    return None


def should_retry_error(exc: Exception) -> bool:
    """Determine if an exception is likely transient and should be retried."""
    status_code = _extract_status_code(exc)
    if status_code in _TRANSIENT_STATUS_CODES:
        return True

    # Anthropic-specific errors
    if anthropic_errors:
        if hasattr(anthropic_errors, "RateLimitError") and isinstance(
            exc, getattr(anthropic_errors, "RateLimitError")
        ):
            return True
        if hasattr(anthropic_errors, "InternalServerError") and isinstance(
            exc, getattr(anthropic_errors, "InternalServerError")
        ):
            return True
        if hasattr(anthropic_errors, "APIError") and isinstance(
            exc, getattr(anthropic_errors, "APIError")
        ):
            if status_code in _TRANSIENT_STATUS_CODES:
                return True

    # HTTP errors
    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and status_code in _TRANSIENT_STATUS_CODES:
        return True

    # Check message for hints
    message = str(exc).lower()
    return any(
        hint in message
        for hint in (
            "temporarily unavailable",
            "overloaded",
            "rate limit",
            "please retry",
        )
    )


async def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 2,
    timeout_seconds: Optional[float] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """Retry a function with exponential backoff.

    Args:
        func: Async function to call
        max_retries: Maximum number of retries
        timeout_seconds: Optional timeout per attempt
        on_retry: Optional callback (attempt, exception, delay) called before retrying

    Returns:
        Result from func

    Raises:
        Last exception if all retries exhausted
    """
    for attempt in range(1, max_retries + 1):
        try:
            if timeout_seconds:
                return await asyncio.wait_for(func(), timeout=timeout_seconds)
            return await func()
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as timeout_exc:
            if attempt == max_retries:
                raise TimeoutError(
                    f"Operation timed out after {timeout_seconds} seconds"
                ) from timeout_exc
            delay = compute_retry_delay(attempt)
            if on_retry:
                on_retry(attempt, timeout_exc, delay)
            await asyncio.sleep(delay)
        except Exception as exc:
            if not should_retry_error(exc) or attempt == max_retries:
                raise
            delay = compute_retry_delay(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            await asyncio.sleep(delay)

    raise RuntimeError("Retry logic exhausted")  # Should never reach here

