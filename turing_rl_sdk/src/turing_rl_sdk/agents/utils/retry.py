"""Retry utilities with exponential backoff."""

import asyncio
import random
from typing import Any, Callable, Optional, TypeVar

import httpx

from ..constants import (
    RETRY_BASE_DELAY_SECONDS,
    RETRY_DEFAULT_MAX_ATTEMPTS,
    RETRY_EXPONENTIAL_BASE,
    RETRY_MAX_DELAY_SECONDS,
    RETRY_TRANSIENT_STATUS_CODES,
)

try:
    from anthropic import errors as anthropic_errors  # type: ignore
except ImportError:
    anthropic_errors = None  # type: ignore

T = TypeVar("T")


def compute_retry_delay(attempt: int, base_delay: float = RETRY_BASE_DELAY_SECONDS) -> float:
    """Calculate exponential backoff delay with random jitter.

    Args:
        attempt (int): Retry attempt number, 1-indexed (1 = first retry).
        base_delay (float): Base delay in seconds, used for both exponential 
            calculation and jitter range. Defaults to RETRY_BASE_DELAY_SECONDS.

    Returns:
        float: Calculated delay in seconds, capped at RETRY_MAX_DELAY_SECONDS. 
            Formula: min(MAX, base * (EXPONENTIAL_BASE^(attempt-1)) + random_jitter).
    """
    exponential = base_delay * (RETRY_EXPONENTIAL_BASE ** (attempt - 1))
    jitter = random.uniform(0, base_delay)
    return min(RETRY_MAX_DELAY_SECONDS, exponential + jitter)


def _extract_status_code(exc: Exception) -> Optional[int]:
    """Best-effort extraction of HTTP status code from exception.
    
    Args:
        exc (Exception): Exception to extract status code from, typically 
            from HTTP clients or API libraries.
    
    Returns:
        Optional[int]: HTTP status code if found, None otherwise. Checks 
            common attribute names like status_code, status, code, and 
            response.status_code.
    """
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
    """Determine if an exception is likely transient and should be retried.
    
    Args:
        exc (Exception): Exception to evaluate for retry eligibility.
    
    Returns:
        bool: True if the error appears transient (rate limits, server errors, 
            network issues, overload) and should be retried. False if the error 
            is likely permanent (bad request, auth failure, etc.). Checks:
            - HTTP status codes (429, 500, 502, 503, 504)
            - Anthropic-specific errors (RateLimitError, InternalServerError)
            - HTTPX errors (RequestError, HTTPStatusError)
            - Error message hints ("rate limit", "overloaded", etc.)
    """
    status_code = _extract_status_code(exc)
    if status_code in RETRY_TRANSIENT_STATUS_CODES:
        return True

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
            if status_code in RETRY_TRANSIENT_STATUS_CODES:
                return True

    if isinstance(exc, httpx.RequestError):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and status_code in RETRY_TRANSIENT_STATUS_CODES:
        return True

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
    max_retries: int = RETRY_DEFAULT_MAX_ATTEMPTS,
    timeout_seconds: Optional[float] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """Retry an async function with exponential backoff for transient errors.

    Args:
        func (Callable[..., Any]): Async callable (no arguments) to execute. 
            Should be a lambda or partial function wrapping the actual call.
        max_retries (int): Maximum number of retry attempts. Must be at least 1. 
            Total calls = max_retries (initial attempt + retries).
        timeout_seconds (Optional[float]): Optional timeout in seconds for each 
            individual attempt. If None, no timeout is applied.
        on_retry (Optional[Callable[[int, Exception, float], None]]): Optional 
            callback invoked before each retry with (attempt_number, exception, 
            delay_seconds). Useful for logging retry attempts.

    Returns:
        Any: The successful result from func().

    Raises:
        ValueError: If max_retries is less than 1.
        asyncio.CancelledError: If the operation is cancelled (not retried).
        TimeoutError: If all attempts timeout and max_retries is exhausted.
        Exception: The last exception encountered if all retries are exhausted 
            and the error is non-transient or max attempts reached.
    """
    if max_retries < 1:
        raise ValueError(
            f"max_retries must be at least 1, got {max_retries}. "
            "Provide a positive integer for retry attempts."
        )
    
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

