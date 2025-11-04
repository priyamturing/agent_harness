"""Adaptive rate limiting with AIMD algorithm for LLM provider calls."""

from __future__ import annotations

import asyncio
import time
from typing import Dict, Optional

_INITIAL_RATE_PER_MINUTE = 1000  # High starting rate to find natural limits
_MIN_RATE_PER_MINUTE = 1  # Never go to zero
_AIMD_ADDITIVE_INCREASE = 1  # Add 1 req/min on success
_AIMD_MULTIPLICATIVE_DECREASE = 0.25  # Reduce to 25% (more aggressive: 75% decrease)
_RATE_LIMIT_COOLDOWN_SECONDS = 5  # Brief cooldown after 429 (not a full pause)
_MIN_ADJUSTMENT_INTERVAL_SECONDS = 0.5  # Minimum time between rate adjustments


class AdaptiveRateLimiter:
    """Per-provider adaptive rate limiter using AIMD algorithm.
    
    Coordinates concurrent requests to a provider and dynamically adjusts
    the rate limit based on 429 responses using Additive Increase Multiplicative
    Decrease (AIMD) algorithm.
    """

    def __init__(self, provider_name: str):
        """Initialize rate limiter for a specific provider.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
        """
        self.provider_name = provider_name
        self._current_rate = float(_INITIAL_RATE_PER_MINUTE)
        self._lock = asyncio.Lock()
        self._last_adjustment_time = 0.0
        self._cooldown_until: Optional[float] = None
        
        # Token bucket for rate limiting
        self._tokens = float(_INITIAL_RATE_PER_MINUTE)
        self._last_refill_time = time.monotonic()
        self._max_tokens = float(_INITIAL_RATE_PER_MINUTE)

    async def _refill_tokens(self) -> None:
        """Refill tokens based on current rate and elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        
        # Add tokens based on rate (requests per minute)
        tokens_to_add = (self._current_rate / 60.0) * elapsed
        self._tokens = min(self._max_tokens, self._tokens + tokens_to_add)
        self._last_refill_time = now

    async def acquire(self) -> None:
        """Acquire permission to make a request.
        
        This will block if the rate limit has been reached. Uses token bucket
        to smooth out request rate according to current limit.
        """
        while True:
            async with self._lock:
                # Refill tokens based on elapsed time
                await self._refill_tokens()
                
                # If we have a token, consume it and proceed
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            
            # No token available, wait briefly before checking again
            # Sleep time based on how long until next token
            tokens_per_second = self._current_rate / 60.0
            if tokens_per_second > 0:
                sleep_time = min(1.0 / tokens_per_second, 1.0)
            else:
                sleep_time = 1.0
            await asyncio.sleep(sleep_time)

    async def on_success(self) -> Optional[float]:
        """Called after a successful request.
        
        Implements additive increase: increases rate by +1 req/min.
        
        Returns:
            The new rate limit if adjusted, None otherwise.
        """
        async with self._lock:
            now = time.monotonic()
            
            # Avoid adjusting too frequently
            if now - self._last_adjustment_time < _MIN_ADJUSTMENT_INTERVAL_SECONDS:
                return None
            
            old_rate = self._current_rate
            self._current_rate += _AIMD_ADDITIVE_INCREASE
            self._max_tokens = self._current_rate
            self._last_adjustment_time = now
            
            return self._current_rate if self._current_rate != old_rate else None

    async def on_rate_limited(self) -> tuple[float, float]:
        """Called when a 429 error is received.
        
        Implements multiplicative decrease: reduces rate to 25% (75% decrease)
        and clears token bucket to create brief cooldown.
        
        Returns:
            Tuple of (new_rate, cooldown_duration_seconds)
        """
        async with self._lock:
            # Aggressive multiplicative decrease: reduce to 25% of current rate
            old_rate = self._current_rate
            self._current_rate = max(
                _MIN_RATE_PER_MINUTE,
                self._current_rate * _AIMD_MULTIPLICATIVE_DECREASE
            )
            self._max_tokens = self._current_rate
            
            # Clear tokens to create brief cooldown, but don't hard-pause
            # Tokens will refill at the new (much slower) rate
            self._tokens = 0.0
            self._last_refill_time = time.monotonic()
            self._last_adjustment_time = time.monotonic()
            
            # Set cooldown period (used for logging, not enforced as hard pause)
            self._cooldown_until = time.monotonic() + _RATE_LIMIT_COOLDOWN_SECONDS
            
            return self._current_rate, _RATE_LIMIT_COOLDOWN_SECONDS

    def get_current_rate(self) -> float:
        """Get the current rate limit in requests per minute."""
        return self._current_rate


class RateLimiterManager:
    """Singleton manager for per-provider rate limiters."""

    _instance: Optional[RateLimiterManager] = None
    _lock = asyncio.Lock()

    def __new__(cls) -> RateLimiterManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._limiters: Dict[str, AdaptiveRateLimiter] = {}
            cls._instance._init_lock = asyncio.Lock()
        return cls._instance

    async def get_limiter(self, provider_name: str) -> AdaptiveRateLimiter:
        """Get or create a rate limiter for the specified provider.
        
        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic")
            
        Returns:
            AdaptiveRateLimiter instance for the provider
        """
        if provider_name not in self._limiters:
            async with self._init_lock:
                # Double-check after acquiring lock
                if provider_name not in self._limiters:
                    self._limiters[provider_name] = AdaptiveRateLimiter(provider_name)
        
        return self._limiters[provider_name]


# Global singleton instance
_rate_limiter_manager = RateLimiterManager()


async def get_rate_limiter(provider_name: str) -> AdaptiveRateLimiter:
    """Get the rate limiter for a specific provider.
    
    Args:
        provider_name: Name of the provider (e.g., "openai", "anthropic")
        
    Returns:
        AdaptiveRateLimiter instance for the provider
    """
    return await _rate_limiter_manager.get_limiter(provider_name)


__all__ = ["AdaptiveRateLimiter", "RateLimiterManager", "get_rate_limiter"]

