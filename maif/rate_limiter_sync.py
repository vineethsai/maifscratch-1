"""
Synchronous Rate Limiting Support for MAIF
=========================================

Provides synchronous rate limiting capabilities alongside the existing async implementation.
Uses thread-safe token bucket algorithm for non-async contexts.
"""

import time
import threading
import functools
import asyncio
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .rate_limiter import RateLimitExceeded, RateLimitConfig


@dataclass
class SyncTokenBucket:
    """Thread-safe token bucket for synchronous rate limiting."""

    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_refill = time.time()

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens from the bucket."""
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on refill rate
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class SyncRateLimiter:
    """Synchronous rate limiter implementation."""

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.buckets: Dict[str, SyncTokenBucket] = {}
        self.global_bucket = SyncTokenBucket(
            capacity=self.config.requests_per_second * 10,
            refill_rate=self.config.requests_per_second,
        )
        self._lock = threading.RLock()
        self._request_history = defaultdict(list)

    def check_rate_limit(self, identifier: str = "global", cost: float = 1.0) -> bool:
        """Check if request is within rate limits."""
        # Check burst protection
        if not self._check_burst_protection(identifier):
            return False

        # Check global rate limit
        if not self.global_bucket.consume(cost):
            return False

        # Check identifier-specific rate limit
        with self._lock:
            if identifier not in self.buckets:
                self.buckets[identifier] = SyncTokenBucket(
                    capacity=self.config.requests_per_second * 2,
                    refill_rate=self.config.requests_per_second / 2,
                )

            bucket = self.buckets[identifier]
            return bucket.consume(cost)

    def _check_burst_protection(self, identifier: str) -> bool:
        """Check burst protection limits."""
        now = time.time()

        with self._lock:
            # Clean old entries
            self._request_history[identifier] = [
                t
                for t in self._request_history[identifier]
                if now - t < self.config.burst_window_seconds
            ]

            # Check burst limit
            if len(self._request_history[identifier]) >= self.config.burst_size:
                return False

            # Record this request
            self._request_history[identifier].append(now)
            return True

    def wait_if_needed(
        self, identifier: str = "global", cost: float = 1.0, max_wait: float = 60.0
    ) -> float:
        """Wait until rate limit allows the request."""
        start_time = time.time()

        while True:
            if self.check_rate_limit(identifier, cost):
                return time.time() - start_time

            # Check if we've waited too long
            if time.time() - start_time > max_wait:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {identifier} after waiting {max_wait}s"
                )

            # Wait a bit before retrying
            time.sleep(0.1)

    def get_wait_time(self, identifier: str = "global", cost: float = 1.0) -> float:
        """Get estimated wait time until request would be allowed."""
        # Check current tokens
        with self._lock:
            if identifier in self.buckets:
                bucket = self.buckets[identifier]
                if bucket.tokens >= cost:
                    return 0.0

                # Calculate wait time
                tokens_needed = cost - bucket.tokens
                wait_time = tokens_needed / bucket.refill_rate
                return wait_time

        return 0.0

    def reset(self, identifier: Optional[str] = None):
        """Reset rate limiter state."""
        with self._lock:
            if identifier:
                if identifier in self.buckets:
                    self.buckets[identifier].tokens = self.buckets[identifier].capacity
                self._request_history[identifier].clear()
            else:
                # Reset all
                for bucket in self.buckets.values():
                    bucket.tokens = bucket.capacity
                self.global_bucket.tokens = self.global_bucket.capacity
                self._request_history.clear()


def rate_limit_sync(
    requests_per_second: float = 10,
    burst_size: int = 20,
    cost: float = 1.0,
    identifier_func: Optional[Callable] = None,
    on_limit_exceeded: Optional[Callable] = None,
    wait: bool = False,
    max_wait: float = 60.0,
):
    """
    Decorator for synchronous rate limiting.

    Args:
        requests_per_second: Sustained request rate
        burst_size: Maximum burst size
        cost: Cost of this operation
        identifier_func: Function to extract identifier from arguments
        on_limit_exceeded: Callback when rate limit is exceeded
        wait: Whether to wait for rate limit to clear
        max_wait: Maximum time to wait
    """
    config = RateLimitConfig(
        requests_per_second=requests_per_second, burst_size=burst_size
    )

    limiter = SyncRateLimiter(config)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = "global"

            # Check or wait for rate limit
            if wait:
                wait_time = limiter.wait_if_needed(identifier, cost, max_wait)
                if wait_time > 0:
                    print(f"Rate limit: waited {wait_time:.2f}s")
            else:
                if not limiter.check_rate_limit(identifier, cost):
                    if on_limit_exceeded:
                        return on_limit_exceeded(*args, **kwargs)
                    else:
                        raise RateLimitExceeded(f"Rate limit exceeded for {identifier}")

            # Execute function
            return func(*args, **kwargs)

        # Add utility methods
        wrapper.reset_rate_limit = limiter.reset
        wrapper.get_wait_time = limiter.get_wait_time
        wrapper.limiter = limiter

        return wrapper

    return decorator


class SyncAdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response times."""

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 1000.0,
        target_latency: float = 1.0,
    ):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_latency = target_latency

        self.limiter = SyncRateLimiter(
            RateLimitConfig(requests_per_second=initial_rate)
        )

        self._latency_window = []
        self._window_size = 100
        self._lock = threading.RLock()
        self._last_adjustment = time.time()
        self._adjustment_interval = 5.0  # Adjust every 5 seconds

    def record_latency(self, latency: float):
        """Record observed latency and adjust rate if needed."""
        with self._lock:
            self._latency_window.append(latency)
            if len(self._latency_window) > self._window_size:
                self._latency_window.pop(0)

            # Check if we should adjust
            now = time.time()
            if now - self._last_adjustment > self._adjustment_interval:
                self._adjust_rate()
                self._last_adjustment = now

    def _adjust_rate(self):
        """Adjust rate based on observed latencies."""
        if len(self._latency_window) < 10:
            return

        avg_latency = sum(self._latency_window) / len(self._latency_window)

        # Adjust rate based on latency
        if avg_latency > self.target_latency * 1.5:
            # Reduce rate
            self.current_rate = max(self.min_rate, self.current_rate * 0.9)
        elif avg_latency < self.target_latency * 0.5:
            # Increase rate
            self.current_rate = min(self.max_rate, self.current_rate * 1.1)

        # Update limiter
        self.limiter.config.requests_per_second = self.current_rate
        self.limiter.global_bucket.refill_rate = self.current_rate

    def check_and_record(self, identifier: str = "global", cost: float = 1.0) -> bool:
        """Check rate limit and prepare to record latency."""
        return self.limiter.check_rate_limit(identifier, cost)


# Example composite rate limiter combining sync and async
class HybridRateLimiter:
    """Rate limiter that supports both sync and async operations."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.sync_limiter = SyncRateLimiter(config)

        # Import async limiter only if needed
        self._async_limiter = None

    def check_sync(self, identifier: str = "global", cost: float = 1.0) -> bool:
        """Synchronous rate limit check."""
        return self.sync_limiter.check_rate_limit(identifier, cost)

    async def check_async(self, identifier: str = "global", cost: float = 1.0) -> bool:
        """Asynchronous rate limit check."""
        if self._async_limiter is None:
            from .rate_limiter import RateLimiter

            self._async_limiter = RateLimiter(self.config)

        return await self._async_limiter.check_rate_limit(identifier, cost)

    def wait_sync(
        self, identifier: str = "global", cost: float = 1.0, max_wait: float = 60.0
    ) -> float:
        """Synchronous wait for rate limit."""
        return self.sync_limiter.wait_if_needed(identifier, cost, max_wait)

    async def wait_async(
        self, identifier: str = "global", cost: float = 1.0, max_wait: float = 60.0
    ) -> float:
        """Asynchronous wait for rate limit."""
        if self._async_limiter is None:
            from .rate_limiter import RateLimiter

            self._async_limiter = RateLimiter(self.config)

        return await self._async_limiter.wait_if_needed(identifier, cost, max_wait)


# Update the existing rate_limit decorator to support both sync and async
def universal_rate_limit(
    requests_per_second: float = 10,
    burst_size: int = 20,
    cost: float = 1.0,
    identifier_func: Optional[Callable] = None,
    wait: bool = False,
):
    """
    Universal rate limit decorator that works with both sync and async functions.
    """
    import asyncio
    import inspect

    config = RateLimitConfig(
        requests_per_second=requests_per_second, burst_size=burst_size
    )

    hybrid_limiter = HybridRateLimiter(config)

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            # Async function
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                identifier = (
                    identifier_func(*args, **kwargs) if identifier_func else "global"
                )

                if wait:
                    await hybrid_limiter.wait_async(identifier, cost)
                else:
                    if not await hybrid_limiter.check_async(identifier, cost):
                        raise RateLimitExceeded(f"Rate limit exceeded for {identifier}")

                return await func(*args, **kwargs)

            return async_wrapper
        else:
            # Sync function
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                identifier = (
                    identifier_func(*args, **kwargs) if identifier_func else "global"
                )

                if wait:
                    hybrid_limiter.wait_sync(identifier, cost)
                else:
                    if not hybrid_limiter.check_sync(identifier, cost):
                        raise RateLimitExceeded(f"Rate limit exceeded for {identifier}")

                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Example 1: Simple sync rate limiting
    @rate_limit_sync(requests_per_second=2, wait=True)
    def slow_api_call(param: str) -> str:
        print(f"Calling API with {param}")
        return f"Result for {param}"

    # Example 2: Adaptive rate limiting
    adaptive_limiter = SyncAdaptiveRateLimiter(initial_rate=10, target_latency=0.5)

    def adaptive_call(data: Any) -> Any:
        start = time.time()

        if not adaptive_limiter.check_and_record("api", 1.0):
            raise RateLimitExceeded("Adaptive rate limit exceeded")

        # Simulate work
        time.sleep(0.3)
        result = f"Processed {data}"

        # Record latency
        adaptive_limiter.record_latency(time.time() - start)

        return result

    # Example 3: Universal decorator
    @universal_rate_limit(requests_per_second=5)
    def sync_function(x: int) -> int:
        return x * 2

    @universal_rate_limit(requests_per_second=5)
    async def async_function(x: int) -> int:
        await asyncio.sleep(0.1)
        return x * 2
