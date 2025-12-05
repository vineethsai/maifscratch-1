"""
Rate Limiting for MAIF Agents
=============================

Provides rate limiting capabilities to prevent abuse and manage costs.
"""

import time
import asyncio
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: int = 20
    window_seconds: int = 60

    @classmethod
    def from_production_config(cls):
        """Create RateLimitConfig from production configuration."""
        from .config import get_config

        config = get_config()

        if config.rate_limit_enabled:
            return cls(
                requests_per_second=config.requests_per_second,
                burst_size=config.burst_size,
                window_seconds=60,  # Default window
            )
        else:
            # Effectively unlimited when disabled
            return cls(
                requests_per_second=float("inf"),
                burst_size=float("inf"),
                window_seconds=60,
            )


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate  # tokens per second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket."""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now

            # Add new tokens based on time elapsed
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens to be available."""
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.rate


class RateLimiter:
    """Rate limiter for MAIF agents."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: Dict[str, TokenBucket] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for key."""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(
                rate=self.config.requests_per_second, capacity=self.config.burst_size
            )
        return self.buckets[key]

    async def check_rate_limit(self, key: str = "default", cost: int = 1) -> bool:
        """Check if request is within rate limit."""
        bucket = self.get_bucket(key)
        allowed = await bucket.acquire(cost)

        # Track request
        self.request_history[key].append(
            {"timestamp": time.time(), "allowed": allowed, "cost": cost}
        )

        if not allowed:
            logger.warning(f"Rate limit exceeded for key: {key}")

        return allowed

    async def wait_if_needed(self, key: str = "default", cost: int = 1):
        """Wait if rate limit exceeded."""
        bucket = self.get_bucket(key)
        wait_time = bucket.get_wait_time(cost)

        if wait_time > 0:
            logger.info(f"Rate limit wait: {wait_time:.2f}s for key: {key}")
            await asyncio.sleep(wait_time)

    def get_metrics(self, key: str = "default") -> Dict[str, any]:
        """Get rate limit metrics."""
        history = list(self.request_history[key])
        if not history:
            return {
                "key": key,
                "total_requests": 0,
                "allowed_requests": 0,
                "rejected_requests": 0,
                "rejection_rate": 0.0,
            }

        total = len(history)
        allowed = sum(1 for r in history if r["allowed"])
        rejected = total - allowed

        return {
            "key": key,
            "total_requests": total,
            "allowed_requests": allowed,
            "rejected_requests": rejected,
            "rejection_rate": rejected / total if total > 0 else 0.0,
            "current_tokens": self.buckets.get(key, TokenBucket(0, 0)).tokens,
        }


class CostBasedRateLimiter(RateLimiter):
    """Rate limiter that accounts for operation costs."""

    def __init__(self, config: RateLimitConfig):
        super().__init__(config)
        self.cost_multipliers = {
            "bedrock_generation": 10,
            "bedrock_embedding": 2,
            "s3_upload": 1,
            "s3_download": 1,
            "dynamodb_write": 2,
            "dynamodb_read": 1,
            "lambda_invoke": 5,
        }

    def get_operation_cost(self, operation: str) -> int:
        """Get cost multiplier for operation."""
        return self.cost_multipliers.get(operation, 1)

    async def check_operation(self, operation: str, key: str = "default") -> bool:
        """Check if operation is allowed."""
        cost = self.get_operation_cost(operation)
        return await self.check_rate_limit(key, cost)


def rate_limit(
    limiter: RateLimiter, key_func: Optional[Callable] = None, cost: int = 1
):
    """Decorator for rate limiting functions."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Determine rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            # Check rate limit
            if not await limiter.check_rate_limit(key, cost):
                raise Exception(f"Rate limit exceeded for key: {key}")

            # Execute function
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, use blocking wait
            loop = asyncio.new_event_loop()

            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            # Check rate limit
            allowed = loop.run_until_complete(limiter.check_rate_limit(key, cost))
            if not allowed:
                raise Exception(f"Rate limit exceeded for key: {key}")

            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter configured from production config."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        config = RateLimitConfig.from_production_config()
        _global_rate_limiter = CostBasedRateLimiter(config)
    return _global_rate_limiter


def reset_rate_limiter():
    """Reset rate limiter (mainly for testing)."""
    global _global_rate_limiter
    _global_rate_limiter = None


# Default rate limiter for backward compatibility
default_rate_limiter = RateLimiter(RateLimitConfig())


# Rate limit exception
class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    pass


# Production-ready decorator
def rate_limited(operation: str = "default", key_func: Optional[Callable] = None):
    """Decorator for rate limiting using production configuration."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Determine key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            # Check rate limit
            if isinstance(limiter, CostBasedRateLimiter):
                allowed = await limiter.check_operation(operation, key)
            else:
                allowed = await limiter.check_rate_limit(key)

            if not allowed:
                raise RateLimitExceeded(f"Rate limit exceeded for {operation}")

            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            raise NotImplementedError(
                "Sync rate limiting not supported. Use async functions."
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Usage example:
"""
# Configure rate limiter
rate_config = RateLimitConfig(
    requests_per_second=10,
    burst_size=20
)
limiter = RateLimiter(rate_config)

# Use as decorator
@rate_limit(limiter, key_func=lambda self: self.agent_id)
async def process_request(self, data):
    # Process data
    pass

# Use directly
if await limiter.check_rate_limit("user_123"):
    # Process request
    pass
else:
    # Handle rate limit
    pass

# Cost-based limiting
cost_limiter = CostBasedRateLimiter(rate_config)
if await cost_limiter.check_operation("bedrock_generation", "user_123"):
    # Make expensive Bedrock call
    pass
"""
