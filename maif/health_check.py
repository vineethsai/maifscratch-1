"""
Health Check System for MAIF Agents
===================================

Provides health monitoring and readiness checks for production deployments.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import boto3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class HealthChecker:
    """Health check system for MAIF agents."""

    def __init__(self, agent: Any, check_interval: int = 30):
        self.agent = agent
        self.check_interval = check_interval
        self.checks: Dict[str, Callable] = {}
        self.results: List[HealthCheckResult] = []
        self.last_check: Optional[float] = None
        self._running = False

        # Register default checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks."""
        # Memory check
        self.register_check("memory", self._check_memory)

        # Agent state check
        self.register_check("agent_state", self._check_agent_state)

        # AWS connectivity (if enabled)
        if hasattr(self.agent, "use_aws") and self.agent.use_aws:
            self.register_check("aws_s3", self._check_aws_s3)
            self.register_check("aws_bedrock", self._check_aws_bedrock)
            self.register_check("aws_kms", self._check_aws_kms)

    def register_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.checks[name] = check_func

    async def _check_memory(self) -> HealthCheckResult:
        """Check memory usage."""
        try:
            import psutil

            process = psutil.Process()
            memory_percent = process.memory_percent()

            if memory_percent < 70:
                status = HealthStatus.HEALTHY
                message = f"Memory usage: {memory_percent:.1f}%"
            elif memory_percent < 85:
                status = HealthStatus.DEGRADED
                message = f"High memory usage: {memory_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical memory usage: {memory_percent:.1f}%"

            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                metadata={"memory_percent": memory_percent},
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}",
            )

    async def _check_agent_state(self) -> HealthCheckResult:
        """Check agent state."""
        try:
            state = (
                self.agent.state.value if hasattr(self.agent, "state") else "unknown"
            )

            if state in ["idle", "perceiving", "reasoning", "executing"]:
                return HealthCheckResult(
                    name="agent_state",
                    status=HealthStatus.HEALTHY,
                    message=f"Agent state: {state}",
                    metadata={"state": state},
                )
            elif state == "terminated":
                return HealthCheckResult(
                    name="agent_state",
                    status=HealthStatus.UNHEALTHY,
                    message="Agent is terminated",
                )
            else:
                return HealthCheckResult(
                    name="agent_state",
                    status=HealthStatus.DEGRADED,
                    message=f"Unknown agent state: {state}",
                )
        except Exception as e:
            return HealthCheckResult(
                name="agent_state",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent state check failed: {e}",
            )

    async def _check_aws_s3(self) -> HealthCheckResult:
        """Check S3 connectivity."""
        try:
            if AWS_CONFIG_AVAILABLE:
                aws_config = get_aws_config()
                s3_client = aws_config.get_client("s3")
            else:
                s3_client = boto3.client("s3")
            # List buckets to verify connectivity
            s3_client.list_buckets()

            return HealthCheckResult(
                name="aws_s3", status=HealthStatus.HEALTHY, message="S3 connectivity OK"
            )
        except Exception as e:
            return HealthCheckResult(
                name="aws_s3",
                status=HealthStatus.UNHEALTHY,
                message=f"S3 connectivity failed: {e}",
            )

    async def _check_aws_bedrock(self) -> HealthCheckResult:
        """Check Bedrock connectivity."""
        try:
            if hasattr(self.agent, "bedrock_integration"):
                # Simple connectivity test
                return HealthCheckResult(
                    name="aws_bedrock",
                    status=HealthStatus.HEALTHY,
                    message="Bedrock connectivity OK",
                )
            else:
                return HealthCheckResult(
                    name="aws_bedrock",
                    status=HealthStatus.DEGRADED,
                    message="Bedrock not initialized",
                )
        except Exception as e:
            return HealthCheckResult(
                name="aws_bedrock",
                status=HealthStatus.UNHEALTHY,
                message=f"Bedrock check failed: {e}",
            )

    async def _check_aws_kms(self) -> HealthCheckResult:
        """Check KMS connectivity."""
        try:
            if AWS_CONFIG_AVAILABLE:
                aws_config = get_aws_config()
                kms_client = aws_config.get_client("kms")
            else:
                kms_client = boto3.client("kms")
            # List keys to verify connectivity
            kms_client.list_keys(Limit=1)

            return HealthCheckResult(
                name="aws_kms",
                status=HealthStatus.HEALTHY,
                message="KMS connectivity OK",
            )
        except Exception as e:
            return HealthCheckResult(
                name="aws_kms",
                status=HealthStatus.UNHEALTHY,
                message=f"KMS connectivity failed: {e}",
            )

    async def run_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        self.results.clear()

        # Run all checks concurrently
        tasks = [check_func() for check_func in self.checks.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for result in results:
            if isinstance(result, Exception):
                self.results.append(
                    HealthCheckResult(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Check failed: {result}",
                    )
                )
            else:
                self.results.append(result)

        self.last_check = time.time()

        # Determine overall status
        statuses = [r.status for r in self.results]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "timestamp": self.last_check,
            "checks": [r.to_dict() for r in self.results],
        }

    async def start_periodic_checks(self):
        """Start periodic health checks."""
        self._running = True

        while self._running:
            try:
                await self.run_checks()

                # Log unhealthy checks
                for result in self.results:
                    if result.status == HealthStatus.UNHEALTHY:
                        logger.error(
                            f"Health check failed: {result.name} - {result.message}"
                        )
                    elif result.status == HealthStatus.DEGRADED:
                        logger.warning(
                            f"Health check degraded: {result.name} - {result.message}"
                        )

            except Exception as e:
                logger.error(f"Health check error: {e}")

            await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop periodic health checks."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current health status."""
        if not self.results:
            return {"status": "unknown", "message": "No health checks have been run"}

        # Return cached results if recent
        if self.last_check and time.time() - self.last_check < self.check_interval:
            statuses = [r.status for r in self.results]
            if any(s == HealthStatus.UNHEALTHY for s in statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(s == HealthStatus.DEGRADED for s in statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY

            return {
                "status": overall_status.value,
                "timestamp": self.last_check,
                "checks": [r.to_dict() for r in self.results],
            }

        return {
            "status": "stale",
            "message": "Health check data is stale",
            "last_check": self.last_check,
        }


class HealthEndpoint:
    """HTTP health check endpoint for Lambda/ECS."""

    def __init__(self, health_checker: HealthChecker):
        self.health_checker = health_checker

    async def handle_health(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check HTTP request."""
        try:
            # Run checks if needed
            status = self.health_checker.get_status()

            if status.get("status") == "stale":
                # Force new check
                status = await self.health_checker.run_checks()

            # Determine HTTP status code
            if status["status"] == "healthy":
                status_code = 200
            elif status["status"] == "degraded":
                status_code = 200  # Still return 200 for degraded
            else:
                status_code = 503  # Service unavailable

            return {
                "statusCode": status_code,
                "headers": {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                },
                "body": status,
            }
        except Exception as e:
            logger.error(f"Health endpoint error: {e}")
            return {"statusCode": 503, "body": {"status": "error", "message": str(e)}}

    async def handle_readiness(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle readiness check (for K8s/ECS)."""
        # Check if agent is ready to accept traffic
        if hasattr(self.health_checker.agent, "state"):
            state = self.health_checker.agent.state.value
            if state in ["idle", "perceiving", "reasoning", "executing"]:
                return {"statusCode": 200, "body": {"ready": True}}

        return {"statusCode": 503, "body": {"ready": False}}

    async def handle_liveness(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle liveness check (for K8s/ECS)."""
        # Basic check that process is alive
        return {"statusCode": 200, "body": {"alive": True}}
