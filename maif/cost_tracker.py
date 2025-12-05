"""
Cost Tracking and Budgeting for MAIF
====================================

Provides real-time cost tracking and budget enforcement for AWS services.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
import json
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class ServiceCost:
    """Cost configuration for a service."""

    service_name: str
    unit_cost: Decimal
    unit_type: str  # e.g., "per_request", "per_gb", "per_second"
    free_tier_limit: int = 0


@dataclass
class Budget:
    """Budget configuration."""

    name: str
    limit: Decimal
    period: str  # "daily", "weekly", "monthly"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    enforce_limit: bool = True


@dataclass
class CostRecord:
    """Individual cost record."""

    timestamp: float
    service: str
    operation: str
    units: float
    unit_cost: Decimal
    total_cost: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)


class CostTracker:
    """Tracks costs for MAIF AWS operations."""

    # AWS service pricing (simplified for demonstration)
    SERVICE_COSTS = {
        "bedrock_claude": ServiceCost("bedrock", Decimal("0.00008"), "per_token", 0),
        "bedrock_embedding": ServiceCost(
            "bedrock", Decimal("0.0001"), "per_1k_chars", 0
        ),
        "s3_storage": ServiceCost("s3", Decimal("0.023"), "per_gb_month", 5),
        "s3_requests": ServiceCost("s3", Decimal("0.0004"), "per_1k_requests", 2000),
        "dynamodb_read": ServiceCost("dynamodb", Decimal("0.00025"), "per_rcu", 25),
        "dynamodb_write": ServiceCost("dynamodb", Decimal("0.00125"), "per_wcu", 25),
        "lambda_requests": ServiceCost(
            "lambda", Decimal("0.0000002"), "per_request", 1000000
        ),
        "lambda_duration": ServiceCost(
            "lambda", Decimal("0.0000166667"), "per_gb_second", 400000
        ),
        "cloudwatch_logs": ServiceCost("cloudwatch", Decimal("0.50"), "per_gb", 5),
        "kinesis_shards": ServiceCost("kinesis", Decimal("0.015"), "per_shard_hour", 0),
        "xray_traces": ServiceCost("xray", Decimal("0.000005"), "per_trace", 100000),
    }

    def __init__(self, budget: Optional[Budget] = None):
        self.budget = budget
        self.costs: List[CostRecord] = []
        self.total_cost = Decimal("0")
        self.cost_by_service: Dict[str, Decimal] = defaultdict(Decimal)
        self.cost_by_period: Dict[str, Decimal] = defaultdict(Decimal)

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

        # Cost optimization suggestions
        self.optimization_suggestions: List[str] = []

        # Background tasks
        self._monitor_task = None
        self._running = False

    async def start(self):
        """Start cost monitoring."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_costs())
        logger.info("Cost tracker started")

    async def stop(self):
        """Stop cost monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Cost tracker stopped")

    def track_cost(
        self,
        service: str,
        operation: str,
        units: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Decimal:
        """Track a cost-incurring operation."""
        # Get service cost config
        cost_config = self.SERVICE_COSTS.get(f"{service}_{operation}")
        if not cost_config:
            logger.warning(f"Unknown service/operation: {service}/{operation}")
            return Decimal("0")

        # Calculate cost
        billable_units = max(0, units - cost_config.free_tier_limit)
        cost = Decimal(str(billable_units)) * cost_config.unit_cost

        # Create cost record
        record = CostRecord(
            timestamp=time.time(),
            service=service,
            operation=operation,
            units=units,
            unit_cost=cost_config.unit_cost,
            total_cost=cost,
            metadata=metadata or {},
        )

        # Update totals
        self.costs.append(record)
        self.total_cost += cost
        self.cost_by_service[service] += cost

        # Update period costs
        period_key = self._get_period_key()
        self.cost_by_period[period_key] += cost

        # Check budget
        if self.budget:
            self._check_budget(cost)

        return cost

    def _get_period_key(self) -> str:
        """Get current period key based on budget."""
        now = datetime.now()

        if not self.budget:
            return now.strftime("%Y-%m-%d")

        if self.budget.period == "daily":
            return now.strftime("%Y-%m-%d")
        elif self.budget.period == "weekly":
            return f"{now.year}-W{now.isocalendar()[1]}"
        elif self.budget.period == "monthly":
            return now.strftime("%Y-%m")
        else:
            return now.strftime("%Y-%m-%d")

    def _check_budget(self, new_cost: Decimal):
        """Check if budget limits are exceeded."""
        period_key = self._get_period_key()
        period_cost = self.cost_by_period[period_key]

        # Check if over budget
        if self.budget.enforce_limit and period_cost > self.budget.limit:
            raise BudgetExceededException(
                f"Budget '{self.budget.name}' exceeded: "
                f"${period_cost:.4f} > ${self.budget.limit:.4f}"
            )

        # Check alert thresholds
        for threshold in self.budget.alert_thresholds:
            threshold_amount = self.budget.limit * Decimal(str(threshold))
            if (
                period_cost >= threshold_amount
                and (period_cost - new_cost) < threshold_amount
            ):
                self._trigger_alert(threshold, period_cost)

    def _trigger_alert(self, threshold: float, current_cost: Decimal):
        """Trigger budget alert."""
        alert = {
            "budget": self.budget.name,
            "threshold": threshold,
            "current_cost": float(current_cost),
            "limit": float(self.budget.limit),
            "percentage": float(current_cost / self.budget.limit * 100),
        }

        logger.warning(f"Budget alert: {alert}")

        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _monitor_costs(self):
        """Background task to monitor costs and generate reports."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Generate cost report
                report = self.generate_report()

                # Check for optimization opportunities
                self._check_optimizations()

                # Log current costs
                logger.info(
                    f"Current period cost: ${report['current_period_cost']:.4f}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cost monitoring error: {e}")

    def _check_optimizations(self):
        """Check for cost optimization opportunities."""
        self.optimization_suggestions.clear()

        # Analyze service costs
        for service, cost in self.cost_by_service.items():
            if service == "bedrock" and cost > Decimal("100"):
                self.optimization_suggestions.append(
                    "Consider using smaller models or caching frequent requests"
                )
            elif service == "s3" and cost > Decimal("50"):
                self.optimization_suggestions.append(
                    "Enable S3 lifecycle policies to move old data to cheaper storage"
                )
            elif service == "dynamodb" and cost > Decimal("75"):
                self.optimization_suggestions.append(
                    "Consider using on-demand pricing or adjusting provisioned capacity"
                )

        # Check for high-frequency operations
        operation_counts = defaultdict(int)
        for record in self.costs[-1000:]:  # Last 1000 operations
            operation_counts[f"{record.service}:{record.operation}"] += 1

        for operation, count in operation_counts.items():
            if count > 100:
                self.optimization_suggestions.append(
                    f"High frequency operation '{operation}' - consider batching"
                )

    def generate_report(self) -> Dict[str, Any]:
        """Generate cost report."""
        period_key = self._get_period_key()

        return {
            "total_cost": float(self.total_cost),
            "current_period_cost": float(self.cost_by_period[period_key]),
            "cost_by_service": {
                service: float(cost) for service, cost in self.cost_by_service.items()
            },
            "budget_status": {
                "name": self.budget.name if self.budget else None,
                "limit": float(self.budget.limit) if self.budget else None,
                "usage_percentage": float(
                    self.cost_by_period[period_key] / self.budget.limit * 100
                )
                if self.budget
                else 0,
            },
            "optimization_suggestions": self.optimization_suggestions,
            "top_operations": self._get_top_operations(),
            "cost_trend": self._get_cost_trend(),
        }

    def _get_top_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cost-incurring operations."""
        operation_costs = defaultdict(Decimal)
        operation_counts = defaultdict(int)

        for record in self.costs:
            key = f"{record.service}:{record.operation}"
            operation_costs[key] += record.total_cost
            operation_counts[key] += 1

        top_ops = sorted(operation_costs.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        return [
            {
                "operation": op,
                "total_cost": float(cost),
                "count": operation_counts[op],
                "average_cost": float(cost / operation_counts[op]),
            }
            for op, cost in top_ops
        ]

    def _get_cost_trend(self) -> List[Dict[str, Any]]:
        """Get cost trend over time."""
        hourly_costs = defaultdict(Decimal)

        for record in self.costs:
            hour = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:00")
            hourly_costs[hour] += record.total_cost

        return [
            {"hour": hour, "cost": float(cost)}
            for hour, cost in sorted(hourly_costs.items())[-24:]  # Last 24 hours
        ]


class BudgetExceededException(Exception):
    """Raised when budget limit is exceeded."""

    pass


class AWSCostExplorer:
    """Integration with AWS Cost Explorer API."""

    def __init__(self):
        if AWS_CONFIG_AVAILABLE:
            aws_config = get_aws_config()
            self.ce_client = aws_config.get_client("ce")
        else:
            self.ce_client = boto3.client("ce")

    async def get_actual_costs(
        self, start_date: str, end_date: str, granularity: str = "DAILY"
    ) -> Dict[str, Any]:
        """Get actual costs from AWS Cost Explorer."""
        try:
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={"Start": start_date, "End": end_date},
                Granularity=granularity,
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            )

            return {
                "costs": response["ResultsByTime"],
                "total": sum(
                    float(result["Total"]["UnblendedCost"]["Amount"])
                    for result in response["ResultsByTime"]
                ),
            }

        except ClientError as e:
            logger.error(f"Cost Explorer error: {e}")
            return {"costs": [], "total": 0}


# Global cost tracker
_cost_tracker = None


def initialize_cost_tracking(budget: Optional[Budget] = None):
    """Initialize global cost tracking."""
    global _cost_tracker

    _cost_tracker = CostTracker(budget)
    asyncio.create_task(_cost_tracker.start())

    return _cost_tracker


def get_cost_tracker() -> CostTracker:
    """Get global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        initialize_cost_tracking()
    return _cost_tracker


def track_cost(service: str, operation: str, units: float, **metadata):
    """Track a cost-incurring operation."""
    tracker = get_cost_tracker()
    return tracker.track_cost(service, operation, units, metadata)


# Cost tracking decorator
def with_cost_tracking(
    service: str, operation: str, unit_calculator: Optional[Callable] = None
):
    """Decorator to track costs for a function."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Calculate units
            if unit_calculator:
                units = unit_calculator(*args, **kwargs)
            else:
                units = 1

            # Track cost
            cost = track_cost(service, operation, units)

            # Execute function
            result = await func(*args, **kwargs)

            # Add cost to result if dict
            if isinstance(result, dict):
                result["_cost"] = float(cost)

            return result

        def sync_wrapper(*args, **kwargs):
            # Calculate units
            if unit_calculator:
                units = unit_calculator(*args, **kwargs)
            else:
                units = 1

            # Track cost
            cost = track_cost(service, operation, units)

            # Execute function
            result = func(*args, **kwargs)

            # Add cost to result if dict
            if isinstance(result, dict):
                result["_cost"] = float(cost)

            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Usage example:
"""
# Initialize with budget
budget = Budget(
    name="production",
    limit=Decimal("1000"),
    period="monthly",
    alert_thresholds=[0.5, 0.8, 0.95],
    enforce_limit=True
)

tracker = initialize_cost_tracking(budget)

# Track costs
tracker.track_cost("bedrock", "claude", 1500)  # 1500 tokens
tracker.track_cost("s3", "requests", 2500)     # 2500 requests
tracker.track_cost("dynamodb", "read", 100)    # 100 RCUs

# Use decorator
@with_cost_tracking("bedrock", "embedding", lambda text: len(text) / 1000)
async def generate_embedding(text: str):
    # Generate embedding
    return embedding

# Get report
report = tracker.generate_report()
print(f"Total cost: ${report['total_cost']:.4f}")
print(f"Budget usage: {report['budget_status']['usage_percentage']:.1f}%")

# AWS Cost Explorer integration
explorer = AWSCostExplorer()
actual_costs = await explorer.get_actual_costs("2024-01-01", "2024-01-31")
"""
