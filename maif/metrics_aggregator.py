"""
Metrics Aggregation for MAIF
============================

Provides metrics collection and aggregation for CloudWatch integration.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
import logging
import boto3
from datetime import datetime
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    unit: str = "Count"
    timestamp: float = field(default_factory=time.time)
    dimensions: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric statistics."""

    name: str
    count: int
    sum: float
    min: float
    max: float
    average: float
    unit: str
    dimensions: Dict[str, str]


class MetricsAggregator:
    """Aggregates metrics for efficient CloudWatch publishing."""

    def __init__(
        self,
        namespace: str = "MAIF",
        flush_interval: int = 60,
        batch_size: int = 20,
        cloudwatch_client: Optional[Any] = None,
    ):
        self.namespace = namespace
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.cloudwatch = cloudwatch_client or boto3.client("cloudwatch")

        # Storage for metrics
        self.metrics_buffer: Dict[str, List[Metric]] = defaultdict(list)
        self.aggregated_metrics: List[AggregatedMetric] = []

        # Performance metrics
        self.total_metrics_sent = 0
        self.total_batches_sent = 0
        self.last_flush_time = time.time()

        # Start background flush task
        self._flush_task = None
        self._running = False

    async def start(self):
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._background_flush())
        logger.info(
            f"MetricsAggregator started with {self.flush_interval}s flush interval"
        )

    async def stop(self):
        """Stop the aggregator and flush remaining metrics."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self.flush()
        logger.info("MetricsAggregator stopped")

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "Count",
        dimensions: Optional[Dict[str, str]] = None,
    ):
        """Record a metric."""
        metric = Metric(name=name, value=value, unit=unit, dimensions=dimensions or {})

        # Create key for aggregation
        key = self._get_metric_key(metric)
        self.metrics_buffer[key].append(metric)

    def record_latency(self, operation: str, duration_ms: float, **dimensions):
        """Record operation latency."""
        self.record_metric(
            f"{operation}_latency",
            duration_ms,
            unit="Milliseconds",
            dimensions=dimensions,
        )

    def record_error(self, operation: str, error_type: str, **dimensions):
        """Record an error occurrence."""
        dims = {"ErrorType": error_type, **dimensions}
        self.record_metric(f"{operation}_errors", 1, unit="Count", dimensions=dims)

    def record_success(self, operation: str, **dimensions):
        """Record a successful operation."""
        self.record_metric(
            f"{operation}_success", 1, unit="Count", dimensions=dimensions
        )

    def _get_metric_key(self, metric: Metric) -> str:
        """Generate aggregation key for metric."""
        dims_str = json.dumps(metric.dimensions, sort_keys=True)
        return f"{metric.name}:{metric.unit}:{dims_str}"

    def _aggregate_metrics(self):
        """Aggregate buffered metrics."""
        self.aggregated_metrics.clear()

        for key, metrics in self.metrics_buffer.items():
            if not metrics:
                continue

            values = [m.value for m in metrics]
            first_metric = metrics[0]

            aggregated = AggregatedMetric(
                name=first_metric.name,
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                average=sum(values) / len(values),
                unit=first_metric.unit,
                dimensions=first_metric.dimensions,
            )

            self.aggregated_metrics.append(aggregated)

        # Clear buffer after aggregation
        self.metrics_buffer.clear()

    async def flush(self):
        """Flush metrics to CloudWatch."""
        try:
            # Aggregate metrics
            self._aggregate_metrics()

            if not self.aggregated_metrics:
                return

            # Prepare CloudWatch metrics
            metric_data = []

            for agg in self.aggregated_metrics:
                # Add statistical values
                base_metric = {
                    "MetricName": agg.name,
                    "Dimensions": [
                        {"Name": k, "Value": v} for k, v in agg.dimensions.items()
                    ],
                    "Timestamp": datetime.utcnow(),
                    "Unit": agg.unit,
                }

                # Send different statistics
                metric_data.extend(
                    [
                        {
                            **base_metric,
                            "Value": agg.sum,
                            "MetricName": f"{agg.name}_sum",
                        },
                        {
                            **base_metric,
                            "Value": agg.average,
                            "MetricName": f"{agg.name}_avg",
                        },
                        {
                            **base_metric,
                            "Value": agg.min,
                            "MetricName": f"{agg.name}_min",
                        },
                        {
                            **base_metric,
                            "Value": agg.max,
                            "MetricName": f"{agg.name}_max",
                        },
                        {
                            **base_metric,
                            "Value": agg.count,
                            "MetricName": f"{agg.name}_count",
                        },
                    ]
                )

            # Send in batches
            for i in range(0, len(metric_data), self.batch_size):
                batch = metric_data[i : i + self.batch_size]

                try:
                    self.cloudwatch.put_metric_data(
                        Namespace=self.namespace, MetricData=batch
                    )
                    self.total_metrics_sent += len(batch)
                    self.total_batches_sent += 1
                except ClientError as e:
                    logger.error(f"Failed to send metrics batch: {e}")

            self.last_flush_time = time.time()
            logger.info(
                f"Flushed {len(metric_data)} metrics in {self.total_batches_sent} batches"
            )

        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

    async def _background_flush(self):
        """Background task to periodically flush metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background flush: {e}")


class MAIFMetrics:
    """High-level metrics interface for MAIF components."""

    def __init__(self, aggregator: MetricsAggregator):
        self.aggregator = aggregator

    def agent_started(self, agent_id: str, agent_type: str):
        """Record agent startup."""
        self.aggregator.record_metric(
            "agent_starts", 1, dimensions={"AgentId": agent_id, "AgentType": agent_type}
        )

    def agent_stopped(self, agent_id: str, agent_type: str, runtime_seconds: float):
        """Record agent shutdown."""
        self.aggregator.record_metric(
            "agent_stops", 1, dimensions={"AgentId": agent_id, "AgentType": agent_type}
        )
        self.aggregator.record_metric(
            "agent_runtime",
            runtime_seconds,
            unit="Seconds",
            dimensions={"AgentId": agent_id, "AgentType": agent_type},
        )

    def perception_processed(self, agent_id: str, data_type: str, duration_ms: float):
        """Record perception processing."""
        self.aggregator.record_latency(
            "perception", duration_ms, AgentId=agent_id, DataType=data_type
        )

    def reasoning_completed(self, agent_id: str, strategy: str, duration_ms: float):
        """Record reasoning completion."""
        self.aggregator.record_latency(
            "reasoning", duration_ms, AgentId=agent_id, Strategy=strategy
        )

    def action_executed(
        self, agent_id: str, action_type: str, success: bool, duration_ms: float
    ):
        """Record action execution."""
        if success:
            self.aggregator.record_success(
                "action", AgentId=agent_id, ActionType=action_type
            )
        else:
            self.aggregator.record_error(
                "action", "ExecutionFailed", AgentId=agent_id, ActionType=action_type
            )

        self.aggregator.record_latency(
            "action", duration_ms, AgentId=agent_id, ActionType=action_type
        )

    def bedrock_request(
        self, model: str, input_tokens: int, output_tokens: int, duration_ms: float
    ):
        """Record Bedrock API request."""
        self.aggregator.record_metric(
            "bedrock_input_tokens", input_tokens, dimensions={"Model": model}
        )
        self.aggregator.record_metric(
            "bedrock_output_tokens", output_tokens, dimensions={"Model": model}
        )
        self.aggregator.record_latency("bedrock_request", duration_ms, Model=model)

    def s3_operation(
        self, operation: str, bucket: str, size_bytes: int, duration_ms: float
    ):
        """Record S3 operation."""
        self.aggregator.record_metric(
            f"s3_{operation}_bytes",
            size_bytes,
            unit="Bytes",
            dimensions={"Bucket": bucket},
        )
        self.aggregator.record_latency(f"s3_{operation}", duration_ms, Bucket=bucket)

    def dynamodb_operation(
        self, operation: str, table: str, items: int, duration_ms: float
    ):
        """Record DynamoDB operation."""
        self.aggregator.record_metric(
            f"dynamodb_{operation}_items", items, dimensions={"Table": table}
        )
        self.aggregator.record_latency(
            f"dynamodb_{operation}", duration_ms, Table=table
        )


# Global metrics instance
_metrics_aggregator = None
_maif_metrics = None


def initialize_metrics(namespace: str = "MAIF", **kwargs):
    """Initialize global metrics collection."""
    global _metrics_aggregator, _maif_metrics

    _metrics_aggregator = MetricsAggregator(namespace=namespace, **kwargs)
    _maif_metrics = MAIFMetrics(_metrics_aggregator)

    # Start aggregator
    asyncio.create_task(_metrics_aggregator.start())

    return _maif_metrics


def get_metrics() -> MAIFMetrics:
    """Get global metrics instance."""
    global _maif_metrics
    if _maif_metrics is None:
        initialize_metrics()
    return _maif_metrics


# Usage example:
"""
# Initialize metrics
metrics = initialize_metrics(
    namespace="MAIF/Production",
    flush_interval=60,
    batch_size=20
)

# Record metrics
metrics.agent_started("agent-123", "reasoning")
metrics.perception_processed("agent-123", "visual", 125.5)
metrics.reasoning_completed("agent-123", "chain-of-thought", 890.2)
metrics.action_executed("agent-123", "query_database", True, 45.8)

# Or use low-level API
aggregator = MetricsAggregator()
await aggregator.start()

aggregator.record_metric("custom_metric", 42.5, unit="Count")
aggregator.record_latency("api_call", 250.0, Service="UserAPI")
aggregator.record_error("database_query", "ConnectionTimeout", Database="users")

await aggregator.stop()
"""
