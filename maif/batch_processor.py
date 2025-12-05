"""
Batch Processing for MAIF
=========================

Provides high-volume batch processing capabilities with AWS integration.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Try to import AWS dependencies
try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_CONFIG_AVAILABLE = True
except ImportError:
    boto3 = None
    ClientError = Exception
    AWS_CONFIG_AVAILABLE = False


def get_aws_config():
    """Get AWS configuration. Returns None if AWS is not available."""
    if not AWS_CONFIG_AVAILABLE:
        return None
    
    class AWSConfig:
        def get_client(self, service_name):
            return boto3.client(service_name)
    
    return AWSConfig()

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchItem(Generic[T]):
    """Individual item in a batch."""

    id: str
    data: T
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    error: Optional[str] = None
    result: Optional[Any] = None


@dataclass
class BatchResult:
    """Result of batch processing."""

    total_items: int
    successful_items: int
    failed_items: int
    duration_seconds: float
    items_per_second: float
    errors: Dict[str, str]


class BatchProcessor(Generic[T, R]):
    """High-performance batch processor."""

    def __init__(
        self,
        process_func: Callable[[T], R],
        batch_size: int = 100,
        max_workers: int = 10,
        max_retries: int = 3,
        use_aws_batch: bool = False,
    ):
        self.process_func = process_func
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.use_aws_batch = use_aws_batch

        # AWS clients
        if use_aws_batch:
            if AWS_CONFIG_AVAILABLE:
                aws_config = get_aws_config()
                self.batch_client = aws_config.get_client("batch")
                self.s3_client = aws_config.get_client("s3")
            else:
                self.batch_client = boto3.client("batch")
                self.s3_client = boto3.client("s3")

        # Metrics
        self.total_processed = 0
        self.total_failed = 0

    async def process_batch(self, items: List[T], parallel: bool = True) -> BatchResult:
        """Process a batch of items."""
        start_time = time.time()

        # Convert to BatchItems
        batch_items = [BatchItem(id=str(uuid.uuid4()), data=item) for item in items]

        if self.use_aws_batch:
            result = await self._process_aws_batch(batch_items)
        elif parallel:
            result = await self._process_parallel(batch_items)
        else:
            result = await self._process_sequential(batch_items)

        duration = time.time() - start_time

        # Calculate metrics
        successful = sum(1 for item in batch_items if item.error is None)
        failed = len(batch_items) - successful

        self.total_processed += successful
        self.total_failed += failed

        return BatchResult(
            total_items=len(batch_items),
            successful_items=successful,
            failed_items=failed,
            duration_seconds=duration,
            items_per_second=len(batch_items) / duration if duration > 0 else 0,
            errors={item.id: item.error for item in batch_items if item.error},
        )

    async def _process_sequential(self, items: List[BatchItem[T]]) -> None:
        """Process items sequentially."""
        for item in items:
            await self._process_item(item)

    async def _process_parallel(self, items: List[BatchItem[T]]) -> None:
        """Process items in parallel."""
        # Use ThreadPoolExecutor for CPU-bound tasks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_item_sync, item): item for item in items
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    item = futures[future]
                    item.error = str(e)
                    logger.error(f"Batch item {item.id} failed: {e}")

    def _process_item_sync(self, item: BatchItem[T]) -> None:
        """Synchronous item processing."""
        for attempt in range(self.max_retries):
            try:
                item.result = self.process_func(item.data)
                item.error = None
                break
            except Exception as e:
                item.retry_count = attempt + 1
                item.error = str(e)

                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Item {item.id} failed after {self.max_retries} attempts: {e}"
                    )

    async def _process_item(self, item: BatchItem[T]) -> None:
        """Asynchronous item processing."""
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(self.process_func):
                    item.result = await self.process_func(item.data)
                else:
                    item.result = self.process_func(item.data)
                item.error = None
                break
            except Exception as e:
                item.retry_count = attempt + 1
                item.error = str(e)

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Item {item.id} failed after {self.max_retries} attempts: {e}"
                    )

    async def _process_aws_batch(self, items: List[BatchItem[T]]) -> None:
        """Process items using AWS Batch."""
        # Upload batch data to S3
        bucket = "maif-batch-processing"
        key = f"batches/{uuid.uuid4()}.json"

        try:
            # Serialize items
            batch_data = {
                "items": [
                    {"id": item.id, "data": item.data, "metadata": item.metadata}
                    for item in items
                ]
            }

            self.s3_client.put_object(
                Bucket=bucket, Key=key, Body=json.dumps(batch_data)
            )

            # Submit batch job
            response = self.batch_client.submit_job(
                jobName=f"maif-batch-{uuid.uuid4()}",
                jobQueue="maif-batch-queue",
                jobDefinition="maif-batch-processor",
                parameters={"bucket": bucket, "key": key},
            )

            job_id = response["jobId"]

            # Wait for completion
            await self._wait_for_batch_job(job_id)

            # Retrieve results
            result_key = f"results/{job_id}.json"
            result_obj = self.s3_client.get_object(Bucket=bucket, Key=result_key)
            results = json.loads(result_obj["Body"].read())

            # Update items with results
            for item in items:
                item_result = results.get(item.id, {})
                item.result = item_result.get("result")
                item.error = item_result.get("error")

        except Exception as e:
            logger.error(f"AWS Batch processing failed: {e}")
            for item in items:
                item.error = str(e)

    async def _wait_for_batch_job(self, job_id: str):
        """Wait for AWS Batch job completion."""
        while True:
            response = self.batch_client.describe_jobs(jobs=[job_id])
            job = response["jobs"][0]
            status = job["status"]

            if status in ["SUCCEEDED", "FAILED"]:
                if status == "FAILED":
                    raise Exception(f"Batch job {job_id} failed")
                break

            await asyncio.sleep(5)


class StreamBatchProcessor(BatchProcessor[T, R]):
    """Batch processor for streaming data."""

    def __init__(
        self,
        process_func: Callable[[T], R],
        batch_size: int = 100,
        batch_timeout: float = 10.0,
        **kwargs,
    ):
        super().__init__(process_func, batch_size, **kwargs)
        self.batch_timeout = batch_timeout
        self.current_batch: List[T] = []
        self.last_batch_time = time.time()
        self._processing = False

    async def add_item(self, item: T):
        """Add item to batch."""
        self.current_batch.append(item)

        # Check if batch should be processed
        if len(self.current_batch) >= self.batch_size:
            await self._flush_batch()
        elif time.time() - self.last_batch_time > self.batch_timeout:
            await self._flush_batch()

    async def _flush_batch(self):
        """Process current batch."""
        if not self.current_batch or self._processing:
            return

        self._processing = True
        batch = self.current_batch[:]
        self.current_batch.clear()
        self.last_batch_time = time.time()

        try:
            result = await self.process_batch(batch)
            logger.info(
                f"Batch processed: {result.successful_items}/{result.total_items} successful"
            )
        finally:
            self._processing = False

    async def start_background_flush(self):
        """Start background task to flush batches on timeout."""
        while True:
            await asyncio.sleep(1)
            if time.time() - self.last_batch_time > self.batch_timeout:
                await self._flush_batch()


class DistributedBatchProcessor:
    """Distributed batch processor using AWS services."""

    def __init__(self, job_definition: str, queue_name: str, bucket_name: str):
        self.job_definition = job_definition
        self.queue_name = queue_name
        self.bucket_name = bucket_name

        # AWS clients
        if AWS_CONFIG_AVAILABLE:
            aws_config = get_aws_config()
            self.batch_client = aws_config.get_client("batch")
            self.s3_client = aws_config.get_client("s3")
            self.sqs_client = aws_config.get_client("sqs")
        else:
            self.batch_client = boto3.client("batch")
            self.s3_client = boto3.client("s3")
            self.sqs_client = boto3.client("sqs")

    async def submit_batch(
        self, items: List[Any], job_name: Optional[str] = None
    ) -> str:
        """Submit batch for distributed processing."""
        batch_id = str(uuid.uuid4())
        job_name = job_name or f"batch-{batch_id}"

        # Upload batch to S3
        key = f"batches/{batch_id}/input.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name, Key=key, Body=json.dumps({"items": items})
        )

        # Submit batch job
        response = self.batch_client.submit_job(
            jobName=job_name,
            jobQueue=self.queue_name,
            jobDefinition=self.job_definition,
            parameters={
                "batch_id": batch_id,
                "input_key": key,
                "output_key": f"batches/{batch_id}/output.json",
            },
            arrayProperties={"size": len(items)} if len(items) > 1 else {},
        )

        return response["jobId"]

    async def get_batch_status(self, job_id: str) -> Dict[str, Any]:
        """Get batch job status."""
        response = self.batch_client.describe_jobs(jobs=[job_id])

        if not response["jobs"]:
            return {"status": "NOT_FOUND"}

        job = response["jobs"][0]

        return {
            "status": job["status"],
            "created_at": job.get("createdAt"),
            "started_at": job.get("startedAt"),
            "stopped_at": job.get("stoppedAt"),
            "status_reason": job.get("statusReason"),
        }

    async def get_batch_results(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get batch processing results."""
        key = f"batches/{batch_id}/output.json"

        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)

            data = json.loads(response["Body"].read())
            return data.get("results", [])

        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise


# Batch processing decorators
def batch_process(batch_size: int = 100, max_workers: int = 10, use_aws: bool = False):
    """Decorator for batch processing."""

    def decorator(func):
        processor = BatchProcessor(
            process_func=func,
            batch_size=batch_size,
            max_workers=max_workers,
            use_aws_batch=use_aws,
        )

        async def wrapper(items: List[Any]) -> BatchResult:
            return await processor.process_batch(items)

        wrapper.processor = processor
        return wrapper

    return decorator


# Usage example:
"""
# Simple batch processing
@batch_process(batch_size=50, max_workers=5)
def process_item(item: dict) -> dict:
    # Process individual item
    return {"id": item["id"], "result": item["value"] * 2}

# Use the batch processor
items = [{"id": i, "value": i} for i in range(1000)]
result = await process_item(items)
print(f"Processed {result.successful_items} items in {result.duration_seconds:.2f}s")

# Stream batch processing
stream_processor = StreamBatchProcessor(
    process_func=lambda x: x.upper(),
    batch_size=20,
    batch_timeout=5.0
)

# Add items as they arrive
await stream_processor.add_item("hello")
await stream_processor.add_item("world")

# Distributed processing
distributed = DistributedBatchProcessor(
    job_definition="maif-batch-job",
    queue_name="maif-batch-queue", 
    bucket_name="maif-batch-data"
)

job_id = await distributed.submit_batch(items)
status = await distributed.get_batch_status(job_id)
results = await distributed.get_batch_results(batch_id)
"""
