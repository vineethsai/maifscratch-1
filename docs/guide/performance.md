# Performance Optimization

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki](https://deepwiki.com/vineethsai/maif)**.
:::

MAIF is designed for high-performance operations. This guide covers optimization techniques for maximum throughput and efficiency.

## Performance Features

MAIF provides:

- **Memory-Mapped I/O**: Efficient file access
- **Write Buffering**: Batched writes for throughput
- **Compression**: Reduce storage and I/O
- **Batch Processing**: Process data efficiently
- **Streaming**: Handle large files

## Memory-Mapped I/O

Enable memory mapping for large files:

```python
from maif.core import MAIFEncoder

# Enable memory mapping
encoder = MAIFEncoder(
    "large.maif",
    agent_id="performance-agent",
    enable_mmap=True
)

# Large write operations are more efficient
for i in range(10000):
    encoder.add_text_block(f"Document {i}")

encoder.finalize()
```

### Block Storage with mmap

```python
from maif.block_storage import BlockStorage

# Direct block access with memory mapping
storage = BlockStorage("large.maif", enable_mmap=True)

# Efficient random access
block = storage.get_block("block-id")
```

## Write Buffering

The encoder uses buffering for better performance:

```python
from maif.core import MAIFEncoder

# Configure buffer size
encoder = MAIFEncoder(
    "buffered.maif",
    agent_id="buffered-agent",
    buffer_size=128 * 1024  # 128KB buffer
)

# Writes are buffered automatically
for i in range(1000):
    encoder.add_text_block(f"Item {i}")

# Buffer flushed on finalize
encoder.finalize()
```

## Compression

Enable compression to reduce file size and I/O:

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(
    "compressed.maif",
    agent_id="compressed-agent",
    enable_compression=True
)

# Data is automatically compressed
encoder.add_text_block("Large document content...")
encoder.finalize()
```

### Compression Manager

```python
from maif.compression_manager import CompressionManager

# Create compression manager
compression = CompressionManager()

# Compress data
compressed = compression.compress(large_data)

# Decompress when needed
original = compression.decompress(compressed)
```

### Embedding Compression

```python
from maif_api import create_maif

maif = create_maif("embedding-agent")

# Compress large embedding sets
embeddings = [[0.1, 0.2, 0.3] for _ in range(10000)]
maif.add_embeddings(embeddings, model_name="bert", compress=True)

maif.save("compressed_embeddings.maif")
```

## Batch Processing

Process data in efficient batches:

```python
from maif.batch_processor import BatchProcessor

# Create batch processor
processor = BatchProcessor(batch_size=100)

# Process items in batches
items = [f"document_{i}" for i in range(10000)]

for batch in processor.process(items):
    # Each batch has up to 100 items
    for item in batch:
        process(item)
```

### Stream Batch Processing

```python
from maif.batch_processor import StreamBatchProcessor

# For streaming data
processor = StreamBatchProcessor(
    batch_size=100,
    timeout=5.0  # Flush after 5 seconds
)

# Add items as they arrive
for item in data_stream:
    processor.add(item)
    
    if processor.is_ready():
        batch = processor.get_batch()
        process_batch(batch)

# Process remaining
final_batch = processor.flush()
```

### Parallel Processing

```python
from maif.batch_processor import DistributedBatchProcessor

# Multi-worker processing
processor = DistributedBatchProcessor(
    batch_size=1000,
    num_workers=4
)

# Process with parallelism
results = processor.process_parallel(items)
```

## Streaming

For large files, use streaming operations:

```python
from maif.streaming import MAIFStreamWriter, MAIFStreamReader

# Write incrementally
with MAIFStreamWriter("large.maif") as writer:
    for i in range(100000):
        writer.write_text_chunk(f"Document {i}")

# Read without loading entire file
with MAIFStreamReader("large.maif") as reader:
    for block in reader.read_blocks():
        process(block)
```

## Hot Buffer

For high-throughput caching:

```python
from maif.hot_buffer import HotBuffer

# Create hot buffer
buffer = HotBuffer(max_size=1024 * 1024)  # 1MB

# Cache frequently accessed data
buffer.put("key1", data1)
buffer.put("key2", data2)

# Fast retrieval
data = buffer.get("key1")
```

## Rate Limiting

Control throughput in production:

```python
from maif.rate_limiter import RateLimiter, RateLimitConfig

# Configure rate limiting
config = RateLimitConfig(
    requests_per_second=100,
    burst_size=20
)
limiter = RateLimiter(config)

# Use rate limiter
with limiter:
    perform_operation()
```

### Cost-Based Rate Limiting

```python
from maif.rate_limiter import CostBasedRateLimiter

# Limit by cost
limiter = CostBasedRateLimiter(max_cost_per_hour=1000)

# Operations have different costs
limiter.track_cost(operation="write", cost=1)
limiter.track_cost(operation="search", cost=5)
```

## Metrics

Track performance metrics:

```python
from maif.metrics_aggregator import (
    MetricsAggregator,
    initialize_metrics,
    get_metrics
)

# Initialize metrics
metrics = initialize_metrics(namespace="my-app")

# Record metrics
metrics.record("write_latency", 0.05)
metrics.record("blocks_written", 100)

# Get aggregated metrics
summary = get_metrics()
print(f"Average latency: {summary['write_latency_avg']}")
```

## Cost Tracking

Track resource usage:

```python
from maif.cost_tracker import (
    CostTracker,
    initialize_cost_tracking,
    with_cost_tracking
)

# Initialize tracking
tracker = initialize_cost_tracking()

# Track operation costs
@with_cost_tracking
def expensive_operation():
    # Operation code
    pass

# Get cost summary
costs = tracker.get_summary()
```

## Performance Tips

### 1. Enable Memory Mapping for Large Files

```python
encoder = MAIFEncoder("fast.maif", agent_id="fast", enable_mmap=True)
```

### 2. Use Compression

```python
encoder = MAIFEncoder("efficient.maif", agent_id="efficient", enable_compression=True)
```

### 3. Batch Operations

```python
# Don't write one at a time
for item in items:
    encoder.add_text_block(item)

# Better: batch writes are efficient
encoder.add_text_block(large_content)
```

### 4. Stream Large Files

```python
# Don't load entire file
with MAIFStreamReader("huge.maif") as reader:
    for block in reader.read_blocks():
        process(block)
```

### 5. Use Parallel Processing

```python
from maif.batch_processor import DistributedBatchProcessor

processor = DistributedBatchProcessor(num_workers=4)
results = processor.process_parallel(items)
```

## Performance Benchmarks

Typical performance characteristics:

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Write (text) | 100K blocks/sec | <1ms |
| Write (binary) | 400+ MB/sec | <5ms |
| Read (sequential) | 500+ MB/sec | <1ms |
| Search (semantic) | <50ms | - |
| Compression | 200+ MB/sec | <10ms |

*Actual performance depends on hardware and configuration.*

## Complete Example

```python
from maif.core import MAIFEncoder
from maif.batch_processor import BatchProcessor
from maif.metrics_aggregator import initialize_metrics
import time

# Initialize metrics
metrics = initialize_metrics()

# Create optimized encoder
encoder = MAIFEncoder(
    "optimized.maif",
    agent_id="optimized",
    enable_mmap=True,
    enable_compression=True,
    buffer_size=256 * 1024
)

# Batch processor
processor = BatchProcessor(batch_size=500)

# Generate test data
documents = [f"Document {i} with content..." for i in range(10000)]

# Process in batches
start = time.time()
for batch in processor.process(documents):
    for doc in batch:
        encoder.add_text_block(doc)

encoder.finalize()
elapsed = time.time() - start

print(f"Processed {len(documents)} documents in {elapsed:.2f}s")
print(f"Throughput: {len(documents)/elapsed:.0f} docs/sec")
```

## Available Performance Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `MAIFEncoder` | `maif.core` | Create files (mmap, buffer options) |
| `MAIFStreamWriter` | `maif.streaming` | Streaming writes |
| `MAIFStreamReader` | `maif.streaming` | Streaming reads |
| `BatchProcessor` | `maif.batch_processor` | Batch processing |
| `DistributedBatchProcessor` | `maif.batch_processor` | Parallel processing |
| `CompressionManager` | `maif.compression_manager` | Data compression |
| `HotBuffer` | `maif.hot_buffer` | High-speed caching |
| `RateLimiter` | `maif.rate_limiter` | Throughput control |
| `MetricsAggregator` | `maif.metrics_aggregator` | Performance metrics |
| `CostTracker` | `maif.cost_tracker` | Resource tracking |

## Next Steps

- **[Streaming →](/guide/streaming)** - Real-time processing
- **[Architecture →](/guide/architecture)** - System design
- **[API Reference →](/api/)** - Complete API documentation
