# Real-time Processing

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki](https://deepwiki.com/vineethsai/maif)**.
:::

MAIF provides streaming capabilities for efficient processing of large files and real-time data ingestion. This guide covers streaming operations.

## Overview

MAIF's streaming features include:

- **Stream Writers**: Write large files incrementally
- **Stream Readers**: Read files without loading entirely into memory
- **Batch Processing**: Process data in batches
- **Memory-Mapped I/O**: Efficient file access

## Stream Writer

Write MAIF files incrementally:

```python
from maif.streaming import MAIFStreamWriter

# Create stream writer
with MAIFStreamWriter("large_output.maif") as writer:
    # Write chunks incrementally
    for i in range(1000):
        writer.write_text_chunk(f"Document {i} content...")

print("Streaming write complete!")
```

### Writing Different Content Types

```python
from maif.streaming import MAIFStreamWriter

with MAIFStreamWriter("multimodal_stream.maif") as writer:
    # Write text chunks
    writer.write_text_chunk("First document")
    writer.write_text_chunk("Second document")
    
    # Write binary data (images, etc.)
    with open("image.png", "rb") as f:
        writer.write_binary_chunk(f.read())
```

## Stream Reader

Read MAIF files efficiently:

```python
from maif.streaming import MAIFStreamReader

# Create stream reader
with MAIFStreamReader("large_file.maif") as reader:
    # Read blocks one at a time
    for block in reader.read_blocks():
        print(f"Block: {block.block_id}")
        print(f"Type: {block.block_type}")
        # Process block without loading entire file
```

### Selective Reading

```python
from maif.streaming import MAIFStreamReader

with MAIFStreamReader("data.maif") as reader:
    # Read only text blocks
    for block in reader.read_blocks_by_type("TEXT"):
        print(f"Text: {block.data[:100]}...")
    
    # Read blocks in range
    for block in reader.read_blocks_range(start=0, end=100):
        process(block)
```

## Batch Processing

Process data in efficient batches:

```python
from maif.batch_processor import BatchProcessor, batch_process

# Create batch processor
processor = BatchProcessor(batch_size=100)

# Process items in batches
items = ["doc1", "doc2", "doc3", ...]

for batch in processor.process(items):
    # Each batch contains up to 100 items
    for item in batch:
        process(item)
```

### Stream Batch Processor

```python
from maif.batch_processor import StreamBatchProcessor

# For streaming data
processor = StreamBatchProcessor(
    batch_size=100,
    timeout=5.0  # Flush batch after 5 seconds
)

# Add items as they arrive
for item in data_stream:
    processor.add(item)
    
    # Process when batch is ready
    if processor.is_ready():
        batch = processor.get_batch()
        process_batch(batch)

# Process any remaining items
final_batch = processor.flush()
```

### Distributed Batch Processing

```python
from maif.batch_processor import DistributedBatchProcessor

# For multi-node processing
processor = DistributedBatchProcessor(
    batch_size=1000,
    num_workers=4
)

# Process with parallelism
results = processor.process_parallel(items)
```

## Memory-Mapped I/O

For large files, enable memory mapping:

```python
from maif.core import MAIFEncoder, MAIFDecoder

# Enable memory mapping when creating
encoder = MAIFEncoder(
    "large.maif",
    agent_id="mmap-demo",
    enable_mmap=True
)

# Large write operations are more efficient
for i in range(10000):
    encoder.add_text_block(f"Content {i}")

encoder.finalize()
```

### Reading with Memory Mapping

```python
from maif.block_storage import BlockStorage

# Use block storage with mmap
storage = BlockStorage("large.maif", enable_mmap=True)

# Efficient random access
block = storage.get_block("block-id")
```

## Write Buffering

The encoder uses write buffering for performance:

```python
from maif.core import MAIFEncoder

# Configure buffer size
encoder = MAIFEncoder(
    "buffered.maif",
    agent_id="buffered",
    buffer_size=128 * 1024  # 128KB buffer
)

# Writes are buffered
for i in range(1000):
    encoder.add_text_block(f"Item {i}")

# Buffer is flushed on finalize
encoder.finalize()
```

## Hot Buffer

For high-throughput scenarios:

```python
from maif.hot_buffer import HotBuffer

# Create hot buffer for caching
buffer = HotBuffer(max_size=1024 * 1024)  # 1MB

# Add items to buffer
buffer.put("key1", data1)
buffer.put("key2", data2)

# Retrieve from buffer
data = buffer.get("key1")

# Buffer handles eviction automatically
```

## Performance Tips

### 1. Use Streaming for Large Files

```python
# Instead of loading entire file
from maif.streaming import MAIFStreamReader

with MAIFStreamReader("huge.maif") as reader:
    for block in reader.read_blocks():
        process(block)
```

### 2. Enable Memory Mapping

```python
encoder = MAIFEncoder("fast.maif", agent_id="fast", enable_mmap=True)
```

### 3. Batch Operations

```python
from maif.batch_processor import batch_process

# Process in batches
results = batch_process(items, batch_size=100)
```

### 4. Use Compression

```python
encoder = MAIFEncoder(
    agent_id="compressed",
    enable_compression=True
)
```

## Complete Streaming Example

```python
from maif.streaming import MAIFStreamWriter, MAIFStreamReader
from maif.batch_processor import BatchProcessor

# Write large dataset in batches
processor = BatchProcessor(batch_size=100)
documents = [f"Document {i}" for i in range(10000)]

with MAIFStreamWriter("large_dataset.maif") as writer:
    for batch in processor.process(documents):
        for doc in batch:
            writer.write_text_chunk(doc)

print("Written 10,000 documents")

# Read back efficiently
with MAIFStreamReader("large_dataset.maif") as reader:
    count = 0
    for block in reader.read_blocks():
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} blocks")

print(f"Total blocks: {count}")
```

## Available Streaming Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `MAIFStreamWriter` | `maif.streaming` | Incremental file writing |
| `MAIFStreamReader` | `maif.streaming` | Efficient file reading |
| `BatchProcessor` | `maif.batch_processor` | Batch processing |
| `StreamBatchProcessor` | `maif.batch_processor` | Streaming batches |
| `DistributedBatchProcessor` | `maif.batch_processor` | Parallel processing |
| `HotBuffer` | `maif.hot_buffer` | High-speed caching |
| `BlockStorage` | `maif.block_storage` | Low-level block access |

## Next Steps

- **[Performance →](/guide/performance)** - Optimization techniques
- **[Multimodal Data →](/guide/multimodal)** - Working with different data types
- **[API Reference →](/api/)** - Complete API documentation
