# Streaming Data Processing

This example demonstrates high-throughput streaming with MAIF's memory-mapped I/O and efficient block processing.

## Overview

MAIF supports efficient streaming for large datasets through:

- **Memory-mapped I/O**: Direct file access without loading into memory
- **Block streaming**: Process blocks as they're read
- **Buffered writes**: Efficient batch writing
- **Compression**: On-the-fly compression for storage efficiency

## Running the Demo

```bash
cd examples/advanced
python3 streaming_demo.py
```

## Code Example

### Streaming Writes

```python
from maif.core import MAIFEncoder

# Create encoder with optimized settings
encoder = MAIFEncoder(
    agent_id="streaming_agent",
    enable_mmap=True,        # Memory-mapped I/O
    enable_compression=True  # Compress blocks
)

# Stream data in batches
data_stream = generate_large_dataset()  # Your data generator

for batch_idx, batch in enumerate(data_stream):
    # Add batch to encoder
    for item in batch:
        encoder.add_text_block(
            item['content'],
            metadata={
                "batch": batch_idx,
                "type": item['type']
            }
        )
    
    # Periodically save checkpoints
    if batch_idx % 100 == 0:
        print(f"Processed batch {batch_idx}")

# Save final artifact
encoder.save("large_dataset.maif")
```

### Streaming Reads

```python
from maif.core import MAIFDecoder

# Open decoder for streaming
decoder = MAIFDecoder("large_dataset.maif")

# Stream blocks without loading entire file
for block in decoder.read_blocks():
    # Process each block individually
    if block.block_type == "TEXT":
        process_text(block.data.decode('utf-8'))
    elif block.block_type == "BINA":
        process_binary(block.data)
    
    # Memory efficient: only one block in memory at a time
```

### Processing Pipeline

```python
from maif_api import create_maif, load_maif
import os

class StreamProcessor:
    """High-throughput stream processor using MAIF."""
    
    def __init__(self, output_dir: str = "streams"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.buffer_size = 1000
        self.current_buffer = []
        self.file_counter = 0
    
    def process_stream(self, data_source):
        """Process a stream of data."""
        for item in data_source:
            self.current_buffer.append(item)
            
            if len(self.current_buffer) >= self.buffer_size:
                self._flush_buffer()
        
        # Flush remaining items
        if self.current_buffer:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Write buffer to MAIF file."""
        self.file_counter += 1
        output_path = f"{self.output_dir}/stream_{self.file_counter:04d}.maif"
        
        artifact = create_maif(f"stream-{self.file_counter}")
        
        for item in self.current_buffer:
            artifact.add_text(
                item.get('content', str(item)),
                title=item.get('title', f"Item {self.file_counter}")
            )
        
        artifact.save(output_path, sign=True)
        print(f"Saved {len(self.current_buffer)} items to {output_path}")
        
        self.current_buffer = []

# Usage
processor = StreamProcessor()

def data_generator():
    for i in range(10000):
        yield {"content": f"Data item {i}", "title": f"Item {i}"}

processor.process_stream(data_generator())
```

## What You'll Learn

- Memory-efficient processing of large datasets
- Streaming reads and writes with MAIF
- Checkpointing strategies
- Compression for storage efficiency

## Performance Tips

1. **Use memory mapping** for files larger than available RAM
2. **Batch operations** for better throughput
3. **Enable compression** to reduce I/O
4. **Stream blocks** instead of loading entire files

## Next Steps

- [Performance Guide](/guide/performance) - Optimization techniques
- [LangGraph Example](/examples/langgraph-rag) - Multi-agent system
