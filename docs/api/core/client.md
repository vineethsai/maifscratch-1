# MAIFClient

The `MAIFClient` class provides native file operations for MAIF artifacts with memory-mapped I/O and write buffering.

## Quick Start

```python
from maif.client import MAIFClient

# Create a client
client = MAIFClient(agent_id="my-agent")

# Create artifact through client
artifact = client.create_artifact("my-artifact")

# Write content
with client.open_file("data.maif", "w") as encoder:
    encoder.add_text_block("Hello, World!")
    encoder.save("data.maif", "manifest.json")

# Read content
content = client.read_content("data.maif")
print(content)

# Always close when done
client.close()
```

## Class Definition

```python
class MAIFClient:
    def __init__(
        self,
        agent_id: str = "default_agent",
        enable_mmap: bool = True,
        buffer_size: int = 64 * 1024
    ):
        """
        Initialize MAIF client.

        Args:
            agent_id: Unique identifier for the agent
            enable_mmap: Enable memory-mapped I/O (default: True)
            buffer_size: Write buffer size in bytes (default: 64KB)
        """
```

## Methods

### create_artifact

Create a new artifact attached to this client.

```python
def create_artifact(self, name: str) -> Artifact:
    """
    Create a new artifact with this client's configuration.

    Args:
        name: Name for the artifact

    Returns:
        Artifact instance configured with this client
    """
```

**Example:**

```python
from maif.client import MAIFClient

client = MAIFClient(agent_id="my-agent")
artifact = client.create_artifact("session-memory")
artifact.add_text("User asked about weather")
artifact.save("session.maif")
client.close()
```

### open_file

Context manager for file access with encoder/decoder.

```python
@contextmanager
def open_file(self, filepath: str, mode: str = 'r'):
    """
    Open a MAIF file.

    Args:
        filepath: Path to the MAIF file
        mode: File mode ('r' for read, 'w' for write)

    Yields:
        MAIFEncoder (for 'w') or MAIFDecoder (for 'r')
    """
```

**Example:**

```python
# Write access
with client.open_file("data.maif", "w") as encoder:
    encoder.add_text_block("Hello world")
    encoder.save("data.maif", "manifest.json")

# Read access  
with client.open_file("data.maif", "r") as decoder:
    for block in decoder.read_blocks():
        print(f"Block type: {block.block_type}")
```

### write_content

Write content to a MAIF file with buffering.

```python
def write_content(
    self,
    filepath: str,
    content: bytes,
    block_type: str = "TEXT"
) -> str:
    """
    Write content to MAIF file.

    Args:
        filepath: Path to the MAIF file
        content: Raw content bytes to write
        block_type: Type of block

    Returns:
        Block ID of the written content
    """
```

### read_content

Read content from a MAIF file.

```python
def read_content(self, filepath: str) -> List[Dict]:
    """
    Read content from MAIF file.

    Args:
        filepath: Path to the MAIF file

    Returns:
        List of block dictionaries with data and metadata
    """
```

### get_file_info

Get information about a MAIF file.

```python
def get_file_info(self, filepath: str) -> Dict:
    """
    Get information about a MAIF file.

    Args:
        filepath: Path to the MAIF file

    Returns:
        Dictionary with file information
    """
```

### flush_all_buffers

Flush all pending write buffers to disk.

```python
def flush_all_buffers(self) -> Dict:
    """
    Flush all pending write buffers.

    Returns:
        Dictionary with flush results
    """
```

### close

Close all open files and clean up resources.

```python
def close(self):
    """Close all open files and release resources."""
```

## Context Manager Support

The client supports the context manager protocol:

```python
from maif.client import MAIFClient

with MAIFClient(agent_id="my-agent") as client:
    artifact = client.create_artifact("data")
    artifact.add_text("Hello")
    artifact.save("data.maif")
# Client is automatically closed here
```

## Simple API Alternative

For most use cases, the simple API is recommended:

```python
from maif_api import create_maif, load_maif

# Create and save
maif = create_maif("my-agent")
maif.add_text("Hello, World!")
maif.save("data.maif", sign=True)

# Load and read
loaded = load_maif("data.maif")
content = loaded.get_content_list()
for item in content:
    print(item)
```

## Performance Tips

1. **Use the simple API**: `maif_api` handles most use cases efficiently
2. **Reuse instances**: Creating new instances has overhead
3. **Batch operations**: Add multiple items before saving
4. **Close when done**: Always close clients to release resources

```python
from maif_api import create_maif

# Good: Batch operations
maif = create_maif("batch-agent")
for item in large_dataset:
    maif.add_text(str(item))
maif.save("batch.maif")  # Single save at the end

# Avoid: Saving after each item
for item in large_dataset:
    maif = create_maif("slow-agent")
    maif.add_text(str(item))
    maif.save("slow.maif")  # Overhead for each item!
```
