# Artifact

The `Artifact` class provides a high-level, object-oriented interface for working with MAIF files. It represents a logical collection of related content items with consistent security and compression settings.

## Quick Start

```python
from maif.artifact import Artifact

# Create an artifact
artifact = Artifact(name="my-memories", agent_id="my-agent")

# Add content
artifact.add_text("Important fact to remember")
artifact.add_image("path/to/image.png", title="screenshot")

# Save to file
artifact.save("memories.maif")

# Load an existing artifact
loaded = Artifact.load("memories.maif")

# Access content
for item in loaded.get_content():
    print(item)
```

## Class Definition

```python
class Artifact:
    def __init__(
        self,
        name: str = "untitled",
        agent_id: str = "default_agent",
        enable_embeddings: bool = True
    ):
        """
        Create a new artifact.

        Args:
            name: Descriptive name for the artifact
            agent_id: Agent identifier for provenance
            enable_embeddings: Generate semantic embeddings for content
        """
```

## Adding Content

### add_text

Add text content to the artifact.

```python
def add_text(
    self,
    text: str,
    title: Optional[str] = None,
    language: str = "en",
    **metadata
) -> str:
    """
    Add text content.

    Args:
        text: The text content to add
        title: Optional title for the text
        language: Language code (default: "en")
        **metadata: Additional custom metadata

    Returns:
        Block ID of the added content
    """
```

**Example:**

```python
from maif.artifact import Artifact

artifact = Artifact(name="notes")

# Simple text
block_id = artifact.add_text("Hello, World!")

# Text with metadata
block_id = artifact.add_text(
    "Important meeting notes",
    title="Q4 Planning",
    language="en",
    priority="high"
)
```

### add_image

Add image content to the artifact.

```python
def add_image(
    self,
    image_path: str,
    title: Optional[str] = None,
    extract_metadata: bool = True
) -> str:
    """
    Add image content.

    Args:
        image_path: Path to the image file
        title: Optional title
        extract_metadata: Extract EXIF metadata

    Returns:
        Block ID of the added content
    """
```

**Example:**

```python
block_id = artifact.add_image(
    "screenshot.png",
    title="Dashboard Screenshot"
)
```

### add_video

Add video content to the artifact.

```python
def add_video(
    self,
    video_path: str,
    title: Optional[str] = None,
    extract_metadata: bool = True
) -> str:
    """
    Add video content.

    Args:
        video_path: Path to the video file
        title: Optional title
        extract_metadata: Extract video metadata

    Returns:
        Block ID of the added content
    """
```

### add_document

Add document content to the artifact.

```python
def add_document(
    self,
    document_path: str,
    title: Optional[str] = None,
    **metadata
) -> str:
    """
    Add document content.

    Args:
        document_path: Path to the document file
        title: Optional title
        **metadata: Additional custom metadata

    Returns:
        Block ID of the added content
    """
```

### add_data

Add arbitrary binary data to the artifact.

```python
def add_data(
    self,
    data: bytes,
    title: Optional[str] = None,
    data_type: str = "binary"
) -> str:
    """
    Add arbitrary data.

    Args:
        data: Raw bytes
        title: Optional title
        data_type: Type descriptor (e.g., "json", "csv", "binary")

    Returns:
        Block ID of the added content
    """
```

**Example:**

```python
import json

# Add JSON data
json_data = json.dumps({"key": "value"}).encode()
block_id = artifact.add_data(
    json_data,
    title="Configuration",
    data_type="json"
)
```

## Reading Content

### get_content

Get content from the artifact.

```python
def get_content(self) -> List[Dict]:
    """
    Get all content from the artifact.

    Returns:
        List of content dictionaries
    """
```

### get_text_content

Get all text content as decoded strings.

```python
def get_text_content(self) -> List[str]:
    """Get all text content as strings."""
```

### get_metadata

Get artifact metadata.

```python
def get_metadata(self) -> Dict:
    """
    Get artifact metadata.

    Returns:
        Dictionary with artifact information
    """
```

## Persistence

### save

Save the artifact to a MAIF file.

```python
def save(self, filepath: str) -> bool:
    """
    Save the artifact to a file.

    Args:
        filepath: Path where to save

    Returns:
        True if successful
    """
```

### load (class method)

Load an artifact from a MAIF file.

```python
@classmethod
def load(cls, filepath: str) -> 'Artifact':
    """
    Load an artifact from a file.

    Args:
        filepath: Path to the MAIF file

    Returns:
        Loaded Artifact instance
    """
```

**Example:**

```python
# Save
artifact = Artifact(name="session")
artifact.add_text("Hello")
artifact.save("session.maif")

# Load
loaded = Artifact.load("session.maif")
print(loaded.get_text_content())
```

## Simple API Alternative

For most use cases, the simple `maif_api` is recommended:

```python
from maif_api import create_maif, load_maif

# Create and populate
maif = create_maif("my-agent")
maif.add_text("Important information")
maif.add_multimodal({"text": "Description", "type": "note"})
maif.save("data.maif", sign=True)

# Load and use
loaded = load_maif("data.maif")
content = loaded.get_content_list()

# Verify integrity
is_valid = loaded.verify_integrity()

# Search content
results = loaded.search("important", top_k=5)
```

## Complete Example

```python
from maif.artifact import Artifact
# Or use the simpler API:
# from maif_api import create_maif, load_maif

# Create a new artifact for an agent session
session = Artifact(
    name="agent-session-001",
    agent_id="support-bot-1"
)

# Log the conversation
session.add_text("Customer: I need help with my order")
session.add_text("Agent: I'd be happy to help. What's your order number?")
session.add_text("Customer: Order #12345")

# Add a screenshot the customer shared
session.add_image(
    "order_screenshot.png",
    title="Order Screenshot"
)

# Save the session
session.save("sessions/session-001.maif")

# Later: Load and analyze
loaded = Artifact.load("sessions/session-001.maif")

print(f"Session: {loaded.name}")
print(f"\nConversation:")
for text in loaded.get_text_content():
    print(f"  {text}")
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Artifact name |
| `agent_id` | `str` | Agent identifier |
| `created_at` | `float` | Creation timestamp |
| `modified_at` | `float` | Last modification timestamp |

## See Also

- [Simple API (maif_api)](/api/) - Recommended for most use cases
- [MAIFEncoder](/api/core/encoder-decoder) - Low-level encoding
- [MAIFDecoder](/api/core/encoder-decoder) - Low-level decoding
