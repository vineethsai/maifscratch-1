# Block Structure

::: danger DEPRECATED
This page is deprecated. For the latest and most accurate documentation, please visit **[DeepWiki - MAIF File Format](https://deepwiki.com/vineethsai/maif/2.1-maif-file-format)**.

DeepWiki documentation is auto-generated from the codebase and always up-to-date.
:::

Blocks are the fundamental unit of data storage in MAIF. This guide explains block types, structure, and how to work with blocks.

## Block Overview

Every MAIF file consists of blocks, each containing:

- **Header**: ID, type, size, hash, and Ed25519 signature
- **Data**: The actual content
- **Metadata**: Additional information

```
┌─────────────────────────────────────┐
│            Block Header              │
│  - Block ID (UUID)                  │
│  - Block Type (4 chars)             │
│  - Data Size + Flags                │
│  - Hash (SHA-256)                   │
│  - Previous Block Hash              │
│  - Ed25519 Signature (64 bytes)     │
│  - Timestamp                        │
├─────────────────────────────────────┤
│            Block Data                │
│  - Content (text, binary, vectors)  │
│  - Compression info (if compressed) │
│  - Encryption info (if encrypted)   │
├─────────────────────────────────────┤
│           Block Metadata             │
│  - Custom JSON metadata             │
│  - Privacy level                    │
└─────────────────────────────────────┘
```

Each block is signed immediately when written, making it immutable and tamper-evident.

## Block Types

MAIF defines several block types:

```python
from maif.block_types import BlockType

# Available block types
BlockType.TEXT_DATA   # "TEXT" - Text content
BlockType.IMAGE_DATA  # "IMAG" - Image binary
BlockType.VIDEO_DATA  # "VIDO" - Video binary
BlockType.AUDIO_DATA  # "AUDI" - Audio binary
BlockType.EMBEDDINGS  # "EMBD" - Vector embeddings
BlockType.METADATA    # "META" - Metadata blocks
BlockType.BINARY      # "BINA" - Generic binary data
```

## Creating Blocks

### Using MAIFEncoder

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder("blocks.maif", agent_id="block-demo")

# Text block
text_id = encoder.add_text_block(
    "Hello, MAIF!",
    metadata={"language": "en"}
)

# Binary block
with open("file.bin", "rb") as f:
    binary_id = encoder.add_binary_block(
        f.read(),
        metadata={"filename": "file.bin"}
    )

# Image block
with open("image.png", "rb") as f:
    image_id = encoder.add_image_block(
        f.read(),
        metadata={"format": "PNG"}
    )

# Embeddings block
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
embed_id = encoder.add_embeddings_block(
    embeddings,
    metadata={"model": "bert", "dimension": 3}
)

encoder.finalize()
```

### Using Simple API

```python
from maif_api import create_maif

maif = create_maif("simple-demo")

# Each add method creates a block
text_id = maif.add_text("Hello!", title="Greeting")
image_id = maif.add_image("photo.jpg", title="Photo")
embed_id = maif.add_embeddings([[0.1, 0.2]], model_name="custom")

maif.save("simple_blocks.maif")
```

## Block Methods

### Text Blocks

```python
# Simple text
encoder.add_text_block("Content")

# With metadata
encoder.add_text_block(
    "Content",
    metadata={
        "title": "My Document",
        "author": "System",
        "tags": ["important"]
    }
)

# With privacy
from maif.privacy import PrivacyLevel, EncryptionMode

encoder.add_text_block(
    "Sensitive content",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    anonymize=True
)

# Update existing block
encoder.update_text_block(
    block_id,
    "Updated content"
)
```

### Binary Blocks

```python
# Generic binary data
encoder.add_binary_block(
    binary_data,
    metadata={"type": "application/octet-stream"}
)
```

### Image Blocks

```python
# Image with metadata extraction
encoder.add_image_block(
    image_data,
    extract_metadata=True,  # Extract EXIF, dimensions
    metadata={"title": "Photo"}
)
```

### Video Blocks

```python
# Video content
encoder.add_video_block(
    video_data,
    extract_metadata=True,
    metadata={"title": "Recording"}
)
```

### Audio Blocks

```python
# Audio content
encoder.add_audio_block(
    audio_data,
    metadata={"title": "Sound Effect"}
)
```

### Embedding Blocks

```python
# Vector embeddings
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
encoder.add_embeddings_block(
    embeddings,
    metadata={
        "model": "sentence-transformers",
        "dimension": 3
    }
)

# Compressed embeddings
encoder.add_compressed_embeddings_block(
    large_embeddings,
    metadata={"compressed": True}
)
```

### Multimodal Blocks

```python
# Cross-modal content
encoder.add_cross_modal_block(
    {
        "text": "Description",
        "visual": visual_features,
        "audio": audio_features
    },
    use_enhanced_acam=True,
    metadata={"type": "multimodal"}
)

# Multimodal block
encoder.add_multimodal_block(
    {
        "text_data": text,
        "image_data": image
    },
    metadata={"combined": True}
)
```

### Special Blocks

```python
# Security block
encoder.add_security_block(security_data, metadata={})

# Provenance block
encoder.add_provenance_block(provenance_data, metadata={})

# Access control block
encoder.add_access_control_block(acl_data, metadata={})

# Lifecycle block
encoder.add_lifecycle_block(lifecycle_data, metadata={})

# Knowledge graph block
encoder.add_knowledge_graph_block(kg_data, metadata={})
```

## Reading Blocks

### Using MAIFDecoder

```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("blocks.maif")

# Read all blocks
for block in decoder.read_blocks():
    print(f"ID: {block.block_id}")
    print(f"Type: {block.block_type}")
    print(f"Metadata: {block.metadata}")

# Get block by ID
block = decoder.get_block_by_id("block-123")

# Get latest version of block
block = decoder.get_latest_block_by_id("block-123")

# Get blocks by type
text_blocks = decoder.get_blocks_by_type("TEXT")
image_blocks = decoder.get_blocks_by_type("IMAG")
embedding_blocks = decoder.get_blocks_by_type("EMBD")

# Get block data
data = decoder.get_block_data("block-123")

# Get block metadata
metadata = decoder.get_block_metadata("block-123")
```

### Block Access Methods

```python
# By hash
block = decoder.get_block_by_hash("sha256:...")

# By offset
block = decoder.get_block_by_offset(1024)

# By index
block = decoder.get_block_by_index(0)

# By version
block = decoder.get_block_by_version("block-id", version=2)

# By timestamp
block = decoder.get_block_by_timestamp(timestamp)

# By agent
blocks = decoder.get_block_by_agent("agent-id")

# By metadata
blocks = decoder.get_block_by_metadata({"key": "value"})
```

### Version History

```python
# Get all versions of a block
history = decoder.get_version_history("block-123")
for version in history:
    print(f"Version: {version.version}")
    print(f"Timestamp: {version.timestamp}")
```

## Block Storage

Direct block storage operations:

```python
from maif.block_storage import BlockStorage

# Create block storage
storage = BlockStorage("artifact.maif", enable_mmap=True)

# Add block
block_id = storage.add_block(
    block_type="TEXT",
    data=b"Content",
    metadata={"title": "Block"}
)

# Get block
block = storage.get_block(block_id)

# Update block
storage.update_block(block_id, new_data)
```

## Block Metadata

### Standard Metadata Fields

```python
metadata = {
    "title": "Document Title",
    "description": "Block description",
    "author": "System",
    "created": "2024-01-15T10:00:00Z",
    "modified": "2024-01-15T11:00:00Z",
    "tags": ["tag1", "tag2"],
    "version": "1.0",
    "privacy_level": "confidential"
}
```

### Custom Metadata

```python
encoder.add_text_block(
    "Content",
    metadata={
        "standard": "value",
        "custom_field": "custom_value",
        "nested": {
            "key": "value"
        },
        "list": [1, 2, 3]
    }
)
```

## Block Integrity

Each block includes a SHA-256 hash for integrity verification:

```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("artifact.maif")

# Blocks are verified during read
for block in decoder.read_blocks():
    # Hash verification happens automatically
    print(f"Block {block.block_id}: verified")
```

## Block with Privacy

```python
from maif.core import MAIFEncoder
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

privacy = PrivacyEngine()
encoder = MAIFEncoder(
    agent_id="private-demo",
    enable_privacy=True,
    privacy_engine=privacy
)

# Encrypted block
encoder.add_text_block(
    "Secret content",
    privacy_level=PrivacyLevel.SECRET,
    encryption_mode=EncryptionMode.AES_GCM
)

# Anonymized block
encoder.add_text_block(
    "John Smith's SSN: 123-45-6789",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    anonymize=True
)

encoder.finalize()
```

## Complete Example

```python
from maif.core import MAIFEncoder, MAIFDecoder
from maif.privacy import PrivacyLevel

# Create encoder (file path is first argument in v3 API)
encoder = MAIFEncoder("complete.maif", agent_id="complete-demo")

# Add various blocks
# Text block
doc_id = encoder.add_text_block(
    "This is a document about AI",
    metadata={
        "title": "AI Overview",
        "author": "System",
        "tags": ["ai", "overview"]
    }
)

# Embedding block
embeddings = [[0.1, 0.2, 0.3, 0.4] for _ in range(10)]
embed_id = encoder.add_embeddings_block(
    embeddings,
    metadata={
        "model": "sentence-transformers",
        "dimension": 4,
        "source": doc_id
    }
)

# Finalize and save
encoder.finalize()

# Read back
decoder = MAIFDecoder("complete.maif")

print("Blocks in artifact:")
for block in decoder.read_blocks():
    print(f"  {block.block_type}: {block.block_id}")
    print(f"    Metadata: {block.metadata}")

# Get specific blocks
text_blocks = decoder.get_blocks_by_type("TEXT")
print(f"\nFound {len(text_blocks)} text blocks")
```

## Best Practices

1. **Use appropriate block types** for better organization and querying
2. **Include meaningful metadata** for searchability
3. **Use versioning** when updating blocks
4. **Enable privacy** for sensitive content
5. **Verify integrity** when reading blocks

## Next Steps

- **[Architecture →](/guide/architecture)** - System design
- **[API Reference →](/api/)** - Complete API documentation
