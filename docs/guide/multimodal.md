# Multi-modal Data

MAIF handles multiple data modalities - text, images, video, audio, and structured data - with unified storage and cross-modal relationships.

## Overview

MAIF supports storing and processing:

- **Text**: Documents, conversations, logs
- **Images**: Photos, diagrams, screenshots
- **Video**: Recordings, presentations
- **Audio**: Speech, music, sound effects
- **Embeddings**: Vector representations
- **Structured Data**: JSON, metadata

## Adding Different Content Types

### Text Content

```python
from maif_api import create_maif

maif = create_maif("multimodal-agent")

# Simple text
maif.add_text("Hello, world!")

# Text with metadata
maif.add_text(
    "This is an important document",
    title="Important Doc"
)

# Encrypted text
maif_secure = create_maif("secure-agent", enable_privacy=True)
maif_secure.add_text(
    "Confidential information",
    title="Secret",
    encrypt=True
)

maif.save("text_content.maif")
```

### Image Content

```python
from maif_api import create_maif

maif = create_maif("image-agent")

# Add image from file path
maif.add_image("photo.jpg", title="My Photo")

# Add with metadata extraction
maif.add_image(
    "document.png",
    title="Scanned Document",
    extract_metadata=True  # Extract EXIF, dimensions, etc.
)

maif.save("images.maif")
```

### Video Content

```python
from maif_api import create_maif

maif = create_maif("video-agent")

# Add video file
maif.add_video("presentation.mp4", title="Q4 Presentation")

# Multiple videos
maif.add_video("intro.mp4", title="Introduction")
maif.add_video("demo.mp4", title="Product Demo")

maif.save("videos.maif")
```

### Embeddings

```python
from maif_api import create_maif

maif = create_maif("embedding-agent")

# Add embedding vectors
embeddings = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 0.1, 0.2, 0.3]
]

# With compression
maif.add_embeddings(
    embeddings,
    model_name="sentence-transformers",
    compress=True
)

maif.save("embeddings.maif")
```

### Multimodal Content

Combine multiple modalities with cross-modal attention:

```python
from maif_api import create_maif

maif = create_maif("combined-agent")

# Add multimodal content with ACAM processing
maif.add_multimodal({
    "text": "A beautiful sunset over the ocean",
    "description": "Nature photography from California",
    "tags": ["sunset", "ocean", "nature", "photography"],
    "metadata": {
        "location": "California",
        "date": "2024-01-15"
    }
}, title="Ocean Sunset", use_acam=True)

maif.save("multimodal.maif")
```

## Using Core API

For more control, use the encoder directly:

### MAIFEncoder Methods

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(agent_id="core-demo")

# Add text block
text_id = encoder.add_text_block(
    "Document content",
    metadata={"type": "document", "version": "1.0"}
)

# Add binary block (for images, files)
with open("image.png", "rb") as f:
    image_id = encoder.add_binary_block(
        f.read(),
        metadata={"type": "image/png", "filename": "image.png"}
    )

# Add image block
with open("photo.jpg", "rb") as f:
    photo_id = encoder.add_image_block(
        f.read(),
        metadata={"title": "Photo"}
    )

# Add video block
with open("video.mp4", "rb") as f:
    video_id = encoder.add_video_block(
        f.read(),
        metadata={"title": "Video"}
    )

# Add audio block
with open("audio.mp3", "rb") as f:
    audio_id = encoder.add_audio_block(
        f.read(),
        metadata={"title": "Audio"}
    )

# Add embeddings block
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
embed_id = encoder.add_embeddings_block(
    embeddings,
    metadata={"model": "bert", "dimension": 3}
)

encoder.save("core_multimodal.maif")
```

### Cross-Modal Block

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(agent_id="cross-modal")

# Add cross-modal content with ACAM
cross_modal_id = encoder.add_cross_modal_block(
    {
        "text": "Description text",
        "visual": "visual_features",
        "semantic": "semantic_data"
    },
    use_enhanced_acam=True,
    metadata={"type": "multimodal_content"}
)

encoder.save("cross_modal.maif")
```

## Reading Multimodal Content

### Using Simple API

```python
from maif_api import load_maif

maif = load_maif("multimodal.maif")

# List all content
contents = maif.get_content_list()
for item in contents:
    print(f"Type: {item['type']}, Title: {item.get('title', 'Untitled')}")

# Search across all content
results = maif.search("sunset")
for result in results:
    print(f"Found: {result}")
```

### Using Decoder

```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("multimodal.maif")

# Read all blocks
for block in decoder.read_blocks():
    print(f"Block ID: {block.block_id}")
    print(f"Type: {block.block_type}")
    print(f"Metadata: {block.metadata}")

# Get blocks by type
text_blocks = decoder.get_blocks_by_type("TEXT")
image_blocks = decoder.get_blocks_by_type("IMAG")
embedding_blocks = decoder.get_blocks_by_type("EMBD")
```

## Cross-Modal Attention

MAIF uses ACAM (Adaptive Cross-Modal Attention) for multimodal understanding:

```python
from maif.semantic import CrossModalAttention

# Create attention mechanism
acam = CrossModalAttention()

# Process multimodal inputs
# ACAM computes attention weights between different modalities
```

### With Enhanced ACAM

```python
from maif.semantic_optimized import AdaptiveCrossModalAttention

# Enhanced version with more features
acam = AdaptiveCrossModalAttention(
    num_heads=8,
    temperature=0.1
)
```

## Complete Example: Document Archive

```python
from maif_api import create_maif, load_maif
import os

def create_document_archive(folder_path: str, output_path: str):
    """Create a MAIF archive from a folder of documents."""
    maif = create_maif("archive-agent")
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        if filename.endswith('.txt'):
            with open(filepath, 'r') as f:
                maif.add_text(f.read(), title=filename)
        
        elif filename.endswith(('.jpg', '.png', '.gif')):
            maif.add_image(filepath, title=filename)
        
        elif filename.endswith(('.mp4', '.avi', '.mov')):
            maif.add_video(filepath, title=filename)
    
    maif.save(output_path)
    return output_path

def list_archive_contents(archive_path: str):
    """List contents of a MAIF archive."""
    maif = load_maif(archive_path)
    
    print(f"Archive: {archive_path}")
    print("-" * 40)
    
    for item in maif.get_content_list():
        print(f"  {item['type']:10} | {item.get('title', 'Untitled')}")

# Usage
create_document_archive("./documents", "archive.maif")
list_archive_contents("archive.maif")
```

## Best Practices

### 1. Use Appropriate Block Types

```python
# Text for documents
encoder.add_text_block(text_content)

# Binary for raw files
encoder.add_binary_block(file_data)

# Specific types for better metadata
encoder.add_image_block(image_data)
encoder.add_video_block(video_data)
encoder.add_audio_block(audio_data)
```

### 2. Include Metadata

```python
# Always include useful metadata
maif.add_image(
    "photo.jpg",
    title="Event Photo"
)

encoder.add_text_block(
    content,
    metadata={
        "author": "System",
        "created": "2024-01-15",
        "tags": ["important", "reviewed"]
    }
)
```

### 3. Use Compression for Large Content

```python
# Compress large embeddings
maif.add_embeddings(embeddings, compress=True)
```

### 4. Enable Privacy for Sensitive Content

```python
maif = create_maif("agent", enable_privacy=True)
maif.add_image("sensitive.jpg", title="Confidential")
```

## Block Type Reference

| Content Type | Simple API Method | Encoder Method | Block Type |
|-------------|-------------------|----------------|------------|
| Text | `add_text()` | `add_text_block()` | `TEXT` |
| Image | `add_image()` | `add_image_block()` | `IMAG` |
| Video | `add_video()` | `add_video_block()` | `VIDO` |
| Embeddings | `add_embeddings()` | `add_embeddings_block()` | `EMBD` |
| Multimodal | `add_multimodal()` | `add_cross_modal_block()` | `MMOD` |
| Binary | - | `add_binary_block()` | `BINA` |
| Audio | - | `add_audio_block()` | `AUDI` |

## Next Steps

- **[Semantic Understanding →](/guide/semantic)** - Cross-modal attention details
- **[Streaming →](/guide/streaming)** - Processing large files
- **[API Reference →](/api/)** - Complete API documentation
