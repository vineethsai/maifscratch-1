# Getting Started

This guide introduces the core concepts of MAIF and shows you how to start using it for your AI applications.

## What is MAIF?

MAIF (Multimodal Artifact File Format) is an AI-native file format designed for storing, managing, and securing multimodal content with built-in:

- **Cryptographic Security**: Digital signatures, encryption, and integrity verification
- **Privacy Features**: Data anonymization, access control, and differential privacy
- **Semantic Understanding**: Embeddings, cross-modal attention, and knowledge graphs
- **Provenance Tracking**: Complete audit trail of all operations
- **Multi-modal Support**: Text, images, video, audio, and structured data

## Core Concepts

### Artifacts and Blocks

A MAIF file is called an **artifact**. Each artifact contains multiple **blocks**:

```
┌─────────────────────────────────────────┐
│           MAIF Artifact (.maif)          │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │ Block 1: Text Data              │    │
│  │  - content, metadata, hash      │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ Block 2: Image Data             │    │
│  │  - binary, metadata, hash       │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ Block 3: Embeddings             │    │
│  │  - vectors, model info, hash    │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  Manifest: index, signatures, metadata   │
└─────────────────────────────────────────┘
```

### Two API Levels

MAIF provides two levels of API:

1. **Simple API** (`maif_api`): Easy-to-use high-level interface
2. **Core API** (`maif.core`): Low-level control for advanced use cases

## Quick Example: Simple API

The simplest way to use MAIF:

```python
from maif_api import MAIF, create_maif, load_maif

# Create a new MAIF artifact
maif = create_maif("my-agent")

# Add text content
maif.add_text("This is my first document.", title="Introduction")
maif.add_text("AI is transforming how we work.", title="AI Overview")

# Save the artifact
maif.save("my_first_artifact.maif")
print("Artifact saved!")

# Load and verify
loaded = load_maif("my_first_artifact.maif")
if loaded.verify_integrity():
    print("Integrity verified!")
```

## Core API Example

For more control, use the core classes directly:

```python
from maif.core import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner

# Create an encoder
encoder = MAIFEncoder(agent_id="my-agent")

# Add blocks with full control
block_id = encoder.add_text_block(
    "Detailed content with metadata",
    metadata={
        "author": "System",
        "version": "1.0",
        "tags": ["important", "reviewed"]
    }
)
print(f"Added block: {block_id}")

# Save with manifest
encoder.save("detailed_artifact.maif")

# Read back with decoder
decoder = MAIFDecoder("detailed_artifact.maif")
blocks = list(decoder.read_blocks())
print(f"Read {len(blocks)} blocks")
```

## Adding Different Content Types

### Text Content

```python
from maif_api import create_maif

maif = create_maif("content-demo")

# Simple text
maif.add_text("Hello, world!")

# Text with title
maif.add_text("This is the body of my document.", title="Document Title")

# Text with encryption (requires privacy enabled)
maif_private = create_maif("private-agent", enable_privacy=True)
maif_private.add_text(
    "Sensitive information here",
    title="Confidential",
    encrypt=True
)
```

### Image Content

```python
from maif_api import create_maif

maif = create_maif("media-demo")

# Add image from file
maif.add_image("photo.jpg", title="My Photo")

# Add with metadata extraction
maif.add_image("document.png", title="Scanned Doc", extract_metadata=True)

maif.save("media_artifact.maif")
```

### Video Content

```python
from maif_api import create_maif

maif = create_maif("video-demo")

# Add video
maif.add_video("presentation.mp4", title="Quarterly Review")

maif.save("video_artifact.maif")
```

### Embeddings

```python
from maif_api import create_maif

maif = create_maif("embedding-demo")

# Add pre-computed embeddings
embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
maif.add_embeddings(embeddings, model_name="my-model", compress=True)

maif.save("embedding_artifact.maif")
```

### Multimodal Content

```python
from maif_api import create_maif

maif = create_maif("multimodal-demo")

# Add multimodal content (uses ACAM - Adaptive Cross-Modal Attention)
maif.add_multimodal({
    "text": "A beautiful sunset over the ocean",
    "description": "Nature photography",
    "tags": ["sunset", "ocean", "nature"]
}, title="Sunset Scene", use_acam=True)

maif.save("multimodal_artifact.maif")
```

## Privacy and Security

### Enable Privacy Features

```python
from maif_api import create_maif

# Create with privacy enabled
maif = create_maif("secure-agent", enable_privacy=True)

# Add encrypted content
maif.add_text(
    "Patient record: John Doe, DOB: 1990-01-15",
    title="Medical Record",
    encrypt=True,
    anonymize=True  # Also anonymize PII
)

maif.save("secure_artifact.maif")

# Check privacy report
report = maif.get_privacy_report()
print(f"Privacy enabled: {report.get('privacy_enabled', False)}")
```

### Using the Privacy Engine Directly

```python
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode
from maif.core import MAIFEncoder

# Create privacy engine
privacy_engine = PrivacyEngine()

# Create encoder with privacy
encoder = MAIFEncoder(
    agent_id="privacy-demo",
    enable_privacy=True,
    privacy_engine=privacy_engine
)

# Add text with privacy settings
encoder.add_text_block(
    "Sensitive data here",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    anonymize=True
)

encoder.save("privacy_demo.maif")
```

## Working with the Core API

### MAIFEncoder

The encoder creates and writes MAIF files:

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(agent_id="encoder-demo")

# Add text block
text_id = encoder.add_text_block("Hello, MAIF!", metadata={"lang": "en"})

# Add binary block (for images, files, etc.)
with open("image.png", "rb") as f:
    image_data = f.read()
binary_id = encoder.add_binary_block(image_data, metadata={"type": "image/png"})

# Add embeddings block
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
embed_id = encoder.add_embeddings_block(embeddings, metadata={"model": "bert"})

# Save the artifact
encoder.save("encoder_demo.maif")
```

### MAIFDecoder

The decoder reads MAIF files:

```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("encoder_demo.maif")

# Read all blocks
for block in decoder.read_blocks():
    print(f"Block ID: {block.block_id}")
    print(f"Block Type: {block.block_type}")
    print(f"Metadata: {block.metadata}")

# Get specific block by ID
block = decoder.get_block_by_id("some-block-id")

# Get blocks by type
text_blocks = decoder.get_blocks_by_type("TEXT")
```

### Security with Signing

```python
from maif.security import MAIFSigner, MAIFVerifier

# Create signer
signer = MAIFSigner(agent_id="signer-demo")

# Sign data
data = b"Important content to sign"
signature = signer.sign_data(data)
print(f"Signature created: {signature[:20]}...")

# Add provenance entry
signer.add_provenance_entry("create", "block-001")

# Verify signature
verifier = MAIFVerifier()
public_key = signer.get_public_key_pem()
is_valid = verifier.verify_signature(data, signature, public_key)
print(f"Signature valid: {is_valid}")
```

## Searching Content

Once you have a MAIF file, you can search its contents:

```python
from maif_api import load_maif

# Load existing artifact
maif = load_maif("my_artifact.maif")

# Search for content
results = maif.search("machine learning", top_k=5)
for result in results:
    print(f"Found: {result['text'][:100]}...")
    print(f"Score: {result.get('score', 'N/A')}")
```

## Complete Example: Document Management System

Here's a more complete example showing a document management workflow:

```python
from maif_api import create_maif, load_maif
from datetime import datetime

def create_document_artifact(title: str, content: str, author: str):
    """Create a new document artifact."""
    maif = create_maif(f"doc-{author}", enable_privacy=True)
    
    # Add document metadata
    maif.add_text(f"Title: {title}\nAuthor: {author}\nDate: {datetime.now()}", 
                  title="Metadata")
    
    # Add main content
    maif.add_text(content, title=title)
    
    # Save with timestamp
    filename = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.maif"
    maif.save(filename)
    
    return filename

def verify_and_read(filename: str):
    """Load, verify, and read a document."""
    maif = load_maif(filename)
    
    # Verify integrity
    if not maif.verify_integrity():
        raise ValueError("Document integrity check failed!")
    
    # List contents
    contents = maif.get_content_list()
    print(f"Document contains {len(contents)} blocks:")
    for item in contents:
        print(f"  - {item.get('title', 'Untitled')} ({item['type']})")
    
    return maif

# Usage
filename = create_document_artifact(
    title="Quarterly Report",
    content="Revenue increased by 25% this quarter...",
    author="analyst"
)
print(f"Created: {filename}")

doc = verify_and_read(filename)
```

## Best Practices

1. **Always verify integrity** after loading a MAIF file
2. **Enable privacy** for sensitive data
3. **Use meaningful titles** for blocks to aid organization
4. **Include metadata** for better searchability
5. **Use the simple API** for most use cases; core API for advanced needs

## Next Steps

- **[Quick Start →](/guide/quick-start)** - More hands-on examples
- **[Security Model →](/guide/security-model)** - Deep dive into security
- **[Privacy Framework →](/guide/privacy)** - Privacy features explained
- **[API Reference →](/api/)** - Complete API documentation
