# Quick Start

Get up and running with MAIF in under 5 minutes.

## Installation

```bash
# Clone and install
git clone https://github.com/vineethsai/maifscratch-1.git
cd maifscratch-1
pip install -e ".[full]"
```

## Your First MAIF File

### Using the Simple API

The easiest way to create a MAIF file:

```python
from maif_api import create_maif

# Create a new artifact
maif = create_maif("my-agent")

# Add some content
maif.add_text("Hello, MAIF!", title="Greeting")
maif.add_text("This is my first artifact.", title="Introduction")

# Save it
maif.save("hello.maif")
print("Created hello.maif!")
```

### Load and Verify

```python
from maif_api import load_maif

# Load existing artifact
maif = load_maif("hello.maif")

# Verify integrity
if maif.verify_integrity():
    print("✓ Integrity verified!")

# List contents
for block in maif.get_content_list():
    print(f"  - {block.get('title', 'Untitled')} ({block['type']})")
```

## Adding Different Content Types

### Text Content

```python
from maif_api import create_maif

maif = create_maif("content-agent")

# Simple text
maif.add_text("Plain text content")

# Text with title
maif.add_text("Document body here...", title="My Document")

maif.save("text_demo.maif")
```

### Images

```python
from maif_api import create_maif

maif = create_maif("image-agent")

# Add image from file
maif.add_image("photo.jpg", title="My Photo")
maif.add_image("diagram.png", title="Architecture Diagram")

maif.save("images_demo.maif")
```

### Video

```python
from maif_api import create_maif

maif = create_maif("video-agent")

# Add video file
maif.add_video("presentation.mp4", title="Q4 Presentation")

maif.save("video_demo.maif")
```

### Embeddings

```python
from maif_api import create_maif

maif = create_maif("embedding-agent")

# Add pre-computed embeddings
embeddings = [
    [0.1, 0.2, 0.3, 0.4],  # First vector
    [0.5, 0.6, 0.7, 0.8],  # Second vector
]
maif.add_embeddings(embeddings, model_name="custom-model", compress=True)

maif.save("embeddings_demo.maif")
```

### Multimodal Content

```python
from maif_api import create_maif

maif = create_maif("multimodal-agent")

# Add multimodal content with cross-modal attention
maif.add_multimodal({
    "text": "A sunset over the mountains",
    "description": "Nature photography from Colorado",
    "tags": ["sunset", "mountains", "nature"]
}, title="Mountain Sunset", use_acam=True)

maif.save("multimodal_demo.maif")
```

## Privacy and Encryption

### Enable Privacy Features

```python
from maif_api import create_maif

# Create with privacy enabled
maif = create_maif("secure-agent", enable_privacy=True)

# Add encrypted content
maif.add_text(
    "Confidential: Annual budget is $1.5M",
    title="Budget Info",
    encrypt=True
)

# Add with anonymization
maif.add_text(
    "Patient John Doe, SSN: 123-45-6789",
    title="Patient Record",
    encrypt=True,
    anonymize=True  # PII will be masked
)

maif.save("secure_demo.maif")

# Check privacy report
report = maif.get_privacy_report()
print(f"Privacy enabled: {report.get('privacy_enabled')}")
```

## Using the Core API

For more control, use the encoder/decoder directly:

### MAIFEncoder

```python
from maif.core import MAIFEncoder

# Create encoder
encoder = MAIFEncoder(agent_id="core-demo")

# Add text block with metadata
block_id = encoder.add_text_block(
    "Detailed content here",
    metadata={
        "author": "system",
        "version": "1.0",
        "tags": ["important"]
    }
)
print(f"Created block: {block_id}")

# Add binary content
with open("image.png", "rb") as f:
    encoder.add_binary_block(f.read(), metadata={"type": "image/png"})

# Save
encoder.save("core_demo.maif")
```

### MAIFDecoder

```python
from maif.core import MAIFDecoder

# Open and read
decoder = MAIFDecoder("core_demo.maif")

# Read all blocks
for block in decoder.read_blocks():
    print(f"Block: {block.block_id}")
    print(f"  Type: {block.block_type}")
    print(f"  Metadata: {block.metadata}")

# Get specific block types
text_blocks = decoder.get_blocks_by_type("TEXT")
print(f"Found {len(text_blocks)} text blocks")
```

## Digital Signatures

```python
from maif.security import MAIFSigner, MAIFVerifier

# Create signer
signer = MAIFSigner(agent_id="signer-demo")

# Sign data
data = b"Important content"
signature = signer.sign_data(data)

# Add provenance
signer.add_provenance_entry("create", "document-001")

# Verify
verifier = MAIFVerifier()
public_key = signer.get_public_key_pem()
is_valid = verifier.verify_signature(data, signature, public_key)
print(f"Valid signature: {is_valid}")
```

## Searching Content

```python
from maif_api import load_maif

maif = load_maif("my_artifact.maif")

# Search for relevant content
results = maif.search("machine learning", top_k=5)

for result in results:
    print(f"Found: {result['text'][:100]}...")
    print(f"Score: {result.get('score', 0):.3f}")
```

## Quick Functions

For one-off operations:

```python
from maif_api import quick_text_maif, quick_multimodal_maif

# Create text MAIF in one line
quick_text_maif("Hello, World!", "hello.maif", title="Greeting")

# Create multimodal MAIF in one line
quick_multimodal_maif(
    {"text": "Sunset photo", "tags": ["nature"]},
    "photo.maif",
    title="Sunset"
)
```

## Complete Example: Note Taking App

```python
from maif_api import create_maif, load_maif
from datetime import datetime

class NoteBook:
    def __init__(self, name: str):
        self.name = name
        self.filename = f"{name}.maif"
        self.maif = create_maif(f"notebook-{name}")
    
    def add_note(self, content: str, title: str = None):
        """Add a new note."""
        if title is None:
            title = f"Note {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.maif.add_text(content, title=title)
        print(f"Added: {title}")
    
    def save(self):
        """Save the notebook."""
        self.maif.save(self.filename)
        print(f"Saved to {self.filename}")
    
    @classmethod
    def open(cls, name: str):
        """Open existing notebook."""
        notebook = cls(name)
        notebook.maif = load_maif(notebook.filename)
        return notebook
    
    def list_notes(self):
        """List all notes."""
        for block in self.maif.get_content_list():
            print(f"  • {block.get('title', 'Untitled')}")

# Usage
notebook = NoteBook("work")
notebook.add_note("Remember to review the Q4 report", title="Task")
notebook.add_note("Meeting with team at 3pm", title="Reminder")
notebook.save()

# Later...
notebook = NoteBook.open("work")
notebook.list_notes()
```

## What's Next?

- **[Getting Started →](/guide/getting-started)** - Deeper dive into concepts
- **[Security Model →](/guide/security-model)** - Understand security features
- **[Privacy Framework →](/guide/privacy)** - Privacy and compliance
- **[Examples →](/examples/)** - More complete examples
- **[API Reference →](/api/)** - Full API documentation
