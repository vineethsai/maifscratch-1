# MAIF Simple API Guide

The MAIF SDK provides an easy-to-use interface for working with Multimodal Artifact File Format (MAIF) files.

## Quick Start

### Installation

```bash
# Clone from GitHub
git clone https://github.com/vineethsai/maifscratch-1.git
cd maifscratch-1

# Install
pip install -e .
```

### Basic Usage

```python
from maif_api import create_maif, load_maif

# Create a new MAIF artifact
maif = create_maif("my_agent")

# Add content
maif.add_text("Hello world!", title="Greeting")
maif.add_text("This is important data", title="Data")

# Save to file
maif.save("my_artifact.maif")

# Load existing artifact
loaded = load_maif("my_artifact.maif")

# Verify integrity
assert loaded.verify_integrity()
```

## API Reference

### `MAIF` Class

The main class for creating and manipulating MAIF files.

#### Constructor

```python
from maif_api import create_maif

maif = create_maif(
    agent_id="my_agent",      # Unique identifier for the agent
    enable_privacy=False      # Enable privacy features
)
```

#### Key Methods

##### `add_text(text, title=None, encrypt=False, anonymize=False)`

Add text content to the artifact.

**Parameters:**
- `text` (str): Text content to add
- `title` (str, optional): Title for the content
- `encrypt` (bool): Whether to encrypt the content
- `anonymize` (bool): Whether to anonymize PII

**Returns:** Block ID (str)

**Example:**
```python
block_id = maif.add_text(
    "This is confidential information",
    title="Secret Document",
    encrypt=True
)
```

##### `add_image(image_path, title=None, extract_metadata=True)`

Add an image to the artifact.

**Parameters:**
- `image_path` (str): Path to image file
- `title` (str, optional): Title for the image
- `extract_metadata` (bool): Whether to extract image metadata

**Returns:** Block ID (str)

**Example:**
```python
block_id = maif.add_image("photo.jpg", title="Vacation Photo")
```

##### `add_video(video_path, title=None, extract_metadata=True)`

Add a video to the artifact.

**Parameters:**
- `video_path` (str): Path to video file
- `title` (str, optional): Title for the video
- `extract_metadata` (bool): Whether to extract video metadata

**Returns:** Block ID (str)

##### `add_multimodal(content, title=None, use_acam=True)`

Add multimodal content using ACAM (Adaptive Cross-Modal Attention) processing.

**Parameters:**
- `content` (dict): Dictionary with different modality data
- `title` (str, optional): Title for the content
- `use_acam` (bool): Whether to use ACAM processing

**Returns:** Block ID (str)

**Example:**
```python
block_id = maif.add_multimodal({
    "text": "A mountain landscape",
    "image_description": "Snow-capped peaks",
    "location": "Swiss Alps",
    "weather": "Clear sky"
}, title="Alpine Scene")
```

##### `add_embeddings(embeddings, model_name="custom", compress=True)`

Add embeddings with optional HSC (Hierarchical Semantic Compression).

**Parameters:**
- `embeddings` (List[List[float]]): List of embedding vectors
- `model_name` (str): Name of the model that generated embeddings
- `compress` (bool): Whether to use HSC compression

**Returns:** Block ID (str)

##### `save(filepath, sign=True)`

Save MAIF to file.

**Parameters:**
- `filepath` (str): Output file path
- `sign` (bool): Whether to cryptographically sign the file

**Returns:** bool (success status)

##### `get_content_list()`

Get list of all content blocks.

**Returns:** List[Dict] - List of content metadata

##### `search(query, top_k=5)`

Search content using semantic similarity.

**Parameters:**
- `query` (str): Search query
- `top_k` (int): Number of results to return

**Returns:** List[Dict] - Search results with similarity scores

##### `verify_integrity()`

Verify MAIF file integrity.

**Returns:** bool - True if integrity check passes

##### `get_privacy_report()`

Get privacy report for the MAIF.

**Returns:** Dict - Privacy status and statistics

### Class Methods

##### `MAIF.load(filepath)`

Load existing MAIF file.

**Parameters:**
- `filepath` (str): Path to MAIF file

**Returns:** MAIF instance

**Example:**
```python
from maif_api import load_maif

maif = load_maif("existing_artifact.maif")
```

## Convenience Functions

### `create_maif(agent_id="default_agent", enable_privacy=False)`

Create a new MAIF instance.

```python
from maif_api import create_maif

maif = create_maif("my_agent")
```

### `load_maif(filepath)`

Load existing MAIF file.

```python
from maif_api import load_maif

maif = load_maif("artifact.maif")
```

### `quick_text_maif(text, output_path, title=None)`

Quickly create a MAIF with just text content.

```python
from maif_api import quick_text_maif

quick_text_maif("Hello world!", "hello.maif", title="Greeting")
```

### `quick_multimodal_maif(content, output_path, title=None)`

Quickly create a MAIF with multimodal content.

```python
from maif_api import quick_multimodal_maif

quick_multimodal_maif(
    {"text": "Description", "category": "Example"},
    "multimodal.maif",
    title="Demo"
)
```

## Examples

### Example 1: Simple Document

```python
from maif_api import create_maif

# Create MAIF
maif = create_maif("document_agent")

# Add content
maif.add_text("This is my document content.", title="My Document")

# Save
maif.save("document.maif")
```

### Example 2: Multimodal Content with ACAM

```python
from maif_api import create_maif

# Create MAIF
maif = create_maif("multimodal_agent")

# Add cross-modal content
maif.add_multimodal({
    "text": "Product description: High-quality headphones",
    "image_description": "Black over-ear headphones on white background",
    "price": "$199.99",
    "category": "Electronics",
    "features": ["Noise cancelling", "Wireless", "30-hour battery"]
}, title="Product Listing", use_acam=True)

# Save
maif.save("product.maif")
```

### Example 3: Privacy-Enabled MAIF

```python
from maif_api import create_maif

# Create MAIF with privacy
maif = create_maif("secure_agent", enable_privacy=True)

# Add encrypted content
maif.add_text(
    "Patient John Doe has condition XYZ",
    title="Medical Record",
    encrypt=True,
    anonymize=True
)

# Save with signing
maif.save("medical_record.maif", sign=True)

# Check privacy report
report = maif.get_privacy_report()
print(f"Encrypted blocks: {report.get('encrypted_blocks', 0)}")
```

### Example 4: Search and Retrieval

```python
from maif_api import load_maif

# Load existing MAIF
maif = load_maif("knowledge_base.maif")

# Search content
results = maif.search("machine learning algorithms", top_k=3)

for result in results:
    print(f"Found: {result}")
```

### Example 5: Working with Embeddings

```python
from maif_api import create_maif

# Create MAIF
maif = create_maif("embedding_agent")

# Add embeddings with compression
embeddings = [
    [0.1, 0.2, 0.3] * 128,  # 384-dimensional
    [0.4, 0.5, 0.6] * 128   # 384-dimensional
]

maif.add_embeddings(
    embeddings,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    compress=True  # Uses HSC compression
)

# Save
maif.save("embeddings.maif")
```

## Advanced Usage

### Using Core Classes Directly

For more control, use the core MAIF classes:

```python
from maif.core import MAIFEncoder, MAIFDecoder

# Encoding
encoder = MAIFEncoder(agent_id="advanced_agent")
encoder.add_text_block("Content", metadata={"custom": "data"})
encoder.save("advanced.maif")

# Decoding
decoder = MAIFDecoder("advanced.maif")
for block in decoder.read_blocks():
    print(f"Block type: {block.block_type}")
```

### Security Features

```python
from maif.security import MAIFSigner, MAIFVerifier

# Sign data
signer = MAIFSigner(agent_id="secure_agent")
signer.add_provenance_entry("create", "artifact_001")

# Verify signatures
verifier = MAIFVerifier()
# verifier.verify_signature(data, signature)
```

### Privacy Features

```python
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

# Create privacy engine
privacy = PrivacyEngine()

# Encrypt data
encrypted = privacy.encrypt_data(
    b"sensitive data",
    EncryptionMode.AES_GCM
)

# Anonymize text
anonymized = privacy.anonymize_data("John Doe, SSN: 123-45-6789")
```

## Novel Algorithms

The API automatically uses MAIF's novel algorithms:

- **ACAM (Adaptive Cross-Modal Attention)**: Used in `add_multimodal()` when `use_acam=True`
- **HSC (Hierarchical Semantic Compression)**: Used in `add_embeddings()` when `compress=True`
- **CSB (Cryptographic Semantic Binding)**: Automatically applied for integrity verification

## Error Handling

```python
from maif_api import create_maif, load_maif

try:
    maif = create_maif("my_agent")
    maif.add_text("Hello world!")
    success = maif.save("output.maif")
    
    if not success:
        print("Failed to save MAIF")
        
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Use descriptive agent IDs**: Choose meaningful identifiers for your agents
2. **Add titles to content**: Makes content easier to identify and search
3. **Enable privacy for sensitive data**: Use `enable_privacy=True` for confidential content
4. **Verify integrity**: Always check `verify_integrity()` after loading files
5. **Use compression for embeddings**: Enable `compress=True` for large embedding sets
6. **Sign important files**: Keep `sign=True` for audit trails and provenance

## Next Steps

- [Getting Started Guide](/guide/getting-started)
- [API Reference](/api/)
- [Examples](/examples/)
