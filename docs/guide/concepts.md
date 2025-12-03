# Core Concepts

This guide introduces the fundamental concepts of MAIF and how they work together.

## What is MAIF?

**MAIF (Multimodal Artifact File Format)** is an AI-native file format designed for:

- Storing multimodal content (text, images, video, audio, embeddings)
- Built-in security and encryption
- Privacy features and data anonymization
- Semantic understanding and search
- Provenance tracking and auditing

## Key Concepts

### 1. Artifacts

An **artifact** is a MAIF file (`.maif`) that serves as a container for related content:

```python
from maif_api import create_maif

# Create an artifact
artifact = create_maif("my-agent")

# Add content
artifact.add_text("Document content")
artifact.add_image("photo.jpg")

# Save the artifact
artifact.save("my_artifact.maif")
```

Artifacts are like enhanced archives that include:
- Multiple data blocks
- Cryptographic signatures
- Metadata and relationships
- Provenance history

### 2. Blocks

**Blocks** are the fundamental units of storage within an artifact:

```
Artifact
├── Block 1 (Text)
├── Block 2 (Image)
├── Block 3 (Embeddings)
└── Block 4 (Metadata)
```

Each block contains:
- **Header**: ID, type, size, hash
- **Data**: The actual content
- **Metadata**: Additional information

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(agent_id="demo")

# Each method adds a block
encoder.add_text_block("Text content")
encoder.add_binary_block(binary_data)
encoder.add_embeddings_block(vectors)
```

### 3. Agents

An **agent ID** identifies the creator of artifacts:

```python
from maif_api import create_maif

# Agent ID associates artifacts with their creator
maif = create_maif("customer-service-agent")
```

Agents are used for:
- Tracking provenance
- Access control
- Audit logging

### 4. Privacy Levels

MAIF supports data classification:

```python
from maif.privacy import PrivacyLevel

# Classification levels
PrivacyLevel.PUBLIC        # No restrictions
PrivacyLevel.INTERNAL      # Internal only
PrivacyLevel.CONFIDENTIAL  # Requires authorization
PrivacyLevel.SECRET        # Highly restricted
PrivacyLevel.TOP_SECRET    # Maximum classification
```

### 5. Encryption

MAIF supports multiple encryption modes:

```python
from maif.privacy import EncryptionMode

# Encryption algorithms
EncryptionMode.AES_GCM           # AES-256-GCM (default)
EncryptionMode.CHACHA20_POLY1305 # ChaCha20
EncryptionMode.HOMOMORPHIC       # Homomorphic (experimental)
```

## Two API Levels

### Simple API (`maif_api`)

High-level, easy-to-use interface:

```python
from maif_api import create_maif, load_maif

# Create
maif = create_maif("agent")
maif.add_text("Hello!")
maif.save("output.maif")

# Load
maif = load_maif("output.maif")
```

### Core API (`maif.core`)

Low-level control for advanced use:

```python
from maif.core import MAIFEncoder, MAIFDecoder

# Encode
encoder = MAIFEncoder(agent_id="agent")
encoder.add_text_block("Hello!", metadata={...})
encoder.save("output.maif")

# Decode
decoder = MAIFDecoder("output.maif")
for block in decoder.read_blocks():
    process(block)
```

## Data Flow

### Writing Data

```
Application
    ↓
create_maif() / MAIFEncoder
    ↓
add_text() / add_image() / etc.
    ↓
[Privacy Engine] → Encryption/Anonymization
    ↓
[Security] → Signatures
    ↓
save() → .maif file
```

### Reading Data

```
.maif file
    ↓
load_maif() / MAIFDecoder
    ↓
[Security] → Verify signatures
    ↓
[Privacy Engine] → Decryption
    ↓
read_blocks() → Application
```

## Security Model

### Integrity

Every block has a SHA-256 hash for verification:

```python
# Verify artifact integrity
if maif.verify_integrity():
    print("File is intact")
```

### Digital Signatures

Sign data with Ed25519:

```python
from maif.security import MAIFSigner, MAIFVerifier

signer = MAIFSigner(agent_id="signer")
signature = signer.sign_data(data)

verifier = MAIFVerifier()
is_valid = verifier.verify_signature(data, signature, public_key)
```

### Access Control

Control who can access what:

```python
from maif.security import AccessControlManager

acm = AccessControlManager()
acm.set_block_permissions("block-id", {
    "read": ["user_a", "user_b"],
    "write": ["admin"]
})
```

## Privacy Features

### Encryption

Protect sensitive data:

```python
maif = create_maif("agent", enable_privacy=True)
maif.add_text("Secret", encrypt=True)
```

### Anonymization

Mask PII automatically:

```python
maif.add_text(
    "John Smith, SSN: 123-45-6789",
    anonymize=True
)
# Becomes: "[NAME], SSN: [SSN]"
```

## Semantic Features

### Embeddings

Store vector representations:

```python
maif.add_embeddings(
    [[0.1, 0.2, 0.3]],
    model_name="bert"
)
```

### Cross-Modal Attention (ACAM)

Understand relationships across modalities:

```python
maif.add_multimodal({
    "text": "A sunset",
    "tags": ["nature"]
}, use_acam=True)
```

## File Structure

A MAIF artifact consists of:

1. **MAIF File** (`.maif`): Binary file with blocks
2. **Manifest** (`_manifest.json`): Index and metadata

```
my_artifact.maif          # Binary data
my_artifact_manifest.json # Metadata, signatures
```

## Common Patterns

### Document Storage

```python
maif = create_maif("doc-agent")
maif.add_text(document_content, title="Report")
maif.save("report.maif")
```

### AI Memory

```python
maif = create_maif("ai-agent")
maif.add_text(conversation_history)
maif.add_embeddings(context_vectors)
maif.save("memory.maif")
```

### Secure Archive

```python
maif = create_maif("secure-agent", enable_privacy=True)
maif.add_text(sensitive_data, encrypt=True)
maif.save("secure.maif")
```

## Terminology

| Term | Description |
|------|-------------|
| **Artifact** | A MAIF file containing blocks |
| **Block** | Unit of storage with data and metadata |
| **Agent** | Creator/owner identifier |
| **Privacy Level** | Data classification |
| **Provenance** | Operation history |
| **ACAM** | Adaptive Cross-Modal Attention Mechanism |
| **HSC** | Hierarchical Semantic Compression |
| **CSB** | Cryptographic Semantic Binding |

## Next Steps

- **[Quick Start →](/guide/quick-start)** - Hands-on examples
- **[Getting Started →](/guide/getting-started)** - Detailed introduction
- **[Architecture →](/guide/architecture)** - System design
- **[API Reference →](/api/)** - Complete API documentation
