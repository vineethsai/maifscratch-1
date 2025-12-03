# MAIF - Complete Feature Documentation

## Overview

MAIF (Multimodal Artifact File Format) is a comprehensive, AI-native file format designed for secure, versioned, and semantically-rich content storage. This document provides a complete overview of all features and capabilities.

## Core Architecture

### File Format Structure
- **Hierarchical Block System**: Similar to ISO BMFF/MP4 with typed blocks
- **Binary Format**: Efficient binary representation with streaming support
- **Metadata Layer**: Rich metadata with custom schemas and provenance tracking
- **Compression**: Multiple algorithms (zlib, LZMA, Brotli) with semantic-aware compression
- **Versioning**: Block-level versioning with append-on-write architecture

### Security Features
- **Digital Signatures**: RSA/ECDSA signatures with certificate chains
- **Cryptographic Provenance**: Immutable audit trails with cryptographic verification
- **Access Control**: Role-based permissions and encryption
- **Integrity Verification**: Multi-level checksums and validation
- **Classified Data Support** (NEW):
  - Mandatory Access Control (Bell-LaPadula model)
  - PKI/CAC/PIV authentication
  - Hardware MFA integration
  - FIPS 140-2 compliant encryption
  - AWS CloudWatch immutable audit trails

## Feature Categories

### 1. Core Functionality (`maif.core`)

#### MAIFEncoder
```python
from maif import MAIFEncoder

encoder = MAIFEncoder(agent_id="my_agent")
encoder.add_text_block("Hello, MAIF!")
encoder.add_binary_block(image_data, "image_data")
encoder.build_maif("output.maif", "manifest.json")
```

**Features:**
- Text and binary block encoding
- Metadata attachment
- Automatic hash generation
- Version tracking
- Agent attribution

#### MAIFDecoder
```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("file.maif", "manifest.json")
blocks = list(decoder.blocks)
metadata = decoder.manifest
```

**Features:**
- Content extraction
- Metadata parsing
- Version history access
- Dependency resolution

### 2. Security & Provenance (`maif.security`)

#### Digital Signatures
```python
from maif.security import MAIFSigner, MAIFVerifier

signer = MAIFSigner(private_key_path="key.pem", agent_id="signer")
signer.add_provenance_entry("create", block_hash)
signed_manifest = signer.sign_maif_manifest(manifest)

verifier = MAIFVerifier()
is_valid = verifier.verify_maif_signature(signed_manifest)
```

**Features:**
- RSA/ECDSA signature support
- Certificate chain validation
- Provenance chain tracking with cryptographic hash chaining
- Timestamp verification
- Non-repudiation guarantees

#### Privacy-Enabled Security
```python
from maif_api import create_maif
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

# Create MAIF with privacy enabled
maif = create_maif("secure-agent", enable_privacy=True)

# Add encrypted content
maif.add_text(
    "Sensitive data here",
    encrypt=True,
    anonymize=True  # Remove PII automatically
)

# Save with digital signature
maif.save("secure.maif", sign=True)

# Verify integrity
loaded = maif.load("secure.maif")
is_valid = loaded.verify_integrity()
```

**Features:**
- Privacy levels (PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED)
- AES-GCM and ChaCha20-Poly1305 encryption
- Automatic PII detection and anonymization
- Digital signatures for non-repudiation
- Cryptographic hash chains for integrity
- Block-level access control

### 3. Semantic Processing (`maif.semantic` & `maif.semantic_optimized`)

#### Embeddings & Knowledge Graphs
```python
from maif.semantic import SemanticEmbedder, KnowledgeGraphBuilder

# Text embeddings
embedder = SemanticEmbedder()
embedding = embedder.embed_text("Hello world")
print(f"Embedding dimension: {len(embedding.vector)}")

# Knowledge graph construction
kg_builder = KnowledgeGraphBuilder()
kg_builder.add_triple("entity1", "knows", "entity2")
graph = kg_builder.build()
```

**Features:**
- Multimodal embeddings (text, image, audio)
- Knowledge graph construction with triples
- Semantic similarity search
- Cross-modal attention mechanisms (ACAM algorithm)
- Entity relationship modeling
- Hierarchical Semantic Compression (HSC)
- Cryptographic Semantic Binding (CSB)

### 4. Compression (`maif.compression`)

#### Advanced Compression
```python
from maif.compression_manager import CompressionManager, CompressionType

compressor = CompressionManager()
compressed = compressor.compress(data, CompressionType.BROTLI)
decompressed = compressor.decompress(compressed)
```

**Supported Algorithms:**
- **zlib**: Fast, general-purpose compression
- **LZMA**: High compression ratio
- **Brotli**: Web-optimized compression
- **Custom**: Pluggable compression algorithms

**Features:**
- Semantic-aware compression
- Delta compression for versions
- Automatic algorithm selection
- Compression ratio optimization

### 5. Binary Format (`maif.binary_format`)

#### Low-Level Binary Operations
```python
from maif.block_storage import BlockStorage, BlockType

# Writing
storage = BlockStorage("file.maif")
block_id = storage.add_block(
    block_type=BlockType.TEXT_DATA,
    data=b"Hello world",
    metadata={"title": "Example"}
)

# Reading
block = storage.get_block(block_id)
data = block.data
```

**Features:**
- Efficient binary serialization
- Streaming support
- Random access capabilities
- Header/footer validation
- Cross-platform compatibility

### 6. Validation & Repair (`maif.validation`)

#### File Validation
```python
from maif.validation import MAIFValidator

validator = MAIFValidator()
report = validator.validate_file("file.maif", "manifest.json")

# Validation results
print(f"Valid: {report.is_valid}")
print(f"Errors: {report.errors}")
print(f"Warnings: {report.warnings}")
```

**Validation Checks:**
- File format integrity
- Block consistency
- Signature verification
- Dependency validation
- Schema compliance
- Performance analysis

**Repair Capabilities:**
- Checksum correction
- Missing block recovery
- Dependency resolution
- Format migration
- Corruption detection

### 7. Metadata Management (`maif.metadata`)

#### Rich Metadata
```python
from maif.metadata import MetadataManager

metadata_mgr = MetadataManager()
metadata = metadata_mgr.create_metadata(
    agent_id="my_agent",
    content_type="text/plain",
    tags=["important"],
    custom={"priority": "high"}
)
```

**Features:**
- Hierarchical metadata structure
- Custom schema support
- Dependency tracking
- Provenance records
- Statistical analysis
- Standards compliance

### 8. Streaming & Performance (`maif.streaming`)

#### High-Performance Streaming
```python
from maif.streaming import StreamingEngine
from maif.core import MAIFDecoder

# Stream processing
decoder = MAIFDecoder("large_file.maif")
streaming = StreamingEngine(buffer_size=8192)

for block in decoder.stream_blocks():
    streaming.process_block(block)
```

**Features:**
- Memory-mapped file access
- Parallel block processing
- Configurable buffering
- Async/await support
- Performance profiling
- Cache management

### 9. Integration & Conversion

#### Using MAIF with AI Frameworks
```python
from maif_api import create_maif, load_maif

# Create MAIF artifact
maif = create_maif("my_agent")

# Add text content
maif.add_text("Document content here", title="Document 1")

# Add multimodal content
maif.add_multimodal({
    "text": "Image description",
    "type": "image_metadata"
})

# Save with signature
maif.save("output.maif", sign=True)

# Load and search
loaded = load_maif("output.maif")
results = loaded.search("relevant content", top_k=5)
```

**LangGraph Integration:**
```python
# See examples/langgraph/ for complete multi-agent RAG example
from maif_api import create_maif

# Use MAIF as durable memory for agents
session_maif = create_maif("langgraph-session")
session_maif.add_text(f"User query: {query}")
session_maif.add_text(f"Retrieved chunks: {chunks}")
session_maif.add_text(f"Model response: {answer}")
session_maif.save("session.maif", sign=True)
```

### 10. Forensics & Analysis (`maif.forensics`)

#### Digital Forensics
```python
from maif.forensics import ForensicsAnalyzer
from maif.core import MAIFDecoder

decoder = MAIFDecoder("suspicious.maif")
analyzer = ForensicsAnalyzer()
report = analyzer.analyze(decoder)

print(f"Total blocks: {report.total_blocks}")
print(f"Suspicious activities: {len(report.suspicious_activities)}")
print(f"Timeline events: {len(report.timeline)}")
```

**Forensic Capabilities:**
- Timeline reconstruction
- Agent activity analysis
- Anomaly detection
- Evidence collection
- Chain of custody
- Tamper detection

### 11. Command Line Interface

#### CLI Tools
```bash
# Create MAIF file
maif create output.maif --text "Hello" --file data.txt --sign

# Verify MAIF file
maif verify file.maif --verbose --repair

# Analyze MAIF file
maif analyze file.maif --forensic --timeline --agents

# Extract content
maif extract file.maif --output-dir ./extracted --type all
```

**Available Commands:**
- `maif create`: Create new MAIF files
- `maif verify`: Validate and repair files
- `maif analyze`: Comprehensive analysis
- `maif extract`: Content extraction

## Advanced Use Cases

### 1. AI Model Artifacts
```python
# Store model weights, metadata, and provenance
encoder = MAIFEncoder(agent_id="training_system")
encoder.add_binary_block(model_weights, "model_weights")
encoder.add_metadata_block({
    "model_type": "transformer",
    "training_data": "dataset_v1.2",
    "accuracy": 0.95,
    "training_time": "4h 32m"
})
```

### 2. Document Versioning
```python
# Track document changes with full history
encoder = MAIFEncoder(agent_id="document_editor")
for version in document_versions:
    block_id = encoder.add_text_block(version.content)
    encoder.add_version_metadata(block_id, version.metadata)
```

### 3. Multimedia Collections
```python
# Store mixed media with semantic relationships
encoder = MAIFEncoder(agent_id="media_curator")
text_id = encoder.add_text_block(description)
image_id = encoder.add_binary_block(image_data, "image_data")
encoder.add_relationship(text_id, "describes", image_id)
```

### 4. Scientific Data
```python
# Research data with provenance and validation
encoder = MAIFEncoder(agent_id="research_lab")
encoder.add_binary_block(experiment_data, "scientific_data")
encoder.add_provenance_chain(experiment_metadata)
encoder.add_validation_schema(data_schema)
```

## Performance Characteristics

### Compression Ratios
- **Text**: 60-80% reduction (Brotli)
- **Binary**: 20-40% reduction (LZMA)
- **Embeddings**: 30-50% reduction (custom)

### Streaming Performance
- **Sequential Read**: 500+ MB/s
- **Parallel Read**: 1.2+ GB/s (4 workers)
- **Random Access**: <1ms seek time

### Validation Speed
- **Basic Validation**: 100+ MB/s
- **Full Forensic**: 50+ MB/s
- **Repair Operations**: 25+ MB/s

## Security Guarantees

### Cryptographic Strength
- **Signatures**: RSA-2048/ECDSA-256
- **Hashing**: SHA-256/SHA-3
- **Encryption**: AES-256-GCM

### Provenance Integrity
- **Immutable History**: Append-only structure
- **Chain Validation**: Cryptographic linking
- **Timestamp Verification**: RFC 3161 compliance

## Standards Compliance

### File Format Standards
- **ISO BMFF**: Base media file format compatibility
- **RFC 3161**: Timestamp protocol
- **JSON Schema**: Metadata validation
- **MIME Types**: Content type identification

### Security Standards
- **FIPS 140-2**: Cryptographic module validation
- **Common Criteria**: Security evaluation
- **NIST Guidelines**: Cryptographic best practices

## Future Roadmap

### Upcoming Features
- **Advanced Analytics**: ML-based anomaly detection
- **Enhanced Cloud Integration**: Extended AWS service support
- **Real-time Streaming**: Live data ingestion improvements
- **Novel Algorithm Enhancements**: Optimized ACAM, HSC, and CSB implementations
- **Cross-Modal AI**: Advanced deep semantic understanding

## Getting Started

### Installation
```bash
# Clone and install from source
git clone https://github.com/vineethsai/maifscratch-1.git
cd maifscratch-1
pip install -e .
```

### Quick Start
```python
from maif_api import create_maif, load_maif

# Create
maif = create_maif("quickstart")
maif.add_text("Hello, MAIF!")
maif.save("hello.maif", sign=True)

# Read
loaded = load_maif("hello.maif")
content = loaded.get_content_list()
for item in content:
    print(item.get("text"))  # "Hello, MAIF!"
```

### Examples
- `examples/basic/simple_api_demo.py`: Basic operations
- `examples/basic/basic_usage.py`: Core encoder/decoder usage
- `examples/advanced/novel_algorithms_demo.py`: ACAM, HSC, CSB algorithms
- `examples/security/privacy_demo.py`: Privacy features
- `examples/langgraph/`: Multi-agent RAG system (featured)

## Support & Documentation

- **Documentation**: See the `/docs` directory for comprehensive guides
- **Examples**: Check `/examples` for usage demonstrations
- **Tests**: Run `pytest tests/` to verify functionality

---

*MAIF - The complete AI-native file format for the future of trustworthy AI systems.*