# MAIF (Multimodal Artifact File Format)

Cryptographically-secure, auditable file format for AI agent data with provenance tracking.

## Overview

MAIF provides:
- **Cryptographic provenance** - Hash-chained blocks for tamper-evident audit trails
- **Multi-agent support** - Shared artifacts with agent-specific logging
- **Multimodal storage** - Text, embeddings, images, video, knowledge graphs
- **Security** - FIPS 140-2 compliant encryption, AWS KMS integration
- **Performance** - Memory-mapped I/O, streaming, compression

## Technical Architecture

### Core Components

1. **Container Format** ([`maif/core.py`](maif/core.py), [`maif/binary_format.py`](maif/binary_format.py))
   - ISO BMFF-inspired hierarchical block structure
   - FourCC block type identifiers
   - Memory-mapped I/O for efficient access
   - Progressive loading with streaming support

2. **Security Layer** ([`maif/security.py`](maif/security.py))
   - AWS KMS integration for key management
   - FIPS 140-2 compliant encryption (AES-256-GCM)
   - Digital signatures using RSA/ECDSA
   - Cryptographic provenance chains
   - Mandatory encryption (no plaintext fallback)

3. **Compliance Logging** ([`maif/compliance_logging.py`](maif/compliance_logging.py))
   - STIG/FIPS validation framework
   - SIEM integration (CloudWatch, Splunk, Elasticsearch)
   - Tamper-evident audit trails using hash chains
   - Support for HIPAA, FISMA, PCI-DSS compliance frameworks

### Performance Characteristics

- **Semantic Search**: ~30ms response time for 1M+ vectors
- **Compression Ratio**: Up to 64× using hierarchical semantic compression
- **Hash Verification**: >500 MB/s throughput
- **Memory Efficiency**: 64KB minimum buffer with streaming

## Installation

```bash
# Basic installation
pip install -e .

# With optional features
pip install -e ".[full]"  # All features
pip install -e ".[ml]"    # Machine learning features
pip install -e ".[dev]"   # Development tools
```

## Quick Start

### Basic Usage

```python
from maif_api import create_maif, load_maif

# Create artifact with cryptographic provenance
maif = create_maif("my_agent")
maif.add_text("Agent conversation data", title="Session 1")
maif.save("agent_memory.maif")

# Load and verify integrity
loaded = load_maif("agent_memory.maif")
assert loaded.verify_integrity()  # Cryptographic hash chain verification
```

### Multi-Agent RAG System (Featured Example)

Complete production-ready system with LangGraph orchestration and MAIF provenance:

```bash
cd examples/langgraph

# Setup
echo "GEMINI_API_KEY=your_key" > .env
pip install -r requirements_enhanced.txt

# Create knowledge base with embeddings
python3 create_kb_enhanced.py

# Run interactive demo
python3 demo_enhanced.py
```

Features: ChromaDB vector search, Gemini API, LLM fact-checking, cryptographic audit trails, multi-turn conversations.

See `examples/langgraph/README.md` for complete documentation.

## Examples

### LangGraph Multi-Agent RAG System
Production-ready multi-agent system with cryptographic provenance:
```bash
cd examples/langgraph
python3 create_kb_enhanced.py  # Create knowledge base
python3 demo_enhanced.py       # Run interactive demo
```

See `examples/langgraph/README.md` for details.

### Basic Examples
- `examples/basic/` - Simple usage patterns
- `examples/security/` - Privacy and encryption
- `examples/aws/` - AWS integration
- `examples/advanced/` - Multi-agent, lifecycle management

## Core Features

- **Cryptographic Provenance** - Hash-chained blocks, tamper-evident
- **Multi-Agent Support** - Shared artifacts, agent-specific logging
- **Multimodal** - Text, embeddings, images, video, knowledge graphs
- **Security** - FIPS 140-2 encryption, AWS KMS, digital signatures
- **Performance** - Memory-mapped I/O, streaming, compression
- **Compliance** - Audit trails, SIEM integration, regulatory support

## Project Structure

```
maifscratch-1/
├── maif/                  # Core library (80 files)
├── maif_api.py           # High-level API
├── examples/
│   ├── langgraph/        # Multi-agent RAG system ⭐
│   ├── basic/            # Simple examples
│   ├── aws/              # AWS integrations
│   ├── security/         # Privacy & encryption
│   └── advanced/         # Multi-agent, lifecycle
├── tests/                # Test suite
├── docs/                 # Documentation
└── benchmarks/           # Performance tests
```

## Use Cases

- **Multi-Agent Systems** - Shared memory with provenance (see `examples/langgraph/`)
- **RAG Systems** - Document storage with embeddings and search
- **Compliance** - Audit trails for regulated industries
- **Research** - Reproducible experiments with full provenance
- **Enterprise AI** - Secure, auditable AI workflows

## Documentation

- **Main Docs**: `docs/README.md`
- **API Reference**: `docs/api/`
- **User Guides**: `docs/guide/`
- **Examples**: See `examples/` directories

## Key Features

### Cryptographic Provenance
Every block linked via `previous_hash` - any tampering breaks the chain

### Multi-Agent Coordination
Shared artifacts enable agent collaboration with full audit trails

### Production Ready
- Thread-safe operations
- Error handling and fallbacks
- AWS integration (optional)
- Comprehensive test suite

## Contributing

Please ensure all contributions:
1. Include comprehensive unit tests.
2. Pass FIPS compliance validation.
3. Include security impact analysis.
4. Update technical documentation.
5. Follow PEP 8 style guidelines.

## References

- [FIPS 140-2 Standards](https://csrc.nist.gov/publications/detail/fips/140/2/final)
- [DISA STIG Requirements](https://public.cyber.mil/stigs/)
- [NIST 800-53 Controls](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- [ISO BMFF Specification](https://www.iso.org/standard/68960.html)

## License

MIT License - See LICENSE file for details
