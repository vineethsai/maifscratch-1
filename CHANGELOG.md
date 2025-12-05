# Changelog

All notable changes to MAIF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive CI/CD pipeline with GitHub Actions
- Security policy and vulnerability disclosure process
- Code of conduct for community contributors

### Changed
- Improved linting and code quality checks

### Fixed
- Fixed 48 undefined name errors (F821) across multiple modules
- Fixed optional dependency handling for boto3 and benchmarks
- Resolved test failures with missing dependencies

## [2.0.0] - 2024-12-05

### Added - Secure Format (v3)
- **Self-contained format:** No external manifest files needed
- **Ed25519 signatures:** Fast 64-byte signatures on every block
- **Embedded provenance:** Complete audit trail in single file
- **Immutable blocks:** Each block signed immediately on write
- **Merkle root:** Fast integrity verification

### Added - Multi-Agent RAG System
- Production-ready LangGraph example with 5 specialized agents
- ChromaDB integration for semantic vector search
- Gemini API integration for LLM reasoning
- Complete MAIF provenance tracking for all agent actions
- Multi-turn conversation support with session management

### Added - Core Features
- `MAIFEncoder` and `MAIFDecoder` for secure format
- `SecureMAIFReader` for integrity verification
- Streaming support for large files (memory-mapped I/O)
- Privacy engine with encryption (AES-GCM, ChaCha20)
- PII detection and anonymization
- Access control with role-based permissions

### Added - Novel Algorithms
- **ACAM (Adaptive Cross-Modal Attention):** Multimodal content fusion
- **HSC (Hierarchical Semantic Compression):** Up to 64× compression ratio
- **CSB (Cryptographic Semantic Binding):** Embedding authenticity verification

### Added - Testing & Quality
- 400+ comprehensive tests with pytest
- Benchmark suite for format comparison
- Integration tests for all major features
- Automated linting with ruff and bandit

### Changed
- Migrated from RSA/ECDSA to Ed25519 for better performance
- Unified block format for consistency
- Improved documentation with VitePress site
- Enhanced error handling and validation

### Performance
- 11× faster read performance vs legacy format
- ~0.1ms integrity verification per file
- 30,000+ signature operations per second (Ed25519)
- Sub-50ms semantic search for 1M+ vectors

### Security
- 100% tamper detection rate in <0.1ms
- Cryptographic provenance for complete audit trails
- Privacy-by-design with built-in encryption
- FIPS 140-2 compatible cryptographic modules

## [1.0.0] - 2024-10-01

### Added
- Initial public release
- Basic MAIF format with external manifests
- RSA/ECDSA signature support
- Text and embedding block types
- Basic integrity verification
- Python SDK with encoder/decoder

### Features
- Multimodal content storage
- Semantic embeddings support
- Knowledge graph integration
- Basic provenance tracking

## [0.9.0-beta] - 2024-08-15

### Added
- Beta release for testing
- Core file format specification
- Basic cryptographic features
- Initial documentation

---

## Version History Summary

- **v2.0.0:** Secure format, multi-agent RAG, Ed25519, novel algorithms
- **v1.0.0:** Initial public release with basic features
- **v0.9.0-beta:** Beta testing release

## Migration Guides

### Migrating from v1.x to v2.0

The v2.0 release introduces the secure format (v3) which is incompatible with v1.x:

**Key Changes:**
- No more external manifest files (self-contained)
- Ed25519 replaces RSA/ECDSA signatures
- New API: `encoder.finalize()` instead of `encoder.save(maif_path, manifest_path)`

**Migration Steps:**

```python
# Old (v1.x)
from maif import MAIFEncoder
encoder = MAIFEncoder(agent_id="agent-1")
encoder.add_text_block("content")
encoder.save("output.maif", "output_manifest.json")

# New (v2.0)
from maif import MAIFEncoder
encoder = MAIFEncoder("output.maif", agent_id="agent-1")
encoder.add_text_block("content")
encoder.finalize()  # Automatically signs and seals
```

See [`docs/guide/migration.md`](docs/guide/migration.md) for complete migration guide.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md).

