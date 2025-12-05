# MAIF File Format Specification v3.0

**Version:** 3.0  
**Status:** Stable  
**Last Updated:** December 2024

## Overview

MAIF (Multimodal Artifact Interchange Format) is a self-contained, cryptographically-secure binary format for storing multimodal AI artifacts with provenance tracking.

### Design Goals

1. **Self-Contained:** All data, signatures, and metadata in one file
2. **Tamper-Evident:** Cryptographic verification of all content
3. **Multimodal:** Support for text, embeddings, images, video, knowledge graphs
4. **Efficient:** Streaming, compression, memory-mapped I/O
5. **Interoperable:** Well-defined binary format for cross-platform use

---

## File Structure

```
┌─────────────────────────────────────┐
│         File Header (32 bytes)       │
├─────────────────────────────────────┤
│         Metadata Section            │
├─────────────────────────────────────┤
│          Block 1                    │
│  ┌──────────────────────────────┐  │
│  │  Block Header (64 bytes)      │  │
│  ├──────────────────────────────┤  │
│  │  Block Data (variable)       │  │
│  ├──────────────────────────────┤  │
│  │  Ed25519 Signature (64 bytes)│  │
│  └──────────────────────────────┘  │
├─────────────────────────────────────┤
│          Block 2                    │
│  ...                                │
├─────────────────────────────────────┤
│      Provenance Chain               │
├─────────────────────────────────────┤
│      Merkle Root (32 bytes)         │
└─────────────────────────────────────┘
```

---

## File Header (32 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | Magic | `MAIF` (0x4D414946) |
| 4 | 2 | Version Major | Format version (3) |
| 6 | 2 | Version Minor | Minor version (0) |
| 8 | 8 | Created Timestamp | Unix timestamp (microseconds) |
| 16 | 8 | File Size | Total file size in bytes |
| 24 | 4 | Block Count | Number of blocks |
| 28 | 4 | Flags | Feature flags (bitfield) |

### Flags Bitfield

| Bit | Flag | Description |
|-----|------|-------------|
| 0 | COMPRESSED | File uses compression |
| 1 | ENCRYPTED | File contains encrypted blocks |
| 2 | SIGNED | All blocks are signed |
| 3 | STREAMING | File optimized for streaming |
| 4-31 | Reserved | Reserved for future use |

---

## Metadata Section

JSON-encoded metadata follows the file header:

```json
{
  "format_version": "3.0",
  "agent_id": "agent-identifier",
  "created_at": "2024-12-05T10:30:00Z",
  "content_type": "multimodal",
  "schema_version": "1.0",
  "custom": {
    // Application-specific metadata
  }
}
```

**Encoding:** UTF-8 JSON  
**Length:** Stored as 4-byte unsigned integer before JSON data  
**Maximum Size:** 64 KB

---

## Block Structure

### Block Header (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 16 | Block ID | UUID v4 |
| 16 | 4 | Block Type | FourCC code (see Block Types) |
| 20 | 8 | Data Offset | Offset to block data |
| 28 | 8 | Data Size | Size of block data |
| 36 | 4 | Compression | Compression algorithm |
| 40 | 4 | Encryption | Encryption algorithm |
| 44 | 8 | Timestamp | Block creation time |
| 52 | 4 | Metadata Length | Length of block metadata |
| 56 | 8 | Previous Hash | SHA-256 of previous block (first 8 bytes) |

### Block Types (FourCC)

| Code | Type | Description |
|------|------|-------------|
| `TEXT` | Text | UTF-8 text content |
| `EMBD` | Embeddings | Semantic vector embeddings |
| `IMAG` | Image | Image data (JPEG, PNG, etc.) |
| `AUDI` | Audio | Audio data (MP3, WAV, etc.) |
| `VIDE` | Video | Video data (MP4, WebM, etc.) |
| `KGRF` | Knowledge Graph | RDF/JSON-LD knowledge graph |
| `BINA` | Binary | Generic binary data |
| `META` | Metadata | Additional metadata |
| `LIFE` | Lifecycle | Agent lifecycle events |

### Compression Algorithms

| Code | Algorithm | Description |
|------|-----------|-------------|
| 0 | None | No compression |
| 1 | GZIP | Standard gzip compression |
| 2 | ZSTD | Zstandard compression |
| 3 | BROTLI | Brotli compression |
| 4 | LZ4 | LZ4 compression (fast) |
| 5 | HSC | Hierarchical Semantic Compression (custom) |

### Encryption Algorithms

| Code | Algorithm | Description |
|------|-----------|-------------|
| 0 | None | No encryption |
| 1 | AES-256-GCM | AES with 256-bit key, GCM mode |
| 2 | ChaCha20-Poly1305 | ChaCha20 stream cipher with Poly1305 MAC |

---

## Block Data

Block data immediately follows the block header. Format depends on block type:

### TEXT Block

```
┌────────────────────┐
│  Encoding (1 byte) │  UTF-8 = 0x01
├────────────────────┤
│  Language (2 bytes)│  ISO 639-1 code
├────────────────────┤
│  Text Data         │
└────────────────────┘
```

### EMBD (Embeddings) Block

```
┌────────────────────┐
│  Dimension (4)     │  Vector dimension
├────────────────────┤
│  Count (4)         │  Number of vectors
├────────────────────┤
│  Data Type (1)     │  float32 = 0x01, float16 = 0x02
├────────────────────┤
│  Vector Data       │  dimension × count × sizeof(type)
└────────────────────┘
```

### IMAG Block

```
┌────────────────────┐
│  Format (4)        │  FourCC: JPEG, PNG_, WEBP
├────────────────────┤
│  Width (4)         │
├────────────────────┤
│  Height (4)        │
├────────────────────┤
│  Image Data        │
└────────────────────┘
```

---

## Ed25519 Signature (64 bytes)

Each block is signed with Ed25519:

```
Signature = Ed25519_Sign(
    private_key,
    SHA-256(block_header || block_data)
)
```

**Public Key Location:** Stored in file metadata section

**Signature Verification:**
```python
is_valid = Ed25519_Verify(
    public_key,
    signature,
    SHA-256(block_header || block_data)
)
```

---

## Provenance Chain

Immutable audit trail of all operations:

```json
{
  "chain": [
    {
      "entry_id": "uuid",
      "action": "create" | "add_block" | "sign" | "verify",
      "agent_id": "agent-identifier",
      "timestamp": "ISO 8601 timestamp",
      "block_id": "related block UUID (optional)",
      "signature": "Ed25519 signature of entry",
      "metadata": {
        // Action-specific metadata
      }
    }
  ]
}
```

**Chain Verification:**
- Each entry signed with agent's private key
- Entries linked via signatures
- Tampering breaks chain verification

---

## Merkle Root (32 bytes)

SHA-256 Merkle root of all block hashes for fast integrity verification:

```
           Root
          /    \
       H01      H23
      /  \      /  \
    H0   H1   H2   H3
    |    |    |    |
   B0   B1   B2   B3
```

**Computation:**
```python
def compute_merkle_root(block_hashes):
    if len(block_hashes) == 1:
        return block_hashes[0]
    
    pairs = []
    for i in range(0, len(block_hashes), 2):
        if i + 1 < len(block_hashes):
            pairs.append(SHA256(block_hashes[i] + block_hashes[i+1]))
        else:
            pairs.append(block_hashes[i])
    
    return compute_merkle_root(pairs)
```

---

## Integrity Verification

### Full Verification

1. **File Header:** Verify magic number and version
2. **Merkle Root:** Recompute and compare
3. **Block Signatures:** Verify all Ed25519 signatures
4. **Hash Chain:** Verify previous_hash links
5. **Provenance Chain:** Verify provenance signatures

### Quick Verification

1. **Merkle Root:** Compare stored vs computed (O(n))
2. Skip full signature verification unless needed

---

## Streaming Support

MAIF supports streaming reads/writes:

### Streaming Write
```python
encoder = MAIFEncoder("output.maif", agent_id="agent-1")
for chunk in data_stream:
    encoder.add_text_block(chunk)  # Signed immediately
encoder.finalize()  # Adds Merkle root
```

### Streaming Read
```python
decoder = MAIFDecoder("input.maif")
for block in decoder.stream_blocks():
    process(block)
```

---

## Compression

### Hierarchical Semantic Compression (HSC)

Custom compression for embeddings:

1. **Clustering:** k-means clustering of vectors
2. **Codebook:** Store cluster centroids
3. **Quantization:** Replace vectors with cluster IDs + residuals
4. **Huffman Encoding:** Compress cluster IDs

**Compression Ratio:** Up to 64× for large embedding sets

---

## Encryption

### Block-Level Encryption

Each block can be independently encrypted:

**AES-256-GCM:**
```
Encrypted = AES-GCM-Encrypt(key, iv, block_data, aad=block_header)
Tag = GCM_Tag (16 bytes)
```

**ChaCha20-Poly1305:**
```
Encrypted = ChaCha20-Encrypt(key, nonce, block_data)
Tag = Poly1305-MAC(key, encrypted, aad=block_header)
```

**Key Derivation:**
```
Key = PBKDF2-SHA256(password, salt, iterations=100000)
```

---

## Version History

| Version | Release | Changes |
|---------|---------|---------|
| 3.0 | 2024-12 | Self-contained format, Ed25519 signatures |
| 2.0 | 2024-10 | Added streaming, improved compression |
| 1.0 | 2024-08 | Initial release with external manifests |

---

## Compatibility

### Forward Compatibility
- Readers SHOULD ignore unknown flags
- Readers SHOULD skip unknown block types
- Readers MUST verify version compatibility

### Backward Compatibility
- Version 3.0 is **not** compatible with 1.x/2.x
- Migration tools available in SDK

---

## Implementation Guidelines

### Recommended Practices

1. **Always verify signatures** before trusting data
2. **Use memory-mapped I/O** for large files
3. **Stream processing** for files > 100MB
4. **Compress embeddings** with HSC for 10+ vectors
5. **Encrypt sensitive blocks** (PII, credentials)

### Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Signature verification | 30,000 ops/sec | Ed25519 |
| Block read | 500+ MB/sec | Memory-mapped |
| Merkle verification | < 0.1ms | For files < 10K blocks |
| Streaming write | 100+ MB/sec | Sequential writes |

---

## Reference Implementation

Python SDK: https://github.com/vineethsai/maif

```python
from maif import MAIFEncoder, MAIFDecoder

# Create
encoder = MAIFEncoder("output.maif", agent_id="agent-1")
encoder.add_text_block("Hello, MAIF!")
encoder.finalize()

# Read
decoder = MAIFDecoder("output.maif")
decoder.load()
is_valid, errors = decoder.verify_integrity()
```

---

## License

This specification is released under CC0 1.0 Universal (Public Domain).

Implementations may use any license.

---

## Contact

- Specification Issues: https://github.com/vineethsai/maif/issues
- Mailing List: dev@maif.ai
- Reference Implementation: https://github.com/vineethsai/maif

