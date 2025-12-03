# Cryptographic Operations

MAIF provides FIPS-compliant cryptographic operations for data encryption, signing, and integrity verification. These capabilities are accessed through `SecurityManager` and `MAIFSigner`/`MAIFVerifier`.

## Overview

| Feature | Algorithm | Standard |
|---------|-----------|----------|
| Encryption | AES-256-GCM | FIPS 140-2 |
| Key Derivation | PBKDF2-HMAC-SHA256 | NIST SP 800-132 |
| Signing | RSA-PSS | PKCS#1 v2.1 |
| Hashing | SHA-256 | FIPS 180-4 |

## Encryption

MAIF uses AES-256-GCM for symmetric encryption with PBKDF2 for key derivation.

### Basic Encryption

```python
from maif.security import SecurityManager

security = SecurityManager(use_kms=False, require_encryption=True)

# Encrypt data
plaintext = b"Sensitive information"
encrypted = security.encrypt_data(plaintext)

# Decrypt data
decrypted = security.decrypt_data(encrypted)
assert decrypted == plaintext
```

### Encryption Format

Encrypted data includes a metadata header:

```
┌────────────────────┬──────────────────────┬────────────────────┐
│  Header Length     │  Metadata JSON       │  Ciphertext        │
│  (4 bytes)         │  (variable)          │  (variable)        │
└────────────────────┴──────────────────────┴────────────────────┘
```

Metadata includes:
- `encryption_method`: "local_fips" or "kms_direct"/"kms_envelope"
- `algorithm`: "AES-256-GCM"
- `kdf`: "PBKDF2-HMAC-SHA256" (for local)
- `iterations`: 100000 (NIST recommended minimum)
- `salt`: Base64-encoded salt
- `iv`: Base64-encoded initialization vector
- `tag`: Base64-encoded GCM authentication tag

### AWS KMS Encryption

When configured with KMS, encryption uses AWS Key Management Service:

```python
security = SecurityManager(
    use_kms=True,
    kms_key_id="arn:aws:kms:us-east-1:123456789:key/abc-123",
    region_name="us-east-1"
)

# Uses KMS for key management
encrypted = security.encrypt_data(b"Sensitive data")
```

KMS supports two modes:
- **Direct encryption** (data ≤ 4KB): Data encrypted directly with KMS
- **Envelope encryption** (data > 4KB): Data encryption key (DEK) encrypted with KMS

## Digital Signatures

MAIF uses RSA-2048 with PSS padding for digital signatures.

### Signing Data

```python
from maif.security import MAIFSigner

signer = MAIFSigner(agent_id="my-agent")

# Sign data
data = b"Important document"
signature = signer.sign_data(data)

# Get public key for verification
public_key_pem = signer.get_public_key_pem()
```

### Signature Format

Signatures are:
- Base64-encoded
- RSA-2048 with PSS padding
- SHA-256 message digest
- PSS salt length = digest length

### Verifying Signatures

```python
from maif.security import MAIFVerifier

verifier = MAIFVerifier()

# Verify signature
is_valid = verifier.verify_signature(data, signature, public_key_pem)
```

### Signing Manifests

MAIF manifests can be signed for integrity verification:

```python
from maif.security import MAIFSigner, MAIFVerifier

signer = MAIFSigner(agent_id="my-agent")

manifest = {
    "blocks": [...],
    "version": "1.0.0"
}

# Sign manifest
signed = signer.sign_maif_manifest(manifest)
# signed contains: signature, public_key, signature_metadata

# Verify
verifier = MAIFVerifier()
is_valid = verifier.verify_maif_signature(signed)
```

## Hash Functions

MAIF uses SHA-256 for content hashing and integrity verification.

### Content Hashing

```python
import hashlib

data = b"Content to hash"
hash_value = hashlib.sha256(data).hexdigest()
```

### Integrity Verification

```python
from maif.security import SecurityManager

security = SecurityManager(use_kms=False)

data = b"Original content"
expected_hash = hashlib.sha256(data).hexdigest()

# Verify integrity
is_valid = security.validate_integrity(data, expected_hash)
```

## Provenance Chains

MAIF creates cryptographically-linked provenance chains:

```python
from maif.security import MAIFSigner

signer = MAIFSigner(agent_id="agent-1")

# Add provenance entries
signer.add_provenance_entry("create", "block-hash-1", {"note": "Initial creation"})
signer.add_provenance_entry("update", "block-hash-2", {"note": "Updated content"})

# Get chain
chain = signer.get_provenance_chain()
```

Each entry contains:
- `timestamp`: When the action occurred
- `agent_id`: Who performed the action
- `action`: What was done (create, update, delete)
- `block_hash`: Hash of the affected block
- `previous_hash`: Link to previous block
- `entry_hash`: Hash of this entry
- `signature`: Digital signature of the entry

### Verifying Provenance

```python
from maif.security import MAIFVerifier

verifier = MAIFVerifier()

# Get chain from somewhere
provenance_data = {"chain": [...]}

# Verify
is_valid, errors = verifier.verify_provenance_chain(provenance_data)
if not is_valid:
    print(f"Chain verification failed: {errors}")
```

## Privacy Engine Encryption

The privacy engine also provides encryption through `EncryptionMode`:

```python
from maif import MAIFEncoder
from maif.privacy import PrivacyLevel, EncryptionMode

encoder = MAIFEncoder(agent_id="agent-1", enable_privacy=True)

# Add encrypted block
encoder.add_text_block(
    "Sensitive text",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM
)
```

Available encryption modes:
- `EncryptionMode.NONE`: No encryption
- `EncryptionMode.AES_GCM`: AES-256-GCM encryption

## Security Events

All cryptographic operations are logged:

```python
security = SecurityManager(use_kms=False)
security.encrypt_data(b"data")

# Check security events
status = security.get_security_status()
print(f"Events logged: {status['events_logged']}")

# Access event log
for event in security.security_events:
    print(f"{event['timestamp']}: {event['type']}")
```

Event types:
- `encrypt_request`: Encryption started
- `encrypt_success`: Encryption completed
- `encrypt_failure`: Encryption failed
- `decrypt_request`: Decryption started
- `decrypt_success`: Decryption completed
- `decrypt_failure`: Decryption failed
- `kms_initialized`: KMS setup completed
- `kms_initialization_failed`: KMS setup failed

## Best Practices

### 1. Use Required Encryption for Sensitive Data

```python
security = SecurityManager(
    use_kms=False,
    require_encryption=True  # Raise error if encryption fails
)
```

### 2. Protect Keys

```python
import os

# Store key paths in environment
key_path = os.environ.get("MAIF_PRIVATE_KEY_PATH")
signer = MAIFSigner(private_key_path=key_path)
```

### 3. Verify Before Trust

```python
from maif.security import MAIFVerifier

verifier = MAIFVerifier()

# Always verify external manifests
is_valid, errors = verifier.verify_maif_manifest(external_manifest)
if not is_valid:
    raise SecurityError(f"Invalid manifest: {errors}")
```

### 4. Use KMS for Production

```python
# Production: Use AWS KMS
security = SecurityManager(
    use_kms=True,
    kms_key_id=os.environ["KMS_KEY_ID"],
    require_encryption=True
)
```

## Algorithm Details

### AES-256-GCM

- **Key size**: 256 bits
- **Block size**: 128 bits
- **IV size**: 96 bits (12 bytes)
- **Tag size**: 128 bits (16 bytes)
- **Mode**: Galois/Counter Mode (authenticated encryption)

### PBKDF2-HMAC-SHA256

- **Hash**: SHA-256
- **Salt size**: 128 bits (16 bytes)
- **Iterations**: 100,000 (minimum)
- **Output key size**: 256 bits

### RSA-PSS

- **Key size**: 2048 bits
- **Hash**: SHA-256
- **Salt length**: Hash digest length
- **Trailer field**: 0xBC
