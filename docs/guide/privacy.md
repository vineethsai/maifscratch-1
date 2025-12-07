# Privacy Framework

::: danger DEPRECATED
This page is deprecated. For the latest and most accurate documentation, please visit **[DeepWiki - Privacy and Encryption](https://deepwiki.com/vineethsai/maif/2.3-privacy-and-encryption)**.

DeepWiki documentation is auto-generated from the codebase and always up-to-date.
:::

MAIF implements privacy-by-design principles with encryption, anonymization, and access control. This guide covers MAIF's privacy features and how to use them effectively.

## Privacy Overview

MAIF's privacy framework provides:

- **Encryption**: AES-GCM and ChaCha20-Poly1305 encryption modes
- **Anonymization**: Automatic PII detection and masking
- **Access Control**: Fine-grained access rules
- **Privacy Policies**: Configurable privacy settings per block
- **Differential Privacy**: Statistical privacy guarantees

## Privacy Engine

The `PrivacyEngine` is the central component for privacy operations:

```python
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

# Create privacy engine
privacy = PrivacyEngine()

# The engine provides:
# - Encryption/decryption
# - Data anonymization
# - Access control
# - Privacy policy management
```

## Encryption

### Encryption Modes

MAIF supports multiple encryption algorithms:

```python
from maif.privacy import EncryptionMode

# Available encryption modes
EncryptionMode.AES_GCM           # AES-256-GCM (default, recommended)
EncryptionMode.CHACHA20_POLY1305 # ChaCha20-Poly1305 (alternative)
EncryptionMode.HOMOMORPHIC       # Homomorphic encryption (experimental)
```

### Encrypting Data

```python
from maif.privacy import PrivacyEngine, EncryptionMode

privacy = PrivacyEngine()

# Encrypt with AES-GCM (default)
sensitive_data = b"Confidential information"
encrypted = privacy.encrypt_data(sensitive_data)

# Decrypt
decrypted = privacy.decrypt_data(encrypted)
assert decrypted == sensitive_data

# Use ChaCha20-Poly1305
encrypted_chacha = privacy.encrypt_data(
    sensitive_data,
    mode=EncryptionMode.CHACHA20_POLY1305
)
decrypted_chacha = privacy.decrypt_data(
    encrypted_chacha,
    mode=EncryptionMode.CHACHA20_POLY1305
)
```

### Batch Encryption

For large datasets:

```python
from maif.privacy import PrivacyEngine

privacy = PrivacyEngine()

# Encrypt multiple items at once
data_items = [b"Item 1", b"Item 2", b"Item 3"]
encrypted_batch = privacy.encrypt_batch(data_items)

# Parallel batch encryption for performance
encrypted_parallel = privacy.encrypt_batch_parallel(data_items)
```

### Encryption in MAIF Files

```python
from maif_api import create_maif

# Enable privacy for encryption
maif = create_maif("secure-agent", enable_privacy=True)

# Add encrypted text
maif.add_text(
    "This content will be encrypted",
    title="Secret Document",
    encrypt=True
)

maif.save("encrypted.maif")
```

## Privacy Levels

MAIF supports classification levels:

```python
from maif.privacy import PrivacyLevel

# Available levels (least to most restrictive)
PrivacyLevel.PUBLIC        # No restrictions
PrivacyLevel.INTERNAL      # Internal use only
PrivacyLevel.CONFIDENTIAL  # Requires authorization
PrivacyLevel.SECRET        # Highly restricted
PrivacyLevel.TOP_SECRET    # Maximum classification
```

### Using Privacy Levels

```python
from maif.core import MAIFEncoder
from maif.privacy import PrivacyEngine, PrivacyLevel, EncryptionMode

privacy = PrivacyEngine()
encoder = MAIFEncoder(
    "classified.maif",
    agent_id="classified",
    enable_privacy=True,
    privacy_engine=privacy
)

# Add blocks with different classification
encoder.add_text_block(
    "Public announcement",
    privacy_level=PrivacyLevel.PUBLIC
)

encoder.add_text_block(
    "Internal memo",
    privacy_level=PrivacyLevel.INTERNAL
)

encoder.add_text_block(
    "Confidential report",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM
)

encoder.finalize()
```

## Data Anonymization

### Automatic PII Detection

The privacy engine can detect and anonymize personally identifiable information:

```python
from maif.privacy import PrivacyEngine

privacy = PrivacyEngine()

# Text with PII
text = "Contact John Smith at john.smith@email.com or call 555-123-4567"

# Anonymize
anonymized = privacy.anonymize_data(text)
print(anonymized)
# Output: "Contact [NAME] at [EMAIL] or call [PHONE]"
```

### Anonymization in MAIF Files

```python
from maif_api import create_maif

maif = create_maif("medical-agent", enable_privacy=True)

# Add with anonymization
maif.add_text(
    "Patient John Doe, SSN: 123-45-6789, diagnosed with condition X",
    title="Medical Record",
    encrypt=True,
    anonymize=True  # PII will be detected and masked
)

maif.save("medical.maif")
```

### Custom Sensitive Patterns

The `_is_sensitive` method checks for common patterns:
- Names
- Email addresses
- Phone numbers
- Social Security Numbers
- Credit card numbers
- Addresses

## Access Control

### Access Rules

Define who can access what:

```python
from maif.privacy import PrivacyEngine, AccessRule
from datetime import datetime, timedelta

privacy = PrivacyEngine()

# Create access rule
rule = AccessRule(
    subject="analyst_a",           # Who
    resource="financial_data",     # What
    permissions=["read"],          # What they can do
    conditions={"department": "finance"},  # Under what conditions
    expiry=datetime.now() + timedelta(days=90)  # When it expires
)

# Add rule to engine
privacy.add_access_rule(rule)

# Check access
can_access = privacy.check_access("analyst_a", "financial_data", "read")
print(f"Access granted: {can_access}")
```

### Access Rule Structure

```python
from maif.privacy import AccessRule
from datetime import datetime

rule = AccessRule(
    subject="user_id",           # User or role identifier
    resource="resource_id",      # Block or artifact identifier
    permissions=["read", "write"],  # List of allowed operations
    conditions={                  # Optional conditions
        "time_range": "business_hours",
        "location": "office_network"
    },
    expiry=datetime(2025, 12, 31)  # Optional expiration date
)
```

## Privacy Policies

### Creating Privacy Policies

```python
from maif.privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode

privacy = PrivacyEngine()

# Create privacy policy
policy = PrivacyPolicy(
    level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    retention_days=365,      # How long to keep data
    anonymization=True,      # Whether to anonymize
    audit=True               # Whether to log access
)

# Set policy for a resource
privacy.set_privacy_policy("document_001", policy)

# Get policy
retrieved_policy = privacy.get_privacy_policy("document_001")
```

### Retention Policy Enforcement

```python
from maif.privacy import PrivacyEngine

privacy = PrivacyEngine()

# Enforce retention policies (delete expired data)
privacy.enforce_retention_policy()
```

## Differential Privacy

MAIF includes differential privacy support for statistical queries:

```python
from maif.privacy import DifferentialPrivacy

# Create differential privacy instance
dp = DifferentialPrivacy(
    epsilon=1.0,    # Privacy budget (lower = more private)
    delta=1e-5      # Probability of privacy failure
)

# Add noise to data for privacy
noisy_data = dp.add_noise(original_data)

# Query with differential privacy
result = dp.private_query(query_function, data)
```

## Secure Multi-Party Computation

For collaborative analysis without revealing data:

```python
from maif.privacy import SecureMultipartyComputation

# Create SMPC instance
smpc = SecureMultipartyComputation()

# Parties can contribute data securely
# Results are computed without revealing individual inputs
```

## Zero-Knowledge Proofs

Prove statements without revealing underlying data:

```python
from maif.privacy import ZeroKnowledgeProof

# Create ZKP instance
zkp = ZeroKnowledgeProof()

# Generate proof that data meets criteria
# without revealing the actual data
```

## Privacy Reports

Generate privacy compliance reports:

```python
from maif.privacy import PrivacyEngine

privacy = PrivacyEngine()

# Generate comprehensive privacy report
report = privacy.generate_privacy_report()

print(f"Encrypted blocks: {report.get('encrypted_count', 0)}")
print(f"Anonymized blocks: {report.get('anonymized_count', 0)}")
print(f"Access rules: {report.get('access_rules_count', 0)}")
```

### Privacy Report with MAIF API

```python
from maif_api import create_maif

maif = create_maif("report-agent", enable_privacy=True)
maif.add_text("Sensitive data", encrypt=True)
maif.save("report.maif")

# Get privacy report
report = maif.get_privacy_report()
print(f"Privacy enabled: {report.get('privacy_enabled')}")
```

## Complete Privacy Example

```python
from maif.core import MAIFEncoder
from maif.privacy import (
    PrivacyEngine, 
    PrivacyLevel, 
    EncryptionMode, 
    PrivacyPolicy,
    AccessRule
)
from datetime import datetime, timedelta

# Setup privacy engine
privacy = PrivacyEngine()

# Define access rules
privacy.add_access_rule(AccessRule(
    subject="data_analyst",
    resource="financial_reports",
    permissions=["read"],
    expiry=datetime.now() + timedelta(days=365)
))

privacy.add_access_rule(AccessRule(
    subject="admin",
    resource="financial_reports",
    permissions=["read", "write", "delete"],
    expiry=datetime.now() + timedelta(days=365)
))

# Create encoder with privacy
encoder = MAIFEncoder(
    "financial_q4.maif",
    agent_id="financial-system",
    enable_privacy=True,
    privacy_engine=privacy
)

# Add public summary
encoder.add_text_block(
    "Q4 2024 Financial Summary: Revenue increased 15%",
    privacy_level=PrivacyLevel.PUBLIC,
    metadata={"type": "summary", "quarter": "Q4-2024"}
)

# Add confidential details
encoder.add_text_block(
    "Detailed revenue breakdown by product line...",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    metadata={"type": "details", "quarter": "Q4-2024"}
)

# Add with PII anonymization
encoder.add_text_block(
    "Top performer: John Smith, Employee ID: 12345",
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    anonymize=True,
    metadata={"type": "personnel"}
)

# Finalize
encoder.finalize()
print("Privacy-protected financial report created")
```

## Best Practices

### 1. Always Enable Privacy for Sensitive Data

```python
# Always use privacy for sensitive data
maif = create_maif("agent", enable_privacy=True)
```

### 2. Use Appropriate Privacy Levels

```python
# Match level to sensitivity
def classify_data(data_type):
    classifications = {
        "public": PrivacyLevel.PUBLIC,
        "internal": PrivacyLevel.INTERNAL,
        "customer_pii": PrivacyLevel.CONFIDENTIAL,
        "financial": PrivacyLevel.SECRET,
        "medical": PrivacyLevel.TOP_SECRET
    }
    return classifications.get(data_type, PrivacyLevel.CONFIDENTIAL)
```

### 3. Set Appropriate Retention Policies

```python
# Don't keep data longer than necessary
policy = PrivacyPolicy(
    level=PrivacyLevel.CONFIDENTIAL,
    retention_days=90,  # Auto-delete after 90 days
    audit=True
)
```

### 4. Use Anonymization for PII

```python
# Always anonymize when possible
maif.add_text(content, anonymize=True)
```

## Available Privacy Components

| Component | Purpose | Import |
|-----------|---------|--------|
| `PrivacyEngine` | Central privacy operations | `from maif.privacy import PrivacyEngine` |
| `PrivacyLevel` | Classification levels | `from maif.privacy import PrivacyLevel` |
| `EncryptionMode` | Encryption algorithms | `from maif.privacy import EncryptionMode` |
| `PrivacyPolicy` | Privacy configuration | `from maif.privacy import PrivacyPolicy` |
| `AccessRule` | Access control rules | `from maif.privacy import AccessRule` |
| `DifferentialPrivacy` | Statistical privacy | `from maif.privacy import DifferentialPrivacy` |
| `SecureMultipartyComputation` | Collaborative computation | `from maif.privacy import SecureMultipartyComputation` |
| `ZeroKnowledgeProof` | Proofs without disclosure | `from maif.privacy import ZeroKnowledgeProof` |

## Next Steps

- **[Security Model →](/guide/security-model)** - Signatures and verification
- **[API Privacy Reference →](/api/privacy/)** - Complete privacy API
- **[Examples →](/examples/)** - Privacy examples in practice
