# Privacy-by-Design Demo

This example demonstrates MAIF's comprehensive privacy features including encryption, anonymization, and access control.

## Overview

The Privacy Demo showcases MAIF's built-in privacy engine with multiple privacy levels, encryption modes, and differential privacy.

### Key Features

- **Encryption**: AES-GCM, ChaCha20-Poly1305 for data at rest
- **Privacy Levels**: PUBLIC, CONFIDENTIAL, SECRET with different protections
- **Anonymization**: Automatic PII detection and redaction
- **Access Control**: Role-based permissions with audit logging
- **Differential Privacy**: Statistical privacy for aggregate queries

## Running the Demo

```bash
cd examples/security
python3 privacy_demo.py
```

## Code Example

```python
from maif.core import MAIFEncoder
from maif.privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode

# Create encoder with privacy enabled
encoder = MAIFEncoder(agent_id="privacy_demo", enable_privacy=True)

# Create privacy policy
policy = PrivacyPolicy(
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    anonymization_required=True
)

# Add encrypted content
confidential_text = "Employee John Smith from john.smith@company.com"
encoder.add_text_block(
    confidential_text,
    metadata={"description": "HR record"},
    privacy_policy=policy,
    anonymize=True
)

# Save with encryption
encoder.build_maif("private.maif", "manifest.json")
```

## What You'll Learn

- How to apply different privacy levels
- Encryption modes and when to use them
- Automatic PII anonymization
- Access control rules
- Differential privacy for statistics
- Audit trail generation
