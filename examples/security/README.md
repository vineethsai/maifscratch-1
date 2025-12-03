# Security & Privacy Examples

Examples demonstrating MAIF's security and privacy features including encryption, access control, and classified data handling.

## Overview

These examples show how to:
- Encrypt sensitive data
- Implement access control
- Handle classified information
- Apply differential privacy
- Manage security clearances
- Audit security events

## Examples

### privacy_demo.py

Comprehensive privacy features demonstration.

**Features demonstrated:**
- AES-GCM encryption for sensitive data
- Differential privacy for statistical queries
- Data anonymization
- Access control with expiry
- Privacy policy enforcement
- Audit trail generation

**Run:**
```bash
python3 privacy_demo.py
```

**Key code:**
```python
from maif.privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel

# Initialize privacy engine
engine = PrivacyEngine()

# Create privacy policy
policy = PrivacyPolicy(
    privacy_level=PrivacyLevel.CONFIDENTIAL,
    encryption_mode=EncryptionMode.AES_GCM,
    anonymization_required=True,
    audit_required=True
)

# Apply to data
encrypted_data = engine.encrypt_data(sensitive_data, "block_id")
anonymized_text = engine.anonymize_data(text_with_pii, "text_block")
```

### classified_api_simple_demo.py

Simplified API for handling classified data.

**Features demonstrated:**
- Classification levels (UNCLASSIFIED, CONFIDENTIAL, SECRET, TOP SECRET)
- Clearance management
- Access control enforcement
- Compartmented information handling
- Need-to-know principle
- Audit logging

**Run:**
```bash
python3 classified_api_simple_demo.py
```

**Key code:**
```python
from maif.classified_api import SecureMAIF

# Create secure instance
maif = SecureMAIF(classification="SECRET")

# Grant clearance
maif.grant_clearance("user_001", "SECRET", 
                     compartments=["CRYPTO", "SIGINT"])

# Store classified data
doc_id = maif.store_classified_data(
    data={"mission": "OPERATION_X"},
    classification="SECRET"
)

# Retrieve with access control
if maif.can_access("user_001", doc_id):
    data = maif.retrieve_classified_data(doc_id)
```

### classified_security_demo.py

Advanced security features for classified environments.

**Features demonstrated:**
- PKI/CAC authentication
- Hardware MFA integration
- Multi-level security (MLS)
- Mandatory access control (MAC)
- Security event logging
- Compliance reporting

**Run:**
```bash
python3 classified_security_demo.py
```

**Key code:**
```python
from maif.classified_security import ClassifiedSecurityManager

# Initialize security manager
security = ClassifiedSecurityManager(
    classification_level="SECRET",
    enable_mfa=True,
    enable_pki=True
)

# PKI authentication
user_id = security.authenticate_with_pki(certificate_pem)

# MFA challenge
challenge = security.create_mfa_challenge(user_id)
verified = security.verify_mfa(challenge, token_code)

# Access control
if security.check_access(user_id, resource_id, "READ"):
    data = security.read_classified(resource_id)
```

## Security Features

### Encryption

MAIF supports multiple encryption modes:

| Mode | Algorithm | Key Size | Use Case |
|------|-----------|----------|----------|
| AES_GCM | AES-256-GCM | 256-bit | General purpose |
| AES_CBC | AES-256-CBC | 256-bit | Legacy compatibility |
| CHACHA20 | ChaCha20-Poly1305 | 256-bit | High performance |

### Access Control

Granular permissions with:
- User-based access control
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Time-based access (expiry)
- Need-to-know enforcement

### Differential Privacy

Protects individual privacy in aggregate queries:
- Laplace mechanism
- Gaussian mechanism
- Configurable epsilon (privacy budget)
- Automatic noise injection

### Audit Logging

All security events logged with:
- Timestamp (microsecond precision)
- User identifier
- Action performed
- Resource accessed
- Success/failure status
- Classification level
- Compliance framework tags

## Classification Levels

### Standard Levels
- **UNCLASSIFIED**: Public information
- **CONFIDENTIAL**: Moderate sensitivity
- **SECRET**: High sensitivity
- **TOP SECRET**: Highest sensitivity

### Custom Levels
Define your own:
```python
from maif.classified_api import ClassificationLevel

custom_level = ClassificationLevel(
    name="RESTRICTED",
    numeric_value=15,
    color_code="YELLOW",
    handling_caveats=["NOFORN", "ORCON"]
)
```

## Compliance

These examples demonstrate compliance with:

- **FIPS 140-2**: Cryptographic module validation
- **HIPAA**: Healthcare data protection
- **GDPR**: European data privacy
- **FISMA**: Federal information security
- **DISA STIG**: Security technical implementation guides

## Configuration

### Environment Variables

```bash
# Optional: Specify encryption keys
export MAIF_ENCRYPTION_KEY=your_key_here

# Optional: Enable FIPS mode
export MAIF_FIPS_MODE=true

# Optional: Audit log destination
export MAIF_AUDIT_LOG=/var/log/maif/audit.log
```

### Programmatic Configuration

```python
from maif.privacy import PrivacyEngine

engine = PrivacyEngine(
    default_encryption=EncryptionMode.AES_GCM,
    enable_differential_privacy=True,
    epsilon=0.1,  # Privacy budget
    audit_all_access=True
)
```

## Best Practices

### 1. Always Verify Integrity
```python
maif = load_maif("artifact.maif")
if not maif.verify_integrity():
    raise SecurityError("Artifact tampered with")
```

### 2. Use Appropriate Classification
Don't over-classify (reduces usability) or under-classify (security risk).

### 3. Implement Need-to-Know
Grant minimum necessary access:
```python
maif.grant_clearance(user_id, level="CONFIDENTIAL", 
                     compartments=["SPECIFIC_PROJECT_ONLY"])
```

### 4. Regular Audits
```python
audit_report = maif.generate_audit_report(
    start_time=yesterday,
    end_time=now
)
```

### 5. Key Rotation
Periodically rotate encryption keys and re-encrypt data.

## Testing

Run security tests:
```bash
cd ../../tests
pytest test_security.py test_privacy.py -v
```

## Limitations

1. **Performance Overhead**: Encryption adds 10-20% overhead
2. **Key Management**: Requires secure key storage
3. **Compliance Scope**: Consult legal team for specific regulations
4. **MFA Integration**: Requires hardware token support

## Production Deployment

### Security Checklist

- [ ] Enable encryption for all sensitive data
- [ ] Implement proper key management
- [ ] Configure audit logging
- [ ] Set up access control policies
- [ ] Enable MFA for privileged operations
- [ ] Regular security audits
- [ ] Incident response procedures
- [ ] Backup and recovery plans

### Monitoring

Monitor security events:
```python
from maif.security import SecurityMonitor

monitor = SecurityMonitor()
monitor.watch_for_anomalies()
monitor.alert_on_unauthorized_access()
```

## Additional Resources

- Security Model: `../../docs/guide/security-model.md`
- Privacy Guide: `../../docs/guide/privacy.md`
- API Reference: `../../docs/api/security/`
- Compliance: `../../docs/MAIF_Security_Verifications_Table.md`

## Support

For security-related questions:
- Review documentation in `docs/guide/security-model.md`
- Check API reference in `docs/api/security/`
- Open GitHub issue with [SECURITY] tag

