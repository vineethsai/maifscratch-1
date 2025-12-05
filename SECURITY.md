# Security Policy

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security vulnerability in MAIF, please report it responsibly:

### How to Report

1. **Email:** Send details to [security@maif.ai](mailto:security@maif.ai)
2. **Include:**
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Response Time:** We aim to respond within 48 hours
- **Updates:** You'll receive updates on the status every 5-7 days
- **Disclosure:** We'll coordinate public disclosure timing with you
- **Credit:** You'll be credited in the security advisory (unless you prefer anonymity)

## Security Considerations

### Cryptographic Security

MAIF uses modern cryptographic standards:

- **Signatures:** Ed25519 (64-byte signatures)
- **Hashing:** SHA-256 for integrity verification
- **Encryption:** AES-256-GCM and ChaCha20-Poly1305
- **Key Derivation:** PBKDF2 with appropriate iterations

### Known Security Features

✅ **Implemented:**
- Cryptographic provenance chains
- Tamper detection (100% detection rate)
- Block-level integrity verification
- Access control and permissions
- Encryption at rest and in transit
- PII detection and anonymization

### Security Best Practices

When using MAIF in production:

1. **Key Management**
   - Never hardcode private keys
   - Use secure key storage (HSM, KMS, or secure vaults)
   - Rotate keys according to your security policy

2. **Access Control**
   - Implement least-privilege access
   - Use role-based access control (RBAC)
   - Log all access attempts

3. **Data Protection**
   - Enable encryption for sensitive data
   - Use anonymization for PII
   - Set appropriate privacy levels

4. **Integrity Verification**
   - Always verify signatures before trusting data
   - Check provenance chains for critical operations
   - Monitor for tamper attempts

5. **Network Security**
   - Use TLS for data in transit
   - Implement proper authentication
   - Rate limit API endpoints

### Dependencies

We regularly update dependencies to patch security vulnerabilities:

- **Core:** `cryptography`, `numpy`, `pydantic`
- **Optional:** `boto3`, `sentence-transformers`, `opencv-python`

To check for known vulnerabilities:

```bash
pip install safety
safety check --file requirements.txt
```

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.0.x   | ✅ Yes             |
| 1.x.x   | ⚠️ Security fixes only |
| < 1.0   | ❌ No              |

## Security Updates

Security updates are released as soon as possible:

- **Critical:** Within 24-48 hours
- **High:** Within 1 week
- **Medium/Low:** Next regular release

Subscribe to [GitHub Security Advisories](https://github.com/vineethsai/maif/security/advisories) for notifications.

## Vulnerability Disclosure Policy

We follow coordinated vulnerability disclosure:

1. **Private Notification:** Vulnerability reported privately
2. **Investigation:** We confirm and develop a fix
3. **Patch Release:** Security update released
4. **Public Disclosure:** Advisory published 7 days after patch
5. **Credit:** Reporter credited in advisory and changelog

## Scope

### In Scope

- Cryptographic vulnerabilities
- Authentication/authorization bypass
- Data leakage or exposure
- Code injection or execution
- Denial of service
- Supply chain attacks

### Out of Scope

- Issues in dependencies (report to upstream)
- Social engineering
- Physical attacks
- Issues requiring physical access to systems

## Security Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities:

<!-- Security researchers will be listed here -->

## Compliance

MAIF is designed with compliance in mind:

- **FIPS 140-2:** Compatible cryptographic modules
- **NIST 800-53:** Security and privacy controls
- **GDPR:** Privacy-by-design features
- **SOC 2:** Audit trail and access controls

For compliance documentation, see [`docs/guide/security-model.md`](docs/guide/security-model.md).

## Questions?

For security questions that aren't vulnerabilities:
- Open a [GitHub Discussion](https://github.com/vineethsai/maif/discussions)
- Email [dev@maif.ai](mailto:dev@maif.ai)

---

**Thank you for helping keep MAIF secure!** 🔒

