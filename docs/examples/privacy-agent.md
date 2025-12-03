# Privacy-Enabled Chat Agent

A realistic agent with memory and built-in privacy features.

## Code

```python
from maif_api import create_maif, load_maif
import os

class PrivateChatAgent:
    """Chat agent with privacy-protected memory."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.memory_path = f"{agent_id}_memory.maif"
        
        # Create or load memory with privacy enabled
        if os.path.exists(self.memory_path):
            self.memory = load_maif(self.memory_path)
        else:
            self.memory = create_maif(agent_id, enable_privacy=True)
    
    def chat(self, message: str, user_id: str) -> str:
        """Process a chat message with privacy protection."""
        
        # Store message with encryption and anonymization
        self.memory.add_text(
            f"User {user_id}: {message}",
            title="User Message",
            encrypt=True,       # AES-GCM encryption
            anonymize=True      # Remove PII automatically
        )
        
        # Search for relevant context
        context = self.memory.search(message, top_k=3)
        
        # Generate response (integrate your LLM here)
        response = f"I understand you're asking about: {message}"
        
        # Store response with same privacy level
        self.memory.add_text(
            f"Agent: {response}",
            title="Agent Response",
            encrypt=True
        )
        
        return response
    
    def save(self):
        """Save memory with cryptographic signing."""
        self.memory.save(self.memory_path, sign=True)
    
    def get_privacy_report(self) -> dict:
        """Get privacy status report."""
        return self.memory.get_privacy_report()

# Usage
agent = PrivateChatAgent("support-bot")
response = agent.chat("How do I reset my password?", "user123")
print(response)

# Check privacy status
report = agent.get_privacy_report()
print(f"Privacy report: {report}")

# Save with signature
agent.save()
```

## Key Concepts

### Privacy Levels

MAIF supports different privacy levels for classifying data sensitivity:

```python
from maif.privacy import PrivacyLevel

# Available levels
PrivacyLevel.PUBLIC       # No protection
PrivacyLevel.INTERNAL     # Internal use only
PrivacyLevel.CONFIDENTIAL # Encrypted storage
PrivacyLevel.RESTRICTED   # Maximum protection
```

### Encryption

Data is encrypted at rest using industry-standard algorithms:

```python
from maif.privacy import EncryptionMode

# Available modes
EncryptionMode.AES_GCM           # AES-256-GCM (default)
EncryptionMode.CHACHA20_POLY1305 # ChaCha20-Poly1305
```

### Anonymization

Automatic PII detection and redaction:

```python
# Original: "John Doe, SSN: 123-45-6789"
# After anonymization: "[PERSON], SSN: [REDACTED]"

artifact.add_text(
    "User John Doe called about order #12345",
    anonymize=True  # PII automatically removed
)
```

### Privacy Report

Get insights into data protection status:

```python
report = artifact.get_privacy_report()
# {
#   "encrypted_blocks": 5,
#   "anonymized_blocks": 3,
#   "privacy_level": "CONFIDENTIAL",
#   "total_blocks": 10
# }
```

## Running the Example

```bash
cd maifscratch-1
pip install -e .

python3 -c "
from maif_api import create_maif

# Create privacy-enabled artifact
memory = create_maif('private-agent', enable_privacy=True)

# Add encrypted content
memory.add_text(
    'User email: john@example.com',
    encrypt=True,
    anonymize=True
)

# Save with signature
memory.save('private.maif', sign=True)

# Check privacy
report = memory.get_privacy_report()
print(f'Privacy report: {report}')
"
```

## What You'll Learn

- Enabling privacy features in MAIF
- Encrypting sensitive data
- Automatic PII anonymization
- Privacy reporting and compliance

## Next Steps

- [Privacy Guide](/guide/privacy) - Full privacy documentation
- [Security Model](/guide/security-model) - Cryptographic details
- [Financial Agent](/examples/financial-agent) - Production patterns
