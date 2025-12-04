# Cookbook

Welcome to the MAIF Cookbook! This collection contains practical patterns and best practices for building production-grade AI agents with MAIF.

## Quick Recipes

### Simple Agent Memory

```python
from maif_api import create_maif, load_maif

# Create agent with persistent memory
memory = create_maif("my-agent")

# Store conversation context
memory.add_text("User prefers brief responses", title="Preference")
memory.add_text("Last topic: Python programming", title="Context")

# Save and reload
memory.save("agent_memory.maif")

# Later: restore memory
memory = load_maif("agent_memory.maif")
```

### Privacy-First Agent

```python
from maif_api import create_maif

# Enable privacy features
secure_memory = create_maif("hipaa-agent", enable_privacy=True)

# Store with encryption and anonymization
secure_memory.add_text(
    "Patient John Doe has condition XYZ",
    title="Patient Record",
    encrypt=True,      # AES-GCM encryption
    anonymize=True     # Remove PII automatically
)

secure_memory.save("secure.maif", sign=True)

# Check privacy status
report = secure_memory.get_privacy_report()
print(f"Encrypted blocks: {report.get('encrypted_blocks', 0)}")
```

### Multi-Modal Content

```python
from maif_api import create_maif

# Create multimodal artifact
artifact = create_maif("media-agent")

# Add different content types
artifact.add_text("Product: High-quality headphones")
artifact.add_image("product.jpg", title="Product Photo")
artifact.add_multimodal({
    "description": "Wireless noise-cancelling headphones",
    "price": 199.99,
    "features": ["Bluetooth 5.0", "40h battery", "ANC"]
}, title="Product Details")

artifact.save("catalog.maif")
```

## Recipe Categories

### Performance Optimization

#### Batch Processing

```python
from maif_api import create_maif

# Process large datasets efficiently
artifact = create_maif("batch-processor")

# Add content in batches
documents = ["doc1", "doc2", "doc3", ...]  # Large list

for i, doc in enumerate(documents):
    artifact.add_text(doc, title=f"Document {i}")
    
    # Save periodically to avoid memory issues
    if i % 1000 == 0:
        artifact.save(f"batch_{i // 1000}.maif")
        print(f"Saved batch {i // 1000}")
```

#### Efficient Search

```python
from maif_api import load_maif

# Load and search efficiently
kb = load_maif("knowledge_base.maif")

# Semantic search with limited results
results = kb.search("machine learning", top_k=5)

for result in results:
    print(f"Score: {result.get('score', 0):.3f}")
    print(f"Content: {result.get('text', '')[:100]}...")
```

### Security Patterns

#### Signed Artifacts

```python
from maif_api import create_maif, load_maif

# Create with signing
artifact = create_maif("secure-agent")
artifact.add_text("Important data")
artifact.save("signed.maif", sign=True)  # Cryptographic signature

# Verify on load
loaded = load_maif("signed.maif")
if loaded.verify_integrity():
    print("✅ Artifact is authentic")
else:
    print("❌ Artifact may be tampered")
```

#### Access Control

```python
from maif.privacy import PrivacyEngine, AccessRule

# Create privacy engine
privacy = PrivacyEngine()

# Add access rules
rule = AccessRule(
    subject="analyst",
    resource="sensitive_data",
    permissions={"read"},
    conditions={"department": "research"}
)
privacy.add_access_rule(rule)

# Check access
if privacy.check_access("analyst", "sensitive_data", "read"):
    print("Access granted")
```

### Agent Memory Patterns

#### Conversation History

```python
from maif_api import create_maif, load_maif
from datetime import datetime
import os

class ConversationMemory:
    """Persistent conversation memory using MAIF."""
    
    def __init__(self, agent_id: str, path: str = "conversations"):
        self.agent_id = agent_id
        self.path = path
        os.makedirs(path, exist_ok=True)
        
        self.artifact_path = f"{path}/{agent_id}.maif"
        
        if os.path.exists(self.artifact_path):
            self.memory = load_maif(self.artifact_path)
        else:
            self.memory = create_maif(agent_id)
    
    def add_turn(self, user_message: str, agent_response: str):
        """Add a conversation turn."""
        timestamp = datetime.now().isoformat()
        
        self.memory.add_text(
            f"User: {user_message}",
            title=f"User message at {timestamp}"
        )
        self.memory.add_text(
            f"Agent: {agent_response}",
            title=f"Agent response at {timestamp}"
        )
    
    def get_context(self, query: str, top_k: int = 3) -> list:
        """Get relevant context for a query."""
        return self.memory.search(query, top_k=top_k)
    
    def save(self):
        """Persist memory to disk."""
        self.memory.save(self.artifact_path, sign=True)

# Usage
memory = ConversationMemory("support-bot")
memory.add_turn("How do I reset my password?", "Go to Settings > Security...")
memory.save()
```

#### Knowledge Base

```python
from maif_api import create_maif, load_maif
import os

class KnowledgeBase:
    """MAIF-backed knowledge base."""
    
    def __init__(self, name: str, path: str = "kb"):
        self.name = name
        self.path = f"{path}/{name}.maif"
        os.makedirs(path, exist_ok=True)
        
        if os.path.exists(self.path):
            self.kb = load_maif(self.path)
        else:
            self.kb = create_maif(f"kb-{name}")
    
    def add_document(self, content: str, title: str, category: str = None):
        """Add a document to the knowledge base."""
        self.kb.add_text(content, title=title)
    
    def search(self, query: str, top_k: int = 5) -> list:
        """Search the knowledge base."""
        return self.kb.search(query, top_k=top_k)
    
    def save(self):
        """Save the knowledge base."""
        self.kb.save(self.path, sign=True)
        return self.kb.verify_integrity()

# Usage
kb = KnowledgeBase("product-docs")
kb.add_document("Returns accepted within 30 days...", "Return Policy", "policy")
kb.add_document("We're open 9am-5pm Monday-Friday...", "Hours", "info")
kb.save()

# Search
results = kb.search("return policy")
```

### Error Handling

```python
from maif_api import create_maif, load_maif
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def safe_load_artifact(path: str):
    """Safely load a MAIF artifact with error handling."""
    try:
        artifact = load_maif(path)
        
        if not artifact.verify_integrity():
            logger.warning(f"Integrity check failed for {path}")
            return None
        
        return artifact
        
    except FileNotFoundError:
        logger.error(f"Artifact not found: {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading artifact {path}: {e}")
        return None

def safe_save_artifact(artifact, path: str) -> bool:
    """Safely save a MAIF artifact with error handling."""
    try:
        success = artifact.save(path, sign=True)
        
        if success:
            logger.info(f"Saved artifact to {path}")
        else:
            logger.error(f"Failed to save artifact to {path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error saving artifact {path}: {e}")
        return False
```

### Monitoring

```python
import time
from contextlib import contextmanager
from maif_api import create_maif

@contextmanager
def timed_operation(name: str):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start) * 1000
        print(f"{name}: {duration:.2f}ms")

# Usage
artifact = create_maif("monitored-agent")

with timed_operation("add_text"):
    artifact.add_text("Sample content")

with timed_operation("save"):
    artifact.save("monitored.maif")
```

## Best Practices

### 1. Agent Memory

- **Persist regularly**: Save after important operations
- **Use meaningful titles**: Makes searching easier
- **Verify on load**: Always check `verify_integrity()`

### 2. Privacy

- **Enable for sensitive data**: Use `enable_privacy=True`
- **Encrypt confidential content**: Use `encrypt=True`
- **Anonymize PII**: Use `anonymize=True` for user data

### 3. Performance

- **Batch large operations**: Process in chunks
- **Limit search results**: Use appropriate `top_k`
- **Save periodically**: Don't accumulate too much in memory

### 4. Security

- **Sign artifacts**: Use `sign=True` on save
- **Verify integrity**: Check before using loaded data
- **Track provenance**: Use meaningful agent IDs

## Quick Reference

```python
from maif_api import create_maif, load_maif

# Create
maif = create_maif("agent-id", enable_privacy=False)

# Add content
maif.add_text("text", title="Title", encrypt=False, anonymize=False)
maif.add_image("path.jpg", title="Image")
maif.add_multimodal({"key": "value"}, title="Data")
maif.add_embeddings([[0.1, 0.2, ...]], model_name="model")

# Save
maif.save("output.maif", sign=True)

# Load
loaded = load_maif("output.maif")

# Verify
is_valid = loaded.verify()

# Search
results = loaded.search("query", top_k=5)

# Get content
content = loaded.get_content_list()

# Privacy report
report = loaded.get_privacy_report()
```

## Next Steps

- [Getting Started Guide](/guide/getting-started)
- [API Reference](/api/)
- [Examples](/examples/)
