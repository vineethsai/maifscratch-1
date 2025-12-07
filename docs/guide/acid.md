# Data Integrity Patterns

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki - Cryptographic Security](https://deepwiki.com/vineethsai/maif/2.2-cryptographic-security)**.
:::

MAIF provides data integrity through cryptographic hashing, verification, and atomic file operations. This guide covers patterns for ensuring data reliability.

## Overview

MAIF ensures data integrity through:

- **Cryptographic Hash Chains**: Each block is linked by hash
- **Digital Signatures**: Optional signing for authenticity
- **Atomic Operations**: File-level atomicity
- **Integrity Verification**: Built-in verification methods

## Integrity Verification

### Basic Integrity Check

```python
from maif_api import load_maif

def verify_artifact(path: str) -> bool:
    """Verify artifact integrity."""
    try:
        artifact = load_maif(path)
        is_valid = artifact.verify()
        
        if is_valid:
            print(f"{path}: Integrity verified")
        else:
            print(f"{path}: Integrity check failed")
        
        return is_valid
    except Exception as e:
        print(f"{path}: Error - {e}")
        return False

# Verify single artifact
verify_artifact("document.maif")
```

### Batch Verification

```python
import os
from maif_api import load_maif

def verify_directory(directory: str) -> dict:
    """Verify all artifacts in a directory."""
    results = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": []
    }
    
    for filename in os.listdir(directory):
        if filename.endswith('.maif'):
            results["total"] += 1
            path = os.path.join(directory, filename)
            
            try:
                artifact = load_maif(path)
                if artifact.verify_integrity():
                    results["valid"] += 1
                else:
                    results["invalid"] += 1
                    results["errors"].append({
                        "file": filename,
                        "error": "Integrity check failed"
                    })
            except Exception as e:
                results["invalid"] += 1
                results["errors"].append({
                    "file": filename,
                    "error": str(e)
                })
    
    return results

# Verify all artifacts
results = verify_directory("./artifacts")
print(f"Valid: {results['valid']}/{results['total']}")
```

## Atomic Operations

### Safe File Writing

Write artifacts atomically to prevent corruption:

```python
import os
import tempfile
import shutil
from maif_api import create_maif

def atomic_save(artifact, path: str):
    """Save artifact atomically using temp file."""
    directory = os.path.dirname(path) or '.'
    
    # Write to temporary file first
    fd, temp_path = tempfile.mkstemp(
        suffix='.maif.tmp',
        dir=directory
    )
    os.close(fd)
    
    try:
        artifact.save(temp_path)
        # Atomic rename
        shutil.move(temp_path, path)
    except Exception:
        # Clean up on failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

# Usage
artifact = create_maif("important-doc")
artifact.add_text("Critical data")
atomic_save(artifact, "important.maif")
```

### Transactional Updates

Implement transaction-like behavior:

```python
import os
import shutil
from maif_api import create_maif, load_maif

class ArtifactTransaction:
    """Transaction-like updates for artifacts."""
    
    def __init__(self, path: str):
        self.path = path
        self.backup_path = f"{path}.backup"
        self.active = False
    
    def begin(self):
        """Start transaction by creating backup."""
        if os.path.exists(self.path):
            shutil.copy2(self.path, self.backup_path)
        self.active = True
    
    def commit(self):
        """Commit transaction by removing backup."""
        if os.path.exists(self.backup_path):
            os.remove(self.backup_path)
        self.active = False
    
    def rollback(self):
        """Rollback transaction by restoring backup."""
        if os.path.exists(self.backup_path):
            shutil.move(self.backup_path, self.path)
        self.active = False
    
    def __enter__(self):
        self.begin()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        else:
            self.commit()
        return False

# Usage with context manager
with ArtifactTransaction("data.maif") as tx:
    # Load and modify artifact
    artifact = load_maif("data.maif") if os.path.exists("data.maif") else create_maif("data")
    artifact.add_text("New content")
    artifact.save("data.maif")
    
    # If anything fails, transaction rolls back automatically
```

## Signed Artifacts

### Creating Signed Artifacts

```python
from maif.security import MAIFSigner, MAIFVerifier
from maif_api import create_maif

# Create signer for your agent/service
signer = MAIFSigner(agent_id="trusted-service")

# Create artifact
artifact = create_maif("signed-document")
artifact.add_text("Important content that must be authenticated")

# Save with signature
artifact.save("signed.maif", sign=True)

print(f"Public key:\n{signer.get_public_key_pem()}")
```

### Verifying Signatures

```python
from maif.security import MAIFVerifier
from maif.core import MAIFDecoder

def verify_signed_artifact(path: str, expected_agent_id: str = None) -> dict:
    """Verify artifact signature and provenance."""
    result = {
        "path": path,
        "integrity": False,
        "signature": False,
        "agent_id": None
    }
    
    try:
        decoder = MAIFDecoder(path)
        
        # Check hash chain integrity
        result["integrity"] = True  # If no error, integrity is OK
        
        # Check manifest for signature info
        manifest = decoder.get_manifest()
        if manifest:
            result["agent_id"] = manifest.get("agent_id")
            result["signature"] = manifest.get("signature") is not None
        
        # Verify agent if expected
        if expected_agent_id and result["agent_id"] != expected_agent_id:
            result["trusted"] = False
        else:
            result["trusted"] = result["signature"]
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# Verify
result = verify_signed_artifact("signed.maif", "trusted-service")
print(f"Trusted: {result.get('trusted', False)}")
```

## Provenance Tracking

### Record Provenance

```python
from maif.security import MAIFSigner
from maif_api import create_maif
from datetime import datetime

class ProvenanceTracker:
    """Track provenance of artifact operations."""
    
    def __init__(self, agent_id: str):
        self.signer = MAIFSigner(agent_id=agent_id)
        self.entries = []
    
    def record_action(self, action: str, artifact_id: str, details: dict = None):
        """Record a provenance entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "artifact_id": artifact_id,
            "details": details or {}
        }
        
        # Add provenance to signer
        self.signer.add_provenance_entry(action, artifact_id)
        self.entries.append(entry)
        
        return entry
    
    def get_chain(self) -> list:
        """Get full provenance chain."""
        return self.entries.copy()

# Usage
tracker = ProvenanceTracker("data-pipeline")

# Track operations
tracker.record_action("create", "doc-001", {"source": "user-upload"})
tracker.record_action("transform", "doc-001", {"operation": "text-extraction"})
tracker.record_action("validate", "doc-001", {"result": "passed"})

# Get provenance
for entry in tracker.get_chain():
    print(f"{entry['timestamp']}: {entry['action']} on {entry['artifact_id']}")
```

## Consistency Patterns

### Read-After-Write Consistency

Ensure reads see recent writes:

```python
import os
from maif_api import create_maif, load_maif

class ConsistentStore:
    """Store with read-after-write consistency."""
    
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
    
    def write(self, name: str, content: str):
        """Write and sync artifact."""
        artifact = create_maif(name)
        artifact.add_text(content)
        
        file_path = os.path.join(self.path, f"{name}.maif")
        artifact.save(file_path)
        
        # Force sync to disk
        with open(file_path, 'rb') as f:
            os.fsync(f.fileno())
        
        return file_path
    
    def read(self, name: str):
        """Read artifact."""
        file_path = os.path.join(self.path, f"{name}.maif")
        return load_maif(file_path)

# Usage
store = ConsistentStore("./consistent_store")
store.write("doc", "Content")

# Immediate read will see the write
doc = store.read("doc")
```

### Optimistic Concurrency

Handle concurrent updates:

```python
import os
import hashlib
from maif_api import create_maif, load_maif

class OptimisticStore:
    """Store with optimistic concurrency control."""
    
    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
    
    def _get_version(self, file_path: str) -> str:
        """Get file version (hash)."""
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_with_version(self, name: str):
        """Load artifact with version info."""
        file_path = os.path.join(self.path, f"{name}.maif")
        version = self._get_version(file_path)
        artifact = load_maif(file_path)
        return artifact, version
    
    def save_with_version(self, name: str, artifact, expected_version: str):
        """Save only if version matches."""
        file_path = os.path.join(self.path, f"{name}.maif")
        current_version = self._get_version(file_path)
        
        if current_version != expected_version:
            raise ConflictError(
                f"Version mismatch: expected {expected_version}, "
                f"found {current_version}"
            )
        
        artifact.save(file_path)

class ConflictError(Exception):
    pass

# Usage
store = OptimisticStore("./optimistic_store")

# Load with version
artifact, version = store.load_with_version("shared-doc")

# Modify
artifact.add_text("Updated content")

# Save with version check
try:
    store.save_with_version("shared-doc", artifact, version)
except ConflictError:
    print("Conflict detected! Reload and retry.")
```

## Data Validation

### Schema Validation

Validate artifact content:

```python
from maif_api import load_maif
import json

class ArtifactValidator:
    """Validate artifact content."""
    
    def __init__(self, schema: dict):
        self.schema = schema
    
    def validate(self, artifact) -> dict:
        """Validate artifact against schema."""
        result = {
            "valid": True,
            "errors": []
        }
        
        contents = artifact.get_content_list()
        
        # Check required fields
        if "required_types" in self.schema:
            found_types = set(c.get("type") for c in contents)
            for req_type in self.schema["required_types"]:
                if req_type not in found_types:
                    result["valid"] = False
                    result["errors"].append(f"Missing required type: {req_type}")
        
        # Check minimum content
        if "min_content_count" in self.schema:
            if len(contents) < self.schema["min_content_count"]:
                result["valid"] = False
                result["errors"].append(
                    f"Minimum {self.schema['min_content_count']} content items required"
                )
        
        return result

# Usage
validator = ArtifactValidator({
    "required_types": ["text"],
    "min_content_count": 1
})

artifact = load_maif("document.maif")
result = validator.validate(artifact)

if not result["valid"]:
    print(f"Validation errors: {result['errors']}")
```

## Recovery Procedures

### Automatic Recovery

```python
import os
import shutil
from maif_api import load_maif

class RecoveryManager:
    """Manage artifact recovery."""
    
    def __init__(self, data_dir: str, backup_dir: str):
        self.data_dir = data_dir
        self.backup_dir = backup_dir
    
    def check_and_recover(self, name: str) -> bool:
        """Check artifact and recover if corrupted."""
        data_path = os.path.join(self.data_dir, f"{name}.maif")
        backup_path = os.path.join(self.backup_dir, f"{name}.maif")
        
        # Check primary
        try:
            artifact = load_maif(data_path)
            if artifact.verify_integrity():
                return True
        except:
            pass
        
        # Primary is corrupted, try backup
        print(f"Primary corrupted, attempting recovery: {name}")
        
        if os.path.exists(backup_path):
            try:
                artifact = load_maif(backup_path)
                if artifact.verify_integrity():
                    shutil.copy2(backup_path, data_path)
                    print(f"Recovered from backup: {name}")
                    return True
            except:
                pass
        
        print(f"Recovery failed: {name}")
        return False
    
    def recover_all(self) -> dict:
        """Attempt recovery of all artifacts."""
        results = {"recovered": 0, "failed": 0}
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.maif'):
                name = filename[:-5]
                if self.check_and_recover(name):
                    results["recovered"] += 1
                else:
                    results["failed"] += 1
        
        return results
```

## Best Practices

1. **Always verify integrity** after loading important artifacts
2. **Use atomic saves** to prevent corruption during writes
3. **Implement backups** for critical data
4. **Sign artifacts** when authenticity matters
5. **Track provenance** for audit requirements

## Next Steps

- **[Security Model →](/guide/security-model)** - Security features
- **[Distributed →](/guide/distributed)** - Distributed patterns
- **[API Reference →](/api/)** - Complete documentation
