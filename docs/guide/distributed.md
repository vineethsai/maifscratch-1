# Distributed Patterns

::: danger DEPRECATED
This page is deprecated. For the latest documentation, please visit **[DeepWiki](https://deepwiki.com/vineethsai/maif)**.
:::

This guide covers patterns for using MAIF in distributed systems, including file synchronization, multi-node architectures, and scaling strategies.

## Overview

MAIF artifacts are files that can be:

- **Shared across nodes** via network file systems
- **Synchronized** using standard tools
- **Replicated** to multiple locations
- **Accessed concurrently** with proper locking

## File-Based Architecture

### Shared Storage Pattern

Use a shared file system for MAIF artifacts:

```python
import os
from maif_api import create_maif, load_maif

class SharedArtifactStore:
    """Store artifacts on shared storage."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def create(self, name: str, content: str):
        """Create artifact on shared storage."""
        artifact = create_maif(name)
        artifact.add_text(content)
        
        path = os.path.join(self.base_path, f"{name}.maif")
        artifact.save(path)
        return path
    
    def load(self, name: str):
        """Load artifact from shared storage."""
        path = os.path.join(self.base_path, f"{name}.maif")
        return load_maif(path)
    
    def list_artifacts(self):
        """List all artifacts in storage."""
        return [
            f[:-5] for f in os.listdir(self.base_path) 
            if f.endswith('.maif')
        ]

# Usage with NFS, S3FS, or other mounted storage
store = SharedArtifactStore("/mnt/shared/artifacts")
store.create("document-001", "Shared document content")
```

### Local Cache Pattern

Cache frequently accessed artifacts locally:

```python
import os
import shutil
from maif_api import load_maif

class CachedArtifactStore:
    """Cache artifacts locally from remote storage."""
    
    def __init__(self, remote_path: str, cache_path: str):
        self.remote_path = remote_path
        self.cache_path = cache_path
        os.makedirs(cache_path, exist_ok=True)
    
    def get(self, name: str):
        """Get artifact, using cache if available."""
        cache_file = os.path.join(self.cache_path, f"{name}.maif")
        remote_file = os.path.join(self.remote_path, f"{name}.maif")
        
        # Check if cached version is current
        if self._is_cache_valid(cache_file, remote_file):
            return load_maif(cache_file)
        
        # Copy from remote to cache
        shutil.copy2(remote_file, cache_file)
        return load_maif(cache_file)
    
    def _is_cache_valid(self, cache_file: str, remote_file: str) -> bool:
        """Check if cache is newer than remote."""
        if not os.path.exists(cache_file):
            return False
        if not os.path.exists(remote_file):
            return True
        return os.path.getmtime(cache_file) >= os.path.getmtime(remote_file)
    
    def invalidate(self, name: str):
        """Invalidate cached artifact."""
        cache_file = os.path.join(self.cache_path, f"{name}.maif")
        if os.path.exists(cache_file):
            os.remove(cache_file)

# Usage
cache = CachedArtifactStore("/mnt/nfs/artifacts", "/tmp/artifact_cache")
artifact = cache.get("frequently-accessed")
```

## Synchronization Patterns

### File Locking

Use file locking for concurrent access:

```python
import fcntl
import os
from contextlib import contextmanager
from maif_api import create_maif, load_maif

@contextmanager
def locked_artifact(path: str, mode: str = 'r'):
    """Context manager for locked artifact access."""
    lock_path = f"{path}.lock"
    lock_file = open(lock_path, 'w')
    
    try:
        # Acquire lock (exclusive for write, shared for read)
        if mode == 'w':
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        else:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
        
        yield path
        
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

# Usage for reading
with locked_artifact("shared.maif", 'r') as path:
    artifact = load_maif(path)
    data = artifact.get_content_list()

# Usage for writing
with locked_artifact("shared.maif", 'w') as path:
    artifact = create_maif("shared")
    artifact.add_text("Updated content")
    artifact.save(path)
```

### Versioned Artifacts

Track versions of artifacts:

```python
import os
import shutil
from datetime import datetime
from maif_api import create_maif, load_maif

class VersionedArtifactStore:
    """Store artifacts with version history."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.versions_path = os.path.join(base_path, "versions")
        os.makedirs(self.versions_path, exist_ok=True)
    
    def save(self, name: str, artifact):
        """Save artifact with versioning."""
        current_path = os.path.join(self.base_path, f"{name}.maif")
        
        # Archive existing version
        if os.path.exists(current_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(
                self.versions_path, 
                f"{name}_{timestamp}.maif"
            )
            shutil.copy2(current_path, archive_path)
        
        # Save new version
        artifact.save(current_path)
    
    def load(self, name: str, version: str = None):
        """Load artifact, optionally a specific version."""
        if version:
            path = os.path.join(self.versions_path, f"{name}_{version}.maif")
        else:
            path = os.path.join(self.base_path, f"{name}.maif")
        return load_maif(path)
    
    def list_versions(self, name: str) -> list:
        """List all versions of an artifact."""
        versions = []
        prefix = f"{name}_"
        for f in os.listdir(self.versions_path):
            if f.startswith(prefix) and f.endswith('.maif'):
                version = f[len(prefix):-5]
                versions.append(version)
        return sorted(versions)

# Usage
store = VersionedArtifactStore("./artifacts")
artifact = create_maif("versioned-doc")
artifact.add_text("Version 1")
store.save("versioned-doc", artifact)

# Later...
artifact.add_text("Version 2")
store.save("versioned-doc", artifact)

# List versions
versions = store.list_versions("versioned-doc")
print(f"Available versions: {versions}")
```

## Multi-Node Patterns

### Leader-Follower Pattern

Designate nodes for writes vs reads:

```python
import os
from maif_api import create_maif, load_maif

class LeaderFollowerStore:
    """Leader handles writes, followers handle reads."""
    
    def __init__(self, is_leader: bool, shared_path: str, local_cache: str = None):
        self.is_leader = is_leader
        self.shared_path = shared_path
        self.local_cache = local_cache
        
        if local_cache:
            os.makedirs(local_cache, exist_ok=True)
    
    def write(self, name: str, content: str):
        """Write artifact (leader only)."""
        if not self.is_leader:
            raise RuntimeError("Only leader can write")
        
        artifact = create_maif(name)
        artifact.add_text(content)
        
        path = os.path.join(self.shared_path, f"{name}.maif")
        artifact.save(path)
        return path
    
    def read(self, name: str):
        """Read artifact (any node)."""
        # Try local cache first
        if self.local_cache:
            cache_path = os.path.join(self.local_cache, f"{name}.maif")
            if os.path.exists(cache_path):
                return load_maif(cache_path)
        
        # Fall back to shared storage
        path = os.path.join(self.shared_path, f"{name}.maif")
        return load_maif(path)
    
    def refresh_cache(self, name: str):
        """Refresh local cache from shared storage."""
        if not self.local_cache:
            return
        
        shared_file = os.path.join(self.shared_path, f"{name}.maif")
        cache_file = os.path.join(self.local_cache, f"{name}.maif")
        
        import shutil
        shutil.copy2(shared_file, cache_file)

# Leader node
leader = LeaderFollowerStore(
    is_leader=True, 
    shared_path="/mnt/shared/artifacts"
)
leader.write("document", "Content from leader")

# Follower nodes
follower = LeaderFollowerStore(
    is_leader=False, 
    shared_path="/mnt/shared/artifacts",
    local_cache="/tmp/cache"
)
follower.refresh_cache("document")
data = follower.read("document")
```

### Partitioned Storage

Partition artifacts across nodes:

```python
import hashlib
import os
from maif_api import create_maif, load_maif

class PartitionedStore:
    """Partition artifacts across multiple storage paths."""
    
    def __init__(self, partitions: list):
        self.partitions = partitions
        for partition in partitions:
            os.makedirs(partition, exist_ok=True)
    
    def _get_partition(self, name: str) -> str:
        """Determine partition for an artifact."""
        hash_val = hashlib.md5(name.encode()).hexdigest()
        partition_idx = int(hash_val, 16) % len(self.partitions)
        return self.partitions[partition_idx]
    
    def save(self, name: str, artifact):
        """Save artifact to appropriate partition."""
        partition = self._get_partition(name)
        path = os.path.join(partition, f"{name}.maif")
        artifact.save(path)
        return path
    
    def load(self, name: str):
        """Load artifact from appropriate partition."""
        partition = self._get_partition(name)
        path = os.path.join(partition, f"{name}.maif")
        return load_maif(path)
    
    def get_distribution(self) -> dict:
        """Get artifact count per partition."""
        distribution = {}
        for partition in self.partitions:
            count = len([f for f in os.listdir(partition) if f.endswith('.maif')])
            distribution[partition] = count
        return distribution

# Usage with 4 partitions
store = PartitionedStore([
    "/data/partition1",
    "/data/partition2", 
    "/data/partition3",
    "/data/partition4"
])

# Artifacts are automatically distributed
for i in range(100):
    artifact = create_maif(f"doc-{i}")
    artifact.add_text(f"Content {i}")
    store.save(f"doc-{i}", artifact)

# Check distribution
print(store.get_distribution())
```

## Replication Patterns

### Simple Replication

Replicate artifacts to multiple locations:

```python
import os
import shutil
from maif_api import create_maif, load_maif

class ReplicatedStore:
    """Replicate artifacts to multiple locations."""
    
    def __init__(self, locations: list):
        self.locations = locations
        for location in locations:
            os.makedirs(location, exist_ok=True)
    
    def save(self, name: str, artifact):
        """Save to all replicas."""
        # Save to first location
        primary_path = os.path.join(self.locations[0], f"{name}.maif")
        artifact.save(primary_path)
        
        # Replicate to other locations
        for location in self.locations[1:]:
            replica_path = os.path.join(location, f"{name}.maif")
            shutil.copy2(primary_path, replica_path)
    
    def load(self, name: str):
        """Load from first available replica."""
        for location in self.locations:
            path = os.path.join(location, f"{name}.maif")
            if os.path.exists(path):
                try:
                    return load_maif(path)
                except:
                    continue
        raise FileNotFoundError(f"Artifact {name} not found in any replica")
    
    def verify_replicas(self, name: str) -> dict:
        """Verify artifact exists and is consistent across replicas."""
        results = {}
        reference_hash = None
        
        for location in self.locations:
            path = os.path.join(location, f"{name}.maif")
            results[location] = {
                "exists": os.path.exists(path),
                "consistent": False
            }
            
            if results[location]["exists"]:
                import hashlib
                with open(path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                if reference_hash is None:
                    reference_hash = file_hash
                results[location]["consistent"] = (file_hash == reference_hash)
        
        return results

# Usage
store = ReplicatedStore([
    "/data/primary",
    "/data/replica1",
    "/data/replica2"
])

artifact = create_maif("important-doc")
artifact.add_text("Critical data")
store.save("important-doc", artifact)

# Verify replicas
status = store.verify_replicas("important-doc")
print(status)
```

## Backup and Recovery

### Automated Backup

```python
import os
import shutil
from datetime import datetime

class BackupManager:
    """Manage artifact backups."""
    
    def __init__(self, source_dir: str, backup_dir: str):
        self.source_dir = source_dir
        self.backup_dir = backup_dir
    
    def create_backup(self) -> str:
        """Create timestamped backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        shutil.copytree(self.source_dir, backup_path)
        return backup_path
    
    def restore_backup(self, backup_name: str):
        """Restore from backup."""
        backup_path = os.path.join(self.backup_dir, backup_name)
        
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup {backup_name} not found")
        
        # Remove current data
        if os.path.exists(self.source_dir):
            shutil.rmtree(self.source_dir)
        
        # Restore from backup
        shutil.copytree(backup_path, self.source_dir)
    
    def list_backups(self) -> list:
        """List available backups."""
        if not os.path.exists(self.backup_dir):
            return []
        return sorted([
            d for d in os.listdir(self.backup_dir) 
            if d.startswith("backup_")
        ])
    
    def cleanup_old_backups(self, keep: int = 5):
        """Remove old backups, keeping the most recent."""
        backups = self.list_backups()
        if len(backups) <= keep:
            return
        
        for backup in backups[:-keep]:
            backup_path = os.path.join(self.backup_dir, backup)
            shutil.rmtree(backup_path)

# Usage
backup_mgr = BackupManager("./artifacts", "./backups")

# Create backup
backup_path = backup_mgr.create_backup()
print(f"Backup created: {backup_path}")

# List backups
print(f"Available backups: {backup_mgr.list_backups()}")

# Cleanup old backups
backup_mgr.cleanup_old_backups(keep=3)
```

## Best Practices

1. **Use file locking** for concurrent access
2. **Implement caching** to reduce network I/O
3. **Version artifacts** for change tracking
4. **Verify integrity** after transfers
5. **Regular backups** for disaster recovery

## Next Steps

- **[Performance →](/guide/performance)** - Optimize distributed operations
- **[Monitoring →](/guide/monitoring)** - Monitor distributed systems
- **[Architecture →](/guide/architecture)** - System design
