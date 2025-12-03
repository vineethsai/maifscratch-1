# Distributed Processing

This example demonstrates patterns for using MAIF in distributed systems, including multi-node coordination, state synchronization, and file sharing.

## Overview

MAIF artifacts can be shared and synchronized across distributed systems using standard file operations and network protocols.

### Key Patterns

- **Shared Storage**: Use network file systems (NFS, S3FS) for artifact sharing
- **State Synchronization**: Keep agent state consistent across nodes
- **File Locking**: Handle concurrent access safely
- **Replication**: Copy artifacts to multiple locations

## Code Example

### Shared Artifact Store

```python
from maif_api import create_maif, load_maif
import os
import shutil
import fcntl
from contextlib import contextmanager

class SharedArtifactStore:
    """Distributed artifact store using shared storage."""
    
    def __init__(self, shared_path: str, local_cache: str = "/tmp/maif_cache"):
        self.shared_path = shared_path
        self.local_cache = local_cache
        os.makedirs(shared_path, exist_ok=True)
        os.makedirs(local_cache, exist_ok=True)
    
    @contextmanager
    def locked_artifact(self, name: str, mode: str = 'r'):
        """Access artifact with file locking."""
        lock_path = f"{self.shared_path}/{name}.lock"
        lock_file = open(lock_path, 'w')
        
        try:
            # Exclusive lock for writes, shared for reads
            if mode == 'w':
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            else:
                fcntl.flock(lock_file, fcntl.LOCK_SH)
            
            yield f"{self.shared_path}/{name}.maif"
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
    
    def create(self, name: str, content: str) -> str:
        """Create artifact with exclusive lock."""
        with self.locked_artifact(name, 'w') as path:
            artifact = create_maif(name)
            artifact.add_text(content)
            artifact.save(path, sign=True)
            return path
    
    def load(self, name: str):
        """Load artifact with shared lock."""
        with self.locked_artifact(name, 'r') as path:
            return load_maif(path)
    
    def load_cached(self, name: str):
        """Load with local caching for performance."""
        cache_path = f"{self.local_cache}/{name}.maif"
        shared_path = f"{self.shared_path}/{name}.maif"
        
        # Check if cache is valid
        if os.path.exists(cache_path):
            cache_time = os.path.getmtime(cache_path)
            shared_time = os.path.getmtime(shared_path)
            if cache_time >= shared_time:
                return load_maif(cache_path)
        
        # Copy to cache and load
        with self.locked_artifact(name, 'r') as path:
            shutil.copy2(path, cache_path)
        
        return load_maif(cache_path)

# Usage
store = SharedArtifactStore("/mnt/shared/maif")

# Create artifact (with locking)
store.create("shared-document", "Content visible to all nodes")

# Load with caching
artifact = store.load_cached("shared-document")
```

### Multi-Node Coordination

```python
from maif_api import create_maif, load_maif
import os
import json
from datetime import datetime

class DistributedAgent:
    """Agent that coordinates with others via shared MAIF artifacts."""
    
    def __init__(self, agent_id: str, shared_path: str):
        self.agent_id = agent_id
        self.shared_path = shared_path
        self.state_file = f"{shared_path}/agent_states.maif"
        
        os.makedirs(shared_path, exist_ok=True)
    
    def register(self):
        """Register this agent in the shared state."""
        # Load or create shared state
        if os.path.exists(self.state_file):
            state = load_maif(self.state_file)
        else:
            state = create_maif("agent-states")
        
        # Add registration
        state.add_text(
            json.dumps({
                "agent_id": self.agent_id,
                "registered_at": datetime.now().isoformat(),
                "status": "active"
            }),
            title=f"Agent Registration: {self.agent_id}"
        )
        
        state.save(self.state_file, sign=True)
    
    def get_active_agents(self) -> list:
        """Get list of active agents."""
        if not os.path.exists(self.state_file):
            return []
        
        state = load_maif(self.state_file)
        content = state.get_content_list()
        
        agents = []
        for item in content:
            if "Agent Registration" in item.get("title", ""):
                try:
                    data = json.loads(item.get("text", "{}"))
                    if data.get("status") == "active":
                        agents.append(data)
                except:
                    pass
        
        return agents
    
    def publish_result(self, task_id: str, result: dict):
        """Publish a result for other agents to see."""
        result_file = f"{self.shared_path}/results/{task_id}.maif"
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        
        artifact = create_maif(f"result-{task_id}")
        artifact.add_text(
            json.dumps({
                "task_id": task_id,
                "agent_id": self.agent_id,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }),
            title=f"Result: {task_id}"
        )
        artifact.save(result_file, sign=True)
    
    def get_result(self, task_id: str) -> dict:
        """Get a result published by any agent."""
        result_file = f"{self.shared_path}/results/{task_id}.maif"
        
        if not os.path.exists(result_file):
            return None
        
        artifact = load_maif(result_file)
        content = artifact.get_content_list()
        
        if content:
            return json.loads(content[0].get("text", "{}"))
        return None

# Usage
agent1 = DistributedAgent("worker-1", "/mnt/shared/coordination")
agent2 = DistributedAgent("worker-2", "/mnt/shared/coordination")

# Register both agents
agent1.register()
agent2.register()

# Agent 1 publishes result
agent1.publish_result("task-001", {"output": "completed", "data": [1, 2, 3]})

# Agent 2 retrieves result
result = agent2.get_result("task-001")
print(f"Result from {result['agent_id']}: {result['result']}")
```

### Replication Pattern

```python
from maif_api import create_maif, load_maif
import os
import shutil

class ReplicatedStore:
    """Store artifacts with replication for reliability."""
    
    def __init__(self, replicas: list):
        self.replicas = replicas
        for path in replicas:
            os.makedirs(path, exist_ok=True)
    
    def save(self, name: str, artifact):
        """Save to all replicas."""
        for replica in self.replicas:
            path = f"{replica}/{name}.maif"
            artifact.save(path, sign=True)
    
    def load(self, name: str):
        """Load from first available replica."""
        for replica in self.replicas:
            path = f"{replica}/{name}.maif"
            if os.path.exists(path):
                try:
                    artifact = load_maif(path)
                    if artifact.verify_integrity():
                        return artifact
                except:
                    continue
        raise FileNotFoundError(f"Artifact {name} not found in any replica")
    
    def verify_replicas(self, name: str) -> dict:
        """Check replica consistency."""
        results = {}
        for replica in self.replicas:
            path = f"{replica}/{name}.maif"
            results[replica] = {
                "exists": os.path.exists(path),
                "valid": False
            }
            if results[replica]["exists"]:
                try:
                    artifact = load_maif(path)
                    results[replica]["valid"] = artifact.verify_integrity()
                except:
                    pass
        return results

# Usage
store = ReplicatedStore([
    "/data/primary",
    "/data/backup1",
    "/data/backup2"
])

artifact = create_maif("important-data")
artifact.add_text("Critical information")
store.save("important-data", artifact)

# Check replicas
status = store.verify_replicas("important-data")
print(f"Replica status: {status}")
```

## What You'll Learn

- File locking for concurrent access
- State synchronization between agents
- Caching strategies for performance
- Replication for reliability

## Key Concepts

### 1. File Locking

```python
import fcntl

# Exclusive lock for writes
fcntl.flock(lock_file, fcntl.LOCK_EX)

# Shared lock for reads
fcntl.flock(lock_file, fcntl.LOCK_SH)
```

### 2. Caching

```python
# Check if local cache is newer than remote
if os.path.getmtime(cache) >= os.path.getmtime(remote):
    use_cache()
else:
    refresh_cache()
```

### 3. Integrity Verification

```python
# Always verify after loading from network
artifact = load_maif(path)
if artifact.verify_integrity():
    # Safe to use
```

## Next Steps

- [Performance Guide](/guide/performance) - Optimization
- [Distributed Patterns](/guide/distributed) - More patterns
