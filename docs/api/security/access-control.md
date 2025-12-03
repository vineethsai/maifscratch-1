# Access Control

MAIF provides two access control mechanisms:
- **AccessController** - Block-level permissions (block_hash → agent_id → permissions)
- **AccessControlManager** - User-resource-permission model with logging

## AccessController

Manages granular access control for individual MAIF blocks.

### Quick Start

```python
from maif.security import AccessController

ac = AccessController()

# Set permissions for a block
ac.set_block_permissions("block-hash-123", "agent-1", ["read", "write"])

# Check permission
if ac.check_permission("block-hash-123", "agent-1", "read"):
    print("Access granted")
```

### Class Definition

```python
class AccessController:
    def __init__(self):
        """Initialize access controller with empty permissions."""
```

### Methods

#### set_block_permissions

Set permissions for a specific block and agent.

```python
def set_block_permissions(
    self,
    block_hash: str,
    agent_id: str,
    permissions: List[str]
):
    """
    Set permissions for a block.

    Args:
        block_hash: Hash of the block
        agent_id: Agent identifier
        permissions: List of permissions (e.g., ["read", "write", "delete"])

    Raises:
        ValueError: If block_hash or agent_id is empty
    """
```

#### check_permission

Check if an agent has a specific permission on a block.

```python
def check_permission(
    self,
    block_hash: str,
    agent_id: str,
    action: str
) -> bool:
    """
    Check if agent has permission.

    Args:
        block_hash: Hash of the block
        agent_id: Agent identifier
        action: Action to check (e.g., "read", "write")

    Returns:
        True if permission is granted, False otherwise
    """
```

::: tip Admin Permission
The "admin" permission grants access to all actions on a block.
:::

#### get_permissions_manifest

Get permissions as a manifest for inclusion in MAIF files.

```python
def get_permissions_manifest(self) -> Dict:
    """
    Get permissions manifest.

    Returns:
        Dictionary with access_control and version keys
    """
```

### Example

```python
from maif.security import AccessController

ac = AccessController()

# Grant permissions to different agents
ac.set_block_permissions("block-001", "agent-reader", ["read"])
ac.set_block_permissions("block-001", "agent-writer", ["read", "write"])
ac.set_block_permissions("block-001", "agent-admin", ["admin"])

# Check permissions
print(ac.check_permission("block-001", "agent-reader", "read"))   # True
print(ac.check_permission("block-001", "agent-reader", "write"))  # False
print(ac.check_permission("block-001", "agent-writer", "write"))  # True
print(ac.check_permission("block-001", "agent-admin", "delete"))  # True (admin)

# Get manifest
manifest = ac.get_permissions_manifest()
print(manifest)
# {
#   "access_control": {
#     "block-001": {
#       "agent-reader": ["read"],
#       "agent-writer": ["read", "write"],
#       "agent-admin": ["admin"]
#     }
#   },
#   "version": "1.0"
# }
```

## AccessControlManager

A more comprehensive access control system with audit logging.

### Quick Start

```python
from maif.security import SecurityManager

# Access via SecurityManager
security = SecurityManager(use_kms=False)
acm = security.access_control

# Grant permission
acm.grant_permission("user-1", "document-abc", "read")

# Check access (logged)
if acm.check_access("user-1", "document-abc", "read"):
    print("Access granted")
```

### Class Definition

```python
class AccessControlManager:
    def __init__(self):
        """Initialize with empty permissions and access logs."""
```

### Methods

#### grant_permission

Grant a permission to a user for a resource.

```python
def grant_permission(
    self,
    user_id: str,
    resource: str,
    permission: str
):
    """
    Grant permission.

    Args:
        user_id: User identifier
        resource: Resource identifier (block ID, file path, etc.)
        permission: Permission to grant (e.g., "read", "write", "delete")

    Raises:
        ValueError: If any argument is empty
    """
```

#### revoke_permission

Revoke a permission from a user.

```python
def revoke_permission(
    self,
    user_id: str,
    resource: str,
    permission: str
):
    """
    Revoke permission.

    Args:
        user_id: User identifier
        resource: Resource identifier
        permission: Permission to revoke
    """
```

#### check_access

Check if a user has access (with logging).

```python
def check_access(
    self,
    user_id: str,
    resource: str,
    permission: str
) -> bool:
    """
    Check user access.

    Args:
        user_id: User identifier
        resource: Resource identifier
        permission: Permission to check

    Returns:
        True if access is granted

    Raises:
        ValueError: If any argument is empty

    Note:
        All access checks are logged in access_logs
    """
```

### Access Logs

Every `check_access` call is logged with:
- `timestamp`: When the check occurred
- `user_id`: Who requested access
- `resource`: What resource was requested
- `permission`: What permission was checked
- `result`: True/False
- `reason`: Why access was granted/denied

```python
acm = AccessControlManager()
acm.grant_permission("user-1", "doc-1", "read")
acm.check_access("user-1", "doc-1", "read")
acm.check_access("user-2", "doc-1", "read")

for log in acm.access_logs:
    print(f"{log['user_id']} -> {log['resource']}: {log['result']} ({log['reason']})")
# user-1 -> doc-1: True (permission_granted)
# user-2 -> doc-1: False (user_not_found)
```

### Example

```python
from maif.security import SecurityManager

security = SecurityManager(use_kms=False)
acm = security.access_control

# Set up permissions for a multi-user scenario
users = ["alice", "bob", "charlie"]
resources = ["project-x", "project-y"]

# Alice is admin on project-x
acm.grant_permission("alice", "project-x", "admin")

# Bob has read/write on project-x
acm.grant_permission("bob", "project-x", "read")
acm.grant_permission("bob", "project-x", "write")

# Charlie has read-only on both projects
acm.grant_permission("charlie", "project-x", "read")
acm.grant_permission("charlie", "project-y", "read")

# Check permissions
def try_access(user, resource, permission):
    result = acm.check_access(user, resource, permission)
    print(f"{user} {permission} {resource}: {'OK' if result else 'DENIED'}")

try_access("alice", "project-x", "delete")  # OK (admin)
try_access("bob", "project-x", "write")     # OK
try_access("bob", "project-x", "delete")    # DENIED
try_access("charlie", "project-x", "read")  # OK
try_access("charlie", "project-x", "write") # DENIED

# Revoke permission
acm.revoke_permission("bob", "project-x", "write")
try_access("bob", "project-x", "write")     # DENIED (revoked)
```

## Standard Permissions

Common permission strings used in MAIF:

| Permission | Description |
|------------|-------------|
| `read` | View content |
| `write` | Modify content |
| `delete` | Remove content |
| `admin` | Full access (grants all permissions) |
| `share` | Share with other users |
| `execute` | Execute/run content |
| `export` | Export content outside system |

## Integration with SecurityManager

The `SecurityManager` provides an integrated `AccessControlManager` instance:

```python
from maif.security import SecurityManager

security = SecurityManager(use_kms=False)

# Access the access control manager
security.access_control.grant_permission("user", "resource", "read")

# Combined with encryption
if security.access_control.check_access("user", "resource", "read"):
    encrypted = security.encrypt_data(b"Sensitive data")
```

## Thread Safety

Both `AccessController` and `AccessControlManager` are thread-safe and use internal locks for concurrent access:

```python
import threading
from maif.security import AccessController

ac = AccessController()

def grant_and_check(agent_id, block_hash):
    ac.set_block_permissions(block_hash, agent_id, ["read"])
    return ac.check_permission(block_hash, agent_id, "read")

# Safe for concurrent use
threads = [
    threading.Thread(target=grant_and_check, args=(f"agent-{i}", f"block-{i}"))
    for i in range(10)
]
for t in threads:
    t.start()
for t in threads:
    t.join()
```
