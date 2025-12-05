"""
Distributed MAIF Architecture
Implements sharding, CRDT support, and distributed coordination for MAIF.
"""

import os
import json
import time
import hashlib
import threading
import socket
import struct
import pickle
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


# CRDT (Conflict-free Replicated Data Type) implementations
class VectorClock:
    """Vector clock for distributed causality tracking."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.clock: Dict[str, int] = defaultdict(int)
        self.clock[node_id] = 0

    def increment(self):
        """Increment this node's clock."""
        self.clock[self.node_id] += 1

    def update(self, other_clock: Dict[str, int]):
        """Update with another vector clock."""
        for node, timestamp in other_clock.items():
            self.clock[node] = max(self.clock[node], timestamp)
        self.increment()

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if this clock happens-before another."""
        for node, timestamp in self.clock.items():
            if timestamp > other.clock.get(node, 0):
                return False
        return any(
            timestamp < other.clock.get(node, 0)
            for node, timestamp in self.clock.items()
        )

    def concurrent_with(self, other: "VectorClock") -> bool:
        """Check if clocks are concurrent."""
        return not self.happens_before(other) and not other.happens_before(self)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return dict(self.clock)


@dataclass
class CRDTOperation:
    """Base class for CRDT operations."""

    op_id: str
    node_id: str
    vector_clock: Dict[str, int]
    timestamp: float

    def __post_init__(self):
        if not self.op_id:
            self.op_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()


class GCounter:
    """Grow-only counter CRDT."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.counts: Dict[str, int] = defaultdict(int)

    def increment(self, value: int = 1):
        """Increment counter."""
        self.counts[self.node_id] += value

    def merge(self, other: "GCounter"):
        """Merge with another counter."""
        for node, count in other.counts.items():
            self.counts[node] = max(self.counts[node], count)

    def value(self) -> int:
        """Get total value."""
        return sum(self.counts.values())


class LWWRegister:
    """Last-Write-Wins Register CRDT."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.value: Any = None
        self.timestamp: float = 0
        self.writer: str = ""

    def set(self, value: Any):
        """Set value with current timestamp."""
        self.value = value
        self.timestamp = time.time()
        self.writer = self.node_id

    def merge(self, other: "LWWRegister"):
        """Merge with another register."""
        if other.timestamp > self.timestamp or (
            other.timestamp == self.timestamp and other.writer > self.writer
        ):
            self.value = other.value
            self.timestamp = other.timestamp
            self.writer = other.writer


class ORSet:
    """Observed-Remove Set CRDT."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.elements: Dict[Any, Set[str]] = defaultdict(set)  # element -> unique tags
        self.tombstones: Set[Tuple[Any, str]] = set()  # (element, tag) pairs

    def add(self, element: Any):
        """Add element to set."""
        tag = f"{self.node_id}:{uuid.uuid4()}"
        self.elements[element].add(tag)

    def remove(self, element: Any):
        """Remove element from set."""
        if element in self.elements:
            for tag in self.elements[element]:
                self.tombstones.add((element, tag))

    def merge(self, other: "ORSet"):
        """Merge with another set."""
        # Merge elements
        for element, tags in other.elements.items():
            self.elements[element].update(tags)

        # Merge tombstones
        self.tombstones.update(other.tombstones)

        # Remove tombstoned elements
        for element, tag in self.tombstones:
            if element in self.elements and tag in self.elements[element]:
                self.elements[element].remove(tag)
                if not self.elements[element]:
                    del self.elements[element]

    def values(self) -> Set[Any]:
        """Get current set values."""
        return set(self.elements.keys())


# Distributed Lock Manager
class DistributedLock:
    """Distributed lock implementation using Lamport timestamps."""

    def __init__(
        self, lock_id: str, node_id: str, coordinator: "DistributedCoordinator"
    ):
        self.lock_id = lock_id
        self.node_id = node_id
        self.coordinator = coordinator
        self.is_locked = False
        self.lock_holder: Optional[str] = None
        self.timestamp: float = 0
        self.queue: List[Tuple[float, str]] = []  # (timestamp, node_id)
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire the distributed lock."""
        with self._lock:
            request_time = time.time()

            # Add to queue
            self.queue.append((request_time, self.node_id))
            self.queue.sort()

            # Broadcast lock request
            self.coordinator.broadcast_lock_request(self.lock_id, request_time)

            # Wait for lock
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if we're at front of queue
                if self.queue and self.queue[0][1] == self.node_id:
                    self.is_locked = True
                    self.lock_holder = self.node_id
                    self.timestamp = request_time
                    return True

                time.sleep(0.1)

            # Timeout - remove from queue
            self.queue = [(t, n) for t, n in self.queue if n != self.node_id]
            return False

    def release(self):
        """Release the distributed lock."""
        with self._lock:
            if self.is_locked and self.lock_holder == self.node_id:
                self.is_locked = False
                self.lock_holder = None

                # Remove from queue
                self.queue = [(t, n) for t, n in self.queue if n != self.node_id]

                # Broadcast release
                self.coordinator.broadcast_lock_release(self.lock_id)

    def handle_request(self, node_id: str, timestamp: float):
        """Handle lock request from another node."""
        with self._lock:
            self.queue.append((timestamp, node_id))
            self.queue.sort()

    def handle_release(self, node_id: str):
        """Handle lock release from another node."""
        with self._lock:
            self.queue = [(t, n) for t, n in self.queue if n != node_id]


# Sharding Manager
class ShardManager:
    """Manages MAIF sharding across nodes."""

    def __init__(self, node_id: str, num_shards: int = 16):
        self.node_id = node_id
        self.num_shards = num_shards
        self.shard_map: Dict[int, str] = {}  # shard_id -> node_id
        self.local_shards: Set[int] = set()
        self.replication_factor = 3
        self._lock = threading.Lock()

    def get_shard_id(self, key: str) -> int:
        """Get shard ID for a key using consistent hashing."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % self.num_shards

    def get_responsible_nodes(self, shard_id: int) -> List[str]:
        """Get nodes responsible for a shard (including replicas)."""
        nodes = []

        # Primary node
        primary = self.shard_map.get(shard_id)
        if primary:
            nodes.append(primary)

        # Replica nodes (next nodes in ring)
        all_nodes = sorted(set(self.shard_map.values()))
        if primary in all_nodes:
            primary_idx = all_nodes.index(primary)
            for i in range(1, self.replication_factor):
                replica_idx = (primary_idx + i) % len(all_nodes)
                nodes.append(all_nodes[replica_idx])

        return nodes

    def rebalance_shards(self, active_nodes: List[str]):
        """Rebalance shards across active nodes."""
        with self._lock:
            if not active_nodes:
                return

            # Clear current assignments
            self.shard_map.clear()
            self.local_shards.clear()

            # Assign shards round-robin
            for shard_id in range(self.num_shards):
                node_idx = shard_id % len(active_nodes)
                assigned_node = active_nodes[node_idx]
                self.shard_map[shard_id] = assigned_node

                if assigned_node == self.node_id:
                    self.local_shards.add(shard_id)

            # Check replicas
            for shard_id in range(self.num_shards):
                responsible_nodes = self.get_responsible_nodes(shard_id)
                if self.node_id in responsible_nodes:
                    self.local_shards.add(shard_id)

    def is_responsible_for(self, key: str) -> bool:
        """Check if this node is responsible for a key."""
        shard_id = self.get_shard_id(key)
        return shard_id in self.local_shards


# Distributed Coordinator
class DistributedCoordinator:
    """Coordinates distributed MAIF operations."""

    def __init__(
        self,
        node_id: str,
        maif_path: str,
        peers: Optional[List[Tuple[str, int]]] = None,
    ):
        self.node_id = node_id
        self.maif_path = Path(maif_path)
        self.peers = peers or []  # [(host, port), ...]

        # Components
        self.shard_manager = ShardManager(node_id)

        # Node tracking
        self.node_last_seen: Dict[str, float] = {}
        self.node_timeout = 30.0  # 30 seconds before considering node dead
        self.node_cleanup_interval = 60.0  # Clean up dead nodes every minute
        self.last_cleanup_time = time.time()
        self.vector_clock = VectorClock(node_id)
        self.locks: Dict[str, DistributedLock] = {}

        # CRDT state
        self.block_counter = GCounter(node_id)
        self.metadata_registers: Dict[str, LWWRegister] = {}
        self.active_nodes = ORSet(node_id)

        # Networking
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self._lock = threading.Lock()

        # Initialize MAIF (v3 format - self-contained)
        from .core import MAIFEncoder, MAIFDecoder

        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
            self.encoder = None
        else:
            self.encoder = MAIFEncoder(
                str(self.maif_path), agent_id="distributed_coordinator"
            )
            self.decoder = None

        # Start coordinator
        self.start()

    def start(self):
        """Start the distributed coordinator."""
        self.running = True
        self.active_nodes.add(self.node_id)

        # Start network listener
        if self.peers:
            threading.Thread(target=self._network_listener, daemon=True).start()

        # Start heartbeat
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()

        # Initial shard rebalance
        self.shard_manager.rebalance_shards(list(self.active_nodes.values()))

    def stop(self):
        """Stop the distributed coordinator."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _network_listener(self):
        """Listen for network messages."""
        # Find available port
        for port in range(9000, 9100):
            try:
                self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server_socket.bind(("", port))
                self.server_socket.listen(5)
                logger.info(f"Distributed coordinator listening on port {port}")
                break
            except (socket.error, OSError):
                continue

        while self.running:
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(
                    target=self._handle_connection, args=(conn,), daemon=True
                ).start()
            except Exception as e:
                logger.debug(f"Heartbeat thread error: {e}")
                break

    def _handle_connection(self, conn: socket.socket):
        """Handle incoming connection."""
        try:
            # Receive message size
            size_data = conn.recv(4)
            if not size_data:
                return

            size = struct.unpack("!I", size_data)[0]

            # Receive message
            data = b""
            while len(data) < size:
                chunk = conn.recv(min(size - len(data), 4096))
                if not chunk:
                    break
                data += chunk

            # Process message
            message = pickle.loads(data)
            self._process_message(message)

        except Exception as e:
            logger.error(f"Error handling connection: {e}")
        finally:
            conn.close()

    def _process_message(self, message: Dict[str, Any]):
        """Process incoming message."""
        msg_type = message.get("type")

        if msg_type == "heartbeat":
            self._handle_heartbeat(message)
        elif msg_type == "lock_request":
            self._handle_lock_request(message)
        elif msg_type == "lock_release":
            self._handle_lock_release(message)
        elif msg_type == "crdt_sync":
            self._handle_crdt_sync(message)
        elif msg_type == "block_replicate":
            self._handle_block_replicate(message)

    def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self.running:
            # Send heartbeat to all peers
            message = {
                "type": "heartbeat",
                "node_id": self.node_id,
                "timestamp": time.time(),
                "active_nodes": list(self.active_nodes.values()),
                "vector_clock": self.vector_clock.to_dict(),
            }

            self._broadcast_message(message)

            # Clean up inactive nodes
            self._cleanup_inactive_nodes()

            time.sleep(5)

    def _cleanup_inactive_nodes(self):
        """Remove nodes that haven't sent heartbeats recently."""
        current_time = time.time()

        # Only run cleanup periodically
        if current_time - self.last_cleanup_time < self.node_cleanup_interval:
            return

        self.last_cleanup_time = current_time

        # Find inactive nodes
        inactive_nodes = []
        for node_id, last_seen in self.node_last_seen.items():
            if current_time - last_seen > self.node_timeout:
                inactive_nodes.append(node_id)

        # Remove inactive nodes
        for node_id in inactive_nodes:
            logger.warning(f"Node {node_id} is inactive, removing from cluster")

            # Remove from tracking
            del self.node_last_seen[node_id]

            # Remove from active nodes
            if node_id in self.active_nodes:
                self.active_nodes.remove(node_id)

            # Remove from CRDT state
            if node_id in self.crdt_state.nodes:
                del self.crdt_state.nodes[node_id]

            # Trigger shard rebalancing
            self._trigger_shard_rebalance()

        if inactive_nodes:
            logger.info(f"Cleaned up {len(inactive_nodes)} inactive nodes")

    def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat message."""
        node_id = message["node_id"]

        # Update active nodes
        self.active_nodes.add(node_id)

        # Update vector clock
        self.vector_clock.update(message["vector_clock"])

        # Check for topology changes
        current_nodes = set(self.active_nodes.values())
        reported_nodes = set(message["active_nodes"])

        if current_nodes != reported_nodes:
            # Merge node sets
            for node in reported_nodes:
                self.active_nodes.add(node)

            # Rebalance shards
            self.shard_manager.rebalance_shards(list(self.active_nodes.values()))

    def _broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all peers."""
        data = pickle.dumps(message)
        size = struct.pack("!I", len(data))

        for host, port in self.peers:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, port))
                sock.sendall(size + data)
                sock.close()
            except (socket.error, OSError):
                pass  # Socket already closed

    def broadcast_lock_request(self, lock_id: str, timestamp: float):
        """Broadcast lock request."""
        message = {
            "type": "lock_request",
            "node_id": self.node_id,
            "lock_id": lock_id,
            "timestamp": timestamp,
            "vector_clock": self.vector_clock.to_dict(),
        }
        self._broadcast_message(message)

    def broadcast_lock_release(self, lock_id: str):
        """Broadcast lock release."""
        message = {
            "type": "lock_release",
            "node_id": self.node_id,
            "lock_id": lock_id,
            "vector_clock": self.vector_clock.to_dict(),
        }
        self._broadcast_message(message)

    def _handle_lock_request(self, message: Dict[str, Any]):
        """Handle lock request."""
        lock_id = message["lock_id"]
        node_id = message["node_id"]
        timestamp = message["timestamp"]

        # Get or create lock
        if lock_id not in self.locks:
            self.locks[lock_id] = DistributedLock(lock_id, self.node_id, self)

        self.locks[lock_id].handle_request(node_id, timestamp)
        self.vector_clock.update(message["vector_clock"])

    def _handle_lock_release(self, message: Dict[str, Any]):
        """Handle lock release."""
        lock_id = message["lock_id"]
        node_id = message["node_id"]

        if lock_id in self.locks:
            self.locks[lock_id].handle_release(node_id)

        self.vector_clock.update(message["vector_clock"])

    def acquire_lock(self, resource_id: str, timeout: float = 30.0) -> bool:
        """Acquire a distributed lock."""
        lock_id = f"maif:{resource_id}"

        if lock_id not in self.locks:
            self.locks[lock_id] = DistributedLock(lock_id, self.node_id, self)

        return self.locks[lock_id].acquire(timeout)

    def release_lock(self, resource_id: str):
        """Release a distributed lock."""
        lock_id = f"maif:{resource_id}"

        if lock_id in self.locks:
            self.locks[lock_id].release()

    def add_block_distributed(
        self, block_data: bytes, block_type: str, metadata: Optional[Dict] = None
    ) -> str:
        """Add a block with distributed coordination."""
        # Generate block ID
        block_id = hashlib.sha256(block_data).hexdigest()

        # Check if responsible for this block
        if not self.shard_manager.is_responsible_for(block_id):
            # Route to responsible node
            responsible_nodes = self.shard_manager.get_responsible_nodes(
                self.shard_manager.get_shard_id(block_id)
            )

            if responsible_nodes:
                # Send to primary node
                self._replicate_block(
                    responsible_nodes[0], block_data, block_type, metadata
                )
                return block_id

        # Add locally
        self.encoder.add_binary_block(block_data, block_type, metadata)

        # Update CRDT counters
        self.block_counter.increment()
        self.vector_clock.increment()

        # Replicate to other responsible nodes
        responsible_nodes = self.shard_manager.get_responsible_nodes(
            self.shard_manager.get_shard_id(block_id)
        )

        for node in responsible_nodes[1:]:  # Skip primary (us)
            self._replicate_block(node, block_data, block_type, metadata)

        return block_id

    def _replicate_block(
        self, node_id: str, block_data: bytes, block_type: str, metadata: Optional[Dict]
    ):
        """Replicate block to another node."""
        message = {
            "type": "block_replicate",
            "node_id": self.node_id,
            "block_data": block_data,
            "block_type": block_type,
            "metadata": metadata,
            "vector_clock": self.vector_clock.to_dict(),
        }

        # In production, would send to specific node
        # For now, broadcast and let node check if responsible
        self._broadcast_message(message)

    def _handle_block_replicate(self, message: Dict[str, Any]):
        """Handle block replication."""
        block_data = message["block_data"]
        block_id = hashlib.sha256(block_data).hexdigest()

        # Check if we're responsible
        if self.shard_manager.is_responsible_for(block_id):
            # Add block
            self.encoder.add_binary_block(
                block_data, message["block_type"], message["metadata"]
            )

            # Update vector clock
            self.vector_clock.update(message["vector_clock"])

    def sync_crdt_state(self):
        """Synchronize CRDT state with peers."""
        message = {
            "type": "crdt_sync",
            "node_id": self.node_id,
            "block_counter": self.block_counter.counts,
            "active_nodes": {
                "elements": dict(self.active_nodes.elements),
                "tombstones": list(self.active_nodes.tombstones),
            },
            "vector_clock": self.vector_clock.to_dict(),
        }

        self._broadcast_message(message)

    def _handle_crdt_sync(self, message: Dict[str, Any]):
        """Handle CRDT synchronization."""
        # Merge block counter
        other_counter = GCounter(message["node_id"])
        other_counter.counts = message["block_counter"]
        self.block_counter.merge(other_counter)

        # Merge active nodes
        other_nodes = ORSet(message["node_id"])
        other_nodes.elements = defaultdict(set, message["active_nodes"]["elements"])
        other_nodes.tombstones = set(message["active_nodes"]["tombstones"])
        self.active_nodes.merge(other_nodes)

        # Update vector clock
        self.vector_clock.update(message["vector_clock"])

    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        return {
            "node_id": self.node_id,
            "active_nodes": list(self.active_nodes.values()),
            "total_blocks": self.block_counter.value(),
            "local_shards": list(self.shard_manager.local_shards),
            "vector_clock": self.vector_clock.to_dict(),
            "locks_held": [
                lid
                for lid, lock in self.locks.items()
                if lock.lock_holder == self.node_id
            ],
        }


# Export classes
__all__ = [
    "DistributedCoordinator",
    "ShardManager",
    "DistributedLock",
    "VectorClock",
    "GCounter",
    "LWWRegister",
    "ORSet",
]
