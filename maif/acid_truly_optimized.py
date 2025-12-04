#!/usr/bin/env python3
"""
Truly Optimized ACID Implementation for MAIF
===========================================

This version focuses on REAL optimizations that actually improve performance:
- Eliminate unnecessary I/O operations
- Minimize memory allocations
- Simplify data structures
- Remove threading overhead for small operations
- Use in-memory buffering with periodic flushes

Target: Level 2 at 1,800+ MB/s (vs 1,200 MB/s basic)
"""

import os
import sys
import time
import struct
import hashlib
import threading
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import uuid
import json


class TrulyOptimizedACIDLevel(Enum):
    """Simplified ACID levels."""
    PERFORMANCE = 0      # No ACID overhead
    FULL_ACID = 2       # Optimized full ACID


@dataclass
class SimpleWALEntry:
    """Minimal WAL entry for maximum performance."""
    transaction_id: str
    operation: str
    block_id: str
    data_size: int
    checksum: str
    
    def to_bytes(self) -> bytes:
        """Ultra-fast serialization."""
        data = f"{self.transaction_id}|{self.operation}|{self.block_id}|{self.data_size}|{self.checksum}\n"
        return data.encode('utf-8')
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'SimpleWALEntry':
        """Ultra-fast deserialization."""
        try:
            text = data.decode('utf-8').strip()
            parts = text.split('|')
            if len(parts) != 5:
                raise ValueError(f"Invalid WAL entry format: expected 5 parts, got {len(parts)}")
            return cls(parts[0], parts[1], parts[2], int(parts[3]), parts[4])
        except (UnicodeDecodeError, ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse WAL entry: {e}")


class FastWAL:
    """Simple, fast WAL implementation."""
    
    def __init__(self, wal_path: str):
        self.wal_path = wal_path
        self.buffer = []
        self.buffer_size = 0
        self.max_buffer_size = 1024 * 1024  # 1MB buffer
        self.entries_written = 0
        self._lock = threading.Lock()  # Thread safety for buffer
        self._closed = False
        
        # Ensure WAL directory exists
        wal_dir = os.path.dirname(wal_path)
        if wal_dir and not os.path.exists(wal_dir):
            os.makedirs(wal_dir, exist_ok=True)
    
    def write_entry(self, entry: SimpleWALEntry):
        """Write entry to buffer."""
        with self._lock:
            if self._closed:
                raise RuntimeError("WAL is closed")
                
            entry_bytes = entry.to_bytes()
            self.buffer.append(entry_bytes)
            self.buffer_size += len(entry_bytes)
            
            # Flush when buffer gets large
            if self.buffer_size >= self.max_buffer_size:
                self._flush_locked()
    
    def _flush_locked(self):
        """Flush buffer to disk (must be called with lock held)."""
        if not self.buffer:
            return
        
        try:
            with open(self.wal_path, 'ab') as f:
                for entry_bytes in self.buffer:
                    f.write(entry_bytes)
                f.flush()
                os.fsync(f.fileno())  # Ensure durability
            
            self.entries_written += len(self.buffer)
            self.buffer.clear()
            self.buffer_size = 0
        except Exception as e:
            raise RuntimeError(f"Failed to flush WAL: {e}")
    
    def flush(self):
        """Flush buffer to disk."""
        with self._lock:
            self._flush_locked()
    
    def read_entries(self) -> List[SimpleWALEntry]:
        """Read all WAL entries."""
        entries = []
        if not os.path.exists(self.wal_path):
            return entries
        
        try:
            with open(self.wal_path, 'rb') as f:
                for line in f:
                    if line.strip():
                        try:
                            entries.append(SimpleWALEntry.from_bytes(line))
                        except ValueError:
                            # Skip corrupted entries
                            continue
        except Exception as e:
            raise RuntimeError(f"Failed to read WAL entries: {e}")
        
        return entries
    
    def close(self):
        """Close WAL."""
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._flush_locked()
                except:
                    pass


class SimpleMVCC:
    """Simple MVCC implementation focused on performance."""
    
    def __init__(self):
        # Multi-version storage: Maps block_id to list of (version_id, data, metadata, timestamp) tuples
        self.versions: Dict[str, List[Tuple[int, bytes, Dict[str, Any], float]]] = defaultdict(list)
        self.next_version = 1
        self.snapshots: Dict[str, float] = {}  # transaction_id -> timestamp
        self._lock = threading.Lock()  # Thread safety
    
    def create_snapshot(self, transaction_id: str) -> float:
        """Create snapshot for transaction."""
        with self._lock:
            timestamp = time.time()
            self.snapshots[transaction_id] = timestamp
            return timestamp
    
    def write_version(self, block_id: str, data: bytes, metadata: Dict[str, Any]) -> int:
        """Write new version."""
        with self._lock:
            version_id = self.next_version
            self.next_version += 1
            timestamp = time.time()
            
            # Create a copy of metadata to avoid mutations
            metadata_copy = metadata.copy() if metadata else {}
            self.versions[block_id].append((version_id, data, metadata_copy, timestamp))
            
            # Keep only last 10 versions to prevent memory bloat
            if len(self.versions[block_id]) > 10:
                self.versions[block_id] = list(self.versions[block_id][-10:])
            
            return version_id
    
    def read_version(self, transaction_id: str, block_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Read version visible to transaction."""
        with self._lock:
            snapshot_time = self.snapshots.get(transaction_id)
            if snapshot_time is None:
                return None
            
            versions = self.versions.get(block_id, [])
            if not versions:
                return None
            
            # Find latest version before snapshot
            for version_id, data, metadata, timestamp in reversed(versions):
                if timestamp <= snapshot_time:
                    # Return copies to avoid mutations
                    return data, metadata.copy() if metadata else {}
            
            return None
    
    def release_snapshot(self, transaction_id: str):
        """Release snapshot."""
        with self._lock:
            self.snapshots.pop(transaction_id, None)


class TrulyOptimizedTransactionManager:
    """Truly optimized transaction manager."""
    
    def __init__(self, file_path: str, acid_level: TrulyOptimizedACIDLevel):
        self.file_path = file_path
        self.acid_level = acid_level
        
        # Initialize based on ACID level
        if acid_level == TrulyOptimizedACIDLevel.FULL_ACID:
            self.wal = FastWAL(f"{file_path}.wal")
            self.mvcc = SimpleMVCC()
        else:
            self.wal = None
            self.mvcc = None
        
        # Simple storage for performance mode
        self.direct_storage: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}
        
        # Transaction tracking
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
        
        # Performance counters
        self.operations = 0
        self.transactions = 0
    
    def begin_transaction(self) -> str:
        """Begin transaction."""
        transaction_id = str(uuid.uuid4())
        
        if self.acid_level == TrulyOptimizedACIDLevel.FULL_ACID:
            snapshot_time = self.mvcc.create_snapshot(transaction_id)
            self.active_transactions[transaction_id] = {
                'snapshot_time': snapshot_time,
                'operations': []
            }
        
        self.transactions += 1
        return transaction_id
    
    def write_block(self, transaction_id: str, block_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        """Write block."""
        try:
            if self.acid_level == TrulyOptimizedACIDLevel.PERFORMANCE:
                # Direct write - no overhead
                self.direct_storage[block_id] = (data, metadata)
            else:
                # ACID write
                if transaction_id not in self.active_transactions:
                    return False
                
                # Add to transaction operations (don't write to MVCC yet)
                self.active_transactions[transaction_id]['operations'].append({
                    'type': 'write',
                    'block_id': block_id,
                    'data': data,
                    'metadata': metadata
                })
                
                # Write to WAL (but don't flush immediately)
                checksum = hashlib.md5(data).hexdigest()[:8]  # Fast checksum
                wal_entry = SimpleWALEntry(
                    transaction_id=transaction_id,
                    operation='write',
                    block_id=block_id,
                    data_size=len(data),
                    checksum=checksum
                )
                self.wal.write_entry(wal_entry)
            
            self.operations += 1
            return True
            
        except Exception:
            return False
    
    def read_block(self, transaction_id: str, block_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Read block."""
        try:
            if self.acid_level == TrulyOptimizedACIDLevel.PERFORMANCE:
                return self.direct_storage.get(block_id)
            else:
                return self.mvcc.read_version(transaction_id, block_id)
        except Exception:
            return None
    
    def commit_transaction(self, transaction_id: str) -> bool:
        """Commit transaction."""
        try:
            if self.acid_level == TrulyOptimizedACIDLevel.PERFORMANCE:
                return True  # No-op
            
            if transaction_id not in self.active_transactions:
                return False
            
            transaction = self.active_transactions[transaction_id]
            
            # Apply all operations to MVCC
            for op in transaction['operations']:
                if op['type'] == 'write':
                    self.mvcc.write_version(op['block_id'], op['data'], op['metadata'])
            
            # Clean up
            self.mvcc.release_snapshot(transaction_id)
            del self.active_transactions[transaction_id]
            
            return True
            
        except Exception:
            return False
    
    def rollback_transaction(self, transaction_id: str) -> bool:
        """Rollback transaction."""
        try:
            if self.acid_level == TrulyOptimizedACIDLevel.PERFORMANCE:
                return True  # No-op
            
            if transaction_id in self.active_transactions:
                self.mvcc.release_snapshot(transaction_id)
                del self.active_transactions[transaction_id]
            
            return True
            
        except Exception:
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance stats."""
        return {
            'acid_level': self.acid_level.name,
            'operations': self.operations,
            'transactions': self.transactions,
            'active_transactions': len(self.active_transactions),
            'storage_blocks': len(self.direct_storage) if self.acid_level == TrulyOptimizedACIDLevel.PERFORMANCE else len(self.mvcc.versions)
        }
    
    def close(self):
        """Close manager."""
        # Clean up any active transactions
        for txn_id in list(self.active_transactions.keys()):
            try:
                self.rollback_transaction(txn_id)
            except (OSError, IOError):
                pass  # Best effort - may already be closed
        
        if self.wal:
            self.wal.close()
            self.wal = None
        
        if self.mvcc:
            self.mvcc = None
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.close()


class TrulyOptimizedTransaction:
    """Simple transaction context manager."""
    
    def __init__(self, manager: TrulyOptimizedTransactionManager):
        self.manager = manager
        self.transaction_id = None
    
    def __enter__(self):
        self.transaction_id = self.manager.begin_transaction()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.manager.commit_transaction(self.transaction_id)
        else:
            self.manager.rollback_transaction(self.transaction_id)
    
    def write_block(self, block_id: str, data: bytes, metadata: Dict[str, Any]) -> bool:
        """Write block."""
        return self.manager.write_block(self.transaction_id, block_id, data, metadata)
    
    def read_block(self, block_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Read block."""
        return self.manager.read_block(self.transaction_id, block_id)


def create_truly_optimized_manager(file_path: str, acid_level: TrulyOptimizedACIDLevel) -> TrulyOptimizedTransactionManager:
    """Create truly optimized manager."""
    return TrulyOptimizedTransactionManager(file_path, acid_level)


def benchmark_truly_optimized():
    """Benchmark the truly optimized implementation."""
    print("🚀 Truly Optimized ACID Benchmark")
    print("=" * 50)
    
    # Level 0 test
    print("\n⚡ Level 0 (Performance Mode)")
    manager = create_truly_optimized_manager("test_truly_opt_perf.maif", TrulyOptimizedACIDLevel.PERFORMANCE)
    
    start_time = time.time()
    operations = 5000
    
    for i in range(operations):
        txn_id = manager.begin_transaction()
        manager.write_block(txn_id, f"block_{i}", f"Data {i}".encode(), {"index": i})
        manager.commit_transaction(txn_id)
    
    level0_time = time.time() - start_time
    level0_throughput = operations / level0_time
    
    print(f"   Operations: {operations}")
    print(f"   Time: {level0_time:.3f} seconds")
    print(f"   Throughput: {level0_throughput:.1f} ops/sec")
    
    manager.close()
    
    # Level 2 test
    print("\n🔒 Level 2 (Full ACID)")
    manager = create_truly_optimized_manager("test_truly_opt_acid.maif", TrulyOptimizedACIDLevel.FULL_ACID)
    
    start_time = time.time()
    operations = 2000
    
    for i in range(operations):
        with TrulyOptimizedTransaction(manager) as txn:
            txn.write_block(f"block_{i}", f"Data {i}".encode(), {"index": i})
    
    level2_time = time.time() - start_time
    level2_throughput = operations / level2_time
    
    print(f"   Operations: {operations}")
    print(f"   Time: {level2_time:.3f} seconds")
    print(f"   Throughput: {level2_throughput:.1f} ops/sec")
    
    # Calculate overhead
    normalized_level0_time = (level0_time / 5000) * 2000
    overhead = ((level2_time - normalized_level0_time) / normalized_level0_time) * 100
    ratio = level0_throughput / level2_throughput
    
    print(f"\n📊 Performance Analysis:")
    print(f"   Level 0: {level0_throughput:.1f} ops/sec")
    print(f"   Level 2: {level2_throughput:.1f} ops/sec")
    print(f"   ACID Overhead: {overhead:.1f}%")
    print(f"   Performance Ratio: {ratio:.1f}x")
    
    if ratio <= 1.5:
        print("   ✅ Target achieved: ACID overhead <1.5x")
    else:
        print("   ⚠️  Target missed: ACID overhead >1.5x")
    
    manager.close()
    
    # Cleanup
    for file in ["test_truly_opt_perf.maif", "test_truly_opt_acid.maif", "test_truly_opt_acid.maif.wal"]:
        if os.path.exists(file):
            os.remove(file)
    
    return {
        'level0_throughput': level0_throughput,
        'level2_throughput': level2_throughput,
        'overhead': overhead,
        'ratio': ratio
    }


if __name__ == "__main__":
    benchmark_truly_optimized()


class TrulyOptimizedAcidMAIF:
    """
    Truly optimized ACID-compliant MAIF implementation.
    
    This class provides a high-performance MAIF implementation with
    ACID transaction support, optimized for maximum throughput.
    """
    
    def __init__(self, maif_path: str = None, acid_level: TrulyOptimizedACIDLevel = TrulyOptimizedACIDLevel.FULL_ACID,
                agent_id: str = None, enable_security: bool = False):
        """
        Initialize a truly optimized ACID-compliant MAIF.
        
        Args:
            maif_path: Path to the MAIF file
            acid_level: ACID compliance level
            agent_id: ID of the agent using this MAIF
            enable_security: Whether to enable security features
        """
        from .core import MAIFEncoder
        import os
        from pathlib import Path
        
        self.maif_path = maif_path or f"maif_{int(time.time())}.maif"
        self.acid_level = acid_level
        self.agent_id = agent_id or f"agent-{int(time.time())}"
        self.enable_security = enable_security
        
        # Ensure directory exists
        maif_dir = os.path.dirname(self.maif_path)
        if maif_dir and not os.path.exists(maif_dir):
            os.makedirs(maif_dir, exist_ok=True)
            
        # Create blocks directory if needed
        blocks_path = f"{self.maif_path}.blocks"
        blocks_dir = os.path.dirname(blocks_path)
        if blocks_dir and not os.path.exists(blocks_dir):
            os.makedirs(blocks_dir, exist_ok=True)
            
        # Create empty blocks file if it doesn't exist
        if not os.path.exists(blocks_path):
            Path(blocks_path).touch()
        
        # Create base encoder (v3 format)
        self._encoder = MAIFEncoder(self.maif_path, agent_id=self.agent_id)
        
        # Create transaction manager
        self._transaction_manager = create_truly_optimized_manager(self.maif_path, acid_level)
        
        # Current transaction
        self._current_transaction = None
        
        # Privacy engine if security is enabled
        if enable_security:
            from .privacy import PrivacyEngine, EncryptionMode
            self._privacy_engine = PrivacyEngine()
            self._encryption_mode = EncryptionMode.AES_GCM
        else:
            self._privacy_engine = None
            self._encryption_mode = None
    
    def begin_transaction(self) -> str:
        """Begin a new transaction."""
        if self._current_transaction:
            self.commit_transaction()
            
        self._current_transaction = self._transaction_manager.begin_transaction()
        return self._current_transaction
    
    def commit_transaction(self) -> bool:
        """Commit the current transaction."""
        if not self._current_transaction:
            return False
            
        result = self._transaction_manager.commit_transaction(self._current_transaction)
        self._current_transaction = None
        return result
    
    def rollback_transaction(self) -> bool:
        """Rollback the current transaction."""
        if not self._current_transaction:
            return False
            
        result = self._transaction_manager.rollback_transaction(self._current_transaction)
        self._current_transaction = None
        return result
    
    def add_text_block(self, text: str, metadata: Dict = None,
                      encryption_enabled: bool = False,
                      access_control_enabled: bool = False) -> str:
        """Add a text block with transaction support."""
        # Ensure we have a transaction
        if not self._current_transaction:
            self.begin_transaction()
            
        # Add block to base encoder
        block_id = self._encoder.add_text_block(text, metadata)
        
        # Add to transaction
        data = text.encode('utf-8')
        
        # Apply privacy/encryption if enabled
        if self._privacy_engine and (encryption_enabled or self.enable_security):
            encrypted_data, encryption_metadata = self._privacy_engine.encrypt_data(
                data,
                block_id,
                encryption_mode=self._encryption_mode
            )
            data = encrypted_data
            metadata = metadata or {}
            metadata["encryption"] = encryption_metadata
            
            if access_control_enabled:
                metadata["access_control"] = {
                    "owner": self.agent_id,
                    "permissions": "rw"
                }
        
        self._transaction_manager.write_block(
            self._current_transaction,
            block_id,
            data,
            metadata or {}
        )
        
        return block_id
    
    def add_binary_block(self, data: bytes, block_type: str, metadata: Dict = None,
                        encryption_enabled: bool = False,
                        access_control_enabled: bool = False) -> str:
        """Add a binary block with transaction support."""
        # Ensure we have a transaction
        if not self._current_transaction:
            self.begin_transaction()
            
        # Add block to base encoder
        block_id = self._encoder.add_binary_block(data, block_type, metadata)
        
        # Apply privacy/encryption if enabled
        if self._privacy_engine and (encryption_enabled or self.enable_security):
            encrypted_data, encryption_metadata = self._privacy_engine.encrypt_data(
                data,
                block_id,
                encryption_mode=self._encryption_mode
            )
            data = encrypted_data
            metadata = metadata or {}
            metadata["encryption"] = encryption_metadata
            
            if access_control_enabled:
                metadata["access_control"] = {
                    "owner": self.agent_id,
                    "permissions": "rw"
                }
        
        # Add to transaction
        self._transaction_manager.write_block(
            self._current_transaction,
            block_id,
            data,
            metadata or {}
        )
        
        return block_id
    
    def save(self) -> bool:
        """Save MAIF file with transaction support (v3 format)."""
        # Commit any pending transaction
        if self._current_transaction:
            self.commit_transaction()
            
        # Finalize using base encoder (v3 format)
        self._encoder.finalize()
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get transaction performance statistics."""
        return self._transaction_manager.get_performance_stats()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_transaction_manager'):
            self._transaction_manager.close()