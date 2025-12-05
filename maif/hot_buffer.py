"""
MAIF 3.0 Hot Buffer Layer
Implements high-frequency write optimization with in-memory buffering and periodic flush.
"""

import time
import threading
import queue
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import struct
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FlushPolicy(Enum):
    """Flush policies for hot buffer."""

    TIME_BASED = "time_based"  # Flush every N seconds
    SIZE_BASED = "size_based"  # Flush when buffer reaches size
    OPERATION_COUNT = "operation_count"  # Flush after N operations
    HYBRID = "hybrid"  # Combination of above


@dataclass
class BufferedOperation:
    """Represents a buffered write operation."""

    operation_id: str
    operation_type: str  # "write", "update", "delete"
    block_type: str
    data: bytes
    metadata: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def size_bytes(self) -> int:
        """Calculate operation size in bytes."""
        return len(self.data) + len(json.dumps(self.metadata).encode())


@dataclass
class HotBufferConfig:
    """Configuration for hot buffer layer."""

    max_buffer_size: int = 10 * 1024 * 1024  # 10MB default
    flush_interval: float = 1.0  # 1 second default
    max_operations: int = 1000  # Max operations before flush
    flush_policy: FlushPolicy = FlushPolicy.HYBRID
    enable_compression: bool = True
    enable_wal: bool = True
    wal_path: Optional[str] = None
    worker_threads: int = 2


class WriteAheadLog:
    """Write-Ahead Log for crash recovery."""

    def __init__(self, wal_path: str):
        self.wal_path = Path(wal_path)
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_file = None
        self.sequence_number = 0
        self._lock = threading.Lock()

    def append(self, operation: BufferedOperation) -> int:
        """Append operation to WAL and return sequence number."""
        with self._lock:
            self.sequence_number += 1

            # Serialize operation
            entry = {
                "sequence": self.sequence_number,
                "operation_id": operation.operation_id,
                "operation_type": operation.operation_type,
                "block_type": operation.block_type,
                "metadata": operation.metadata,
                "timestamp": operation.timestamp,
                "data_size": len(operation.data),
                "data_hash": hashlib.sha256(operation.data).hexdigest(),
            }

            # Write to WAL file
            wal_file = self.wal_path / f"wal_{int(time.time())}.log"
            with open(wal_file, "ab") as f:
                # Write entry header
                entry_json = json.dumps(entry).encode()
                f.write(struct.pack(">I", len(entry_json)))
                f.write(entry_json)

                # Write data
                f.write(struct.pack(">I", len(operation.data)))
                f.write(operation.data)

                f.flush()

            return self.sequence_number

    def recover(self) -> List[BufferedOperation]:
        """Recover operations from WAL files."""
        operations = []

        # Find all WAL files
        wal_files = sorted(self.wal_path.glob("wal_*.log"))

        for wal_file in wal_files:
            try:
                with open(wal_file, "rb") as f:
                    while True:
                        # Read entry header size
                        size_data = f.read(4)
                        if not size_data:
                            break

                        entry_size = struct.unpack(">I", size_data)[0]
                        entry_json = f.read(entry_size).decode()
                        entry = json.loads(entry_json)

                        # Read data
                        data_size = struct.unpack(">I", f.read(4))[0]
                        data = f.read(data_size)

                        # Verify data integrity
                        if hashlib.sha256(data).hexdigest() == entry["data_hash"]:
                            operations.append(
                                BufferedOperation(
                                    operation_id=entry["operation_id"],
                                    operation_type=entry["operation_type"],
                                    block_type=entry["block_type"],
                                    data=data,
                                    metadata=entry["metadata"],
                                    timestamp=entry["timestamp"],
                                )
                            )
            except Exception as e:
                logger.warning(f"Error recovering from WAL {wal_file}: {e}")

        return operations

    def cleanup(self, before_timestamp: float):
        """Clean up old WAL files."""
        for wal_file in self.wal_path.glob("wal_*.log"):
            try:
                file_time = int(wal_file.stem.split("_")[1])
                if file_time < before_timestamp:
                    wal_file.unlink()
            except Exception:
                pass


class HotBufferLayer:
    """
    High-performance hot buffer layer for MAIF 3.0.
    Supports 1000+ operations per second with configurable flush policies.
    """

    def __init__(self, config: HotBufferConfig, flush_callback: Callable):
        self.config = config
        self.flush_callback = flush_callback

        # Buffer storage
        self.buffer: List[BufferedOperation] = []
        self.buffer_size = 0
        self.operation_count = 0
        self._buffer_lock = threading.Lock()

        # Write-Ahead Log
        self.wal = None
        if config.enable_wal:
            wal_path = config.wal_path or "./maif_wal"
            self.wal = WriteAheadLog(wal_path)

            # Recover from WAL
            recovered_ops = self.wal.recover()
            if recovered_ops:
                logger.info(f"Recovered {len(recovered_ops)} operations from WAL")
                self.buffer.extend(recovered_ops)
                self.buffer_size = sum(op.size_bytes() for op in recovered_ops)
                self.operation_count = len(recovered_ops)

        # Performance metrics
        self.total_operations = 0
        self.total_flushes = 0
        self.last_flush_time = time.time()

        # Background flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()

        # Worker pool for compression
        self._compression_queue = queue.Queue()
        self._compression_workers = []
        if config.enable_compression:
            for _ in range(config.worker_threads):
                worker = threading.Thread(target=self._compression_worker, daemon=True)
                worker.start()
                self._compression_workers.append(worker)

    def write(
        self, block_type: str, data: bytes, metadata: Optional[Dict] = None
    ) -> str:
        """
        Write operation to hot buffer.
        Returns operation ID for tracking.
        """
        import uuid

        operation_id = str(uuid.uuid4())
        metadata = metadata or {}

        operation = BufferedOperation(
            operation_id=operation_id,
            operation_type="write",
            block_type=block_type,
            data=data,
            metadata=metadata,
        )

        # Write to WAL first if enabled
        if self.wal:
            self.wal.append(operation)

        # Add to buffer
        with self._buffer_lock:
            self.buffer.append(operation)
            self.buffer_size += operation.size_bytes()
            self.operation_count += 1
            self.total_operations += 1

            # Check if immediate flush needed
            should_flush = self._should_flush()

        # Trigger flush if needed (outside lock)
        if should_flush:
            self._trigger_flush()

        return operation_id

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed based on policy."""
        if self.config.flush_policy == FlushPolicy.TIME_BASED:
            return False  # Handled by background thread

        elif self.config.flush_policy == FlushPolicy.SIZE_BASED:
            return self.buffer_size >= self.config.max_buffer_size

        elif self.config.flush_policy == FlushPolicy.OPERATION_COUNT:
            return self.operation_count >= self.config.max_operations

        elif self.config.flush_policy == FlushPolicy.HYBRID:
            return (
                self.buffer_size >= self.config.max_buffer_size
                or self.operation_count >= self.config.max_operations
            )

        return False

    def _flush_worker(self):
        """Background thread for time-based flushing."""
        while self._running:
            time.sleep(self.config.flush_interval)

            if self.config.flush_policy in [FlushPolicy.TIME_BASED, FlushPolicy.HYBRID]:
                with self._buffer_lock:
                    if self.buffer:
                        self._trigger_flush()

    def _trigger_flush(self):
        """Trigger buffer flush to MAIF."""
        # Get operations to flush
        with self._buffer_lock:
            if not self.buffer:
                return

            operations_to_flush = self.buffer[:]
            self.buffer.clear()
            self.buffer_size = 0
            self.operation_count = 0

        # Compress if enabled
        if self.config.enable_compression:
            operations_to_flush = self._compress_operations(operations_to_flush)

        # Execute flush callback
        try:
            self.flush_callback(operations_to_flush)
            self.total_flushes += 1
            self.last_flush_time = time.time()

            # Clean up WAL after successful flush
            if self.wal:
                self.wal.cleanup(time.time() - 3600)  # Keep 1 hour of WAL

        except Exception as e:
            logger.error(f"Flush failed: {e}")
            # Re-add operations to buffer on failure
            with self._buffer_lock:
                self.buffer.extend(operations_to_flush)
                self.buffer_size = sum(op.size_bytes() for op in operations_to_flush)
                self.operation_count = len(operations_to_flush)

    def _compress_operations(
        self, operations: List[BufferedOperation]
    ) -> List[BufferedOperation]:
        """Compress operations using streaming compression."""
        import zlib

        compressed_ops = []

        for op in operations:
            try:
                # Compress data
                compressed_data = zlib.compress(op.data, level=6)

                # Update metadata
                compressed_metadata = op.metadata.copy()
                compressed_metadata["compression"] = "zlib"
                compressed_metadata["original_size"] = len(op.data)
                compressed_metadata["compressed_size"] = len(compressed_data)

                compressed_op = BufferedOperation(
                    operation_id=op.operation_id,
                    operation_type=op.operation_type,
                    block_type=op.block_type,
                    data=compressed_data,
                    metadata=compressed_metadata,
                    timestamp=op.timestamp,
                )

                compressed_ops.append(compressed_op)

            except Exception as e:
                logger.warning(
                    f"Compression failed for operation {op.operation_id}: {e}"
                )
                compressed_ops.append(op)  # Use uncompressed

        return compressed_ops

    def _compression_worker(self):
        """Worker thread for parallel compression."""
        while self._running:
            try:
                operation = self._compression_queue.get(timeout=1)
                if operation is None:
                    break
                # Compress operation
                # (Implementation depends on specific needs)
            except queue.Empty:
                continue

    def flush(self):
        """Force immediate flush of buffer."""
        self._trigger_flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get hot buffer statistics."""
        with self._buffer_lock:
            return {
                "total_operations": self.total_operations,
                "total_flushes": self.total_flushes,
                "current_buffer_size": self.buffer_size,
                "current_operation_count": self.operation_count,
                "last_flush_time": self.last_flush_time,
                "operations_per_second": self.total_operations
                / (time.time() - self.last_flush_time)
                if time.time() > self.last_flush_time
                else 0,
            }

    def shutdown(self):
        """Shutdown hot buffer layer."""
        self._running = False

        # Final flush
        self.flush()

        # Stop threads
        if self._flush_thread.is_alive():
            self._flush_thread.join()

        for worker in self._compression_workers:
            self._compression_queue.put(None)

        for worker in self._compression_workers:
            if worker.is_alive():
                worker.join()


# Integration with MAIF
class MAIFHotBufferIntegration:
    """Integrates hot buffer with MAIF encoder."""

    def __init__(self, maif_encoder, config: Optional[HotBufferConfig] = None):
        self.maif_encoder = maif_encoder
        self.config = config or HotBufferConfig()

        # Create hot buffer
        self.hot_buffer = HotBufferLayer(self.config, self._flush_to_maif)

    def _flush_to_maif(self, operations: List[BufferedOperation]):
        """Flush operations to MAIF encoder."""
        for op in operations:
            if op.operation_type == "write":
                if op.block_type == "text":
                    text = op.data.decode("utf-8")
                    self.maif_encoder.add_text_block(text, op.metadata)
                else:
                    self.maif_encoder.add_binary_block(
                        op.data, op.block_type, op.metadata
                    )

    def write_text(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Write text through hot buffer."""
        return self.hot_buffer.write("text", text.encode("utf-8"), metadata)

    def write_binary(
        self, data: bytes, block_type: str, metadata: Optional[Dict] = None
    ) -> str:
        """Write binary data through hot buffer."""
        return self.hot_buffer.write(block_type, data, metadata)

    def flush(self):
        """Force flush to MAIF."""
        self.hot_buffer.flush()

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.hot_buffer.get_stats()

    def shutdown(self):
        """Shutdown hot buffer."""
        self.hot_buffer.shutdown()
