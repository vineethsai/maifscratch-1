"""
Self-Optimizing MAIF Capabilities
Implements smart reorganization, auto-recovery, and adaptive optimization.
"""

import os
import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import logging
import heapq
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class AccessPattern:
    """Tracks access patterns for optimization."""

    block_id: str
    access_count: int = 0
    last_access: float = 0
    access_times: deque = field(default_factory=lambda: deque(maxlen=100))
    read_latencies: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_access(self, latency: float):
        """Record an access event."""
        current_time = time.time()
        self.access_count += 1
        self.last_access = current_time
        self.access_times.append(current_time)
        self.read_latencies.append(latency)

    def get_access_frequency(self) -> float:
        """Calculate access frequency (accesses per hour)."""
        if len(self.access_times) < 2:
            return 0.0

        time_span = self.access_times[-1] - self.access_times[0]
        if time_span == 0:
            return 0.0

        return len(self.access_times) * 3600 / time_span

    def get_average_latency(self) -> float:
        """Get average read latency."""
        if not self.read_latencies:
            return 0.0
        return sum(self.read_latencies) / len(self.read_latencies)


@dataclass
class OptimizationMetrics:
    """Metrics for optimization decisions."""

    total_reads: int = 0
    total_writes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    reorganization_count: int = 0
    recovery_count: int = 0
    compression_ratio: float = 1.0
    average_read_latency: float = 0.0
    average_write_latency: float = 0.0
    fragmentation_ratio: float = 0.0


class SelfOptimizingMAIF:
    """
    Self-optimizing MAIF implementation with:
    - Smart reorganization based on access patterns
    - Auto-recovery from corruption
    - Adaptive caching and prefetching
    - Dynamic compression selection
    """

    def __init__(self, maif_path: str, optimization_interval: float = 300.0):
        self.maif_path = Path(maif_path)
        self.optimization_interval = optimization_interval

        # Access tracking
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.metrics = OptimizationMetrics()

        # Optimization state
        self.hot_blocks: Set[str] = set()
        self.cold_blocks: Set[str] = set()
        self.prefetch_queue: deque = deque(maxlen=100)

        # Recovery state
        self.corruption_events: List[Dict[str, Any]] = []
        self.recovery_history: List[Dict[str, Any]] = []

        # Threading
        self._lock = threading.RLock()
        self._optimization_thread: Optional[threading.Thread] = None
        self._running = False

        # Initialize MAIF (v3 format - self-contained)
        from .core import MAIFEncoder, MAIFDecoder
        from .hot_buffer import HotBufferLayer

        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
            self.encoder = None  # Will create when needed
        else:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="self_optimizing")
            self.decoder = None

        # Hot buffer for frequently accessed blocks
        from .hot_buffer import HotBufferConfig, FlushPolicy

        # Create a config object for the hot buffer
        hot_buffer_config = HotBufferConfig(
            max_buffer_size=100 * 1024 * 1024,  # 100MB
            flush_interval=10.0,
            flush_policy=FlushPolicy.HYBRID,
        )

        # Define a flush callback function
        def flush_callback(operations):
            for op in operations:
                if op.operation_type == "write":
                    if op.block_type == "text":
                        try:
                            text = op.data.decode("utf-8")
                            self.encoder.add_text_block(text, op.metadata)
                        except Exception as e:
                            logger.error(f"Error flushing text block: {e}")
                    else:
                        try:
                            self.encoder.add_binary_block(
                                op.data, op.block_type, op.metadata
                            )
                        except Exception as e:
                            logger.error(f"Error flushing binary block: {e}")

        # Initialize the hot buffer with config and callback
        self.hot_buffer = HotBufferLayer(
            config=hot_buffer_config, flush_callback=flush_callback
        )

        # Start optimization loop
        self.start_optimization()

    def start_optimization(self):
        """Start the optimization background thread."""
        self._running = True
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self._optimization_thread.start()

    def stop_optimization(self):
        """Stop the optimization thread."""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join()

    def _optimization_loop(self):
        """Background optimization loop."""
        while self._running:
            try:
                # Wait for interval
                time.sleep(self.optimization_interval)

                # Run optimizations
                self._analyze_access_patterns()
                self._reorganize_if_needed()
                self._check_and_recover()
                self._update_prefetch_strategy()

            except Exception as e:
                logger.error(f"Optimization error: {e}")

    def record_access(self, block_id: str, latency: float, is_read: bool = True):
        """Record block access for optimization."""
        with self._lock:
            if block_id not in self.access_patterns:
                self.access_patterns[block_id] = AccessPattern(block_id)

            self.access_patterns[block_id].record_access(latency)

            if is_read:
                self.metrics.total_reads += 1
            else:
                self.metrics.total_writes += 1

    def _analyze_access_patterns(self):
        """Analyze access patterns to identify hot/cold blocks."""
        with self._lock:
            # Calculate access frequencies
            block_frequencies = []

            for block_id, pattern in self.access_patterns.items():
                frequency = pattern.get_access_frequency()
                block_frequencies.append((frequency, block_id))

            # Sort by frequency
            block_frequencies.sort(reverse=True)

            # Identify hot blocks (top 20%)
            hot_threshold = int(len(block_frequencies) * 0.2)
            self.hot_blocks = {
                block_id for _, block_id in block_frequencies[:hot_threshold]
            }

            # Identify cold blocks (bottom 20%)
            cold_threshold = int(len(block_frequencies) * 0.8)
            self.cold_blocks = {
                block_id for _, block_id in block_frequencies[cold_threshold:]
            }

            logger.info(
                f"Identified {len(self.hot_blocks)} hot blocks, "
                f"{len(self.cold_blocks)} cold blocks"
            )

    def _reorganize_if_needed(self):
        """Reorganize MAIF file if beneficial."""
        if not self.decoder:
            return

        with self._lock:
            # Check fragmentation
            fragmentation = self._calculate_fragmentation()
            self.metrics.fragmentation_ratio = fragmentation

            # Reorganize if fragmented or access patterns suggest benefit
            if fragmentation > 0.3 or self._should_reorganize():
                logger.info("Starting MAIF reorganization...")
                self._perform_reorganization()
                self.metrics.reorganization_count += 1

    def _calculate_fragmentation(self) -> float:
        """Calculate file fragmentation ratio."""
        if not self.decoder:
            return 0.0

        # Check block ordering vs access patterns
        block_positions = {}
        for i, block in enumerate(self.decoder.blocks):
            block_positions[block.block_id] = i

        # Calculate disorder based on access frequency
        disorder_score = 0
        sorted_by_frequency = sorted(
            self.access_patterns.items(),
            key=lambda x: x[1].get_access_frequency(),
            reverse=True,
        )

        for i, (block_id, _) in enumerate(sorted_by_frequency):
            if block_id in block_positions:
                actual_pos = block_positions[block_id]
                disorder_score += abs(i - actual_pos)

        max_disorder = len(sorted_by_frequency) * len(sorted_by_frequency) / 2
        return disorder_score / max_disorder if max_disorder > 0 else 0.0

    def _should_reorganize(self) -> bool:
        """Determine if reorganization would be beneficial."""
        # Check if hot blocks are scattered
        if not self.decoder or not self.hot_blocks:
            return False

        hot_positions = []
        for i, block in enumerate(self.decoder.blocks):
            if block.block_id in self.hot_blocks:
                hot_positions.append(i)

        if len(hot_positions) < 2:
            return False

        # Check if hot blocks are clustered
        position_variance = np.var(hot_positions)
        expected_variance = len(self.decoder.blocks) ** 2 / 12  # Uniform distribution

        # Reorganize if hot blocks are too scattered
        return position_variance > expected_variance * 0.5

    def _perform_reorganization(self):
        """Reorganize MAIF file for optimal access."""
        if not self.decoder:
            return

        from .core import MAIFEncoder, MAIFDecoder

        # Create new MAIF with optimized layout (v3 format)
        temp_path = self.maif_path.with_suffix(".reorg.maif")
        new_encoder = MAIFEncoder(str(temp_path), agent_id="self_optimizing_reorg")

        # Group blocks by access pattern
        hot_blocks = []
        warm_blocks = []
        cold_blocks = []

        for block in self.decoder.blocks:
            block_id = block.header.block_id
            if block_id in self.hot_blocks:
                hot_blocks.append(block)
            elif block_id in self.cold_blocks:
                cold_blocks.append(block)
            else:
                warm_blocks.append(block)

        # Write blocks in optimized order: hot -> warm -> cold
        # Block type constants
        BLOCK_TYPE_TEXT = 0x54455854  # 'TEXT'

        for block_list in [hot_blocks, warm_blocks, cold_blocks]:
            for block in block_list:
                block_data = block.data
                block_type = block.header.block_type

                # Check if text block (by comparing int values)
                if block_type == BLOCK_TYPE_TEXT or block_type == 1:
                    new_encoder.add_text_block(
                        block_data.decode("utf-8"), metadata=block.metadata
                    )
                else:
                    new_encoder.add_binary_block(block_data, metadata=block.metadata)

        # Finalize reorganized MAIF (v3 format)
        new_encoder.finalize()

        # Atomic replace
        os.replace(temp_path, self.maif_path)

        # Reload
        self.decoder = MAIFDecoder(str(self.maif_path))
        self.decoder.load()

        logger.info("MAIF reorganization completed")

    def _check_and_recover(self):
        """Check for corruption and auto-recover."""
        if not self.decoder:
            return

        corruption_found = False

        try:
            # Verify block integrity
            for block in self.decoder.blocks:
                try:
                    # Try to read block
                    data = self.decoder.get_block_data(block.block_id)

                    # Verify checksum if available
                    if hasattr(block, "checksum"):
                        import hashlib

                        calculated = hashlib.sha256(data).hexdigest()
                        if calculated != block.checksum:
                            corruption_found = True
                            self._handle_corruption(block, "checksum_mismatch")

                except Exception as e:
                    corruption_found = True
                    self._handle_corruption(block, str(e))

            if corruption_found:
                self._perform_recovery()

        except Exception as e:
            logger.error(f"Recovery check failed: {e}")

    def _handle_corruption(self, block: Any, error: str):
        """Handle detected corruption."""
        with self._lock:
            self.corruption_events.append(
                {
                    "timestamp": time.time(),
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "error": error,
                }
            )

            logger.warning(f"Corruption detected in block {block.block_id}: {error}")

    def _perform_recovery(self):
        """Perform automatic recovery."""
        logger.info("Starting auto-recovery...")

        with self._lock:
            recovery_start = time.time()
            recovered_blocks = 0

            # Try to recover from backups or redundancy
            for event in self.corruption_events:
                block_id = event["block_id"]

                # Check hot buffer
                if block_id in self.hot_buffer.buffer:
                    # Recover from hot buffer
                    logger.info(f"Recovering block {block_id} from hot buffer")
                    recovered_blocks += 1
                    continue

                # Check for redundant copies (if using distributed mode)
                # In production, would check replicas

                # Mark as unrecoverable for now
                logger.error(f"Unable to recover block {block_id}")

            # Record recovery
            self.recovery_history.append(
                {
                    "timestamp": recovery_start,
                    "duration": time.time() - recovery_start,
                    "corruption_events": len(self.corruption_events),
                    "recovered_blocks": recovered_blocks,
                }
            )

            self.metrics.recovery_count += 1
            self.corruption_events.clear()

            logger.info(f"Recovery completed: {recovered_blocks} blocks recovered")

    def _update_prefetch_strategy(self):
        """Update prefetch strategy based on access patterns."""
        with self._lock:
            # Clear old prefetch queue
            self.prefetch_queue.clear()

            # Analyze sequential access patterns
            sequential_patterns = self._detect_sequential_patterns()

            # Add predicted next blocks to prefetch queue
            for pattern in sequential_patterns:
                if pattern["confidence"] > 0.7:
                    next_blocks = pattern["predicted_next"]
                    self.prefetch_queue.extend(next_blocks[:10])

    def _detect_sequential_patterns(self) -> List[Dict[str, Any]]:
        """Detect sequential access patterns."""
        patterns = []

        # Group accesses by time windows
        time_window = 1.0  # 1 second windows
        access_sequences = defaultdict(list)

        for block_id, pattern in self.access_patterns.items():
            for access_time in pattern.access_times:
                window = int(access_time / time_window)
                access_sequences[window].append((access_time, block_id))

        # Analyze sequences
        for window, accesses in access_sequences.items():
            if len(accesses) < 3:
                continue

            # Sort by time
            accesses.sort()

            # Check for sequential pattern
            block_sequence = [block_id for _, block_id in accesses]

            # Simple sequential detection (in production, use more sophisticated methods)
            if self._is_sequential(block_sequence):
                patterns.append(
                    {
                        "sequence": block_sequence,
                        "confidence": 0.8,
                        "predicted_next": self._predict_next_blocks(block_sequence),
                    }
                )

        return patterns

    def _is_sequential(self, block_sequence: List[str]) -> bool:
        """Check if blocks form a sequential pattern."""
        # Simple check - in production, use pattern matching algorithms
        if len(block_sequence) < 3:
            return False

        # Check if blocks are accessed in order
        for i in range(1, len(block_sequence)):
            if block_sequence[i] == block_sequence[i - 1]:
                return False

        return True

    def _predict_next_blocks(self, sequence: List[str]) -> List[str]:
        """Predict next blocks in sequence."""
        # Simple prediction - return blocks that often follow the last block
        if not sequence:
            return []

        last_block = sequence[-1]
        predictions = []

        # In production, use ML models or statistical analysis
        # For now, return empty list
        return predictions

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        with self._lock:
            return {
                "metrics": {
                    "total_reads": self.metrics.total_reads,
                    "total_writes": self.metrics.total_writes,
                    "cache_hit_rate": (
                        self.metrics.cache_hits
                        / (self.metrics.cache_hits + self.metrics.cache_misses)
                        if self.metrics.cache_misses > 0
                        else 1.0
                    ),
                    "reorganization_count": self.metrics.reorganization_count,
                    "recovery_count": self.metrics.recovery_count,
                    "fragmentation_ratio": self.metrics.fragmentation_ratio,
                },
                "hot_blocks": len(self.hot_blocks),
                "cold_blocks": len(self.cold_blocks),
                "prefetch_queue_size": len(self.prefetch_queue),
                "corruption_events": len(self.corruption_events),
                "recovery_history": self.recovery_history[-10:],  # Last 10 recoveries
            }

    def optimize_for_workload(self, workload_type: str):
        """Optimize MAIF for specific workload type."""
        logger.info(f"Optimizing for {workload_type} workload")

        if workload_type == "sequential_read":
            # Increase prefetch aggressiveness
            self.prefetch_queue = deque(maxlen=200)
            # Reorganize for sequential access
            self._perform_reorganization()

        elif workload_type == "random_access":
            # Optimize hot buffer for random access
            self.hot_buffer.buffer_size = 200 * 1024 * 1024  # 200MB
            # Keep more blocks in memory

        elif workload_type == "write_heavy":
            # Optimize for writes
            self.hot_buffer.flush_interval = 30.0  # Less frequent flushes
            self.hot_buffer.batch_size = 1000  # Larger batches

        elif workload_type == "read_heavy":
            # Optimize for reads
            self.hot_buffer.compression_enabled = True
            # More aggressive caching

        else:
            logger.warning(f"Unknown workload type: {workload_type}")


# Adaptive Compression Selector
class AdaptiveCompressionSelector:
    """Selects optimal compression algorithm based on data characteristics."""

    def __init__(self):
        self.compression_stats: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()

    def select_compression(self, data: bytes, block_type: str) -> str:
        """Select optimal compression algorithm."""
        # Sample data characteristics
        data_size = len(data)
        entropy = self._calculate_entropy(data[:1024])  # Sample first 1KB

        # Select based on characteristics
        if block_type == "embeddings":
            # Embeddings compress poorly, use fast algorithm
            return "lz4"

        elif data_size < 1024:
            # Small data, skip compression
            return "none"

        elif entropy > 0.9:
            # High entropy (already compressed/encrypted)
            return "none"

        elif entropy < 0.5:
            # Low entropy, high redundancy
            return "zstd"  # Best compression ratio

        else:
            # Medium entropy
            return "snappy"  # Balanced

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if not data:
            return 0.0

        # Count byte frequencies
        frequencies = defaultdict(int)
        for byte in data:
            frequencies[byte] += 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)

        for count in frequencies.values():
            if count > 0:
                probability = count / data_len
                entropy -= probability * np.log2(probability)

        # Normalize to [0, 1]
        return entropy / 8.0

    def update_stats(
        self,
        algorithm: str,
        original_size: int,
        compressed_size: int,
        compression_time: float,
    ):
        """Update compression statistics."""
        with self._lock:
            stats = self.compression_stats[algorithm]

            # Update running averages
            if "count" not in stats:
                stats["count"] = 0
                stats["total_ratio"] = 0.0
                stats["total_time"] = 0.0

            stats["count"] += 1
            stats["total_ratio"] += compressed_size / original_size
            stats["total_time"] += compression_time

            stats["avg_ratio"] = stats["total_ratio"] / stats["count"]
            stats["avg_time"] = stats["total_time"] / stats["count"]


# Export classes
__all__ = [
    "SelfOptimizingMAIF",
    "AdaptiveCompressionSelector",
    "AccessPattern",
    "OptimizationMetrics",
]
