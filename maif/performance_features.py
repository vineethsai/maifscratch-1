"""
Performance Features for MAIF
Implements shared dictionaries and hardware-optimized I/O
"""

import os
import mmap
import struct
import hashlib
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import time
import pickle
import zlib

try:
    import zstandard as zstd

    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# Try to import hardware acceleration libraries
try:
    import pyarrow as pa

    ARROW_AVAILABLE = True
except ImportError:
    ARROW_AVAILABLE = False

try:
    import numba
    from numba import cuda

    CUDA_AVAILABLE = cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False
    numba = None


@dataclass
class DictionaryStats:
    """Statistics for shared dictionary performance."""

    dictionary_id: str
    size: int
    entry_count: int
    hit_rate: float
    compression_improvement: float
    last_updated: float


class SharedDictionaryManager:
    """
    Manages shared dictionaries for improved compression across similar data.
    Implements dictionary training and caching for Zstandard compression.
    """

    def __init__(
        self, max_dictionaries: int = 10, max_dictionary_size: int = 100 * 1024
    ):
        self.max_dictionaries = max_dictionaries
        self.max_dictionary_size = max_dictionary_size
        self.dictionaries: Dict[str, bytes] = {}
        self.dictionary_stats: Dict[str, DictionaryStats] = {}
        self.training_data: Dict[str, List[bytes]] = defaultdict(list)
        self._lock = threading.RLock()

        # Zstandard context managers
        self.compression_contexts: Dict[str, Any] = {}
        self.decompression_contexts: Dict[str, Any] = {}

    def train_dictionary(
        self, data_type: str, samples: List[bytes], force_update: bool = False
    ) -> Optional[str]:
        """Train a shared dictionary from sample data."""
        if not ZSTD_AVAILABLE:
            return None

        with self._lock:
            # Add samples to training data
            self.training_data[data_type].extend(samples)

            # Check if we have enough samples
            if len(self.training_data[data_type]) < 100 and not force_update:
                return None

            # Limit training data size
            training_samples = self.training_data[data_type][-1000:]

            try:
                # Train dictionary using Zstandard
                dict_data = zstd.train_dictionary(
                    self.max_dictionary_size, training_samples
                )

                # Generate dictionary ID
                dict_id = f"{data_type}_{hashlib.md5(dict_data).hexdigest()[:8]}"

                # Store dictionary
                self.dictionaries[dict_id] = dict_data

                # Create compression/decompression contexts
                self.compression_contexts[dict_id] = zstd.ZstdCompressor(
                    dict_data=zstd.ZstdCompressionDict(dict_data)
                )
                self.decompression_contexts[dict_id] = zstd.ZstdDecompressor(
                    dict_data=zstd.ZstdCompressionDict(dict_data)
                )

                # Initialize stats
                self.dictionary_stats[dict_id] = DictionaryStats(
                    dictionary_id=dict_id,
                    size=len(dict_data),
                    entry_count=len(training_samples),
                    hit_rate=0.0,
                    compression_improvement=0.0,
                    last_updated=time.time(),
                )

                # Evict old dictionaries if needed
                self._evict_if_needed()

                return dict_id

            except Exception as e:
                print(f"Dictionary training failed: {e}")
                return None

    def compress_with_dictionary(self, data: bytes, dict_id: str) -> Optional[bytes]:
        """Compress data using a shared dictionary."""
        if not ZSTD_AVAILABLE or dict_id not in self.compression_contexts:
            return None

        try:
            with self._lock:
                compressor = self.compression_contexts[dict_id]
                compressed = compressor.compress(data)

                # Update stats
                stats = self.dictionary_stats[dict_id]
                stats.hit_rate = (stats.hit_rate * 0.95) + 0.05  # Moving average

                # Calculate compression improvement
                standard_compressed = zlib.compress(data)
                improvement = (
                    len(standard_compressed) / len(compressed) if compressed else 1.0
                )
                stats.compression_improvement = (
                    stats.compression_improvement * 0.9 + improvement * 0.1
                )

                return compressed

        except Exception:
            return None

    def decompress_with_dictionary(
        self, compressed_data: bytes, dict_id: str
    ) -> Optional[bytes]:
        """Decompress data using a shared dictionary."""
        if not ZSTD_AVAILABLE or dict_id not in self.decompression_contexts:
            return None

        try:
            with self._lock:
                decompressor = self.decompression_contexts[dict_id]
                return decompressor.decompress(compressed_data)
        except Exception:
            return None

    def get_best_dictionary(self, data_type: str) -> Optional[str]:
        """Get the best performing dictionary for a data type."""
        with self._lock:
            candidates = [
                (dict_id, stats)
                for dict_id, stats in self.dictionary_stats.items()
                if dict_id.startswith(data_type)
            ]

            if not candidates:
                return None

            # Sort by compression improvement
            candidates.sort(key=lambda x: x[1].compression_improvement, reverse=True)
            return candidates[0][0]

    def _evict_if_needed(self):
        """Evict least recently used dictionaries if limit exceeded."""
        if len(self.dictionaries) <= self.max_dictionaries:
            return

        # Sort by last updated time
        sorted_dicts = sorted(
            self.dictionary_stats.items(), key=lambda x: x[1].last_updated
        )

        # Evict oldest
        for dict_id, _ in sorted_dicts[
            : len(self.dictionaries) - self.max_dictionaries
        ]:
            del self.dictionaries[dict_id]
            del self.dictionary_stats[dict_id]
            del self.compression_contexts[dict_id]
            del self.decompression_contexts[dict_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get dictionary manager statistics."""
        with self._lock:
            return {
                "dictionary_count": len(self.dictionaries),
                "total_size": sum(len(d) for d in self.dictionaries.values()),
                "dictionaries": {
                    dict_id: {
                        "size": stats.size,
                        "hit_rate": stats.hit_rate,
                        "compression_improvement": stats.compression_improvement,
                        "age": time.time() - stats.last_updated,
                    }
                    for dict_id, stats in self.dictionary_stats.items()
                },
            }


class HardwareOptimizedIO:
    """
    Hardware-optimized I/O operations for MAIF.
    Leverages memory mapping, direct I/O, and GPU acceleration where available.
    """

    def __init__(
        self,
        enable_mmap: bool = True,
        enable_direct_io: bool = True,
        enable_gpu: bool = True,
    ):
        self.enable_mmap = enable_mmap
        self.enable_direct_io = enable_direct_io
        self.enable_gpu = enable_gpu and CUDA_AVAILABLE

        # Memory-mapped file cache
        self.mmap_cache: Dict[str, mmap.mmap] = {}
        self._cache_lock = threading.Lock()

        # Direct I/O alignment
        self.alignment = 4096  # 4KB alignment for most systems

        # GPU memory pool if available
        if self.enable_gpu and numba:
            self.gpu_memory_pool = cuda.MemoryPool()
            cuda.set_memory_manager(self.gpu_memory_pool)

    def read_optimized(
        self, filepath: str, offset: int = 0, size: Optional[int] = None
    ) -> bytes:
        """Read data using hardware-optimized methods."""
        # Try memory-mapped I/O first
        if self.enable_mmap:
            data = self._read_mmap(filepath, offset, size)
            if data is not None:
                return data

        # Fall back to direct I/O
        if self.enable_direct_io:
            data = self._read_direct(filepath, offset, size)
            if data is not None:
                return data

        # Standard I/O as fallback
        with open(filepath, "rb") as f:
            f.seek(offset)
            return f.read(size) if size else f.read()

    def write_optimized(
        self, filepath: str, data: bytes, offset: int = 0, sync: bool = True
    ) -> int:
        """Write data using hardware-optimized methods."""
        bytes_written = 0

        # Use direct I/O for large writes
        if self.enable_direct_io and len(data) >= self.alignment:
            bytes_written = self._write_direct(filepath, data, offset, sync)
            if bytes_written > 0:
                return bytes_written

        # Standard I/O as fallback
        mode = "r+b" if os.path.exists(filepath) else "wb"
        with open(filepath, mode) as f:
            f.seek(offset)
            bytes_written = f.write(data)
            if sync:
                f.flush()
                os.fsync(f.fileno())

        return bytes_written

    def _read_mmap(
        self, filepath: str, offset: int, size: Optional[int]
    ) -> Optional[bytes]:
        """Read using memory-mapped file."""
        try:
            with self._cache_lock:
                # Check cache
                if filepath in self.mmap_cache:
                    mm = self.mmap_cache[filepath]
                    mm.seek(offset)
                    return mm.read(size) if size else mm.read()

                # Create new memory map
                with open(filepath, "rb") as f:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    self.mmap_cache[filepath] = mm
                    mm.seek(offset)
                    return mm.read(size) if size else mm.read()

        except Exception:
            return None

    def _read_direct(
        self, filepath: str, offset: int, size: Optional[int]
    ) -> Optional[bytes]:
        """Read using direct I/O (O_DIRECT on Linux)."""
        try:
            # Align offset and size
            aligned_offset = (offset // self.alignment) * self.alignment
            offset_delta = offset - aligned_offset

            if size:
                aligned_size = (
                    (size + offset_delta + self.alignment - 1) // self.alignment
                ) * self.alignment
            else:
                aligned_size = os.path.getsize(filepath) - aligned_offset

            # Open with O_DIRECT if available (Linux)
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT

            fd = os.open(filepath, flags)
            try:
                os.lseek(fd, aligned_offset, os.SEEK_SET)

                # Allocate aligned buffer
                buffer = bytearray(aligned_size)
                bytes_read = os.read(fd, aligned_size)

                # Extract requested portion
                if size:
                    return bytes(buffer[offset_delta : offset_delta + size])
                else:
                    return bytes(buffer[offset_delta:bytes_read])

            finally:
                os.close(fd)

        except Exception:
            return None

    def _write_direct(self, filepath: str, data: bytes, offset: int, sync: bool) -> int:
        """Write using direct I/O."""
        try:
            # Align data
            aligned_offset = (offset // self.alignment) * self.alignment
            offset_delta = offset - aligned_offset

            # Pad data to alignment
            padded_size = (
                (len(data) + offset_delta + self.alignment - 1) // self.alignment
            ) * self.alignment
            padded_data = bytearray(padded_size)

            # Read existing data if needed
            if offset_delta > 0 or padded_size > len(data) + offset_delta:
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        f.seek(aligned_offset)
                        existing = f.read(padded_size)
                        padded_data[: len(existing)] = existing

            # Insert new data
            padded_data[offset_delta : offset_delta + len(data)] = data

            # Open with O_DIRECT if available
            flags = os.O_WRONLY | os.O_CREAT
            if hasattr(os, "O_DIRECT"):
                flags |= os.O_DIRECT
            if sync:
                flags |= os.O_SYNC

            fd = os.open(filepath, flags, 0o644)
            try:
                os.lseek(fd, aligned_offset, os.SEEK_SET)
                bytes_written = os.write(fd, padded_data)
                return min(bytes_written - offset_delta, len(data))
            finally:
                os.close(fd)

        except Exception:
            return 0

    def process_on_gpu(self, data: np.ndarray, operation: str) -> Optional[np.ndarray]:
        """Process data on GPU if available."""
        if not self.enable_gpu or not numba:
            return None

        try:
            if operation == "compress_prepare":
                return self._gpu_compress_prepare(data)
            elif operation == "decompress_prepare":
                return self._gpu_decompress_prepare(data)
            else:
                return None
        except Exception:
            return None

    @staticmethod
    @numba.cuda.jit if numba else lambda: None
    def _gpu_compress_kernel(data, output):
        """GPU kernel for compression preprocessing."""
        idx = numba.cuda.grid(1)
        if idx < data.size:
            # Simple delta encoding
            if idx == 0:
                output[idx] = data[idx]
            else:
                output[idx] = data[idx] - data[idx - 1]

    def _gpu_compress_prepare(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for compression on GPU."""
        if not numba:
            return data

        # Transfer to GPU
        d_data = cuda.to_device(data)
        d_output = cuda.device_array_like(d_data)

        # Configure kernel
        threads_per_block = 256
        blocks_per_grid = (data.size + threads_per_block - 1) // threads_per_block

        # Run kernel
        self._gpu_compress_kernel[blocks_per_grid, threads_per_block](d_data, d_output)

        # Copy back
        return d_output.copy_to_host()

    def _gpu_decompress_prepare(self, data: np.ndarray) -> np.ndarray:
        """Prepare compressed data for decompression on GPU."""
        # Inverse of compression preparation
        if not numba:
            return data

        result = np.empty_like(data)
        result[0] = data[0]

        # Cumulative sum to reverse delta encoding
        for i in range(1, len(data)):
            result[i] = result[i - 1] + data[i]

        return result

    def close(self):
        """Clean up resources."""
        with self._cache_lock:
            for mm in self.mmap_cache.values():
                mm.close()
            self.mmap_cache.clear()


# Global instances
_shared_dict_manager = None
_hw_optimized_io = None


def get_shared_dictionary_manager() -> SharedDictionaryManager:
    """Get global shared dictionary manager instance."""
    global _shared_dict_manager
    if _shared_dict_manager is None:
        _shared_dict_manager = SharedDictionaryManager()
    return _shared_dict_manager


def get_hardware_optimized_io() -> HardwareOptimizedIO:
    """Get global hardware-optimized I/O instance."""
    global _hw_optimized_io
    if _hw_optimized_io is None:
        _hw_optimized_io = HardwareOptimizedIO()
    return _hw_optimized_io


# Integration with MAIF compression
class EnhancedCompressor:
    """Enhanced compressor that uses shared dictionaries and hardware optimization."""

    def __init__(self):
        self.dict_manager = get_shared_dictionary_manager()
        self.hw_io = get_hardware_optimized_io()

    def compress_with_features(
        self,
        data: bytes,
        data_type: str = "binary",
        use_dictionary: bool = True,
        use_gpu: bool = True,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress with advanced features."""
        metadata = {
            "original_size": len(data),
            "data_type": data_type,
            "features_used": [],
        }

        # Try GPU preprocessing for numeric data
        if use_gpu and data_type in ["embeddings", "numeric"]:
            try:
                # Convert to numpy array
                arr = np.frombuffer(data, dtype=np.float32)
                processed = self.hw_io.process_on_gpu(arr, "compress_prepare")
                if processed is not None:
                    data = processed.tobytes()
                    metadata["features_used"].append("gpu_preprocessing")
            except Exception:
                pass

        # Try dictionary compression
        if use_dictionary and ZSTD_AVAILABLE:
            dict_id = self.dict_manager.get_best_dictionary(data_type)
            if dict_id:
                compressed = self.dict_manager.compress_with_dictionary(data, dict_id)
                if compressed:
                    metadata["dictionary_id"] = dict_id
                    metadata["features_used"].append("shared_dictionary")
                    metadata["compressed_size"] = len(compressed)
                    metadata["compression_ratio"] = len(data) / len(compressed)
                    return compressed, metadata

        # Fallback to standard compression
        compressed = zlib.compress(data)
        metadata["compressed_size"] = len(compressed)
        metadata["compression_ratio"] = len(data) / len(compressed)
        metadata["features_used"].append("zlib")

        return compressed, metadata

    def decompress_with_features(
        self, compressed_data: bytes, metadata: Dict[str, Any]
    ) -> bytes:
        """Decompress with advanced features."""
        # Check for dictionary compression
        if "dictionary_id" in metadata:
            dict_id = metadata["dictionary_id"]
            decompressed = self.dict_manager.decompress_with_dictionary(
                compressed_data, dict_id
            )
            if decompressed:
                compressed_data = decompressed
            else:
                # Fallback to standard decompression
                compressed_data = zlib.decompress(compressed_data)
        elif "zlib" in metadata.get("features_used", []):
            compressed_data = zlib.decompress(compressed_data)

        # Reverse GPU preprocessing if used
        if "gpu_preprocessing" in metadata.get("features_used", []):
            try:
                arr = np.frombuffer(compressed_data, dtype=np.float32)
                processed = self.hw_io.process_on_gpu(arr, "decompress_prepare")
                if processed is not None:
                    compressed_data = processed.tobytes()
            except Exception:
                pass

        return compressed_data


# Export classes and functions
__all__ = [
    "SharedDictionaryManager",
    "HardwareOptimizedIO",
    "EnhancedCompressor",
    "get_shared_dictionary_manager",
    "get_hardware_optimized_io",
]
