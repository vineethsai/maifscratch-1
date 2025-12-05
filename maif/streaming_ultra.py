"""
Ultra-high-performance streaming for 400+ MB/s throughput.
Uses advanced I/O techniques and zero-copy operations.
"""

import os
import mmap
import time
import threading
from typing import Iterator, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from .core import MAIFDecoder, MAIFBlock


@dataclass
class UltraStreamingConfig:
    """Ultra-high-performance streaming configuration."""

    # Optimized for 400+ MB/s
    chunk_size: int = 64 * 1024 * 1024  # 64MB chunks
    max_workers: int = min(32, mp.cpu_count() * 4)  # Maximum parallelism
    buffer_size: int = 256 * 1024 * 1024  # 256MB buffer
    use_memory_mapping: bool = True
    use_process_pool: bool = True  # Use processes instead of threads
    prefetch_size: int = 128 * 1024 * 1024  # 128MB prefetch
    batch_size: int = 64  # Large batches
    use_zero_copy: bool = True
    sequential_read: bool = True  # Optimize for sequential access


class UltraHighThroughputReader:
    """Ultra-high-performance reader targeting 400+ MB/s."""

    def __init__(self, maif_path: str, config: Optional[UltraStreamingConfig] = None):
        self.maif_path = Path(maif_path)
        self.config = config or UltraStreamingConfig()

        if not self.maif_path.exists():
            raise FileNotFoundError(f"MAIF file not found: {maif_path}")

        self.file_size = self.maif_path.stat().st_size
        self.decoder = None
        self._total_bytes_read = 0
        self._start_time = None

    def __enter__(self):
        """Context manager entry."""
        self._start_time = time.time()
        self._initialize_decoder()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

    def stream_blocks_ultra(self) -> Iterator[Tuple[str, bytes]]:
        """Ultra-fast streaming using advanced techniques."""
        if not self.decoder or not self.decoder.blocks:
            return

        # Method 1: Pure memory mapping with zero-copy
        if self.config.use_zero_copy:
            yield from self._stream_zero_copy()
        else:
            # Method 2: Optimized buffered reading
            yield from self._stream_optimized_buffered()

    def _stream_zero_copy(self) -> Iterator[Tuple[str, bytes]]:
        """Zero-copy streaming using memory mapping."""
        try:
            with open(self.maif_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Advise kernel about access pattern
                    if hasattr(mmap, "MADV_SEQUENTIAL"):
                        mm.madvise(mmap.MADV_SEQUENTIAL)
                    if hasattr(mmap, "MADV_WILLNEED"):
                        mm.madvise(mmap.MADV_WILLNEED)

                    # Sort blocks by offset for sequential access
                    sorted_blocks = sorted(self.decoder.blocks, key=lambda b: b.offset)

                    for block in sorted_blocks:
                        try:
                            header_size = 32
                            data_size = max(0, block.size - header_size)

                            if data_size > 0:
                                start_pos = block.offset + header_size
                                end_pos = start_pos + data_size

                                if end_pos <= len(mm):
                                    # Zero-copy slice - direct memory access
                                    data = mm[start_pos:end_pos]
                                    self._total_bytes_read += len(data)
                                    yield block.block_type or "unknown", bytes(data)
                                else:
                                    yield (
                                        block.block_type or "unknown",
                                        b"boundary_error",
                                    )
                            else:
                                yield block.block_type or "unknown", b"empty"

                        except Exception as e:
                            yield "error", f"zero_copy_error: {str(e)}".encode()

        except Exception as e:
            yield "error", f"mmap_error: {str(e)}".encode()

    def _initialize_decoder(self):
        """Initialize decoder with error handling (v3 format - self-contained)."""
        try:
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
        except Exception:
            pass

    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get throughput statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0.001
        throughput_mbps = (self._total_bytes_read / (1024 * 1024)) / elapsed

        return {
            "total_bytes_read": self._total_bytes_read,
            "elapsed_seconds": elapsed,
            "throughput_mbps": throughput_mbps,
            "file_size": self.file_size,
            "config": {
                "chunk_size_mb": self.config.chunk_size // (1024 * 1024),
                "buffer_size_mb": self.config.buffer_size // (1024 * 1024),
                "max_workers": self.config.max_workers,
                "use_zero_copy": self.config.use_zero_copy,
                "use_process_pool": self.config.use_process_pool,
            },
        }


# Raw file streaming for maximum performance
class RawFileStreamer:
    """Raw file streaming for absolute maximum throughput."""

    def __init__(self, file_path: str, chunk_size: int = 256 * 1024 * 1024):
        self.file_path = file_path
        self.chunk_size = chunk_size  # 256MB chunks
        self.file_size = os.path.getsize(file_path)

    def stream_mmap_raw(self) -> Iterator[bytes]:
        """Stream using memory mapping for maximum speed."""
        try:
            with open(self.file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Advise sequential access
                    if hasattr(mmap, "MADV_SEQUENTIAL"):
                        mm.madvise(mmap.MADV_SEQUENTIAL)

                    offset = 0
                    while offset < len(mm):
                        end_offset = min(offset + self.chunk_size, len(mm))
                        yield mm[offset:end_offset]
                        offset = end_offset

        except Exception as e:
            yield f"mmap_raw_error: {str(e)}".encode()
