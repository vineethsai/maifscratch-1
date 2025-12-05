"""
Compression Manager for MAIF
============================

Provides a unified interface for compression operations across the MAIF system.
This module serves as a facade for the underlying compression implementations,
making it easier to use compression functionality throughout the codebase.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import json
import logging
from enum import Enum
from dataclasses import dataclass

from .compression import (
    MAIFCompressor,
    CompressionAlgorithm,
    CompressionConfig,
    CompressionResult,
    ZSTD_AVAILABLE,
    LZ4_AVAILABLE,
    BROTLI_AVAILABLE,
)

# Initialize logger
logger = logging.getLogger(__name__)


class CompressionManager:
    """
    Manages compression operations across the MAIF system.
    Provides a unified interface for compression and decompression.
    """

    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize the compression manager with optional configuration."""
        self.config = config or CompressionConfig()
        self.compressor = MAIFCompressor(self.config)
        self.compression_stats = {}

    def compress(
        self,
        data: bytes,
        algorithm: Union[str, CompressionAlgorithm] = None,
        level: int = None,
    ) -> bytes:
        """
        Compress data using the specified algorithm and level.

        Args:
            data: The data to compress
            algorithm: The compression algorithm to use (defaults to config)
            level: The compression level to use (defaults to config)

        Returns:
            Compressed bytes
        """
        if algorithm is None:
            algorithm = self.config.algorithm

        return self.compressor.compress(data, algorithm, level)

    def decompress(
        self, data: bytes, algorithm: Union[str, CompressionAlgorithm]
    ) -> bytes:
        """
        Decompress data using the specified algorithm.

        Args:
            data: The compressed data
            algorithm: The algorithm used for compression

        Returns:
            Decompressed bytes
        """
        return self.compressor.decompress(data, algorithm)

    def compress_with_metadata(
        self,
        data: bytes,
        data_type: str = "binary",
        semantic_context: Optional[Dict] = None,
    ) -> CompressionResult:
        """
        Compress data with full metadata and semantic awareness.

        Args:
            data: The data to compress
            data_type: Type of data (binary, text, json, embeddings, etc.)
            semantic_context: Optional context for semantic compression

        Returns:
            CompressionResult object with compressed data and metadata
        """
        return self.compressor.compress_data(data, data_type, semantic_context)

    def decompress_with_metadata(self, result: CompressionResult) -> bytes:
        """
        Decompress data using metadata from CompressionResult.

        Args:
            result: CompressionResult from previous compression

        Returns:
            Decompressed bytes
        """
        return self.compressor.decompress_data(result.compressed_data, result.metadata)

    def get_compression_ratio(self, original: bytes, compressed: bytes) -> float:
        """Calculate compression ratio between original and compressed data."""
        return self.compressor.get_compression_ratio(original, compressed)

    def benchmark_algorithms(
        self, data: bytes, data_type: str = "binary"
    ) -> Dict[str, Any]:
        """
        Benchmark all available compression algorithms on the given data.

        Args:
            data: Sample data to benchmark
            data_type: Type of data for semantic algorithms

        Returns:
            Dictionary of algorithm names to benchmark results
        """
        return self.compressor.benchmark_algorithms(data)

    def get_optimal_algorithm(
        self, data: bytes, data_type: str = "binary"
    ) -> CompressionAlgorithm:
        """
        Determine the optimal compression algorithm for the given data.

        Args:
            data: Sample data to analyze
            data_type: Type of data (binary, text, json, embeddings, etc.)

        Returns:
            The recommended CompressionAlgorithm
        """
        return self.compressor._select_optimal_algorithm(data, data_type, None)

    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported compression algorithms."""
        return [algo.value for algo in self.compressor.supported_algorithms]

    def compress_zstd(self, data: bytes) -> bytes:
        """
        Compress data using ZSTD algorithm.

        Args:
            data: The data to compress

        Returns:
            Compressed bytes
        """
        return self.compress(data, CompressionAlgorithm.ZSTANDARD)

    def decompress_zstd(self, data: bytes) -> bytes:
        """
        Decompress data using ZSTD algorithm.

        Args:
            data: The compressed data

        Returns:
            Decompressed bytes
        """
        return self.decompress(data, CompressionAlgorithm.ZSTANDARD)

    def compress_gzip(self, data: bytes) -> bytes:
        """
        Compress data using GZIP algorithm.

        Args:
            data: The data to compress

        Returns:
            Compressed bytes
        """
        return self.compress(data, CompressionAlgorithm.GZIP)

    def decompress_gzip(self, data: bytes) -> bytes:
        """
        Decompress data using GZIP algorithm.

        Args:
            data: The compressed data

        Returns:
            Decompressed bytes
        """
        return self.decompress(data, CompressionAlgorithm.GZIP)

    def compress_lz4(self, data: bytes) -> bytes:
        """
        Compress data using LZ4 algorithm.

        Args:
            data: The data to compress

        Returns:
            Compressed bytes
        """
        return self.compress(data, CompressionAlgorithm.LZ4)

    def decompress_lz4(self, data: bytes) -> bytes:
        """
        Decompress data using LZ4 algorithm.

        Args:
            data: The compressed data

        Returns:
            Decompressed bytes
        """
        return self.decompress(data, CompressionAlgorithm.LZ4)

    def compress_snappy(self, data: bytes) -> bytes:
        """
        Compress data using Snappy algorithm.

        Args:
            data: The data to compress

        Returns:
            Compressed bytes
        """
        # Snappy is not directly supported in the CompressionAlgorithm enum
        # We'll need to add support for it or use a fallback
        try:
            import snappy  # type: ignore[import-not-found]

            return snappy.compress(data)
        except ImportError:
            # Fallback to LZ4 if available, otherwise ZLIB
            if LZ4_AVAILABLE:
                return self.compress(data, CompressionAlgorithm.LZ4)
            else:
                return self.compress(data, CompressionAlgorithm.ZLIB)

    def decompress_snappy(self, data: bytes) -> bytes:
        """
        Decompress data using Snappy algorithm.

        Args:
            data: The compressed data

        Returns:
            Decompressed bytes
        """
        try:
            import snappy  # type: ignore[import-not-found]

            return snappy.decompress(data)
        except ImportError:
            # Try LZ4 first, then ZLIB as fallback
            try:
                if LZ4_AVAILABLE:
                    return self.decompress(data, CompressionAlgorithm.LZ4)
                else:
                    return self.decompress(data, CompressionAlgorithm.ZLIB)
            except (ValueError, Exception) as e:
                # Last resort: return data as-is
                logger.debug(f"All decompression attempts failed: {e}")
                return data
