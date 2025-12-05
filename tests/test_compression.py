"""
Comprehensive tests for MAIF compression functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from maif.compression import (
    CompressionMetadata,
    CompressionAlgorithm,
    MAIFCompressor,
    SemanticAwareCompressor,
)


class TestCompressionMetadata:
    """Test CompressionMetadata data structure."""

    def test_compression_metadata_creation(self):
        """Test basic CompressionMetadata creation."""
        metadata = CompressionMetadata(
            algorithm="zlib",
            level=6,
            original_size=1000,
            compressed_size=300,
            ratio=3.33,
        )

        assert metadata.algorithm == "zlib"
        assert metadata.level == 6
        assert metadata.original_size == 1000
        assert metadata.compressed_size == 300
        assert metadata.ratio == 3.33


class TestCompressionAlgorithm:
    """Test CompressionAlgorithm enum."""

    def test_compression_algorithm_values(self):
        """Test CompressionAlgorithm enum values."""
        assert CompressionAlgorithm.ZLIB.value == "zlib"
        assert CompressionAlgorithm.GZIP.value == "gzip"
        assert CompressionAlgorithm.BZIP2.value == "bzip2"
        assert CompressionAlgorithm.LZMA.value == "lzma"
        assert CompressionAlgorithm.BROTLI.value == "brotli"
        assert CompressionAlgorithm.LZ4.value == "lz4"
        assert CompressionAlgorithm.ZSTANDARD.value == "zstandard"
        assert CompressionAlgorithm.HSC.value == "hsc"


class TestMAIFCompressor:
    """Test MAIFCompressor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MAIFCompressor()

    def test_compressor_initialization(self):
        """Test MAIFCompressor initialization."""
        assert hasattr(self.compressor, "supported_algorithms")
        assert CompressionAlgorithm.ZLIB in self.compressor.supported_algorithms
        assert CompressionAlgorithm.GZIP in self.compressor.supported_algorithms
        assert CompressionAlgorithm.BZIP2 in self.compressor.supported_algorithms
        assert CompressionAlgorithm.LZMA in self.compressor.supported_algorithms

    def test_zlib_compression(self):
        """Test zlib compression and decompression."""
        test_data = (
            b"Hello, MAIF compression world! " * 100
        )  # Repetitive for better compression

        # Compress
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.ZLIB)

        assert compressed != test_data
        assert len(compressed) < len(test_data)  # Should be smaller

        # Decompress
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == test_data

    def test_gzip_compression(self):
        """Test gzip compression and decompression."""
        test_data = b"GZIP compression test data. " * 50

        # Compress
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.GZIP)

        assert compressed != test_data
        assert len(compressed) < len(test_data)

        # Decompress
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.GZIP)

        assert decompressed == test_data

    def test_bzip2_compression(self):
        """Test bzip2 compression and decompression."""
        test_data = b"BZIP2 compression test with repeated patterns. " * 30

        # Compress
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.BZIP2)

        assert compressed != test_data
        assert len(compressed) < len(test_data)

        # Decompress
        decompressed = self.compressor.decompress(
            compressed, CompressionAlgorithm.BZIP2
        )

        assert decompressed == test_data

    def test_lzma_compression(self):
        """Test LZMA compression and decompression."""
        test_data = b"LZMA compression test data with patterns. " * 40

        # Compress
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.LZMA)

        assert compressed != test_data
        assert len(compressed) < len(test_data)

        # Decompress
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.LZMA)

        assert decompressed == test_data

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("brotli") is None,
        reason="Brotli not available",
    )
    def test_brotli_compression(self):
        """Test Brotli compression and decompression."""
        import brotli

        test_data = b"Brotli compression test data. " * 60

        # Compress
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.BROTLI)

        assert compressed != test_data
        assert len(compressed) < len(test_data)

        # Decompress
        decompressed = self.compressor.decompress(
            compressed, CompressionAlgorithm.BROTLI
        )

        assert decompressed == test_data

    def test_compression_levels(self):
        """Test compression with different levels."""
        test_data = b"Compression level test data. " * 100

        # Test different compression levels
        levels = [1, 6, 9]
        compressed_sizes = []

        for level in levels:
            compressed = self.compressor.compress(
                test_data, CompressionAlgorithm.ZLIB, level=level
            )
            compressed_sizes.append(len(compressed))

            # Verify decompression works
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            assert decompressed == test_data

        # Higher compression levels should generally produce smaller files
        # (though this isn't guaranteed for all data)
        assert all(size > 0 for size in compressed_sizes)

    def test_get_compression_ratio(self):
        """Test compression ratio calculation."""
        original = b"x" * 1000
        compressed = b"y" * 300

        ratio = self.compressor.get_compression_ratio(original, compressed)

        expected_ratio = 1000 / 300
        assert abs(ratio - expected_ratio) < 0.01

    def test_benchmark_algorithms(self):
        """Test algorithm benchmarking."""
        test_data = b"Benchmark test data with repeated patterns. " * 50

        results = self.compressor.benchmark_algorithms(test_data)

        assert isinstance(results, dict)
        assert "zlib" in results
        assert "gzip" in results
        assert "bzip2" in results
        assert "lzma" in results

        # Check result structure
        for algorithm, result in results.items():
            if getattr(result, "metadata", {}).get("success", True):
                assert hasattr(result, "compressed_size")
                assert hasattr(result, "compression_ratio")
                assert "compression_time" in result.metadata
                # Only check decompression_time if present
                if "decompression_time" in result.metadata:
                    assert result.metadata["decompression_time"] >= 0
                assert result.compressed_size > 0
                assert result.compression_ratio > 0

    def test_empty_data_compression(self):
        """Test compression of empty data."""
        empty_data = b""

        # Should handle empty data gracefully
        compressed = self.compressor.compress(empty_data, CompressionAlgorithm.ZLIB)
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == empty_data

    def test_small_data_compression(self):
        """Test compression of very small data."""
        small_data = b"x"

        compressed = self.compressor.compress(small_data, CompressionAlgorithm.ZLIB)
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == small_data
        # Small data might actually be larger after compression due to headers
        assert len(compressed) >= 0

    def test_large_data_compression(self):
        """Test compression of large data."""
        large_data = b"Large data compression test. " * 10000  # ~300KB

        compressed = self.compressor.compress(large_data, CompressionAlgorithm.ZLIB)
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == large_data
        assert len(compressed) < len(large_data)  # Should achieve good compression

    def test_random_data_compression(self):
        """Test compression of random (incompressible) data."""
        import random

        # Generate random data (should be difficult to compress)
        random_data = bytes([random.randint(0, 255) for _ in range(1000)])

        compressed = self.compressor.compress(random_data, CompressionAlgorithm.ZLIB)
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == random_data
        # Random data might not compress well, but should still work
        assert len(compressed) > 0


class TestSemanticAwareCompressor:
    """Test SemanticAwareCompressor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = SemanticAwareCompressor()

    def test_semantic_compressor_initialization(self):
        """Test SemanticAwareCompressor initialization."""
        assert isinstance(
            self.compressor, MAIFCompressor
        )  # Should inherit from MAIFCompressor
        assert hasattr(self.compressor, "supported_algorithms")
        assert CompressionAlgorithm.HSC in self.compressor.supported_algorithms

    def test_compress_with_semantic_preservation(self):
        """Test compression with semantic preservation."""
        test_data = b"This is semantic text data that should preserve meaning during compression."

        compressed = self.compressor.compress_with_semantic_preservation(
            data=test_data, data_type="text", algorithm=CompressionAlgorithm.ZLIB
        )

        assert compressed != test_data
        assert len(compressed) <= len(test_data)  # Should be compressed

        # Should be able to decompress back
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)
        assert decompressed == test_data

    def test_compress_text_semantic(self):
        """Test semantic text compression."""
        text_data = b"Natural language text with semantic meaning and structure."

        compressed = self.compressor._compress_text_semantic(
            text_data, CompressionAlgorithm.ZLIB
        )

        assert compressed != text_data
        assert len(compressed) > 0

    def test_compress_embeddings_semantic(self):
        """Test semantic embeddings compression."""
        # Simulate embedding data (JSON-encoded list of floats)
        import json

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        embedding_data = json.dumps(embeddings).encode("utf-8")

        compressed = self.compressor._compress_embeddings_semantic(
            embedding_data, CompressionAlgorithm.ZLIB
        )

        assert compressed != embedding_data
        assert len(compressed) > 0

    @patch("maif.semantic.HierarchicalSemanticCompression")
    def test_hsc_compression(self, mock_hsc_class):
        """Test Hierarchical Semantic Compression (HSC)."""
        # Mock the HSC class
        mock_hsc = Mock()
        mock_hsc.compress_embeddings.return_value = {
            "compressed_data": b"compressed_embedding_data",
            "metadata": {"algorithm": "hsc", "compression_ratio": 2.5},
        }
        mock_hsc_class.return_value = mock_hsc

        test_data = b"embedding data to compress with HSC"

        compressed = self.compressor._compress_hsc(test_data)

        assert compressed is not None
        assert len(compressed) > 0
        mock_hsc.compress_embeddings.assert_called_once()

    @patch("maif.semantic.HierarchicalSemanticCompression")
    def test_hsc_decompression(self, mock_hsc_class):
        """Test HSC decompression."""
        # Mock the HSC class
        mock_hsc = Mock()
        mock_hsc.decompress_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_hsc_class.return_value = mock_hsc

        # Simulate compressed HSC data
        import json

        compressed_data = json.dumps(
            {"compressed_data": "base64_encoded_data", "metadata": {"algorithm": "hsc"}}
        ).encode("utf-8")

        decompressed = self.compressor._decompress_hsc(compressed_data)

        assert decompressed is not None
        mock_hsc.decompress_embeddings.assert_called_once()

    def test_semantic_preservation_text_types(self):
        """Test semantic preservation with different text types."""
        test_cases = [
            (b"Scientific research paper abstract with technical terms.", "scientific"),
            (b"Legal document with formal language and clauses.", "legal"),
            (b"Medical report with patient information and diagnoses.", "medical"),
            (b"General business communication and correspondence.", "business"),
        ]

        for text_data, data_type in test_cases:
            compressed = self.compressor.compress_with_semantic_preservation(
                data=text_data, data_type=data_type, algorithm=CompressionAlgorithm.ZLIB
            )

            assert compressed != text_data
            assert len(compressed) > 0

            # Verify decompression
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            assert decompressed == text_data

    def test_semantic_preservation_embedding_types(self):
        """Test semantic preservation with different embedding types."""
        import json

        test_embeddings = [
            ([[0.1, 0.2], [0.3, 0.4]], "word_embeddings"),
            ([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]], "sentence_embeddings"),
            ([[0.1] * 512], "image_embeddings"),
            ([[0.2] * 768], "multimodal_embeddings"),
        ]

        for embeddings, data_type in test_embeddings:
            embedding_data = json.dumps(embeddings).encode("utf-8")

            compressed = self.compressor.compress_with_semantic_preservation(
                data=embedding_data,
                data_type=data_type,
                algorithm=CompressionAlgorithm.ZLIB,
            )

            assert compressed != embedding_data
            assert len(compressed) > 0


class TestCompressionIntegration:
    """Test compression integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MAIFCompressor()
        self.semantic_compressor = SemanticAwareCompressor()

    def test_algorithm_comparison(self):
        """Test comparison of different compression algorithms."""
        test_data = b"Compression algorithm comparison test data. " * 100

        algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.BZIP2,
            CompressionAlgorithm.LZMA,
        ]

        results = {}

        for algorithm in algorithms:
            try:
                compressed = self.compressor.compress(test_data, algorithm)
                decompressed = self.compressor.decompress(compressed, algorithm)

                assert decompressed == test_data

                ratio = self.compressor.get_compression_ratio(test_data, compressed)
                results[algorithm.value] = {
                    "compressed_size": len(compressed),
                    "ratio": ratio,
                    "success": True,
                }
            except Exception as e:
                results[algorithm.value] = {"success": False, "error": str(e)}

        # At least zlib should work
        assert results["zlib"]["success"] is True
        assert results["zlib"]["ratio"] > 1.0  # Should achieve some compression

    def test_compression_with_different_data_types(self):
        """Test compression with various data types."""
        test_cases = [
            (b"Plain text data for compression testing.", "text"),
            (b'{"json": "data", "numbers": [1, 2, 3]}', "json"),
            (b"<xml><tag>XML data</tag></xml>", "xml"),
            (b"\x89PNG\r\n\x1a\n" + b"fake_png_data" * 50, "binary"),
            (b"Repeated pattern " * 200, "repetitive"),
        ]

        for data, data_type in test_cases:
            # Test regular compression
            compressed = self.compressor.compress(data, CompressionAlgorithm.ZLIB)
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            assert decompressed == data

            # Test semantic-aware compression
            semantic_compressed = (
                self.semantic_compressor.compress_with_semantic_preservation(
                    data, data_type, CompressionAlgorithm.ZLIB
                )
            )
            semantic_decompressed = self.semantic_compressor.decompress(
                semantic_compressed, CompressionAlgorithm.ZLIB
            )
            assert semantic_decompressed == data

    def test_compression_performance_characteristics(self):
        """Test compression performance with different data sizes."""
        import time

        data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB

        for size in data_sizes:
            test_data = b"Performance test data. " * (size // 23)  # Approximate size

            start_time = time.time()
            compressed = self.compressor.compress(test_data, CompressionAlgorithm.ZLIB)
            compression_time = time.time() - start_time

            start_time = time.time()
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            decompression_time = time.time() - start_time

            assert decompressed == test_data
            assert compression_time < 5.0  # Should be reasonably fast
            assert decompression_time < 5.0

            ratio = self.compressor.get_compression_ratio(test_data, compressed)
            assert ratio > 1.0  # Should achieve compression


class TestCompressionErrorHandling:
    """Test compression error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MAIFCompressor()

    def test_invalid_algorithm(self):
        """Test handling of invalid compression algorithms."""
        test_data = b"test data"

        with pytest.raises((ValueError, AttributeError)):
            self.compressor.compress(test_data, "invalid_algorithm")

    def test_corrupted_compressed_data(self):
        """Test handling of corrupted compressed data."""
        corrupted_data = b"this_is_not_valid_compressed_data"

        with pytest.raises(Exception):
            self.compressor.decompress(corrupted_data, CompressionAlgorithm.ZLIB)

    def test_wrong_decompression_algorithm(self):
        """Test decompression with wrong algorithm."""
        test_data = b"test data for wrong algorithm"

        # Compress with zlib
        compressed = self.compressor.compress(test_data, CompressionAlgorithm.ZLIB)

        # Try to decompress with gzip (should fail)
        with pytest.raises(Exception):
            self.compressor.decompress(compressed, CompressionAlgorithm.GZIP)

    def test_none_data_input(self):
        """Test handling of None data input."""
        with pytest.raises((TypeError, AttributeError)):
            self.compressor.compress(None, CompressionAlgorithm.ZLIB)

    def test_invalid_compression_level(self):
        """Test handling of invalid compression levels."""
        test_data = b"test data"

        # Very high compression level (might be invalid)
        try:
            compressed = self.compressor.compress(
                test_data, CompressionAlgorithm.ZLIB, level=100
            )
            # If it doesn't raise an exception, that's also acceptable
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            assert decompressed == test_data
        except (ValueError, OSError):
            # Expected for invalid compression levels
            pass


class TestCompressionPerformance:
    """Test compression performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MAIFCompressor()

    def test_compression_speed_benchmark(self):
        """Test compression speed with various algorithms."""
        import time

        # Create test data with good compression potential
        test_data = b"Compression speed test data with repeated patterns. " * 1000

        algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.BZIP2,
            CompressionAlgorithm.LZMA,
        ]

        performance_results = {}

        for algorithm in algorithms:
            try:
                # Measure compression time
                start_time = time.time()
                compressed = self.compressor.compress(test_data, algorithm)
                compression_time = time.time() - start_time

                # Measure decompression time
                start_time = time.time()
                decompressed = self.compressor.decompress(compressed, algorithm)
                decompression_time = time.time() - start_time

                assert decompressed == test_data

                ratio = self.compressor.get_compression_ratio(test_data, compressed)

                performance_results[algorithm.value] = {
                    "compression_time": compression_time,
                    "decompression_time": decompression_time,
                    "compression_ratio": ratio,
                    "compressed_size": len(compressed),
                }

                # Performance should be reasonable
                assert compression_time < 10.0  # Should complete in under 10 seconds
                assert decompression_time < 5.0  # Decompression should be faster

            except ImportError:
                # Some algorithms might not be available
                performance_results[algorithm.value] = {"available": False}

        # At least zlib should be available and performant
        assert "zlib" in performance_results
        assert performance_results["zlib"].get("compression_ratio", 0) > 1.0

    def test_memory_efficiency(self):
        """Test memory efficiency with large data."""
        # Create moderately large test data
        large_data = b"Memory efficiency test data. " * 50000  # ~1.5MB

        # Should handle large data without excessive memory usage
        compressed = self.compressor.compress(large_data, CompressionAlgorithm.ZLIB)
        decompressed = self.compressor.decompress(compressed, CompressionAlgorithm.ZLIB)

        assert decompressed == large_data
        assert len(compressed) < len(large_data)  # Should achieve compression

    def test_compression_ratio_optimization(self):
        """Test compression ratio optimization."""
        # Test data with different characteristics
        test_cases = [
            (b"A" * 10000, "highly_repetitive"),  # Should compress very well
            (b"The quick brown fox jumps over the lazy dog. " * 200, "natural_text"),
            (b'{"key": "value", "number": 123}' * 500, "structured_data"),
        ]

        for data, data_type in test_cases:
            compressed = self.compressor.compress(data, CompressionAlgorithm.ZLIB)
            ratio = self.compressor.get_compression_ratio(data, compressed)

            if data_type == "highly_repetitive":
                assert ratio > 50.0  # Should achieve excellent compression
            elif data_type in ["natural_text", "structured_data"]:
                assert ratio > 2.0  # Should achieve good compression

            # Verify correctness
            decompressed = self.compressor.decompress(
                compressed, CompressionAlgorithm.ZLIB
            )
            assert decompressed == data


if __name__ == "__main__":
    pytest.main([__file__])
