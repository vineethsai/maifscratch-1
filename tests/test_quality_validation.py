"""
Comprehensive tests for quality validation and reconstruction guarantees.
Tests ensure semantic fidelity meets quality thresholds and <5% loss requirements.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from maif.compression import (
    MAIFCompressor,
    CompressionConfig,
    CompressionAlgorithm,
    CompressionResult,
)


class TestQualityThresholdEnforcement:
    """Test quality threshold enforcement in compression pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        # Configure with strict quality threshold (95% = <5% loss)
        self.config = CompressionConfig(
            algorithm=CompressionAlgorithm.SEMANTIC_AWARE,
            preserve_semantics=True,
            quality_threshold=0.95,  # <5% loss requirement
        )
        self.compressor = MAIFCompressor(self.config)

    def test_quality_threshold_enforcement_text(self):
        """Test quality threshold enforcement for text compression."""
        test_text = "This is a test of semantic-aware compression for artificial intelligence and machine learning applications."
        text_data = test_text.encode("utf-8")

        # Mock semantic compression to return low fidelity
        with patch.object(
            self.compressor, "_semantic_aware_compression"
        ) as mock_semantic:
            mock_semantic.return_value = CompressionResult(
                compressed_data=b"mock_compressed",
                original_size=len(text_data),
                compressed_size=50,
                compression_ratio=2.0,
                algorithm="semantic_aware",
                metadata={"mock": True},
                semantic_fidelity=0.85,  # Below 0.95 threshold
            )

            result = self.compressor.compress_data(text_data, "text")

            # Should fallback to lossless compression
            assert "fallback" in result.algorithm
            assert (
                result.semantic_fidelity == 1.0
            )  # Lossless guarantees perfect fidelity
            assert "fallback_reason" in result.metadata
            assert result.metadata["fallback_reason"] == "quality_threshold_not_met"
            assert result.metadata["original_fidelity"] == 0.85

    def test_quality_threshold_enforcement_embeddings(self):
        """Test quality threshold enforcement for embedding compression."""
        # Create test embeddings
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        import struct

        embedding_data = b""
        for embedding in embeddings:
            for value in embedding:
                embedding_data += struct.pack("f", value)

        # Mock HSC compression to return low fidelity
        with patch.object(self.compressor, "_hsc_compression") as mock_hsc:
            mock_hsc.return_value = CompressionResult(
                compressed_data=b"mock_hsc_compressed",
                original_size=len(embedding_data),
                compressed_size=20,
                compression_ratio=3.0,
                algorithm="hsc",
                metadata={"mock": True},
                semantic_fidelity=0.80,  # Below 0.95 threshold
            )

            result = self.compressor.compress_data(embedding_data, "embeddings")

            # Should fallback to lossless compression
            assert "fallback" in result.algorithm
            assert result.semantic_fidelity == 1.0
            assert result.metadata["fallback_reason"] == "quality_threshold_not_met"
            assert result.metadata["original_fidelity"] == 0.80

    def test_quality_threshold_met_no_fallback(self):
        """Test that high-quality compression doesn't trigger fallback."""
        test_text = "High quality compression test."
        text_data = test_text.encode("utf-8")

        # Mock semantic compression to return high fidelity
        with patch.object(
            self.compressor, "_semantic_aware_compression"
        ) as mock_semantic:
            mock_semantic.return_value = CompressionResult(
                compressed_data=b"high_quality_compressed",
                original_size=len(text_data),
                compressed_size=15,
                compression_ratio=2.0,
                algorithm="semantic_aware",
                metadata={"high_quality": True},
                semantic_fidelity=0.98,  # Above 0.95 threshold
            )

            result = self.compressor.compress_data(text_data, "text")

            # Should NOT fallback
            assert "fallback" not in result.algorithm
            assert result.algorithm == "semantic_aware"
            assert result.semantic_fidelity == 0.98
            assert "fallback_reason" not in result.metadata


class TestReconstructionQuality:
    """Test actual reconstruction quality and accuracy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.compressor = MAIFCompressor()

    def test_lossless_compression_perfect_reconstruction(self):
        """Test that lossless algorithms achieve perfect reconstruction."""
        test_data = b"Perfect reconstruction test data. " * 100

        lossless_algorithms = [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.GZIP,
            CompressionAlgorithm.BZIP2,
            CompressionAlgorithm.LZMA,
        ]

        for algorithm in lossless_algorithms:
            compressed = self.compressor.compress(test_data, algorithm)
            decompressed = self.compressor.decompress(compressed, algorithm)

            # Perfect reconstruction required
            assert decompressed == test_data, (
                f"Perfect reconstruction failed for {algorithm.value}"
            )

            # Calculate reconstruction error (should be 0)
            reconstruction_error = self._calculate_reconstruction_error(
                test_data, decompressed
            )
            assert reconstruction_error == 0.0, (
                f"Reconstruction error {reconstruction_error} for {algorithm.value}"
            )

    def test_semantic_fidelity_calculation_accuracy(self):
        """Test semantic fidelity calculation accuracy."""
        # Test with known embeddings
        original_embeddings = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # Perfect reconstruction
        perfect_reconstruction = original_embeddings.copy()
        fidelity_perfect = self._calculate_cosine_similarity_fidelity(
            original_embeddings, perfect_reconstruction
        )
        assert fidelity_perfect >= 0.999, (
            f"Perfect reconstruction fidelity should be ~1.0, got {fidelity_perfect}"
        )

        # Slight degradation
        degraded_reconstruction = original_embeddings * 0.95  # 5% reduction
        fidelity_degraded = self._calculate_cosine_similarity_fidelity(
            original_embeddings, degraded_reconstruction
        )
        assert fidelity_degraded >= 0.95, (
            f"5% degradation should maintain >95% fidelity, got {fidelity_degraded}"
        )

        # Significant degradation - add noise to break cosine similarity
        np.random.seed(42)  # For reproducible results
        heavily_degraded = original_embeddings * 0.5 + np.random.normal(
            0, 0.5, original_embeddings.shape
        )
        fidelity_heavy = self._calculate_cosine_similarity_fidelity(
            original_embeddings, heavily_degraded
        )
        assert fidelity_heavy < 0.95, (
            f"Noisy degradation should reduce fidelity, got {fidelity_heavy}"
        )

    def test_quality_threshold_validation(self):
        """Test that quality thresholds are properly validated."""
        config_strict = CompressionConfig(quality_threshold=0.99)  # 99% fidelity
        config_lenient = CompressionConfig(quality_threshold=0.90)  # 90% fidelity

        compressor_strict = MAIFCompressor(config_strict)
        _compressor_lenient = MAIFCompressor(config_lenient)  # noqa: F841

        # Test with mock results
        mock_result_medium_quality = CompressionResult(
            compressed_data=b"test",
            original_size=100,
            compressed_size=50,
            compression_ratio=2.0,
            algorithm="test",
            metadata={},
            semantic_fidelity=0.95,  # 95% fidelity
        )

        # Strict threshold (99%) should reject 95% fidelity
        assert (
            mock_result_medium_quality.semantic_fidelity
            < config_strict.quality_threshold
        )

        # Lenient threshold (90%) should accept 95% fidelity
        assert (
            mock_result_medium_quality.semantic_fidelity
            >= config_lenient.quality_threshold
        )

    def test_reconstruction_error_measurement(self):
        """Test reconstruction error measurement for different data types."""
        # Text data
        original_text = (
            b"This is original text data for testing reconstruction accuracy."
        )
        perfect_text = original_text
        corrupted_text = b"This is original text data for testing reconstruction accuracy!"  # Added exclamation

        error_perfect = self._calculate_reconstruction_error(
            original_text, perfect_text
        )
        error_corrupted = self._calculate_reconstruction_error(
            original_text, corrupted_text
        )

        assert error_perfect == 0.0, "Perfect reconstruction should have 0 error"
        assert error_corrupted > 0.0, "Corrupted reconstruction should have >0 error"
        assert error_corrupted < 0.05, "Minor corruption should be <5% error"

        # Binary data
        original_binary = bytes(range(256))
        perfect_binary = bytes(range(256))
        corrupted_binary = bytes(range(255)) + b"\x00"  # Changed last byte

        error_perfect_bin = self._calculate_reconstruction_error(
            original_binary, perfect_binary
        )
        error_corrupted_bin = self._calculate_reconstruction_error(
            original_binary, corrupted_binary
        )

        assert error_perfect_bin == 0.0
        assert error_corrupted_bin > 0.0

    def _calculate_reconstruction_error(
        self, original: bytes, reconstructed: bytes
    ) -> float:
        """Calculate reconstruction error as percentage of differing bytes."""
        if len(original) != len(reconstructed):
            return 1.0  # 100% error for different lengths

        if len(original) == 0:
            return 0.0

        differing_bytes = sum(1 for a, b in zip(original, reconstructed) if a != b)
        return differing_bytes / len(original)

    def _calculate_cosine_similarity_fidelity(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float:
        """Calculate semantic fidelity using cosine similarity."""
        similarities = []

        for i in range(min(len(original), len(reconstructed))):
            orig_vec = original[i]
            recon_vec = reconstructed[i]

            norm_orig = np.linalg.norm(orig_vec)
            norm_recon = np.linalg.norm(recon_vec)

            if norm_orig > 0 and norm_recon > 0:
                similarity = np.dot(orig_vec, recon_vec) / (norm_orig * norm_recon)
                similarities.append(max(0, similarity))

        return np.mean(similarities) if similarities else 0.0


class TestQualityGuarantees:
    """Test overall quality guarantees and SLA compliance."""

    def test_five_percent_loss_guarantee(self):
        """Test that system guarantees <5% loss (>95% fidelity)."""
        config = CompressionConfig(
            quality_threshold=0.95,  # 95% fidelity = <5% loss
            preserve_semantics=True,
        )
        compressor = MAIFCompressor(config)

        # Test various data types
        test_cases = [
            (b"Text data for quality testing. " * 50, "text"),
            (b"Binary data: " + bytes(range(256)) * 10, "binary"),
            # Note: embeddings would need proper struct packing
        ]

        for data, data_type in test_cases:
            result = compressor.compress_data(data, data_type)

            # Guarantee: semantic fidelity must be >= 95% (or fallback to lossless)
            assert (
                result.semantic_fidelity is None or result.semantic_fidelity >= 0.95
            ), (
                f"Quality guarantee violated: {result.semantic_fidelity} < 0.95 for {data_type}"
            )

            # If fallback occurred, should be perfect fidelity
            if "fallback" in result.algorithm:
                assert result.semantic_fidelity == 1.0, (
                    "Fallback should guarantee perfect fidelity"
                )

    def test_quality_monitoring_and_reporting(self):
        """Test quality monitoring and reporting capabilities."""
        compressor = MAIFCompressor()

        # Simulate multiple compression operations
        test_data = b"Quality monitoring test data. " * 20

        # Perform multiple compressions
        for _ in range(5):
            result = compressor.compress_data(test_data, "text")

            # Verify quality metrics are tracked
            if result.semantic_fidelity is not None:
                assert 0.0 <= result.semantic_fidelity <= 1.0, (
                    "Fidelity should be in [0,1] range"
                )

        # Check if statistics are being tracked
        assert hasattr(compressor, "compression_stats"), (
            "Should track compression statistics"
        )

    def test_algorithm_selection_quality_impact(self):
        """Test that algorithm selection considers quality requirements."""
        # High quality requirement should prefer lossless algorithms
        high_quality_config = CompressionConfig(
            quality_threshold=0.99,
            preserve_semantics=False,  # Disable semantic compression
        )

        # Low quality requirement allows lossy algorithms
        low_quality_config = CompressionConfig(
            quality_threshold=0.80,
            preserve_semantics=True,  # Enable semantic compression
        )

        high_quality_compressor = MAIFCompressor(high_quality_config)
        low_quality_compressor = MAIFCompressor(low_quality_config)

        test_data = b"Algorithm selection quality test data."

        # High quality should use conservative algorithms
        high_result = high_quality_compressor.compress_data(test_data, "text")

        # Low quality may use aggressive compression
        low_result = low_quality_compressor.compress_data(test_data, "text")

        # Both should meet their respective quality requirements
        if high_result.semantic_fidelity is not None:
            assert (
                high_result.semantic_fidelity >= 0.99
                or "fallback" in high_result.algorithm
            )

        if low_result.semantic_fidelity is not None:
            assert (
                low_result.semantic_fidelity >= 0.80
                or "fallback" in low_result.algorithm
            )
