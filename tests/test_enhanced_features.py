"""
Comprehensive test suite for enhanced MAIF features (v3 format).
Tests the improved implementations.
"""

import os
import json
import time
import tempfile
import shutil
import pytest
from pathlib import Path

from maif import MAIFEncoder, MAIFDecoder, BlockType


def test_enhanced_block_types():
    """Test the v3 block type system."""
    print("Testing Enhanced Block Types...")

    try:
        # Test block types exist
        assert hasattr(BlockType, "TEXT")
        assert hasattr(BlockType, "EMBEDDINGS")
        assert hasattr(BlockType, "BINARY")
        assert hasattr(BlockType, "VIDEO")
        assert hasattr(BlockType, "AUDIO")
        assert hasattr(BlockType, "IMAGE")

        # Test block type values are integers
        for bt in BlockType:
            assert isinstance(bt.value, int)

        print("✓ Enhanced Block Types: PASSED")

    except Exception as e:
        print(f"✗ Enhanced Block Types: FAILED - {e}")
        pytest.fail(str(e))


def test_enhanced_semantic_algorithms():
    """Test semantic algorithms if available."""
    print("Testing Enhanced Semantic Algorithms...")

    try:
        from maif.semantic_optimized import (
            AdaptiveCrossModalAttention,
            HierarchicalSemanticCompression,
            CryptographicSemanticBinding,
        )
        import numpy as np

        # Test ACAM
        acam = AdaptiveCrossModalAttention(embedding_dim=384)
        embeddings = {"text": np.random.rand(384), "image": np.random.rand(384)}
        trust_scores = {"text": 1.0, "image": 0.8}

        attention_weights = acam.compute_attention_weights(embeddings, trust_scores)
        assert hasattr(attention_weights, "normalized_weights")

        # Test HSC
        hsc = HierarchicalSemanticCompression(target_compression_ratio=0.4)
        test_embeddings = [[float(i + j) for i in range(384)] for j in range(10)]
        compressed_result = hsc.compress_embeddings(
            test_embeddings, preserve_fidelity=True
        )

        assert "compressed_data" in compressed_result
        assert "metadata" in compressed_result

        # Test CSB
        csb = CryptographicSemanticBinding()
        test_embedding = [float(i) for i in range(384)]
        test_source = "Test data"

        commitment = csb.create_semantic_commitment(test_embedding, test_source)
        assert "commitment_hash" in commitment

        print("✓ Enhanced Semantic Algorithms: PASSED")

    except ImportError:
        print("⚠ Semantic algorithms not available - skipping")
        pytest.skip("Semantic algorithms not available")
    except Exception as e:
        print(f"✗ Enhanced Semantic Algorithms: FAILED - {e}")
        pytest.fail(str(e))


def test_semantic_aware_compression():
    """Test compression system."""
    print("Testing Compression...")

    try:
        from maif.compression import MAIFCompressor, CompressionAlgorithm

        compressor = MAIFCompressor()

        # Test text compression
        test_text = "This is a test " * 100
        text_data = test_text.encode("utf-8")

        compressed = compressor.compress(text_data, CompressionAlgorithm.ZLIB)
        assert len(compressed) < len(text_data)

        decompressed = compressor.decompress(compressed, CompressionAlgorithm.ZLIB)
        assert decompressed == text_data

        print("✓ Compression: PASSED")

    except Exception as e:
        print(f"✗ Compression: FAILED - {e}")
        pytest.fail(str(e))


def test_advanced_forensics():
    """Test forensic analysis capabilities."""
    print("Testing Advanced Forensics...")

    try:
        from maif.forensics import ForensicAnalyzer

        with tempfile.TemporaryDirectory() as temp_dir:
            maif_path = os.path.join(temp_dir, "test_forensics.maif")

            encoder = MAIFEncoder(maif_path, agent_id="test_agent")
            encoder.add_text_block("Test forensic data")
            encoder.add_text_block("More test data")
            encoder.finalize()

            # Analyze with forensics
            analyzer = ForensicAnalyzer()
            result = analyzer.analyze_maif_file(maif_path)

            assert "version_analysis" in result
            assert "integrity_analysis" in result
            assert "temporal_analysis" in result

            print("✓ Advanced Forensics: PASSED")

    except Exception as e:
        print(f"✗ Advanced Forensics: FAILED - {e}")
        pytest.fail(str(e))


def test_enhanced_integration():
    """Test enhanced integration module."""
    print("Testing Enhanced Integration...")

    try:
        from maif.integration_enhanced import EnhancedMAIF

        with tempfile.TemporaryDirectory() as temp_dir:
            maif_path = os.path.join(temp_dir, "enhanced.maif")

            enhanced = EnhancedMAIF(maif_path, agent_id="test")
            enhanced.add_text_block("Test content")
            enhanced.save()

            assert os.path.exists(maif_path)

            print("✓ Enhanced Integration: PASSED")

    except Exception as e:
        print(f"✗ Enhanced Integration: FAILED - {e}")
        pytest.fail(str(e))


def test_performance_benchmarks():
    """Test performance."""
    print("Testing Performance Benchmarks...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            maif_path = os.path.join(temp_dir, "perf_test.maif")

            # Test creation performance
            start_time = time.time()
            encoder = MAIFEncoder(maif_path, agent_id="perf_test")

            for i in range(100):
                encoder.add_text_block(f"Block {i} content " * 10)

            encoder.finalize()
            creation_time = time.time() - start_time

            assert creation_time < 10.0  # Should complete within 10 seconds

            # Test reading performance
            start_time = time.time()
            decoder = MAIFDecoder(maif_path)
            decoder.load()
            read_time = time.time() - start_time

            assert read_time < 5.0  # Should read within 5 seconds
            assert len(decoder.blocks) == 100

            print("✓ Performance Benchmarks: PASSED")

    except Exception as e:
        print(f"✗ Performance Benchmarks: FAILED - {e}")
        pytest.fail(str(e))


def test_privacy_enhancements():
    """Test enhanced privacy features."""
    print("Testing Privacy Enhancements...")

    try:
        from maif.privacy import PrivacyEngine, EncryptionMode

        engine = PrivacyEngine()

        test_data = b"This is sensitive test data"

        # Test AES-GCM
        encrypted, metadata = engine.encrypt_data(
            test_data, "test_block", EncryptionMode.AES_GCM
        )
        assert len(encrypted) > 0
        assert "algorithm" in metadata

        decrypted = engine.decrypt_data(encrypted, "test_block", metadata)
        assert decrypted == test_data

        # Test anonymization
        sensitive_text = "John Doe works at john.doe@company.com"
        anonymized = engine.anonymize_data(sensitive_text, "test_context")
        assert "john.doe@company.com" not in anonymized

        print("✓ Privacy Enhancements: PASSED")

    except Exception as e:
        print(f"✗ Privacy Enhancements: FAILED - {e}")
        pytest.fail(str(e))


class TestEnhancedMAIFFeatures:
    """Test class for enhanced features."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multi_block_file(self):
        """Test creating file with multiple block types."""
        encoder = MAIFEncoder(self.maif_path, agent_id="multi-block")

        encoder.add_text_block("Text content")
        encoder.add_binary_block(b"Binary content", BlockType.BINARY)
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
        encoder.add_binary_block(b"Video data", BlockType.VIDEO)

        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert len(decoder.blocks) == 4

        types = [b.header.block_type for b in decoder.blocks]
        assert BlockType.TEXT in types
        assert BlockType.BINARY in types
        assert BlockType.EMBEDDINGS in types
        assert BlockType.VIDEO in types

    def test_provenance_tracking(self):
        """Test provenance chain tracking."""
        encoder = MAIFEncoder(self.maif_path, agent_id="provenance-test")

        encoder.add_text_block("Block 1")
        encoder.add_text_block("Block 2")
        encoder.add_text_block("Block 3")

        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        provenance = decoder.get_provenance()

        # Should have genesis + 3 adds + finalize = 5+ entries
        assert len(provenance) >= 5

        # Verify chain links
        for i in range(1, len(provenance)):
            assert provenance[i].previous_entry_hash == provenance[i - 1].entry_hash

    def test_integrity_verification(self):
        """Test integrity verification."""
        encoder = MAIFEncoder(self.maif_path, agent_id="integrity-test")
        encoder.add_text_block("Test content")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_tamper_detection(self):
        """Test tamper detection."""
        encoder = MAIFEncoder(self.maif_path, agent_id="tamper-test")
        encoder.add_text_block("Original content")
        encoder.finalize()

        # Tamper with file
        with open(self.maif_path, "r+b") as f:
            f.seek(500)
            f.write(b"TAMPERED")

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is False or len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
