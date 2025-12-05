#!/usr/bin/env python3
"""
Working MAIF Functionality Tests
Tests the actual working features of the MAIF v3 system.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path

# Import MAIF modules
from maif import MAIFEncoder, MAIFDecoder, BlockType
from maif.privacy import PrivacyEngine
from maif.compression import MAIFCompressor, CompressionAlgorithm
from maif.security import MAIFSigner
from maif.metadata import MAIFMetadataManager
from maif.validation import MAIFValidator


class TestWorkingMAIFCore:
    """Test core MAIF v3 functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_maif_encoder_creation(self):
        """Test MAIF encoder creation and basic operations."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test_agent")
        assert encoder.agent_id == "test_agent"
        assert len(encoder.blocks) == 0

        # Add text block
        encoder.add_text_block("Hello MAIF!", metadata={"test": True})
        assert len(encoder.blocks) == 1

        # Add binary block
        encoder.add_binary_block(b"test_data", BlockType.BINARY)
        assert len(encoder.blocks) == 2

    def test_maif_file_creation_and_reading(self):
        """Test complete MAIF file creation and reading cycle."""
        # Create MAIF file
        encoder = MAIFEncoder(self.maif_path, agent_id="test_agent")
        encoder.add_text_block("Hello MAIF!", metadata={"test": True})
        encoder.add_binary_block(b"test_data", BlockType.BINARY)
        encoder.finalize()

        # Verify file exists
        assert os.path.exists(self.maif_path)

        # Read MAIF file
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert len(decoder.blocks) == 2

        # Read text content
        text = decoder.get_text_content(0)
        assert text == "Hello MAIF!"


class TestWorkingPrivacy:
    """Test privacy functionality that we know works."""

    def test_privacy_engine_creation(self):
        """Test privacy engine creation."""
        privacy = PrivacyEngine()
        assert privacy is not None

    def test_data_encryption_decryption(self):
        """Test data encryption and decryption."""
        privacy = PrivacyEngine()

        test_data = b"Sensitive data to encrypt"

        encrypted, metadata = privacy.encrypt_data(test_data, block_id="test_block")
        assert encrypted != test_data

        # Pass metadata from encryption to decryption
        decrypted = privacy.decrypt_data(
            encrypted, block_id="test_block", metadata=metadata
        )
        assert decrypted == test_data


class TestWorkingCompression:
    """Test compression functionality that we know works."""

    def test_compressor_creation(self):
        """Test compressor creation."""
        compressor = MAIFCompressor()
        assert compressor is not None

    def test_data_compression_decompression(self):
        """Test data compression and decompression."""
        compressor = MAIFCompressor()

        test_data = b"This is test data that should be compressed." * 100

        compressed = compressor.compress(test_data, CompressionAlgorithm.ZLIB)
        assert len(compressed) < len(test_data)

        decompressed = compressor.decompress(compressed, CompressionAlgorithm.ZLIB)
        assert decompressed == test_data


class TestWorkingSecurity:
    """Test security functionality with Ed25519."""

    def test_signer_creation(self):
        """Test signer creation."""
        signer = MAIFSigner(agent_id="test_agent")
        assert signer is not None
        assert signer.agent_id == "test_agent"

    def test_public_key_generation(self):
        """Test public key generation."""
        signer = MAIFSigner(agent_id="test_agent")
        public_key = signer.get_public_key_hex()

        assert public_key is not None
        assert len(public_key) == 64  # Ed25519 public key hex

    def test_data_signing_verification(self):
        """Test data signing and verification."""
        signer = MAIFSigner(agent_id="test_agent")

        test_data = b"Data to sign"
        signature = signer.sign_data(test_data)

        assert signature is not None
        assert len(signature) == 64  # Ed25519 signature

        is_valid = signer.verify_signature(test_data, signature)
        assert is_valid is True

    def test_multiple_signers(self):
        """Test multiple signers work independently."""
        signer1 = MAIFSigner(agent_id="agent1")
        signer2 = MAIFSigner(agent_id="agent2")

        test_data = b"Shared data"

        sig1 = signer1.sign_data(test_data)
        sig2 = signer2.sign_data(test_data)

        # Different keys produce different signatures
        assert sig1 != sig2

        # Each can verify their own
        assert signer1.verify_signature(test_data, sig1) is True
        assert signer2.verify_signature(test_data, sig2) is True


class TestWorkingMetadata:
    """Test metadata functionality that we know works."""

    def test_metadata_manager_creation(self):
        """Test metadata manager creation."""
        manager = MAIFMetadataManager()
        assert manager is not None

    def test_set_get_metadata(self):
        """Test setting and getting metadata."""
        manager = MAIFMetadataManager()

        # Use block metadata
        manager.add_block_metadata("block1", {"key1": "value1"})
        manager.add_block_metadata("block2", {"key2": {"nested": "value"}})

        summary = manager.get_metadata_summary()
        assert summary is not None


class TestIntegrationWorkingFeatures:
    """Integration tests for working features."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_full_maif_workflow(self):
        """Test complete MAIF workflow with all features."""
        # Create with signing
        encoder = MAIFEncoder(self.maif_path, agent_id="workflow_agent")

        # Add multiple block types
        encoder.add_text_block("Text block 1", metadata={"index": 1})
        encoder.add_text_block("Text block 2", metadata={"index": 2})
        encoder.add_embeddings_block([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        # Finalize
        encoder.finalize()

        # Verify and read
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Verify integrity
        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True

        # Check content
        assert len(decoder.blocks) == 3
        assert decoder.get_text_content(0) == "Text block 1"
        assert decoder.get_text_content(1) == "Text block 2"

        # Check provenance
        assert len(decoder.provenance) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
