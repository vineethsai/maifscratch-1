"""
Comprehensive tests for MAIF v3 binary format functionality.
"""

import pytest
import tempfile
import os
import struct
import hashlib
import shutil

from maif import MAIFEncoder, MAIFDecoder, BlockType


class TestBinaryFormat:
    """Test MAIF v3 binary format structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_binary_header_format(self):
        """Test binary header format with MAIF magic number."""
        encoder = MAIFEncoder(self.maif_path, agent_id="binary_test")
        encoder.add_text_block("Test content", metadata={"test": True})
        encoder.finalize()

        # Read binary header
        with open(self.maif_path, "rb") as f:
            header_data = f.read(32)

        # Should have MAIF magic number
        assert header_data[:4] == b"MAIF"

    def test_binary_block_structure(self):
        """Test binary block structure."""
        encoder = MAIFEncoder(self.maif_path, agent_id="binary_test")
        encoder.add_text_block("Block structure test", metadata={"block_test": True})
        encoder.add_binary_block(
            b"Binary data test", BlockType.BINARY, metadata={"binary_test": True}
        )
        encoder.finalize()

        # Verify blocks can be read back
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert len(decoder.blocks) == 2
        assert decoder.blocks[0].header.block_type == BlockType.TEXT
        assert decoder.blocks[1].header.block_type == BlockType.BINARY

    def test_block_data_integrity(self):
        """Test that block data maintains integrity."""
        original_text = "Test content for integrity check"
        original_binary = b"Binary data 12345"

        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block(original_text)
        encoder.add_binary_block(original_binary, BlockType.BINARY)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.get_text_content(0) == original_text
        assert decoder.blocks[1].data == original_binary

    def test_file_integrity_verification(self):
        """Test file integrity verification."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content for verification")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_tamper_detection(self):
        """Test that tampering is detected."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content to tamper")
        encoder.finalize()

        # Tamper with the file
        with open(self.maif_path, "r+b") as f:
            f.seek(500)
            f.write(b"TAMPERED")

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        # Should detect tampering
        assert is_valid is False or len(errors) > 0


class TestBinaryBlockTypes:
    """Test different binary block types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_text_block(self):
        """Test TEXT block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Test text content")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.TEXT

    def test_embeddings_block(self):
        """Test EMBEDDINGS block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_embeddings_block([[0.1, 0.2, 0.3, 0.4]])
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.EMBEDDINGS

    def test_binary_block(self):
        """Test BINARY block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"binary data", BlockType.BINARY)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.BINARY

    def test_video_block(self):
        """Test VIDEO block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"video content", BlockType.VIDEO)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.VIDEO

    def test_audio_block(self):
        """Test AUDIO block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"audio content", BlockType.AUDIO)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.AUDIO

    def test_image_block(self):
        """Test IMAGE block type."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"image content", BlockType.IMAGE)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.IMAGE


class TestBlockMetadata:
    """Test block metadata handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_metadata(self):
        """Test simple metadata storage."""
        metadata = {"key": "value", "number": 42}

        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content", metadata=metadata)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].metadata["key"] == "value"
        assert decoder.blocks[0].metadata["number"] == 42

    def test_nested_metadata(self):
        """Test nested metadata structures."""
        metadata = {"nested": {"inner": "value"}, "array": [1, 2, 3]}

        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content", metadata=metadata)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].metadata["nested"]["inner"] == "value"
        assert decoder.blocks[0].metadata["array"] == [1, 2, 3]

    def test_no_metadata(self):
        """Test blocks without metadata."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content without metadata")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Should still work without metadata
        assert decoder.blocks[0].data is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
