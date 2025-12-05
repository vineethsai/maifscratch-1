"""
Tests to ensure block type validation for v3 format.
"""

import pytest
import tempfile
import os
import shutil
from maif import MAIFEncoder, MAIFDecoder, BlockType


class TestBlockTypeFormat:
    """Test that all block types are properly formatted."""

    def test_all_block_types_defined(self):
        """Ensure all expected BlockType enum values exist."""
        expected_types = [
            "TEXT",
            "EMBEDDINGS",
            "IMAGE",
            "AUDIO",
            "VIDEO",
            "KNOWLEDGE",
            "BINARY",
            "METADATA",
        ]

        for type_name in expected_types:
            assert hasattr(BlockType, type_name), f"BlockType.{type_name} should exist"

    def test_block_type_values_are_integers(self):
        """Ensure all BlockType values are integers (FourCC-like)."""
        for block_type in BlockType:
            assert isinstance(block_type.value, int), (
                f"BlockType.{block_type.name} should have int value"
            )


class TestBlockTypeMapping:
    """Test that block types work correctly in MAIF files."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_text_block_type(self):
        """Test TEXT block type creation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Test content")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.TEXT

    def test_embeddings_block_type(self):
        """Test EMBEDDINGS block type creation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.EMBEDDINGS

    def test_binary_block_type(self):
        """Test BINARY block type creation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"binary data", BlockType.BINARY)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.BINARY

    def test_video_block_type(self):
        """Test VIDEO block type creation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_binary_block(b"video data", BlockType.VIDEO)
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        assert decoder.blocks[0].header.block_type == BlockType.VIDEO


class TestBlockValidation:
    """Test block validation in v3 format."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_block_integrity(self):
        """Test that blocks maintain integrity after save/load."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Test content", metadata={"key": "value"})
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True

    def test_multiple_block_types_in_same_file(self):
        """Test file with multiple block types."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Text content")
        encoder.add_binary_block(b"binary", BlockType.BINARY)
        encoder.add_embeddings_block([[0.1, 0.2]])
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        types = [b.header.block_type for b in decoder.blocks]
        assert BlockType.TEXT in types
        assert BlockType.BINARY in types
        assert BlockType.EMBEDDINGS in types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
