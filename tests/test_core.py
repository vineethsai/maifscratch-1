"""
Tests for MAIF Core Module - Secure Format v3

Tests the new self-contained MAIF format with Ed25519 signatures.
"""

import pytest
import os
import tempfile
import struct

from maif import (
    MAIFEncoder,
    MAIFDecoder,
    MAIFParser,
    BlockType,
    BlockFlags,
    FileFlags,
    MAIFBlock,
    MAIFVersion,
    MAIFHeader,
    ProvenanceEntry,
    SecureBlock,
    verify_maif,
    create_maif,
    quick_create,
    quick_verify,
    quick_read,
    MAGIC_HEADER,
    MAGIC_FOOTER,
)


class TestMAIFEncoder:
    """Tests for MAIFEncoder (SecureMAIFWriter)."""

    def test_create_simple_file(self, tmp_path):
        """Test creating a simple MAIF file."""
        output = tmp_path / "test.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Hello, world!")
        encoder.finalize()

        assert output.exists()
        assert output.stat().st_size > 0

    def test_create_with_multiple_blocks(self, tmp_path):
        """Test creating file with multiple blocks."""
        output = tmp_path / "multi.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("First block")
        encoder.add_text_block("Second block")
        encoder.add_text_block("Third block")
        encoder.finalize()

        # Verify we can read all blocks
        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert len(decoder.blocks) == 3

    def test_create_with_embeddings(self, tmp_path):
        """Test adding embeddings block."""
        output = tmp_path / "embed.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        encoder.add_embeddings_block(embeddings)
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert len(decoder.blocks) == 1
        assert decoder.blocks[0].header.block_type == BlockType.EMBEDDINGS

    def test_create_with_metadata(self, tmp_path):
        """Test adding blocks with metadata."""
        output = tmp_path / "meta.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block(
            "Text with metadata",
            metadata={"source": "test", "language": "en", "version": 1},
        )
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.blocks[0].metadata is not None
        assert decoder.blocks[0].metadata.get("source") == "test"

    def test_file_header_magic(self, tmp_path):
        """Test that file starts with MAIF magic."""
        output = tmp_path / "magic.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Test")
        encoder.finalize()

        with open(output, "rb") as f:
            magic = f.read(4)

        assert magic == MAGIC_HEADER

    def test_blocks_are_signed(self, tmp_path):
        """Test that blocks are signed on creation."""
        output = tmp_path / "signed.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Signed content")
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()

        # Check SIGNED flag
        assert decoder.blocks[0].header.flags & BlockFlags.SIGNED

    def test_cannot_add_after_finalize(self, tmp_path):
        """Test that adding blocks after finalize raises error."""
        output = tmp_path / "final.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Before finalize")
        encoder.finalize()

        with pytest.raises(RuntimeError):
            encoder.add_text_block("After finalize")

    def test_provenance_entries_created(self, tmp_path):
        """Test that provenance entries are created."""
        output = tmp_path / "prov.maif"

        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Block 1")
        encoder.add_text_block("Block 2")
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()

        # Should have: genesis + 2 add_text_block + finalize = 4
        assert len(decoder.provenance) >= 4


class TestMAIFDecoder:
    """Tests for MAIFDecoder (SecureMAIFReader)."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample MAIF file for testing."""
        output = tmp_path / "sample.maif"
        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Test content one")
        encoder.add_text_block("Test content two")
        encoder.finalize()
        return str(output)

    def test_load_file(self, sample_file):
        """Test loading a MAIF file."""
        decoder = MAIFDecoder(sample_file)
        result = decoder.load()
        assert result is True

    def test_get_blocks(self, sample_file):
        """Test getting blocks from file."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        blocks = decoder.get_blocks()
        assert len(blocks) == 2

    def test_get_text_content(self, sample_file):
        """Test reading text content."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        text = decoder.get_text_content(0)
        assert text == "Test content one"

        text = decoder.get_text_content(1)
        assert text == "Test content two"

    def test_get_provenance(self, sample_file):
        """Test getting provenance chain."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        provenance = decoder.get_provenance()
        assert len(provenance) > 0
        assert provenance[0].action == "genesis"

    def test_get_file_info(self, sample_file):
        """Test getting file info."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        info = decoder.get_file_info()
        assert info["version"] == "2.1"
        assert info["block_count"] == 2
        assert info["is_signed"] is True
        assert info["is_finalized"] is True

    def test_get_security_info(self, sample_file):
        """Test getting security info."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        security = decoder.get_security_info()
        assert "public_key" in security
        assert security.get("key_algorithm") == "Ed25519"


class TestIntegrity:
    """Tests for file integrity verification."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample MAIF file."""
        output = tmp_path / "integrity.maif"
        encoder = MAIFEncoder(str(output), agent_id="test-agent")
        encoder.add_text_block("Integrity test")
        encoder.finalize()
        return str(output)

    def test_verify_valid_file(self, sample_file):
        """Test that valid file passes verification."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True
        assert len(errors) == 0

    def test_verify_maif_function(self, sample_file):
        """Test verify_maif convenience function."""
        is_valid, report = verify_maif(sample_file)
        assert is_valid is True

    def test_detect_tampering(self, sample_file):
        """Test that tampering is detected."""
        # Tamper with the file
        with open(sample_file, "r+b") as f:
            f.seek(500)  # Seek to data area
            f.write(b"TAMPERED")

        decoder = MAIFDecoder(sample_file)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is False or decoder.is_tampered()

    def test_is_tampered_method(self, sample_file):
        """Test is_tampered method on valid file."""
        decoder = MAIFDecoder(sample_file)
        decoder.load()

        assert decoder.is_tampered() is False


class TestMAIFParser:
    """Tests for MAIFParser high-level interface."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample file."""
        output = tmp_path / "parser.maif"
        encoder = MAIFEncoder(str(output), agent_id="parser-test")
        encoder.add_text_block("Parser test content")
        encoder.finalize()
        return str(output)

    def test_parser_load(self, sample_file):
        """Test loading with parser."""
        parser = MAIFParser(sample_file)
        result = parser.load()
        assert result is True

    def test_parser_blocks(self, sample_file):
        """Test accessing blocks through parser."""
        parser = MAIFParser(sample_file)
        parser.load()

        assert len(parser.blocks) == 1
        assert isinstance(parser.blocks[0], MAIFBlock)

    def test_parser_provenance(self, sample_file):
        """Test accessing provenance through parser."""
        parser = MAIFParser(sample_file)
        parser.load()

        assert len(parser.provenance) > 0

    def test_parser_verify(self, sample_file):
        """Test verification through parser."""
        parser = MAIFParser(sample_file)
        parser.load()

        is_valid, errors = parser.verify()
        assert is_valid is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_quick_create(self, tmp_path):
        """Test quick_create function."""
        output = tmp_path / "quick.maif"

        result = quick_create(
            str(output), texts=["Hello", "World"], agent_id="quick-test"
        )

        assert os.path.exists(result)

    def test_quick_verify(self, tmp_path):
        """Test quick_verify function."""
        output = tmp_path / "verify.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_text_block("Test")
        encoder.finalize()

        assert quick_verify(str(output)) is True

    def test_quick_read(self, tmp_path):
        """Test quick_read function."""
        output = tmp_path / "read.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_text_block("Read test")
        encoder.finalize()

        data = quick_read(str(output))
        assert "blocks" in data
        assert "provenance" in data


class TestDataClasses:
    """Tests for data classes."""

    def test_maif_block_creation(self):
        """Test MAIFBlock dataclass."""
        block = MAIFBlock(block_type="TEXT", size=100, hash_value="abc123")

        assert block.block_type == "TEXT"
        assert block.size == 100
        assert block.block_id is not None  # Auto-generated

    def test_maif_block_to_dict(self):
        """Test MAIFBlock.to_dict()."""
        block = MAIFBlock(block_type="TEXT", size=100, hash_value="abc123")

        d = block.to_dict()
        assert d["type"] == "TEXT"
        assert d["size"] == 100

    def test_maif_version_creation(self):
        """Test MAIFVersion dataclass."""
        version = MAIFVersion(
            version=1,
            timestamp=1234567890.0,
            agent_id="test",
            operation="create",
            block_hash="abc123",
        )

        assert version.version == 1
        assert version.version_number == 1  # Alias

    def test_maif_header_creation(self):
        """Test MAIFHeader dataclass."""
        header = MAIFHeader(version="2.1", creator_id="test")

        assert header.version == "2.1"
        assert header.created_timestamp is not None


class TestBlockTypes:
    """Tests for different block types."""

    def test_text_block(self, tmp_path):
        """Test TEXT block type."""
        output = tmp_path / "text.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_text_block("Hello")
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.blocks[0].header.block_type == BlockType.TEXT

    def test_embeddings_block(self, tmp_path):
        """Test EMBEDDINGS block type."""
        output = tmp_path / "embed.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.blocks[0].header.block_type == BlockType.EMBEDDINGS

    def test_binary_block(self, tmp_path):
        """Test BINARY block type."""
        output = tmp_path / "binary.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_binary_block(b"binary data")
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.blocks[0].header.block_type == BlockType.BINARY


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text_block(self, tmp_path):
        """Test adding empty text block."""
        output = tmp_path / "empty.maif"

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_text_block("")
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.get_text_content(0) == ""

    def test_unicode_text(self, tmp_path):
        """Test Unicode text content."""
        output = tmp_path / "unicode.maif"

        text = "Hello 世界 🌍 مرحبا"
        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_text_block(text)
        encoder.finalize()

        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert decoder.get_text_content(0) == text

    def test_large_embeddings(self, tmp_path):
        """Test large embeddings block."""
        output = tmp_path / "large_embed.maif"

        # 100 vectors of 384 dimensions
        embeddings = [[float(i) / 1000 for i in range(384)] for _ in range(100)]

        encoder = MAIFEncoder(str(output), agent_id="test")
        encoder.add_embeddings_block(embeddings)
        encoder.finalize()

        # Should not raise and should be readable
        decoder = MAIFDecoder(str(output))
        decoder.load()
        assert len(decoder.blocks) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
