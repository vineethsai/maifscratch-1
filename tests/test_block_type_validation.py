"""
Tests to ensure block type validation and FourCC compliance.
These tests prevent file format changes that could break compatibility.
"""

import pytest
import tempfile
import os
from maif.core import MAIFEncoder
from maif.block_types import BlockType, BlockHeader, BlockValidator
from maif.validation import MAIFValidator


class TestBlockTypeFourCCCompliance:
    """Test that all block types comply with FourCC (4-character) requirements."""
    
    def test_all_block_types_are_fourcc_compliant(self):
        """Ensure all BlockType enum values are exactly 4 characters."""
        for block_type in BlockType:
            assert len(block_type.value) == 4, f"Block type {block_type.name} value '{block_type.value}' is not 4 characters"
            # Ensure they are ASCII printable characters
            assert block_type.value.isascii(), f"Block type {block_type.name} value '{block_type.value}' contains non-ASCII characters"
    
    def test_block_type_mapping_compliance(self):
        """Ensure all mapped block types resolve to valid FourCC codes."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        for original_type, mapped_type in encoder.BLOCK_TYPE_MAPPING.items():
            assert len(mapped_type) == 4, f"Mapped type '{mapped_type}' for '{original_type}' is not 4 characters"
            assert mapped_type.isascii(), f"Mapped type '{mapped_type}' for '{original_type}' contains non-ASCII characters"
    
    def test_block_header_validation_enforces_fourcc(self):
        """Test that block header validation properly enforces FourCC requirements."""
        # Valid 4-character type should pass
        valid_header = BlockHeader(size=100, type="VDAT", version=1, flags=0)
        errors = BlockValidator.validate_block_header(valid_header)
        fourcc_errors = [e for e in errors if "Block type must be 4 characters" in e]
        assert len(fourcc_errors) == 0, f"Valid FourCC 'VDAT' should not produce FourCC errors: {fourcc_errors}"
        
        # Invalid length types should fail
        invalid_headers = [
            BlockHeader(size=100, type="", version=1, flags=0),  # Empty
            BlockHeader(size=100, type="ABC", version=1, flags=0),  # Too short
            BlockHeader(size=100, type="ABCDE", version=1, flags=0),  # Too long
            BlockHeader(size=100, type="video_data", version=1, flags=0),  # Way too long
        ]
        
        for header in invalid_headers:
            errors = BlockValidator.validate_block_header(header)
            fourcc_errors = [e for e in errors if "Block type must be 4 characters" in e]
            assert len(fourcc_errors) > 0, f"Invalid type '{header.type}' should produce FourCC validation error"


class TestBlockTypeMapping:
    """Test that block type mapping works correctly and maintains compatibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = MAIFEncoder(agent_id="test_agent")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_common_block_types_map_correctly(self):
        """Test that commonly used block types map to correct FourCC codes."""
        expected_mappings = {
            "text": "TEXT",
            "text_data": "TEXT", 
            "binary": "BDAT",
            "binary_data": "BDAT",
            "data": "BDAT",
            "embedding": "EMBD",
            "embeddings": "EMBD",
            "video_data": "VDAT",
            "audio_data": "AUDI",
            "image_data": "IDAT",
            "cross_modal": "XMOD",
            "semantic_binding": "SBND",
            "compressed_embeddings": "CEMB",
            "knowledge_graph": "KGRF",
            "security": "SECU",
            "provenance": "PROV",
            "access_control": "ACLS",
            "lifecycle": "LIFE",
        }
        
        for original_type, expected_fourcc in expected_mappings.items():
            mapped_type = self.encoder.BLOCK_TYPE_MAPPING.get(original_type)
            assert mapped_type == expected_fourcc, f"Block type '{original_type}' should map to '{expected_fourcc}', got '{mapped_type}'"
    
    def test_video_data_block_creation_uses_correct_fourcc(self):
        """Test that video_data blocks use the correct VDAT FourCC."""
        # Add a video block
        block_id = self.encoder.add_video_block(b"fake video data", metadata={"test": True})
        
        # Find the block in the encoder
        video_block = None
        for block in self.encoder.blocks:
            if block.block_id == block_id:
                video_block = block
                break
        
        assert video_block is not None, "Video block should be created"
        assert video_block.block_type == "VDAT", "Block type should be 'VDAT'"
        
        # Build the MAIF file and check the actual header
        maif_path = os.path.join(self.temp_dir, "test_video.maif")
        manifest_path = os.path.join(self.temp_dir, "test_video_manifest.json")
        
        self.encoder.build_maif(maif_path, manifest_path)
        
        # Read the file and check the header
        with open(maif_path, 'rb') as f:
            # File starts directly with the first block (no file header)
            f.seek(0)
            header_data = f.read(32)  # Read block header
            
            # Parse the block header: [size(4)][type(4)][version(4)][flags(4)][uuid(16)]
            import struct
            if len(header_data) >= 16:
                size, type_bytes, version, flags = struct.unpack('>I4sII', header_data[:16])
                block_type = type_bytes.decode('ascii').rstrip('\0')
                
                assert block_type == "VDAT", f"Video block should use VDAT FourCC, got '{block_type}'"
            else:
                assert False, f"Not enough header data: got {len(header_data)} bytes, expected at least 16"
    
    def test_data_block_creation_uses_correct_fourcc(self):
        """Test that 'data' blocks use the correct BDAT FourCC."""
        # Add a data block
        block_id = self.encoder.add_binary_block(b"test binary data", "data", metadata={"test": True})
        
        # Build the MAIF file and check the actual header
        maif_path = os.path.join(self.temp_dir, "test_data.maif")
        manifest_path = os.path.join(self.temp_dir, "test_data_manifest.json")
        
        self.encoder.build_maif(maif_path, manifest_path)
        
        # Read the file and check the header
        with open(maif_path, 'rb') as f:
            # File starts directly with the first block (no file header)
            f.seek(0)
            header_data = f.read(32)  # Read block header
            
            # Parse the block header: [size(4)][type(4)][version(4)][flags(4)][uuid(16)]
            import struct
            if len(header_data) >= 16:
                size, type_bytes, version, flags = struct.unpack('>I4sII', header_data[:16])
                block_type = type_bytes.decode('ascii').rstrip('\0')
                
                assert block_type == "BDAT", f"Data block should use BDAT FourCC, got '{block_type}'"
            else:
                assert False, f"Not enough header data: got {len(header_data)} bytes, expected at least 16"


class TestFileFormatStability:
    """Test that file format changes don't break existing functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_all_block_types_create_valid_files(self):
        """Test that all supported block types create valid MAIF files."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # Test each mapped block type
        test_data = {
            "text": b"Hello, world!",
            "binary": b"binary data",
            "data": b"generic data",
            "embeddings": [[0.1, 0.2, 0.3]],  # Special case for embeddings
            "video_data": b"fake video data",
        }
        
        for block_type, data in test_data.items():
            if block_type == "embeddings":
                encoder.add_embeddings_block(data, metadata={"type": block_type})
            elif block_type in ["text"]:
                encoder.add_text_block(data.decode('utf-8'), metadata={"type": block_type})
            elif block_type == "video_data":
                encoder.add_video_block(data, metadata={"type": block_type})
            else:
                encoder.add_binary_block(data, block_type, metadata={"type": block_type})
        
        # Build the MAIF file
        maif_path = os.path.join(self.temp_dir, "test_all_types.maif")
        manifest_path = os.path.join(self.temp_dir, "test_all_types_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        # Validate the file
        validator = MAIFValidator()
        result = validator.validate_file(maif_path, manifest_path)
        
        assert result.is_valid, f"MAIF file with all block types should be valid. Errors: {result.errors}"
        assert len(result.errors) == 0, f"No validation errors expected. Got: {result.errors}"
    
    def test_fourcc_validation_prevents_invalid_types(self):
        """Test that the validation system prevents invalid FourCC types."""
        # This test ensures that our fix doesn't allow invalid types to slip through
        
        # Create a header with an invalid type directly
        invalid_header = BlockHeader(size=100, type="TOOLONG", version=1, flags=0)
        errors = BlockValidator.validate_block_header(invalid_header)
        
        # Should have FourCC validation error
        fourcc_errors = [e for e in errors if "Block type must be 4 characters" in e]
        assert len(fourcc_errors) > 0, "Invalid FourCC type should be caught by validation"
    
    def test_backward_compatibility_maintained(self):
        """Test that existing block type names still work."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # These are the original block types that should still work
        legacy_types = ["text", "binary", "embeddings"]
        
        for block_type in legacy_types:
            if block_type == "text":
                block_id = encoder.add_text_block("test text", metadata={"legacy": True})
            elif block_type == "embeddings":
                block_id = encoder.add_embeddings_block([[0.1, 0.2]], metadata={"legacy": True})
            else:
                block_id = encoder.add_binary_block(b"test data", block_type, metadata={"legacy": True})
            
            assert block_id is not None, f"Legacy block type '{block_type}' should still work"
        
        # Build and validate
        maif_path = os.path.join(self.temp_dir, "test_legacy.maif")
        manifest_path = os.path.join(self.temp_dir, "test_legacy_manifest.json")
        
        encoder.build_maif(maif_path, manifest_path)
        
        validator = MAIFValidator()
        result = validator.validate_file(maif_path, manifest_path)
        
        assert result.is_valid, f"Legacy block types should create valid files. Errors: {result.errors}"


class TestFourCCErrorHandling:
    """Test that FourCC errors are handled gracefully."""
    
    def test_long_block_type_handling(self):
        """Test that long block types are handled without crashing."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # This should work because our fix handles long types
        try:
            # The encoder should handle this gracefully by truncating or mapping
            block_id = encoder.add_binary_block(b"test data", "very_long_block_type_name", metadata={"test": True})
            assert block_id is not None, "Long block type should be handled gracefully"
        except Exception as e:
            pytest.fail(f"Long block type should not cause an exception: {e}")
    
    def test_empty_block_type_handling(self):
        """Test that empty block types are handled properly."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # Empty block type should raise a ValueError
        with pytest.raises(ValueError, match="Block type cannot be empty"):
            encoder.add_binary_block(b"test data", "", metadata={"test": True})
    
    def test_none_block_type_handling(self):
        """Test that None block types are handled properly."""
        encoder = MAIFEncoder(agent_id="test_agent")
        
        # None block type should raise a ValueError
        with pytest.raises((ValueError, TypeError)):
            encoder.add_binary_block(b"test data", None, metadata={"test": True})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])