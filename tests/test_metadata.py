"""
Comprehensive tests for MAIF metadata functionality.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from maif.metadata import (
    MAIFVersion,
    ContentType,
    CompressionType,
    MAIFHeader,
    BlockMetadata,
    ProvenanceRecord,
    MAIFMetadataManager,
)


class TestMAIFVersion:
    """Test MAIFVersion enum."""

    def test_maif_version_values(self):
        """Test MAIFVersion enum values."""
        assert MAIFVersion.V1_0.value == "1.0"
        assert MAIFVersion.V2_0.value == "2.0"


class TestContentType:
    """Test ContentType enum."""

    def test_content_type_values(self):
        """Test ContentType enum values."""
        assert ContentType.TEXT.value == "text"
        assert ContentType.BINARY.value == "binary"
        assert ContentType.EMBEDDINGS.value == "embeddings"
        assert ContentType.KNOWLEDGE_GRAPH.value == "knowledge_graph"
        assert ContentType.CROSS_MODAL.value == "cross_modal"


class TestCompressionType:
    """Test CompressionType enum."""

    def test_compression_type_values(self):
        """Test CompressionType enum values."""
        assert CompressionType.NONE.value == "none"
        assert CompressionType.ZLIB.value == "zlib"
        assert CompressionType.GZIP.value == "gzip"
        assert CompressionType.LZMA.value == "lzma"


class TestMAIFHeader:
    """Test MAIFHeader data structure."""

    def test_maif_header_creation(self):
        """Test basic MAIFHeader creation."""
        header = MAIFHeader(
            version="2.0",
            creator_agent="test_agent",
            timestamp=1234567890.0,
            file_id="test_file_123",
            compression=CompressionType.ZLIB,
            encryption_enabled=True,
        )

        assert header.version == "2.0"
        assert header.creator_agent == "test_agent"
        assert header.timestamp == 1234567890.0
        assert header.file_id == "test_file_123"
        assert header.compression == CompressionType.ZLIB
        assert header.encryption_enabled is True

    def test_maif_header_post_init(self):
        """Test MAIFHeader post-initialization validation."""
        import time

        # Test with valid timestamp
        header = MAIFHeader(
            version="2.0", creator_agent="test_agent", timestamp=time.time()
        )

        assert header.timestamp > 0

        # Test with invalid timestamp (should be corrected)
        header_invalid = MAIFHeader(
            version="2.0",
            creator_agent="test_agent",
            timestamp=-1.0,  # Invalid timestamp
        )

        # Should be corrected to current time
        assert header_invalid.timestamp > 0


class TestBlockMetadata:
    """Test BlockMetadata data structure."""

    def test_block_metadata_creation(self):
        """Test basic BlockMetadata creation."""
        metadata = BlockMetadata(
            block_id="block_123",
            content_type=ContentType.TEXT,
            size=1024,
            hash="abc123def456",
            compression=CompressionType.ZLIB,
            encrypted=True,
            created_at=1234567890.0,
        )

        assert metadata.block_id == "block_123"
        assert metadata.content_type == ContentType.TEXT
        assert metadata.size == 1024
        assert metadata.hash == "abc123def456"
        assert metadata.compression == CompressionType.ZLIB
        assert metadata.encrypted is True
        assert metadata.created_at == 1234567890.0

    def test_block_metadata_post_init(self):
        """Test BlockMetadata post-initialization validation."""
        # Test with valid size
        metadata = BlockMetadata(
            block_id="block_123", content_type=ContentType.TEXT, size=1024
        )

        assert metadata.size == 1024

        # Test with invalid size (should be corrected)
        metadata_invalid = BlockMetadata(
            block_id="block_123",
            content_type=ContentType.TEXT,
            size=-100,  # Invalid size
        )

        # Should be corrected to 0
        assert metadata_invalid.size == 0


class TestProvenanceRecord:
    """Test ProvenanceRecord data structure."""

    def test_provenance_record_creation(self):
        """Test basic ProvenanceRecord creation."""
        record = ProvenanceRecord(
            operation_type="create",
            agent_id="test_agent",
            timestamp=1234567890.0,
            block_id="block_123",
            previous_hash="prev_hash_456",
            operation_hash="op_hash_789",
        )

        assert record.operation_type == "create"
        assert record.agent_id == "test_agent"
        assert record.timestamp == 1234567890.0
        assert record.block_id == "block_123"
        assert record.previous_hash == "prev_hash_456"
        assert record.operation_hash == "op_hash_789"

    def test_provenance_record_post_init(self):
        """Test ProvenanceRecord post-initialization validation."""
        import time

        # Test with valid timestamp
        record = ProvenanceRecord(
            operation_type="create", agent_id="test_agent", timestamp=time.time()
        )

        assert record.timestamp > 0

        # Test with invalid timestamp (should be corrected)
        record_invalid = ProvenanceRecord(
            operation_type="create",
            agent_id="test_agent",
            timestamp=-1.0,  # Invalid timestamp
        )

        # Should be corrected to current time
        assert record_invalid.timestamp > 0


class TestMAIFMetadataManager:
    """Test MAIFMetadataManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metadata_manager = MAIFMetadataManager(version="2.0")

    def test_metadata_manager_initialization(self):
        """Test MAIFMetadataManager initialization."""
        assert self.metadata_manager.version == "2.0"
        assert self.metadata_manager.blocks == {}
        assert self.metadata_manager.provenance == []
        assert self.metadata_manager.dependencies == {}
        assert self.metadata_manager.custom_schemas == {}

    def test_create_header(self):
        """Test header creation."""
        header = self.metadata_manager.create_header(
            creator_agent="test_agent",
            file_id="test_file_123",
            compression=CompressionType.ZLIB,
        )

        assert isinstance(header, MAIFHeader)
        assert header.version == "2.0"
        assert header.creator_agent == "test_agent"
        assert header.file_id == "test_file_123"
        assert header.compression == CompressionType.ZLIB
        assert header.timestamp > 0

    def test_add_block_metadata(self):
        """Test adding block metadata."""
        result = self.metadata_manager.add_block_metadata(
            block_id="block_123",
            content_type=ContentType.TEXT,
            size=1024,
            hash="abc123def456",
            compression=CompressionType.ZLIB,
        )

        assert result is True
        assert "block_123" in self.metadata_manager.blocks

        block_meta = self.metadata_manager.blocks["block_123"]
        assert block_meta.block_id == "block_123"
        assert block_meta.content_type == ContentType.TEXT
        assert block_meta.size == 1024
        assert block_meta.hash == "abc123def456"

    def test_update_block_metadata(self):
        """Test updating block metadata."""
        # First add a block
        self.metadata_manager.add_block_metadata(
            block_id="block_123", content_type=ContentType.TEXT, size=1024
        )

        # Update the block
        result = self.metadata_manager.update_block_metadata(
            block_id="block_123", size=2048, encrypted=True
        )

        assert result is True

        block_meta = self.metadata_manager.blocks["block_123"]
        assert block_meta.size == 2048
        assert block_meta.encrypted is True

    def test_update_nonexistent_block(self):
        """Test updating non-existent block metadata."""
        result = self.metadata_manager.update_block_metadata(
            block_id="nonexistent_block", size=1024
        )

        assert result is False

    def test_add_provenance_record(self):
        """Test adding provenance records."""
        result = self.metadata_manager.add_provenance_record(
            operation_type="create",
            agent_id="test_agent",
            block_id="block_123",
            operation_hash="op_hash_789",
        )

        assert result is True
        assert len(self.metadata_manager.provenance) == 1

        record = self.metadata_manager.provenance[0]
        assert record.operation_type == "create"
        assert record.agent_id == "test_agent"
        assert record.block_id == "block_123"
        assert record.operation_hash == "op_hash_789"

    def test_block_dependencies(self):
        """Test block dependency management."""
        # Add blocks with dependencies
        self.metadata_manager.add_block_metadata(
            block_id="block_1",
            content_type=ContentType.TEXT,
            dependencies=["block_2", "block_3"],
        )

        self.metadata_manager.add_block_metadata(
            block_id="block_2", content_type=ContentType.TEXT
        )

        self.metadata_manager.add_block_metadata(
            block_id="block_3", content_type=ContentType.TEXT
        )

        # Test getting dependencies
        deps = self.metadata_manager.get_block_dependencies("block_1")
        assert "block_2" in deps
        assert "block_3" in deps

        # Test getting dependents
        dependents = self.metadata_manager.get_block_dependents("block_2")
        assert "block_1" in dependents

    def test_validate_dependencies(self):
        """Test dependency validation."""
        # Add blocks with valid dependencies
        self.metadata_manager.add_block_metadata(
            block_id="block_1", content_type=ContentType.TEXT, dependencies=["block_2"]
        )

        self.metadata_manager.add_block_metadata(
            block_id="block_2", content_type=ContentType.TEXT
        )

        # Should have no validation errors
        errors = self.metadata_manager.validate_dependencies()
        assert len(errors) == 0

        # Add block with missing dependency
        self.metadata_manager.add_block_metadata(
            block_id="block_3",
            content_type=ContentType.TEXT,
            dependencies=["nonexistent_block"],
        )

        # Should have validation errors
        errors = self.metadata_manager.validate_dependencies()
        assert len(errors) > 0
        assert any("nonexistent_block" in error for error in errors)

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        # Create circular dependency
        self.metadata_manager.add_block_metadata(
            block_id="block_1", content_type=ContentType.TEXT, dependencies=["block_2"]
        )

        self.metadata_manager.add_block_metadata(
            block_id="block_2",
            content_type=ContentType.TEXT,
            dependencies=["block_1"],  # Circular dependency
        )

        # Should detect circular dependency
        has_circular = self.metadata_manager._has_circular_dependency("block_1")
        assert has_circular is True

    def test_metadata_summary(self):
        """Test metadata summary generation."""
        # Add some test data
        self.metadata_manager.add_block_metadata(
            block_id="block_1", content_type=ContentType.TEXT, size=1024
        )

        self.metadata_manager.add_block_metadata(
            block_id="block_2", content_type=ContentType.BINARY, size=2048
        )

        self.metadata_manager.add_provenance_record(
            operation_type="create", agent_id="test_agent", block_id="block_1"
        )

        summary = self.metadata_manager.get_metadata_summary()

        assert summary["total_blocks"] == 2
        assert summary["total_size"] == 3072  # 1024 + 2048
        assert summary["content_types"]["text"] == 1
        assert summary["content_types"]["binary"] == 1
        assert summary["provenance_records"] == 1

    def test_export_import_manifest(self):
        """Test manifest export and import."""
        # Add test data
        self.metadata_manager.add_block_metadata(
            block_id="block_1", content_type=ContentType.TEXT, size=1024, hash="abc123"
        )

        self.metadata_manager.add_provenance_record(
            operation_type="create", agent_id="test_agent", block_id="block_1"
        )

        # Export manifest
        manifest = self.metadata_manager.export_manifest()

        assert "version" in manifest
        assert "blocks" in manifest
        assert "provenance" in manifest
        assert len(manifest["blocks"]) == 1
        assert len(manifest["provenance"]) == 1

        # Import to new manager
        new_manager = MAIFMetadataManager()
        result = new_manager.import_manifest(manifest)

        assert result is True
        assert len(new_manager.blocks) == 1
        assert len(new_manager.provenance) == 1
        assert "block_1" in new_manager.blocks

    def test_custom_schema_management(self):
        """Test custom schema management."""
        # Add custom schema
        schema = {
            "type": "object",
            "properties": {
                "custom_field": {"type": "string"},
                "custom_number": {"type": "number"},
            },
            "required": ["custom_field"],
        }

        result = self.metadata_manager.add_custom_schema("custom_schema", schema)
        assert result is True
        assert "custom_schema" in self.metadata_manager.custom_schemas

        # Add block with custom metadata
        self.metadata_manager.add_block_metadata(
            block_id="custom_block",
            content_type=ContentType.TEXT,
            custom_metadata={"custom_field": "test_value", "custom_number": 42},
        )

        # Validate custom metadata
        errors = self.metadata_manager.validate_custom_metadata(
            "custom_block", "custom_schema"
        )
        assert len(errors) == 0  # Should be valid

        # Test with invalid custom metadata
        self.metadata_manager.add_block_metadata(
            block_id="invalid_custom_block",
            content_type=ContentType.TEXT,
            custom_metadata={
                "custom_number": 42  # Missing required field
            },
        )

        errors = self.metadata_manager.validate_custom_metadata(
            "invalid_custom_block", "custom_schema"
        )
        assert len(errors) > 0  # Should have validation errors

    def test_statistics_generation(self):
        """Test statistics generation."""
        # Add diverse test data
        self.metadata_manager.add_block_metadata(
            block_id="text_block_1",
            content_type=ContentType.TEXT,
            size=1024,
            compression=CompressionType.ZLIB,
        )

        self.metadata_manager.add_block_metadata(
            block_id="text_block_2",
            content_type=ContentType.TEXT,
            size=2048,
            compression=CompressionType.GZIP,
        )

        self.metadata_manager.add_block_metadata(
            block_id="binary_block_1",
            content_type=ContentType.BINARY,
            size=4096,
            encrypted=True,
        )

        # Add provenance records
        for i, block_id in enumerate(
            ["text_block_1", "text_block_2", "binary_block_1"]
        ):
            self.metadata_manager.add_provenance_record(
                operation_type="create", agent_id=f"agent_{i}", block_id=block_id
            )

        stats = self.metadata_manager.get_statistics()

        # Check basic statistics
        assert stats["blocks"]["total"] == 3
        assert stats["blocks"]["by_type"]["text"] == 2
        assert stats["blocks"]["by_type"]["binary"] == 1
        assert stats["blocks"]["total_size"] == 7168  # 1024 + 2048 + 4096

        # Check compression statistics
        assert stats["compression"]["zlib"] == 1
        assert stats["compression"]["gzip"] == 1
        assert stats["compression"]["none"] == 1

        # Check encryption statistics
        assert stats["encryption"]["encrypted"] == 1
        assert stats["encryption"]["unencrypted"] == 2

        # Check provenance statistics
        assert stats["provenance"]["total_records"] == 3
        assert stats["provenance"]["by_operation"]["create"] == 3


class TestMetadataErrorHandling:
    """Test metadata error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metadata_manager = MAIFMetadataManager()

    def test_duplicate_block_id(self):
        """Test handling of duplicate block IDs."""
        # Add first block
        result1 = self.metadata_manager.add_block_metadata(
            block_id="duplicate_block", content_type=ContentType.TEXT, size=1024
        )
        assert result1 is True

        # Try to add block with same ID
        result2 = self.metadata_manager.add_block_metadata(
            block_id="duplicate_block", content_type=ContentType.BINARY, size=2048
        )

        # Should handle gracefully (either reject or update)
        assert isinstance(result2, bool)

    def test_invalid_manifest_import(self):
        """Test import of invalid manifest."""
        invalid_manifests = [
            {},  # Empty manifest
            {"version": "1.0"},  # Missing required fields
            {"blocks": "not_a_list"},  # Invalid data types
            {"provenance": "not_a_list"},  # Invalid data types
        ]

        for invalid_manifest in invalid_manifests:
            result = self.metadata_manager.import_manifest(invalid_manifest)
            assert result is False

    def test_invalid_custom_schema(self):
        """Test handling of invalid custom schemas."""
        invalid_schemas = [
            "not_a_dict",  # Not a dictionary
            {},  # Empty schema
            {"type": "invalid_type"},  # Invalid JSON schema
        ]

        for invalid_schema in invalid_schemas:
            result = self.metadata_manager.add_custom_schema("invalid", invalid_schema)
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__])
