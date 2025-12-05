"""
Integration tests for MAIF enhanced features (v3 format).
"""

import pytest
import tempfile
import os
import json
import shutil

from maif import MAIFEncoder, MAIFDecoder
from maif.integration_enhanced import EnhancedMAIF


class TestEnhancedMAIF:
    """Test EnhancedMAIF basic functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_enhanced_maif_initialization(self):
        """Test EnhancedMAIF initialization."""
        enhanced = EnhancedMAIF(self.maif_path, agent_id="test-agent")

        assert enhanced is not None
        assert enhanced.agent_id == "test-agent"

    def test_add_text_content(self):
        """Test adding text content."""
        enhanced = EnhancedMAIF(self.maif_path, agent_id="test")

        block_id = enhanced.add_text_block("Test content")
        assert block_id is not None

    def test_add_multiple_blocks(self):
        """Test adding multiple blocks."""
        enhanced = EnhancedMAIF(self.maif_path, agent_id="test")

        enhanced.add_text_block("Block 1")
        enhanced.add_text_block("Block 2")
        enhanced.add_text_block("Block 3")

        assert len(enhanced.encoder.blocks) == 3

    def test_save_and_load(self):
        """Test save and load functionality."""
        enhanced = EnhancedMAIF(self.maif_path, agent_id="test")
        enhanced.add_text_block("Test content to save")
        enhanced.save()

        # Verify file exists
        assert os.path.exists(self.maif_path)

        # Load and verify
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()
        assert len(decoder.blocks) >= 1


class TestMAIFConverter:
    """Test MAIF format conversion."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_convert_json_to_maif(self):
        """Test JSON to MAIF conversion."""
        # Create test JSON file
        json_content = {
            "title": "Test Document",
            "content": "This is test content for conversion",
            "metadata": {"author": "Test Author"},
        }

        json_path = os.path.join(self.temp_dir, "test.json")
        with open(json_path, "w") as f:
            json.dump(json_content, f)

        maif_path = os.path.join(self.temp_dir, "converted.maif")

        # Create MAIF from JSON manually
        encoder = MAIFEncoder(maif_path, agent_id="converter")
        encoder.add_text_block(
            json.dumps(json_content), metadata={"source": json_path, "format": "json"}
        )
        encoder.finalize()

        assert os.path.exists(maif_path)

    def test_convert_text_to_maif(self):
        """Test text to MAIF conversion."""
        text_content = "This is plain text content."
        text_path = os.path.join(self.temp_dir, "test.txt")

        with open(text_path, "w") as f:
            f.write(text_content)

        maif_path = os.path.join(self.temp_dir, "converted.maif")

        encoder = MAIFEncoder(maif_path, agent_id="converter")
        encoder.add_text_block(text_content, metadata={"source": text_path})
        encoder.finalize()

        assert os.path.exists(maif_path)

        # Verify content
        decoder = MAIFDecoder(maif_path)
        decoder.load()
        assert decoder.get_text_content(0) == text_content

    def test_export_maif_to_json(self):
        """Test MAIF to JSON export."""
        maif_path = os.path.join(self.temp_dir, "export_test.maif")

        # Create MAIF file
        encoder = MAIFEncoder(maif_path, agent_id="test_agent")
        encoder.add_text_block("Test content for export", metadata={"id": 1})
        encoder.finalize()

        # Export to JSON
        json_output = os.path.join(self.temp_dir, "exported.json")

        decoder = MAIFDecoder(maif_path)
        decoder.load()

        export_data = decoder.export_manifest()

        with open(json_output, "w") as f:
            json.dump(export_data, f, indent=2)

        assert os.path.exists(json_output)

        # Verify JSON content
        with open(json_output, "r") as f:
            loaded = json.load(f)

        assert "blocks" in loaded
        assert len(loaded["blocks"]) >= 1


class TestIntegrationScenarios:
    """Test integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test full integration workflow."""
        maif_path = os.path.join(self.temp_dir, "workflow.maif")

        # Create
        encoder = MAIFEncoder(maif_path, agent_id="workflow-agent")
        encoder.add_text_block("Step 1: Initial content")
        encoder.add_text_block("Step 2: More content")
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
        encoder.finalize()

        # Verify
        decoder = MAIFDecoder(maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True
        assert len(decoder.blocks) == 3

    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        # Create multiple MAIF files
        for i in range(3):
            path = os.path.join(self.temp_dir, f"batch_{i}.maif")
            encoder = MAIFEncoder(path, agent_id=f"batch-agent-{i}")
            encoder.add_text_block(f"Content from batch {i}")
            encoder.finalize()

        # Verify all files
        for i in range(3):
            path = os.path.join(self.temp_dir, f"batch_{i}.maif")
            decoder = MAIFDecoder(path)
            decoder.load()

            is_valid, _ = decoder.verify_integrity()
            assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
