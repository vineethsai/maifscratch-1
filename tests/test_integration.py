"""
Comprehensive tests for MAIF integration functionality (v3 format).
"""

import pytest
import tempfile
import os
import json
import shutil

from maif import MAIFEncoder, MAIFDecoder


class TestEnhancedMAIFProcessor:
    """Test enhanced MAIF processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_maif_from_json(self):
        """Test creating MAIF from JSON content."""
        json_content = {
            "title": "Test Document",
            "content": "This is test content",
            "metadata": {"author": "Test"},
        }

        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block(json.dumps(json_content), metadata={"format": "json"})
        encoder.finalize()

        assert os.path.exists(self.maif_path)

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()
        assert len(decoder.blocks) == 1

    def test_create_maif_from_text(self):
        """Test creating MAIF from text content."""
        text_content = "Simple text content"

        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block(text_content)
        encoder.finalize()

        assert os.path.exists(self.maif_path)

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()
        assert decoder.get_text_content(0) == text_content

    def test_convert_json_to_maif(self):
        """Test JSON to MAIF conversion."""
        json_path = os.path.join(self.temp_dir, "input.json")
        with open(json_path, "w") as f:
            json.dump({"key": "value", "data": [1, 2, 3]}, f)

        # Read JSON and convert to MAIF
        with open(json_path, "r") as f:
            content = f.read()

        encoder = MAIFEncoder(self.maif_path, agent_id="converter")
        encoder.add_text_block(content, metadata={"source_format": "json"})
        encoder.finalize()

        assert os.path.exists(self.maif_path)

    def test_convert_xml_to_maif(self):
        """Test XML to MAIF conversion."""
        xml_content = """<?xml version="1.0"?>
        <root><item>Test</item></root>"""

        xml_path = os.path.join(self.temp_dir, "input.xml")
        with open(xml_path, "w") as f:
            f.write(xml_content)

        encoder = MAIFEncoder(self.maif_path, agent_id="converter")
        encoder.add_text_block(xml_content, metadata={"source_format": "xml"})
        encoder.finalize()

        assert os.path.exists(self.maif_path)

    def test_export_maif_to_json(self):
        """Test exporting MAIF to JSON."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")
        encoder.add_text_block("Content to export")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        export_data = decoder.export_manifest()
        json_path = os.path.join(self.temp_dir, "export.json")

        with open(json_path, "w") as f:
            json.dump(export_data, f)

        assert os.path.exists(json_path)

        with open(json_path, "r") as f:
            loaded = json.load(f)

        assert "blocks" in loaded


class TestIntegrationFormats:
    """Test format integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_csv_to_maif(self):
        """Test CSV to MAIF conversion."""
        csv_content = "name,value\ntest,123\ndata,456"
        csv_path = os.path.join(self.temp_dir, "data.csv")

        with open(csv_path, "w") as f:
            f.write(csv_content)

        maif_path = os.path.join(self.temp_dir, "output.maif")

        encoder = MAIFEncoder(maif_path, agent_id="csv-converter")
        encoder.add_text_block(csv_content, metadata={"format": "csv"})
        encoder.finalize()

        decoder = MAIFDecoder(maif_path)
        decoder.load()

        assert decoder.get_text_content(0) == csv_content

    def test_multi_format_maif(self):
        """Test MAIF with multiple format types."""
        maif_path = os.path.join(self.temp_dir, "multi.maif")

        encoder = MAIFEncoder(maif_path, agent_id="multi")
        encoder.add_text_block("Plain text", metadata={"format": "txt"})
        encoder.add_text_block('{"json": true}', metadata={"format": "json"})
        encoder.add_text_block("<xml/>", metadata={"format": "xml"})
        encoder.finalize()

        decoder = MAIFDecoder(maif_path)
        decoder.load()

        assert len(decoder.blocks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
