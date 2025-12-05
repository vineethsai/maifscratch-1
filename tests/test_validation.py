"""
Tests for MAIF validation module (v3 format).
"""

import pytest
import tempfile
import os
import shutil

from maif import MAIFEncoder, MAIFDecoder
from maif.validation import (
    MAIFValidator,
    MAIFRepairTool,
    ValidationResult,
    validate_maif,
    get_validation_report,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_valid(self):
        """Test valid result."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid is True
        assert bool(result) is True
        assert len(result.errors) == 0

    def test_validation_result_invalid(self):
        """Test invalid result."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        assert result.is_valid is False
        assert bool(result) is False
        assert len(result.errors) == 2

    def test_validation_result_with_warnings(self):
        """Test result with warnings."""
        result = ValidationResult(is_valid=True, warnings=["Warning 1"])
        assert result.is_valid is True
        assert len(result.warnings) == 1


class TestMAIFValidator:
    """Test MAIFValidator functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def valid_maif(self, temp_dir):
        """Create a valid MAIF file."""
        path = os.path.join(temp_dir, "valid.maif")
        encoder = MAIFEncoder(path, agent_id="test-agent")
        encoder.add_text_block("Test content")
        encoder.finalize()
        return path

    def test_validate_valid_file(self, valid_maif):
        """Test validation of a valid file."""
        validator = MAIFValidator()
        result = validator.validate(valid_maif)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_nonexistent_file(self, temp_dir):
        """Test validation of non-existent file."""
        validator = MAIFValidator()
        result = validator.validate(os.path.join(temp_dir, "nonexistent.maif"))

        assert result.is_valid is False
        assert any("not found" in e.lower() for e in result.errors)

    def test_validate_empty_file(self, temp_dir):
        """Test validation of empty file."""
        path = os.path.join(temp_dir, "empty.maif")
        open(path, "w").close()

        validator = MAIFValidator()
        result = validator.validate(path)

        assert result.is_valid is False

    def test_validate_multiple_blocks(self, temp_dir):
        """Test validation of file with multiple blocks."""
        path = os.path.join(temp_dir, "multi.maif")
        encoder = MAIFEncoder(path, agent_id="test")
        encoder.add_text_block("Block 1")
        encoder.add_text_block("Block 2")
        encoder.add_text_block("Block 3")
        encoder.finalize()

        validator = MAIFValidator()
        result = validator.validate(path)

        assert result.is_valid is True
        assert result.details.get("block_count") == 3

    def test_validate_with_embeddings(self, temp_dir):
        """Test validation of file with embeddings."""
        path = os.path.join(temp_dir, "embed.maif")
        encoder = MAIFEncoder(path, agent_id="test")
        encoder.add_text_block("Text")
        encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
        encoder.finalize()

        validator = MAIFValidator()
        result = validator.validate(path)

        assert result.is_valid is True

    def test_detect_tampering(self, valid_maif):
        """Test detection of tampered file."""
        # Tamper with the file
        with open(valid_maif, "r+b") as f:
            f.seek(500)
            f.write(b"TAMPERED")

        validator = MAIFValidator()
        result = validator.validate(valid_maif)

        # Should detect tampering
        assert result.is_valid is False or len(result.errors) > 0

    def test_strict_mode(self, valid_maif):
        """Test strict validation mode."""
        validator_strict = MAIFValidator(strict=True)
        validator_lenient = MAIFValidator(strict=False)

        result_strict = validator_strict.validate(valid_maif)
        result_lenient = validator_lenient.validate(valid_maif)

        # Both should pass for valid file
        assert result_strict.is_valid is True
        assert result_lenient.is_valid is True

    def test_legacy_validate_file(self, valid_maif):
        """Test legacy validate_file method."""
        validator = MAIFValidator()
        result = validator.validate_file(valid_maif, None)

        assert result.is_valid is True


class TestMAIFRepairTool:
    """Test MAIFRepairTool functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def valid_maif(self, temp_dir):
        """Create a valid MAIF file."""
        path = os.path.join(temp_dir, "valid.maif")
        encoder = MAIFEncoder(path, agent_id="test-agent")
        encoder.add_text_block("Test content")
        encoder.finalize()
        return path

    def test_repair_valid_file(self, valid_maif):
        """Test repair of already valid file."""
        repair_tool = MAIFRepairTool()
        result = repair_tool.repair_file(valid_maif)

        assert result is True

    def test_analyze_damage(self, valid_maif):
        """Test damage analysis."""
        repair_tool = MAIFRepairTool()
        analysis = repair_tool.analyze_damage(valid_maif)

        assert analysis["is_valid"] is True
        assert analysis["error_count"] == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def valid_maif(self, temp_dir):
        """Create a valid MAIF file."""
        path = os.path.join(temp_dir, "valid.maif")
        encoder = MAIFEncoder(path, agent_id="test-agent")
        encoder.add_text_block("Test content")
        encoder.finalize()
        return path

    def test_validate_maif(self, valid_maif):
        """Test validate_maif function."""
        result = validate_maif(valid_maif)
        assert result is True

    def test_get_validation_report(self, valid_maif):
        """Test get_validation_report function."""
        report = get_validation_report(valid_maif)

        assert "is_valid" in report
        assert "errors" in report
        assert "warnings" in report
        assert report["is_valid"] is True


class TestValidationIntegration:
    """Integration tests for validation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    def test_validate_after_create(self, temp_dir):
        """Test validation immediately after creation."""
        path = os.path.join(temp_dir, "test.maif")

        encoder = MAIFEncoder(path, agent_id="test")
        encoder.add_text_block("Content")
        encoder.finalize()

        result = validate_maif(path)
        assert result is True

    def test_validate_large_file(self, temp_dir):
        """Test validation of larger file."""
        path = os.path.join(temp_dir, "large.maif")

        encoder = MAIFEncoder(path, agent_id="test")
        for i in range(50):
            encoder.add_text_block(f"Block {i} with some content")
        encoder.finalize()

        result = validate_maif(path)
        assert result is True

    def test_validate_unicode_content(self, temp_dir):
        """Test validation of file with unicode content."""
        path = os.path.join(temp_dir, "unicode.maif")

        encoder = MAIFEncoder(path, agent_id="test")
        encoder.add_text_block("Hello 世界 🌍 مرحبا")
        encoder.finalize()

        result = validate_maif(path)
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
