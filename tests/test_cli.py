"""
Comprehensive tests for MAIF CLI functionality (v3 format).
"""

import pytest
import tempfile
import os
import json
import shutil
from click.testing import CliRunner

from maif.cli import (
    create_privacy_maif,
    access_privacy_maif,
    manage_privacy,
    create_maif,
    verify_maif,
    analyze_maif,
    extract_content,
    main,
)


class TestCLICommands:
    """Test CLI command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_privacy_maif_command(self):
        """Test create-privacy-maif command."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content for privacy MAIF creation")

        output_file = os.path.join(self.temp_dir, "output.maif")

        result = self.runner.invoke(
            create_privacy_maif,
            [
                "--input",
                input_file,
                "--output",
                output_file,
                "--privacy-level",
                "medium",
                "--agent-id",
                "test_agent",
            ],
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)

    def test_access_privacy_maif_command(self):
        """Test access-privacy-maif command."""
        # First create a MAIF file
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content for privacy access")

        maif_file = os.path.join(self.temp_dir, "test.maif")

        create_result = self.runner.invoke(
            create_privacy_maif,
            [
                "--input",
                input_file,
                "--output",
                maif_file,
                "--privacy-level",
                "low",
                "--agent-id",
                "test_agent",
            ],
        )

        assert create_result.exit_code == 0

        # Now test access command
        result = self.runner.invoke(
            access_privacy_maif,
            [
                "--maif-file",
                maif_file,
                "--user-id",
                "test_user",
                "--permission",
                "read",
            ],
        )

        assert result.exit_code == 0

    def test_manage_privacy_command(self):
        """Test manage-privacy command."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content for privacy management")

        maif_file = os.path.join(self.temp_dir, "manage_test.maif")

        create_result = self.runner.invoke(
            create_privacy_maif,
            ["--input", input_file, "--output", maif_file, "--agent-id", "test_agent"],
        )

        assert create_result.exit_code == 0

        result = self.runner.invoke(
            manage_privacy, ["--maif-file", maif_file, "--action", "status"]
        )

        assert result.exit_code == 0
        assert "Blocks:" in result.output

    def test_create_maif_command(self):
        """Test create-maif command."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content for basic MAIF")

        output_file = os.path.join(self.temp_dir, "basic.maif")

        result = self.runner.invoke(
            create_maif,
            [
                "--input",
                input_file,
                "--output",
                output_file,
                "--agent-id",
                "test_agent",
            ],
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)

    def test_verify_maif_command(self):
        """Test verify-maif command."""
        # Create a file first
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content")

        maif_file = os.path.join(self.temp_dir, "verify_test.maif")

        self.runner.invoke(create_maif, ["--input", input_file, "--output", maif_file])

        result = self.runner.invoke(verify_maif, ["--maif-file", maif_file])

        assert result.exit_code == 0
        assert "VALID" in result.output

    def test_analyze_maif_command(self):
        """Test analyze-maif command."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Test content for analysis")

        maif_file = os.path.join(self.temp_dir, "analyze_test.maif")

        self.runner.invoke(create_maif, ["--input", input_file, "--output", maif_file])

        result = self.runner.invoke(analyze_maif, ["--maif-file", maif_file])

        assert result.exit_code == 0
        assert "Blocks:" in result.output

    def test_extract_content_command(self):
        """Test extract-content command."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Content to extract")

        maif_file = os.path.join(self.temp_dir, "extract_test.maif")
        output_dir = os.path.join(self.temp_dir, "extracted")

        self.runner.invoke(create_maif, ["--input", input_file, "--output", maif_file])

        result = self.runner.invoke(
            extract_content, ["--maif-file", maif_file, "--output-dir", output_dir]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_dir)

    def test_cli_with_json_input(self):
        """Test CLI with JSON input."""
        json_file = os.path.join(self.temp_dir, "input.json")
        with open(json_file, "w") as f:
            json.dump({"key": "value"}, f)

        output_file = os.path.join(self.temp_dir, "json.maif")

        result = self.runner.invoke(
            create_maif, ["--input", json_file, "--output", output_file]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)

    def test_cli_with_compression(self):
        """Test CLI with compression flag."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Compressible content " * 100)

        output_file = os.path.join(self.temp_dir, "compressed.maif")

        result = self.runner.invoke(
            create_maif, ["--input", input_file, "--output", output_file, "--compress"]
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)

    def test_cli_with_encryption(self):
        """Test CLI with encryption."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Encrypted content")

        output_file = os.path.join(self.temp_dir, "encrypted.maif")

        result = self.runner.invoke(
            create_privacy_maif,
            [
                "--input",
                input_file,
                "--output",
                output_file,
                "--encryption",
                "aes_gcm",
                "--agent-id",
                "test",
            ],
        )

        assert result.exit_code == 0
        assert os.path.exists(output_file)


class TestCLIIntegration:
    """Test CLI integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_verify_analyze_workflow(self):
        """Test complete workflow: create -> verify -> analyze."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Workflow test content")

        maif_file = os.path.join(self.temp_dir, "workflow.maif")

        # Create
        create_result = self.runner.invoke(
            create_maif, ["--input", input_file, "--output", maif_file]
        )
        assert create_result.exit_code == 0

        # Verify
        verify_result = self.runner.invoke(verify_maif, ["--maif-file", maif_file])
        assert verify_result.exit_code == 0
        assert "VALID" in verify_result.output

        # Analyze
        analyze_result = self.runner.invoke(analyze_maif, ["--maif-file", maif_file])
        assert analyze_result.exit_code == 0

    def test_privacy_workflow(self):
        """Test privacy workflow: create-privacy -> access-privacy -> manage-privacy."""
        input_file = os.path.join(self.temp_dir, "input.txt")
        with open(input_file, "w") as f:
            f.write("Privacy workflow test")

        maif_file = os.path.join(self.temp_dir, "privacy_workflow.maif")

        # Create with privacy
        create_result = self.runner.invoke(
            create_privacy_maif,
            [
                "--input",
                input_file,
                "--output",
                maif_file,
                "--privacy-level",
                "confidential",
                "--agent-id",
                "privacy_test",
            ],
        )
        assert create_result.exit_code == 0

        # Access
        access_result = self.runner.invoke(
            access_privacy_maif,
            ["--maif-file", maif_file, "--user-id", "user1", "--permission", "read"],
        )
        assert access_result.exit_code == 0

        # Manage - status
        status_result = self.runner.invoke(
            manage_privacy, ["--maif-file", maif_file, "--action", "status"]
        )
        assert status_result.exit_code == 0

        # Manage - audit
        audit_result = self.runner.invoke(
            manage_privacy, ["--maif-file", maif_file, "--action", "audit"]
        )
        assert audit_result.exit_code == 0

    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        files = []
        for i in range(3):
            input_file = os.path.join(self.temp_dir, f"input_{i}.txt")
            with open(input_file, "w") as f:
                f.write(f"Batch content {i}")
            files.append(input_file)

        for i, input_file in enumerate(files):
            output_file = os.path.join(self.temp_dir, f"batch_{i}.maif")
            result = self.runner.invoke(
                create_maif, ["--input", input_file, "--output", output_file]
            )
            assert result.exit_code == 0
            assert os.path.exists(output_file)


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_input_file(self):
        """Test handling of missing input file."""
        output_file = os.path.join(self.temp_dir, "output.maif")

        result = self.runner.invoke(
            create_maif, ["--input", "/nonexistent/file.txt", "--output", output_file]
        )

        assert result.exit_code != 0

    def test_missing_maif_file_verify(self):
        """Test verify with missing MAIF file."""
        result = self.runner.invoke(
            verify_maif, ["--maif-file", "/nonexistent/file.maif"]
        )

        assert result.exit_code != 0

    def test_missing_maif_file_analyze(self):
        """Test analyze with missing MAIF file."""
        result = self.runner.invoke(
            analyze_maif, ["--maif-file", "/nonexistent/file.maif"]
        )

        assert result.exit_code != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
