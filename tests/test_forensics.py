"""
Comprehensive tests for MAIF forensics functionality (v3 format).
"""

import pytest
import tempfile
import os
import json
import hashlib
import shutil

from maif.validation import MAIFValidator, MAIFRepairTool, ValidationResult
from maif import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner, MAIFVerifier


class TestForensicAnalysis:
    """Test forensic analysis capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")
        self.validator = MAIFValidator()
        self.repair_tool = MAIFRepairTool()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forensic_validation_chain(self):
        """Test forensic validation of provenance chain."""
        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_test")
        encoder.add_text_block(
            "Forensic evidence data", metadata={"evidence_id": "E001"}
        )
        encoder.finalize()

        result = self.validator.validate(self.maif_path)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_forensic_integrity_verification(self):
        """Test forensic integrity verification."""
        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_test")
        encoder.add_text_block("Chain of custody data", metadata={"custody_id": "C001"})
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True

    def test_forensic_signature_verification(self):
        """Test forensic signature verification."""
        signer = MAIFSigner(agent_id="forensic_signer")

        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_test")
        encoder.add_text_block("Signed evidence", metadata={"signature_required": True})
        encoder.finalize()

        # Verify file is signed
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid is True

        # Check security info
        security_info = decoder.get_security_info()
        assert security_info.get("key_algorithm") == "Ed25519"

    def test_forensic_tamper_detection(self):
        """Test detection of tampering."""
        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_test")
        encoder.add_text_block("Original evidence", metadata={"tamper_test": True})
        encoder.finalize()

        # First verify clean file passes
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()
        is_valid, _ = decoder.verify_integrity()
        assert is_valid is True

        # Tamper with the file
        with open(self.maif_path, "r+b") as f:
            f.seek(500)
            f.write(b"TAMPERED")

        # Re-verify - should detect tampering
        tampered_decoder = MAIFDecoder(self.maif_path)
        tampered_decoder.load()
        is_valid, errors = tampered_decoder.verify_integrity()

        # Should detect tampering (may be invalid or have errors)
        assert is_valid is False or len(errors) > 0

    def test_forensic_repair_attempt(self):
        """Test forensic repair capabilities."""
        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_test")
        encoder.add_text_block("Repair test data", metadata={"repair_test": True})
        encoder.finalize()

        repair_success = self.repair_tool.repair_file(self.maif_path)
        assert isinstance(repair_success, bool)

    def test_forensic_metadata_analysis(self):
        """Test forensic metadata analysis."""
        encoder = MAIFEncoder(self.maif_path, agent_id="forensic_analyst")
        encoder.add_text_block(
            "Evidence with metadata",
            metadata={
                "case_id": "CASE-2024-001",
                "evidence_type": "digital",
                "chain_of_custody": ["Officer A", "Lab Tech B", "Analyst C"],
                "collection_timestamp": "2024-01-01T12:00:00Z",
                "hash_verified": True,
            },
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        block = decoder.blocks[0]
        assert block.metadata["case_id"] == "CASE-2024-001"
        assert len(block.metadata["chain_of_custody"]) == 3


class TestForensicCompliance:
    """Test forensic compliance features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "compliance.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forensic_chain_of_custody(self):
        """Test chain of custody tracking."""
        encoder = MAIFEncoder(self.maif_path, agent_id="custody_agent")
        encoder.add_text_block("Evidence item 1", metadata={"custody_entry": 1})
        encoder.add_text_block("Evidence item 2", metadata={"custody_entry": 2})
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Verify provenance chain exists
        provenance = decoder.get_provenance()
        assert len(provenance) >= 3  # genesis + 2 blocks + finalize

        # Verify chain links
        for i in range(1, len(provenance)):
            assert provenance[i].previous_entry_hash == provenance[i - 1].entry_hash

    def test_forensic_audit_trail(self):
        """Test audit trail generation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="audit_agent")
        encoder.add_text_block("Audit entry 1")
        encoder.add_text_block("Audit entry 2")
        encoder.add_text_block("Audit entry 3")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Get file info for audit
        file_info = decoder.get_file_info()
        assert file_info["block_count"] == 3
        assert file_info["is_signed"] is True
        assert file_info["is_finalized"] is True

    def test_forensic_evidence_integrity(self):
        """Test evidence integrity verification."""
        evidence_data = "Critical forensic evidence content"

        encoder = MAIFEncoder(self.maif_path, agent_id="evidence_agent")
        encoder.add_text_block(evidence_data, metadata={"evidence": True})
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Verify content hash
        block = decoder.blocks[0]
        content_hash = block.get_content_hash()
        stored_hash = block.header.content_hash

        assert content_hash == stored_hash


class TestForensicReporting:
    """Test forensic reporting capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "report.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forensic_validation_report(self):
        """Test forensic validation report generation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="report_agent")
        encoder.add_text_block("Report content")
        encoder.finalize()

        validator = MAIFValidator()
        result = validator.validate(self.maif_path)

        assert result.is_valid is True
        assert "block_count" in result.details

    def test_forensic_summary_generation(self):
        """Test forensic summary generation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="summary_agent")

        for i in range(5):
            encoder.add_text_block(f"Summary block {i}", metadata={"index": i})

        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Generate summary
        file_info = decoder.get_file_info()
        security_info = decoder.get_security_info()
        provenance = decoder.get_provenance()

        summary = {
            "file_info": file_info,
            "security": security_info,
            "provenance_count": len(provenance),
            "block_count": len(decoder.blocks),
        }

        assert summary["block_count"] == 5
        assert summary["provenance_count"] >= 6  # genesis + 5 adds + finalize

    def test_forensic_export_manifest(self):
        """Test forensic manifest export."""
        encoder = MAIFEncoder(self.maif_path, agent_id="export_agent")
        encoder.add_text_block("Export test", metadata={"export": True})
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        manifest = decoder.export_manifest()

        assert "blocks" in manifest
        assert "provenance" in manifest
        assert "file_info" in manifest
        assert len(manifest["blocks"]) >= 1


class TestForensicTimeline:
    """Test forensic timeline analysis."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "timeline.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_forensic_event_timeline(self):
        """Test forensic event timeline creation."""
        encoder = MAIFEncoder(self.maif_path, agent_id="timeline_agent")

        import time

        for i in range(3):
            encoder.add_text_block(f"Event {i}")
            time.sleep(0.01)  # Small delay between events

        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        provenance = decoder.get_provenance()

        # Verify chronological order
        timestamps = [p.timestamp for p in provenance]
        assert timestamps == sorted(timestamps)

    def test_forensic_agent_activity(self):
        """Test forensic agent activity tracking."""
        encoder = MAIFEncoder(self.maif_path, agent_id="activity_agent")
        encoder.add_text_block("Activity 1")
        encoder.add_text_block("Activity 2")
        encoder.finalize()

        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        provenance = decoder.get_provenance()

        # All actions should be by the same agent
        for entry in provenance:
            assert entry.agent_id == "activity_agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
