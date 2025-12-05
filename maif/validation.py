"""
MAIF File Validation Module

Provides comprehensive validation for MAIF v3 self-contained format.
Validates file structure, block integrity, signatures, and provenance.
"""

import os
import hashlib
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field

from .core import MAIFDecoder, BlockType, BlockFlags


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __bool__(self):
        return self.is_valid


class MAIFValidator:
    """
    Comprehensive validator for MAIF v3 files.

    Validates:
    - File structure (header, blocks, footer)
    - Block integrity (signatures, content hashes)
    - Provenance chain
    - Merkle root

    Usage:
        validator = MAIFValidator()
        result = validator.validate("file.maif")
        if not result:
            print("Errors:", result.errors)
    """

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, warnings are treated as errors
        """
        self.strict = strict
        self.validation_rules: List[Callable] = [
            self._validate_file_exists,
            self._validate_file_structure,
            self._validate_block_integrity,
            self._validate_signatures,
            self._validate_provenance_chain,
            self._validate_merkle_root,
        ]

    def validate(self, maif_path: str) -> ValidationResult:
        """
        Validate a MAIF file.

        Args:
            maif_path: Path to MAIF file

        Returns:
            ValidationResult with errors, warnings, and details
        """
        errors = []
        warnings = []
        details = {"file_path": maif_path}

        # Check file exists
        if not os.path.exists(maif_path):
            return ValidationResult(
                is_valid=False, errors=[f"File not found: {maif_path}"]
            )

        # Load file
        try:
            decoder = MAIFDecoder(maif_path)
            decoder.load()
            details["block_count"] = len(decoder.blocks)
            details["provenance_count"] = len(decoder.provenance)
        except Exception as e:
            return ValidationResult(
                is_valid=False, errors=[f"Failed to load file: {str(e)}"]
            )

        # Run validation rules
        for rule in self.validation_rules:
            try:
                rule_errors, rule_warnings = rule(decoder, maif_path)
                errors.extend(rule_errors)
                warnings.extend(rule_warnings)
            except Exception as e:
                errors.append(f"Validation rule {rule.__name__} failed: {str(e)}")

        # Determine validity
        if self.strict:
            is_valid = len(errors) == 0 and len(warnings) == 0
        else:
            is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, details=details
        )

    # Legacy compatibility
    def validate_file(
        self, maif_path: str, manifest_path: str = None
    ) -> ValidationResult:
        """Legacy method - manifest_path is ignored in v3."""
        return self.validate(maif_path)

    def _validate_file_exists(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Check file exists and is readable."""
        errors = []
        warnings = []

        if not os.path.exists(maif_path):
            errors.append(f"File not found: {maif_path}")
        elif os.path.getsize(maif_path) == 0:
            errors.append("File is empty")

        return errors, warnings

    def _validate_file_structure(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Validate file structure."""
        errors = []
        warnings = []

        # Check header
        if decoder.file_header is None:
            errors.append("Missing file header")
            return errors, warnings

        # Check magic number
        if decoder.file_header.magic != b"MAIF":
            errors.append(f"Invalid magic number: {decoder.file_header.magic}")

        # Check version
        if decoder.file_header.version_major < 2:
            warnings.append(
                f"Old format version: {decoder.file_header.version_major}.{decoder.file_header.version_minor}"
            )

        # Check block count matches
        if decoder.file_header.block_count != len(decoder.blocks):
            errors.append(
                f"Block count mismatch: header={decoder.file_header.block_count}, "
                f"actual={len(decoder.blocks)}"
            )

        return errors, warnings

    def _validate_block_integrity(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Validate block integrity via content hashes."""
        errors = []
        warnings = []

        for i, block in enumerate(decoder.blocks):
            # Calculate actual content hash
            calculated_hash = block.get_content_hash()
            stored_hash = block.header.content_hash

            if calculated_hash != stored_hash:
                errors.append(
                    f"Block {i}: content hash mismatch - "
                    f"expected {stored_hash.hex()[:16]}..., "
                    f"got {calculated_hash.hex()[:16]}..."
                )

            # Check block has data
            if not block.data:
                warnings.append(f"Block {i}: empty data")

        return errors, warnings

    def _validate_signatures(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Validate Ed25519 signatures."""
        errors = []
        warnings = []

        # Use decoder's built-in verification
        is_valid, verify_errors = decoder.verify_integrity()

        if not is_valid:
            for err in verify_errors:
                if "TAMPERED" in err.upper():
                    errors.append(err)
                else:
                    warnings.append(err)

        # Check all blocks are signed
        for i, block in enumerate(decoder.blocks):
            if not (block.header.flags & BlockFlags.SIGNED):
                warnings.append(f"Block {i}: not signed")

        return errors, warnings

    def _validate_provenance_chain(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Validate provenance chain integrity."""
        errors = []
        warnings = []

        if not decoder.provenance:
            warnings.append("No provenance entries found")
            return errors, warnings

        # Check chain links
        prev_hash = ""
        for i, entry in enumerate(decoder.provenance):
            if i == 0:
                if entry.action != "genesis":
                    warnings.append("First provenance entry is not genesis")
            else:
                if entry.previous_entry_hash != prev_hash:
                    errors.append(
                        f"Provenance chain broken at entry {i}: "
                        f"expected prev={prev_hash[:16]}..., "
                        f"got {entry.previous_entry_hash[:16] if entry.previous_entry_hash else 'None'}..."
                    )
            prev_hash = entry.entry_hash

        return errors, warnings

    def _validate_merkle_root(
        self, decoder: MAIFDecoder, maif_path: str
    ) -> Tuple[List[str], List[str]]:
        """Validate Merkle root."""
        errors = []
        warnings = []

        if not decoder.blocks:
            return errors, warnings

        # Calculate Merkle root
        hashes = [block.header.content_hash for block in decoder.blocks]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashlib.sha256(hashes[i] + hashes[i + 1]).digest()
                new_hashes.append(combined)
            hashes = new_hashes

        calculated_root = hashes[0] if hashes else b"\x00" * 32
        stored_root = decoder.file_header.merkle_root

        if calculated_root != stored_root:
            errors.append(
                f"Merkle root mismatch: "
                f"expected {stored_root.hex()[:16]}..., "
                f"got {calculated_root.hex()[:16]}..."
            )

        return errors, warnings


class MAIFRepairTool:
    """
    Tool for repairing damaged MAIF files.

    Note: In v3 format, most repairs require re-signing which
    requires access to the original private key.
    """

    def __init__(self):
        self.validator = MAIFValidator(strict=False)

    def repair_file(self, maif_path: str, manifest_path: str = None) -> bool:
        """
        Attempt to repair a MAIF file.

        Args:
            maif_path: Path to MAIF file
            manifest_path: Ignored (v3 format is self-contained)

        Returns:
            True if repair was successful
        """
        # First validate
        result = self.validator.validate(maif_path)

        if result.is_valid:
            return True  # Nothing to repair

        # In v3 format, repairs are limited without the private key
        # We can only fix structural issues, not re-sign blocks

        try:
            decoder = MAIFDecoder(maif_path)
            decoder.load()

            # Check what can be repaired
            repairable = []
            unrepairable = []

            for error in result.errors:
                if "TAMPERED" in error or "signature" in error.lower():
                    unrepairable.append(error)
                else:
                    repairable.append(error)

            if unrepairable:
                print(f"Cannot repair {len(unrepairable)} issues without private key:")
                for err in unrepairable[:3]:
                    print(f"  - {err}")
                return False

            # For now, we can only report issues
            print(
                f"Validation found {len(result.errors)} errors, {len(result.warnings)} warnings"
            )
            return len(result.errors) == 0

        except Exception as e:
            print(f"Repair failed: {e}")
            return False

    def analyze_damage(self, maif_path: str) -> Dict[str, Any]:
        """
        Analyze file damage without attempting repair.

        Returns dict with damage assessment.
        """
        result = self.validator.validate(maif_path)

        return {
            "file_path": maif_path,
            "is_valid": result.is_valid,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "errors": result.errors,
            "warnings": result.warnings,
            "repairable": [
                e
                for e in result.errors
                if "TAMPERED" not in e and "signature" not in e.lower()
            ],
            "requires_resigning": [
                e for e in result.errors if "TAMPERED" in e or "signature" in e.lower()
            ],
        }


# Convenience functions
def validate_maif(maif_path: str) -> bool:
    """Quick validation check."""
    validator = MAIFValidator()
    return validator.validate(maif_path).is_valid


def get_validation_report(maif_path: str) -> Dict[str, Any]:
    """Get detailed validation report."""
    validator = MAIFValidator()
    result = validator.validate(maif_path)
    return {
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "details": result.details,
    }


__all__ = [
    "MAIFValidator",
    "MAIFRepairTool",
    "ValidationResult",
    "validate_maif",
    "get_validation_report",
]
