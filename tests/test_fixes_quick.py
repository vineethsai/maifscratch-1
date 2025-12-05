#!/usr/bin/env python3
"""Quick test to verify v3 format fixes."""

import os
import tempfile
from maif import MAIFEncoder, MAIFDecoder
from maif.validation import MAIFValidator


def test_validation_fix():
    """Test that validation now works correctly."""
    print("Testing validation fix...")

    with tempfile.TemporaryDirectory() as temp_dir:
        maif_path = os.path.join(temp_dir, "test.maif")

        # Create a simple MAIF file (v3 format)
        encoder = MAIFEncoder(maif_path, agent_id="test_agent")
        encoder.add_text_block("Test content", metadata={"test": True})
        encoder.finalize()

        # Validate it
        validator = MAIFValidator()
        result = validator.validate(maif_path)

        print(f"  Validation result: is_valid={result.is_valid}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Warnings: {len(result.warnings)}")

        if result.errors:
            for error in result.errors:
                print(f"    Error: {error}")

        success = result.is_valid
        assert success, f"Validation failed: {result.errors}"


def test_semantic_fix():
    """Test that semantic understanding works."""
    print("Testing semantic fix...")

    try:
        from maif.semantic import DeepSemanticUnderstanding

        dsu = DeepSemanticUnderstanding()

        inputs = {"text": "test text", "metadata": {"test": True}}
        result = dsu.process_multimodal_input(inputs)

        print(f"  Result keys: {list(result.keys())}")
        has_understanding_score = "understanding_score" in result
        print(f"  Has understanding_score: {has_understanding_score}")

        assert has_understanding_score, "Semantic understanding test failed"
    except Exception as e:
        print(f"  Error: {e}")
        assert False, f"Semantic test failed: {e}"


if __name__ == "__main__":
    print("Running quick fix verification tests...\n")

    test_validation_fix()
    print("Validation fix: ✓ PASS\n")

    test_semantic_fix()
    print("Semantic fix: ✓ PASS\n")

    print("🎉 All fixes appear to be working!")
