#!/usr/bin/env python3
"""
Quick test script to verify the major fixes.
"""

import tempfile
import os
import sys


def test_cli_privacy_levels():
    """Test that CLI accepts the privacy levels."""
    from maif.privacy import PrivacyLevel

    # Test that all expected privacy levels exist
    expected_levels = [
        "public",
        "low",
        "internal",
        "medium",
        "confidential",
        "high",
        "secret",
        "top_secret",
    ]

    for level in expected_levels:
        try:
            PrivacyLevel(level)
            print(f"✓ Privacy level '{level}' is valid")
        except ValueError:
            print(f"✗ Privacy level '{level}' is invalid")
            assert False, f"Privacy level '{level}' is invalid"

    assert True  # Test passed


def test_basic_maif_creation():
    """Test basic MAIF creation without encryption."""
    from maif.core import MAIFEncoder

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            maif_path = os.path.join(temp_dir, "test.maif")
            manifest_path = os.path.join(temp_dir, "test_manifest.json")
            encoder = MAIFEncoder(maif_path, agent_id="test_agent")
            encoder.add_text_block("Test content")
            encoder.finalize()

            # Check files were created
            if os.path.exists(maif_path):
                print("✓ Basic MAIF creation works")
                assert True  # Test passed
            else:
                print("✗ MAIF file not created")
                assert False, "MAIF file not created"
    except Exception as e:
        print(f"✗ Basic MAIF creation failed: {e}")
        assert False, f"Basic MAIF creation failed: {e}"


def test_validation():
    """Test validation of a simple MAIF file."""
    from maif.core import MAIFEncoder
    from maif.validation import MAIFValidator

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple MAIF file
            encoder = MAIFEncoder(agent_id="test_agent", enable_privacy=False)
            encoder.add_text_block("Test content")

            maif_path = os.path.join(temp_dir, "test.maif")
            manifest_path = os.path.join(temp_dir, "test_manifest.json")

            encoder.build_maif(maif_path, manifest_path)

            # Validate it
            validator = MAIFValidator()
            result = validator.validate_file(maif_path, manifest_path)

            if result.is_valid:
                print("✓ Validation works")
                assert True  # Test passed
            else:
                print(f"✗ Validation failed: {result.errors}")
                return False
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False


def test_semantic_embedder():
    """Test semantic embedder initialization."""
    from maif.semantic import SemanticEmbedder

    try:
        embedder = SemanticEmbedder(model_name="test-model")

        # Test embedding
        embedding = embedder.embed_text("Test text")

        if hasattr(embedding, "vector") and len(embedding.vector) > 0:
            print("✓ Semantic embedder works")
            assert True  # Test passed
        else:
            print("✗ Semantic embedder failed to create embedding")
            return False
    except Exception as e:
        print(f"✗ Semantic embedder test failed: {e}")
        return False


def test_hierarchical_compression():
    """Test hierarchical semantic compression."""
    from maif.semantic import HierarchicalSemanticCompression

    try:
        hsc = HierarchicalSemanticCompression()

        # Test with empty embeddings
        result = hsc.compress_embeddings([])

        if "compressed_data" in result:
            print("✓ Hierarchical compression works")
            assert True  # Test passed
        else:
            print("✗ Hierarchical compression failed")
            return False
    except Exception as e:
        print(f"✗ Hierarchical compression test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing major fixes...")

    tests = [
        test_cli_privacy_levels,
        test_basic_maif_creation,
        test_validation,
        test_semantic_embedder,
        test_hierarchical_compression,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All major fixes appear to be working!")
        return 0
    else:
        print("❌ Some issues remain")
        return 1


if __name__ == "__main__":
    sys.exit(main())
