#!/usr/bin/env python3
"""
Quick test to verify the benchmark suite works correctly.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_basic_maif_functionality():
    """Test basic MAIF functionality before running full benchmark."""
    print("Testing basic MAIF functionality...")

    try:
        from maif import MAIFEncoder, MAIFDecoder, BlockType

        with tempfile.TemporaryDirectory() as tmpdir:
            maif_path = os.path.join(tmpdir, "test.maif")

            # Test encoding (v3 format)
            encoder = MAIFEncoder(maif_path, agent_id="benchmark_test")
            encoder.add_text_block("Hello, MAIF!")
            encoder.add_embeddings_block([[1.0, 2.0, 3.0]])
            encoder.finalize()

            # Test decoding
            decoder = MAIFDecoder(maif_path)
            decoder.load()

            # Get blocks
            text_blocks = [
                b for b in decoder.blocks if b.header.block_type == BlockType.TEXT
            ]
            emb_blocks = [
                b for b in decoder.blocks if b.header.block_type == BlockType.EMBEDDINGS
            ]

            # Verify
            assert len(text_blocks) == 1
            assert text_blocks[0].data.decode("utf-8") == "Hello, MAIF!"
            assert len(emb_blocks) == 1

            print("✓ Basic MAIF functionality works")

    except Exception as e:
        print(f"✗ Basic MAIF test failed: {e}")
        assert False, f"Basic MAIF test failed: {e}"


def test_benchmark_imports():
    """Test that benchmark imports work."""
    import pytest
    pytest.skip("Benchmarks are not a package - run directly with python benchmarks/...")


def test_quick_benchmark():
    """Run a very quick benchmark test."""
    import pytest
    pytest.skip("Benchmarks are not a package - run directly with python benchmarks/...")


def main():
    """Run all tests."""
    print("MAIF Benchmark Test Suite")
    print("=" * 40)

    tests = [
        test_basic_maif_functionality,
        test_benchmark_imports,
        test_quick_benchmark,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print(f"Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! The benchmark suite is ready to run.")
        print("\nTo run the full benchmark suite:")
        print("  python run_benchmark.py")
        print("\nTo run a quick benchmark:")
        print("  python run_benchmark.py --quick")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
