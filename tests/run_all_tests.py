"""
Comprehensive test runner for the entire MAIF codebase.
"""

import pytest
import sys
import os
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_suite():
    """Run the complete MAIF test suite."""

    print("🧪 MAIF Comprehensive Test Suite")
    print("=" * 50)

    # Test modules to run
    test_modules = [
        "tests/test_core.py",
        "tests/test_security.py",
        "tests/test_privacy.py",
        "tests/test_semantic.py",
        "tests/test_compression.py",
        "tests/test_streaming.py",
        "tests/test_validation.py",
        "tests/test_integration.py",
        "tests/test_metadata.py",
        "tests/test_forensics.py",
        "tests/test_binary_format.py",
        "tests/test_cli.py",
    ]

    # Test configuration
    pytest_args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings for cleaner output
        "-x",  # Stop on first failure (optional)
    ]

    total_start_time = time.time()

    print(f"📋 Running {len(test_modules)} test modules...")
    print()

    # Run each test module
    results = {}

    for module in test_modules:
        if os.path.exists(module):
            print(f"🔍 Testing {module}...")
            start_time = time.time()

            # Run the specific test module
            result = pytest.main(pytest_args + [module])

            end_time = time.time()
            duration = end_time - start_time

            results[module] = {
                "result": result,
                "duration": duration,
                "status": "PASSED" if result == 0 else "FAILED",
            }

            print(f"   ✅ {module}: {results[module]['status']} ({duration:.2f}s)")
        else:
            print(f"   ⚠️  {module}: SKIPPED (file not found)")
            results[module] = {"result": -1, "duration": 0, "status": "SKIPPED"}

    total_end_time = time.time()
    total_duration = total_end_time - total_start_time

    # Print summary
    print()
    print("📊 Test Summary")
    print("=" * 50)

    passed = sum(1 for r in results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in results.values() if r["status"] == "FAILED")
    skipped = sum(1 for r in results.values() if r["status"] == "SKIPPED")

    print(f"Total modules: {len(test_modules)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Skipped: {skipped}")
    print(f"⏱️  Total time: {total_duration:.2f}s")

    # Detailed results
    if failed > 0:
        print()
        print("❌ Failed Tests:")
        for module, result in results.items():
            if result["status"] == "FAILED":
                print(f"   - {module}")

    # Performance summary
    print()
    print("⚡ Performance Summary:")
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["duration"], reverse=True
    )
    for module, result in sorted_results[:5]:  # Top 5 slowest
        if result["status"] != "SKIPPED":
            print(f"   {result['duration']:6.2f}s - {module}")

    print()

    # Overall result
    if failed == 0:
        print("🎉 All tests passed! MAIF codebase is ready for production.")
        return 0
    else:
        print(f"💥 {failed} test module(s) failed. Please review and fix issues.")
        return 1


def run_specific_tests():
    """Run specific test categories."""

    test_categories = {
        "core": ["tests/test_core.py"],
        "security": ["tests/test_security.py", "tests/test_privacy.py"],
        "performance": ["tests/test_compression.py", "tests/test_streaming.py"],
        "validation": ["tests/test_validation.py", "tests/test_forensics.py"],
        "integration": ["tests/test_integration.py", "tests/test_cli.py"],
        "semantic": ["tests/test_semantic.py"],
        "format": ["tests/test_binary_format.py", "tests/test_metadata.py"],
    }

    print("🎯 MAIF Test Categories")
    print("=" * 30)

    for category, modules in test_categories.items():
        print(f"{category}: {', '.join(modules)}")

    print()
    category = (
        input("Enter category to test (or 'all' for everything): ").strip().lower()
    )

    if category == "all":
        return run_test_suite()
    elif category in test_categories:
        modules = test_categories[category]
        existing_modules = [m for m in modules if os.path.exists(m)]

        if existing_modules:
            print(f"🔍 Running {category} tests...")
            result = pytest.main(["-v", "--tb=short"] + existing_modules)
            return result
        else:
            print(f"❌ No test files found for category: {category}")
            return 1
    else:
        print(f"❌ Unknown category: {category}")
        return 1


def run_quick_tests():
    """Run a quick subset of tests for rapid feedback."""

    quick_tests = [
        "tests/test_core.py::TestMAIFEncoder::test_add_text_block",
        "tests/test_security.py::TestMAIFSigner::test_sign_data",
        "tests/test_privacy.py::TestPrivacyEngine::test_encrypt_decrypt_aes_gcm",
        "tests/test_compression.py::TestMAIFCompressor::test_zlib_compression",
        "tests/test_validation.py::TestMAIFValidator::test_validate_valid_file",
    ]

    print("⚡ MAIF Quick Test Suite")
    print("=" * 30)
    print("Running essential tests for rapid feedback...")

    existing_tests = []
    for test in quick_tests:
        module = test.split("::")[0]
        if os.path.exists(module):
            existing_tests.append(test)

    if existing_tests:
        result = pytest.main(["-v", "--tb=short"] + existing_tests)

        if result == 0:
            print("✅ Quick tests passed! Core functionality is working.")
        else:
            print("❌ Quick tests failed! Check core functionality.")

        return result
    else:
        print("❌ No quick test files found.")
        return 1


def run_coverage_analysis():
    """Run tests with coverage analysis."""

    try:
        import coverage
    except ImportError:
        print("❌ Coverage package not installed. Install with: pip install coverage")
        return 1

    print("📊 MAIF Test Coverage Analysis")
    print("=" * 35)

    # Coverage configuration
    cov = coverage.Coverage(
        source=["maif"],
        omit=["*/tests/*", "*/test_*", "*/__pycache__/*", "*/venv/*", "*/env/*"],
    )

    cov.start()

    # Run tests
    result = run_test_suite()

    cov.stop()
    cov.save()

    # Generate coverage report
    print()
    print("📈 Coverage Report:")
    print("-" * 20)

    cov.report(show_missing=True)

    # Generate HTML report
    html_dir = "htmlcov"
    cov.html_report(directory=html_dir)
    print(f"📄 HTML coverage report generated in: {html_dir}/")

    return result


def main():
    """Main test runner entry point."""

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "quick":
            return run_quick_tests()
        elif command == "category":
            return run_specific_tests()
        elif command == "coverage":
            return run_coverage_analysis()
        elif command == "help":
            print_help()
            return 0
        else:
            print(f"❌ Unknown command: {command}")
            print_help()
            return 1
    else:
        return run_test_suite()


def print_help():
    """Print help information."""

    print("🧪 MAIF Test Runner")
    print("=" * 20)
    print()
    print("Usage: python tests/run_all_tests.py [command]")
    print()
    print("Commands:")
    print("  (none)     - Run all tests")
    print("  quick      - Run quick essential tests")
    print("  category   - Run tests by category")
    print("  coverage   - Run tests with coverage analysis")
    print("  help       - Show this help message")
    print()
    print("Examples:")
    print("  python tests/run_all_tests.py")
    print("  python tests/run_all_tests.py quick")
    print("  python tests/run_all_tests.py coverage")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
