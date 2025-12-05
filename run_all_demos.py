#!/usr/bin/env python3
"""
Run all MAIF demos (except langgraph) and collect output files in one folder.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Change to the maif root directory
MAIF_ROOT = Path(__file__).parent
os.chdir(MAIF_ROOT)

# Create output folder for all .maif files
OUTPUT_DIR = MAIF_ROOT / "demo_output"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

print("=" * 70)
print("MAIF DEMO RUNNER")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

# List of demos to run (excluding langgraph as requested)
DEMOS = [
    # Basic demos
    ("Basic Usage", "examples/basic/basic_usage.py"),
    ("Simple API Demo", "examples/basic/simple_api_demo.py"),
    ("Secure MAIF Demo", "examples/basic/secure_maif_demo.py"),
    # Security demos
    ("Privacy Demo", "examples/security/privacy_demo.py"),
    ("Classified API Demo", "examples/security/classified_api_simple_demo.py"),
    # Advanced demos
    ("Video Demo", "examples/advanced/video_demo.py"),
    ("Versioning Demo", "examples/advanced/versioning_demo.py"),
    ("Advanced Features", "examples/advanced/advanced_features_demo.py"),
    ("Hybrid Architecture", "examples/advanced/hybrid_architecture_demo.py"),
    ("Agent Demo", "examples/advanced/maif_agent_demo.py"),
    ("Lifecycle Management", "examples/advanced/lifecycle_management_demo.py"),
    ("Enhanced Lifecycle", "examples/advanced/enhanced_lifecycle_demo.py"),
    ("Novel Algorithms", "examples/advanced/novel_algorithms_demo.py"),
    ("Agent State Restoration", "examples/advanced/agent_state_restoration_demo.py"),
    ("Integrated Features", "examples/advanced/integrated_features_demo.py"),
    # Skipped: multi_agent_consortium_demo.py (3895 lines, very complex)
]

# Track results
results = []


def run_demo(name: str, script_path: str) -> bool:
    """Run a single demo and return success status."""
    print(f"\n{'=' * 70}")
    print(f"Running: {name}")
    print(f"Script: {script_path}")
    print("-" * 70)

    try:
        # Set up environment with PYTHONPATH pointing to maif root
        env = os.environ.copy()
        pythonpath = str(MAIF_ROOT)
        if "PYTHONPATH" in env:
            pythonpath = f"{pythonpath}:{env['PYTHONPATH']}"
        env["PYTHONPATH"] = pythonpath

        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=MAIF_ROOT,
            env=env,
            capture_output=False,
            timeout=300,  # 5 minute timeout
        )

        success = result.returncode == 0
        print(f"\n{'✅ PASSED' if success else '❌ FAILED'}: {name}")
        return success

    except subprocess.TimeoutExpired:
        print(f"\n⏱️ TIMEOUT: {name}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {name} - {e}")
        return False


def collect_maif_files():
    """Collect all .maif files from the root and move to output folder."""
    print(f"\n{'=' * 70}")
    print("Collecting .maif files...")
    print("-" * 70)

    collected = 0

    # Check current directory and subdirectories
    search_paths = [
        MAIF_ROOT,
        MAIF_ROOT / "examples" / "basic",
        MAIF_ROOT / "examples" / "security",
        MAIF_ROOT / "examples" / "advanced",
        MAIF_ROOT / "demo_workspace",
    ]

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for maif_file in search_path.glob("*.maif"):
            dest = OUTPUT_DIR / maif_file.name
            # Handle duplicate names
            if dest.exists():
                base = maif_file.stem
                ext = maif_file.suffix
                i = 1
                while dest.exists():
                    dest = OUTPUT_DIR / f"{base}_{i}{ext}"
                    i += 1

            shutil.move(str(maif_file), str(dest))
            print(f"  Moved: {maif_file.name} -> {dest.name}")
            collected += 1

        # Note: We no longer need manifest files - the new format is self-contained!

    print(f"\nCollected {collected} files to {OUTPUT_DIR}")
    return collected


def main():
    """Run all demos."""

    # Run each demo
    for name, script in DEMOS:
        success = run_demo(name, script)
        results.append((name, success))

    # Collect output files
    collect_maif_files()

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")

    print(f"\n{passed}/{total} demos passed")

    # List output files
    print(f"\nOutput files in {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")

    print(f"\nTo clean up: rm -rf {OUTPUT_DIR}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
