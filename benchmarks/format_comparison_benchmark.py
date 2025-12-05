#!/usr/bin/env python3
"""
MAIF Format Comparison Benchmark

Compares different configurations of the secure MAIF format to measure
baseline performance metrics and demonstrate the secure format capabilities.

Note: Both "legacy" and "secure" now use the same SecureMAIFWriter API,
as the v3 format is self-contained with Ed25519 signatures.

Metrics compared:
1. File creation speed
2. File size overhead
3. Read/decode speed
4. Integrity verification speed
5. Tamper detection capability
6. Scalability with block count
"""

import os
import sys
import time
import json
import tempfile
import statistics
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import random
import string

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from maif.core import MAIFEncoder, MAIFDecoder
from maif.secure_format import SecureMAIFWriter, SecureMAIFReader, SecureBlockType


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""

    name: str
    legacy_value: float
    secure_value: float
    unit: str
    better_is: str = "lower"  # or "higher"

    @property
    def improvement(self) -> float:
        """Calculate improvement percentage (positive = secure is better)."""
        if self.better_is == "lower":
            if self.secure_value == 0:
                return 100.0 if self.legacy_value > 0 else 0.0
            return ((self.legacy_value - self.secure_value) / self.legacy_value) * 100
        else:
            if self.legacy_value == 0:
                return 100.0 if self.secure_value > 0 else 0.0
            return ((self.secure_value - self.legacy_value) / self.legacy_value) * 100

    @property
    def winner(self) -> str:
        if self.better_is == "lower":
            return "Secure" if self.secure_value < self.legacy_value else "Legacy"
        else:
            return "Secure" if self.secure_value > self.legacy_value else "Legacy"


class FormatComparisonBenchmark:
    """
    Benchmark suite for the secure MAIF format (v3).

    Note: Both "legacy" and "secure" test cases now use SecureMAIFWriter,
    as the v3 format is self-contained with Ed25519 signatures.
    This benchmark measures baseline performance characteristics.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics: List[BenchmarkMetrics] = []
        self.detailed_results: Dict[str, Any] = {}

        # Test data sizes
        self.text_sizes = [100, 1000, 10000, 100000]  # bytes
        self.block_counts = [1, 10, 50, 100]
        self.iterations = 5  # Number of iterations for timing

    def _generate_random_text(self, length: int) -> str:
        """Generate random text of specified length."""
        chars = string.ascii_letters + string.digits + " \n"
        return "".join(random.choice(chars) for _ in range(length))

    def _generate_test_data(self, text_size: int, block_count: int) -> List[str]:
        """Generate test data blocks."""
        return [self._generate_random_text(text_size) for _ in range(block_count)]

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all comparison benchmarks."""
        print("\n" + "=" * 80)
        print("MAIF FORMAT COMPARISON BENCHMARK")
        print("Legacy (External Manifest) vs Secure (Ed25519, Self-Contained)")
        print("=" * 80)

        # Run benchmarks
        self._benchmark_file_creation()
        self._benchmark_file_size()
        self._benchmark_read_performance()
        self._benchmark_integrity_verification()
        self._benchmark_tamper_detection()
        self._benchmark_scalability()
        self._benchmark_security_features()

        # Generate report
        return self._generate_report()

    def _benchmark_file_creation(self):
        """Benchmark file creation speed for both formats."""
        print("\n--- File Creation Speed ---")

        legacy_times = []
        secure_times = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for size in self.text_sizes:
                test_data = self._generate_test_data(size, 10)

                # Legacy format timing
                legacy_times_for_size = []
                for i in range(self.iterations):
                    start = time.time()
                    maif_path = os.path.join(tmpdir, f"legacy_{size}_{i}.maif")
                    encoder = MAIFEncoder(maif_path, agent_id="benchmark-agent")
                    for text in test_data:
                        encoder.add_text_block(text)
                    encoder.finalize()
                    legacy_times_for_size.append(time.time() - start)
                legacy_times.extend(legacy_times_for_size)

                # Secure format timing
                secure_times_for_size = []
                for i in range(self.iterations):
                    start = time.time()
                    writer = SecureMAIFWriter(
                        os.path.join(tmpdir, f"secure_{size}_{i}.maif"),
                        agent_id="benchmark-agent",
                    )
                    for text in test_data:
                        writer.add_text_block(text)
                    writer.finalize()
                    secure_times_for_size.append(time.time() - start)
                secure_times.extend(secure_times_for_size)

                print(
                    f"  Size {size:,} bytes: Legacy {statistics.mean(legacy_times_for_size) * 1000:.2f}ms, "
                    f"Secure {statistics.mean(secure_times_for_size) * 1000:.2f}ms"
                )

        avg_legacy = statistics.mean(legacy_times) * 1000
        avg_secure = statistics.mean(secure_times) * 1000

        self.metrics.append(
            BenchmarkMetrics(
                name="File Creation Time",
                legacy_value=avg_legacy,
                secure_value=avg_secure,
                unit="ms",
                better_is="lower",
            )
        )

        self.detailed_results["file_creation"] = {
            "legacy_times_ms": [t * 1000 for t in legacy_times],
            "secure_times_ms": [t * 1000 for t in secure_times],
            "legacy_avg_ms": avg_legacy,
            "secure_avg_ms": avg_secure,
        }

        print(
            f"  ✓ Average: Legacy {avg_legacy:.2f}ms, Secure (Ed25519) {avg_secure:.2f}ms"
        )

    def _benchmark_file_size(self):
        """Benchmark file size overhead for both formats."""
        print("\n--- File Size Comparison ---")

        legacy_sizes = []
        secure_sizes = []
        data_sizes = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for size in self.text_sizes:
                for block_count in [5, 20]:
                    test_data = self._generate_test_data(size, block_count)
                    raw_data_size = sum(len(t.encode("utf-8")) for t in test_data)
                    data_sizes.append(raw_data_size)

                    # Legacy format (now using SecureMAIFWriter API)
                    maif_path = os.path.join(
                        tmpdir, f"legacy_{size}_{block_count}.maif"
                    )
                    encoder = MAIFEncoder(maif_path, agent_id="benchmark-agent")
                    for text in test_data:
                        encoder.add_text_block(text)
                    encoder.finalize()
                    legacy_total = os.path.getsize(maif_path)
                    legacy_sizes.append(legacy_total)

                    # Secure format
                    secure_path = os.path.join(
                        tmpdir, f"secure_{size}_{block_count}.maif"
                    )
                    writer = SecureMAIFWriter(secure_path, agent_id="benchmark-agent")
                    for text in test_data:
                        writer.add_text_block(text)
                    writer.finalize()
                    secure_total = os.path.getsize(secure_path)
                    secure_sizes.append(secure_total)

                    legacy_overhead = (
                        (legacy_total - raw_data_size) / raw_data_size
                    ) * 100
                    secure_overhead = (
                        (secure_total - raw_data_size) / raw_data_size
                    ) * 100

                    print(
                        f"  {block_count} blocks × {size:,}B: Legacy {legacy_total:,}B ({legacy_overhead:.1f}% overhead), "
                        f"Secure {secure_total:,}B ({secure_overhead:.1f}% overhead)"
                    )

        avg_legacy_overhead = statistics.mean(
            [((ls - ds) / ds) * 100 for ls, ds in zip(legacy_sizes, data_sizes)]
        )
        avg_secure_overhead = statistics.mean(
            [((ss - ds) / ds) * 100 for ss, ds in zip(secure_sizes, data_sizes)]
        )

        self.metrics.append(
            BenchmarkMetrics(
                name="File Size Overhead",
                legacy_value=avg_legacy_overhead,
                secure_value=avg_secure_overhead,
                unit="%",
                better_is="lower",
            )
        )

        self.detailed_results["file_size"] = {
            "legacy_sizes": legacy_sizes,
            "secure_sizes": secure_sizes,
            "raw_data_sizes": data_sizes,
            "legacy_avg_overhead_pct": avg_legacy_overhead,
            "secure_avg_overhead_pct": avg_secure_overhead,
        }

        print(
            f"  ✓ Average overhead: Legacy {avg_legacy_overhead:.1f}%, Secure {avg_secure_overhead:.1f}%"
        )

    def _benchmark_read_performance(self):
        """Benchmark file read/decode performance."""
        print("\n--- Read Performance ---")

        legacy_times = []
        secure_times = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for size in [1000, 10000, 50000]:
                test_data = self._generate_test_data(size, 20)

                # Create files (both formats now use same API)
                legacy_path = os.path.join(tmpdir, f"legacy_{size}.maif")
                encoder = MAIFEncoder(legacy_path, agent_id="benchmark-agent")
                for text in test_data:
                    encoder.add_text_block(text)
                encoder.finalize()

                secure_path = os.path.join(tmpdir, f"secure_{size}.maif")
                writer = SecureMAIFWriter(secure_path, agent_id="benchmark-agent")
                for text in test_data:
                    writer.add_text_block(text)
                writer.finalize()

                # Benchmark reading
                legacy_read_times = []
                for _ in range(self.iterations):
                    start = time.time()
                    decoder = MAIFDecoder(legacy_path)
                    decoder.load()
                    blocks = decoder.get_blocks()
                    legacy_read_times.append(time.time() - start)
                legacy_times.extend(legacy_read_times)

                secure_read_times = []
                for _ in range(self.iterations):
                    start = time.time()
                    reader = SecureMAIFReader(secure_path)
                    reader.load()
                    _blocks = reader.get_blocks()  # noqa: F841
                    secure_read_times.append(time.time() - start)
                secure_times.extend(secure_read_times)

                print(
                    f"  Size {size:,}: Legacy {statistics.mean(legacy_read_times) * 1000:.2f}ms, "
                    f"Secure {statistics.mean(secure_read_times) * 1000:.2f}ms"
                )

        avg_legacy = statistics.mean(legacy_times) * 1000
        avg_secure = statistics.mean(secure_times) * 1000

        self.metrics.append(
            BenchmarkMetrics(
                name="Read/Decode Time",
                legacy_value=avg_legacy,
                secure_value=avg_secure,
                unit="ms",
                better_is="lower",
            )
        )

        self.detailed_results["read_performance"] = {
            "legacy_avg_ms": avg_legacy,
            "secure_avg_ms": avg_secure,
        }

        print(f"  ✓ Average: Legacy {avg_legacy:.2f}ms, Secure {avg_secure:.2f}ms")

    def _benchmark_integrity_verification(self):
        """Benchmark integrity verification speed."""
        print("\n--- Integrity Verification ---")

        legacy_times = []
        secure_times = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for block_count in [10, 50, 100]:
                test_data = self._generate_test_data(5000, block_count)

                # Create files (both formats now use same API)
                legacy_path = os.path.join(tmpdir, f"legacy_{block_count}.maif")
                encoder = MAIFEncoder(legacy_path, agent_id="benchmark-agent")
                for text in test_data:
                    encoder.add_text_block(text)
                encoder.finalize()

                secure_path = os.path.join(tmpdir, f"secure_{block_count}.maif")
                writer = SecureMAIFWriter(secure_path, agent_id="benchmark-agent")
                for text in test_data:
                    writer.add_text_block(text)
                writer.finalize()

                # Benchmark verification
                legacy_verify_times = []
                for _ in range(self.iterations):
                    start = time.time()
                    decoder = MAIFDecoder(legacy_path)
                    is_valid, errors = decoder.verify_integrity()
                    legacy_verify_times.append(time.time() - start)
                legacy_times.extend(legacy_verify_times)

                secure_verify_times = []
                for _ in range(self.iterations):
                    start = time.time()
                    reader = SecureMAIFReader(secure_path)
                    is_valid, errors = reader.verify_integrity()
                    secure_verify_times.append(time.time() - start)
                secure_times.extend(secure_verify_times)

                print(
                    f"  {block_count} blocks: Legacy {statistics.mean(legacy_verify_times) * 1000:.2f}ms, "
                    f"Secure {statistics.mean(secure_verify_times) * 1000:.2f}ms"
                )

        avg_legacy = statistics.mean(legacy_times) * 1000
        avg_secure = statistics.mean(secure_times) * 1000

        self.metrics.append(
            BenchmarkMetrics(
                name="Integrity Verification",
                legacy_value=avg_legacy,
                secure_value=avg_secure,
                unit="ms",
                better_is="lower",
            )
        )

        self.detailed_results["integrity_verification"] = {
            "legacy_avg_ms": avg_legacy,
            "secure_avg_ms": avg_secure,
        }

        print(f"  ✓ Average: Legacy {avg_legacy:.2f}ms, Secure {avg_secure:.2f}ms")

    def _benchmark_tamper_detection(self):
        """Benchmark tamper detection capabilities."""
        print("\n--- Tamper Detection ---")

        legacy_detections = 0
        secure_detections = 0
        total_tests = 0

        legacy_detection_times = []
        secure_detection_times = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for test_num in range(20):
                test_data = self._generate_test_data(1000, 10)
                total_tests += 1

                # Create legacy file (now uses secure API)
                legacy_path = os.path.join(tmpdir, f"legacy_tamper_{test_num}.maif")
                encoder = MAIFEncoder(legacy_path, agent_id="benchmark-agent")
                for text in test_data:
                    encoder.add_text_block(text)
                encoder.finalize()

                # Create secure file
                secure_path = os.path.join(tmpdir, f"secure_tamper_{test_num}.maif")
                writer = SecureMAIFWriter(secure_path, agent_id="benchmark-agent")
                for text in test_data:
                    writer.add_text_block(text)
                writer.finalize()

                # Tamper with both files (modify bytes in data section)
                legacy_size = os.path.getsize(legacy_path)
                secure_size = os.path.getsize(secure_path)

                # Tamper legacy file
                if legacy_size > 100:
                    tamper_offset = min(100, legacy_size - 10)
                    with open(legacy_path, "r+b") as f:
                        f.seek(tamper_offset)
                        f.write(b"TAMPERED!")

                # Tamper secure file (in data section after headers)
                if secure_size > 900:
                    tamper_offset = 900  # After file header (444) + block header (372)
                    with open(secure_path, "r+b") as f:
                        f.seek(tamper_offset)
                        f.write(b"TAMPERED!")

                # Test legacy detection
                start = time.time()
                try:
                    decoder = MAIFDecoder(legacy_path)
                    is_valid, errors = decoder.verify_integrity()
                    if not is_valid:
                        legacy_detections += 1
                except Exception:
                    legacy_detections += 1  # Exception counts as detection
                legacy_detection_times.append(time.time() - start)

                # Test secure detection
                start = time.time()
                reader = SecureMAIFReader(secure_path)
                is_valid, errors = reader.verify_integrity()
                if not is_valid or reader.is_tampered():
                    secure_detections += 1
                secure_detection_times.append(time.time() - start)

        legacy_rate = (legacy_detections / total_tests) * 100
        secure_rate = (secure_detections / total_tests) * 100

        self.metrics.append(
            BenchmarkMetrics(
                name="Tamper Detection Rate",
                legacy_value=legacy_rate,
                secure_value=secure_rate,
                unit="%",
                better_is="higher",
            )
        )

        avg_legacy_time = statistics.mean(legacy_detection_times) * 1000
        avg_secure_time = statistics.mean(secure_detection_times) * 1000

        self.metrics.append(
            BenchmarkMetrics(
                name="Tamper Detection Time",
                legacy_value=avg_legacy_time,
                secure_value=avg_secure_time,
                unit="ms",
                better_is="lower",
            )
        )

        self.detailed_results["tamper_detection"] = {
            "total_tests": total_tests,
            "legacy_detections": legacy_detections,
            "secure_detections": secure_detections,
            "legacy_rate_pct": legacy_rate,
            "secure_rate_pct": secure_rate,
            "legacy_avg_time_ms": avg_legacy_time,
            "secure_avg_time_ms": avg_secure_time,
        }

        print(
            f"  ✓ Detection rate: Legacy {legacy_rate:.1f}%, Secure {secure_rate:.1f}%"
        )
        print(
            f"  ✓ Detection time: Legacy {avg_legacy_time:.2f}ms, Secure {avg_secure_time:.2f}ms"
        )

    def _benchmark_scalability(self):
        """Benchmark scalability with increasing block count."""
        print("\n--- Scalability (by block count) ---")

        scalability_results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            for block_count in [10, 50, 100, 200]:
                test_data = self._generate_test_data(2000, block_count)

                # Legacy (now uses secure API)
                legacy_path = os.path.join(tmpdir, f"scale_legacy_{block_count}.maif")
                start = time.time()
                encoder = MAIFEncoder(legacy_path, agent_id="benchmark-agent")
                for text in test_data:
                    encoder.add_text_block(text)
                encoder.finalize()
                legacy_create_time = time.time() - start

                # Secure
                start = time.time()
                secure_path = os.path.join(tmpdir, f"scale_secure_{block_count}.maif")
                writer = SecureMAIFWriter(secure_path, agent_id="benchmark-agent")
                for text in test_data:
                    writer.add_text_block(text)
                writer.finalize()
                secure_create_time = time.time() - start

                legacy_size = os.path.getsize(legacy_path)
                secure_size = os.path.getsize(secure_path)

                scalability_results[block_count] = {
                    "legacy_create_ms": legacy_create_time * 1000,
                    "secure_create_ms": secure_create_time * 1000,
                    "legacy_size_kb": legacy_size / 1024,
                    "secure_size_kb": secure_size / 1024,
                }

                print(
                    f"  {block_count} blocks: Legacy {legacy_create_time * 1000:.1f}ms/{legacy_size / 1024:.1f}KB, "
                    f"Secure {secure_create_time * 1000:.1f}ms/{secure_size / 1024:.1f}KB"
                )

        self.detailed_results["scalability"] = scalability_results

    def _benchmark_security_features(self):
        """Compare security features availability."""
        print("\n--- Security Features ---")

        features = {
            "Block-level signatures": (False, True),
            "Self-contained provenance": (False, True),
            "Merkle root integrity": (False, True),
            "Embedded public key": (False, True),
            "Chain linking (blocks)": (False, True),
            "Immutable blocks": (False, True),
            "External manifest required": (True, False),  # True for legacy is worse
            "Cryptographic tamper detection": (False, True),
        }

        legacy_score = 0
        secure_score = 0

        for feature, (legacy_has, secure_has) in features.items():
            legacy_str = "✓" if legacy_has else "✗"
            secure_str = "✓" if secure_has else "✗"

            # Score: having the feature is good, except "External manifest required"
            if feature == "External manifest required":
                if not legacy_has:
                    legacy_score += 1
                if not secure_has:
                    secure_score += 1
            else:
                if legacy_has:
                    legacy_score += 1
                if secure_has:
                    secure_score += 1

            print(f"  {feature}: Legacy {legacy_str}, Secure {secure_str}")

        max_score = len(features)

        self.metrics.append(
            BenchmarkMetrics(
                name="Security Features Score",
                legacy_value=legacy_score,
                secure_value=secure_score,
                unit=f"/{max_score}",
                better_is="higher",
            )
        )

        self.detailed_results["security_features"] = {
            "features": features,
            "legacy_score": legacy_score,
            "secure_score": secure_score,
            "max_score": max_score,
        }

        print(
            f"  ✓ Score: Legacy {legacy_score}/{max_score}, Secure {secure_score}/{max_score}"
        )

    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)

        report = {
            "timestamp": time.time(),
            "metrics": [],
            "detailed_results": self.detailed_results,
            "summary": {},
        }

        secure_wins = 0
        legacy_wins = 0

        print(
            "\n{:<30} {:>15} {:>15} {:>12} {:>10}".format(
                "Metric", "Legacy", "Secure", "Winner", "Diff"
            )
        )
        print("-" * 82)

        for metric in self.metrics:
            winner = metric.winner
            if winner == "Secure":
                secure_wins += 1
            else:
                legacy_wins += 1

            diff_str = f"{abs(metric.improvement):.1f}%"
            if metric.improvement > 0:
                diff_str = f"+{diff_str}"
            else:
                diff_str = f"-{diff_str}"

            legacy_str = f"{metric.legacy_value:.2f}{metric.unit}"
            secure_str = f"{metric.secure_value:.2f}{metric.unit}"

            print(
                "{:<30} {:>15} {:>15} {:>12} {:>10}".format(
                    metric.name, legacy_str, secure_str, winner, diff_str
                )
            )

            report["metrics"].append(
                {
                    "name": metric.name,
                    "legacy_value": metric.legacy_value,
                    "secure_value": metric.secure_value,
                    "unit": metric.unit,
                    "winner": winner,
                    "improvement_pct": metric.improvement,
                }
            )

        print("-" * 82)
        print(
            f"\nOverall: Secure wins {secure_wins}/{len(self.metrics)}, Legacy wins {legacy_wins}/{len(self.metrics)}"
        )

        report["summary"] = {
            "secure_wins": secure_wins,
            "legacy_wins": legacy_wins,
            "total_metrics": len(self.metrics),
            "recommendation": self._get_recommendation(secure_wins, legacy_wins),
        }

        # Print recommendation
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\n{report['summary']['recommendation']}")

        # Save report
        report_path = self.output_dir / "format_comparison_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Detailed report saved to: {report_path}")

        return report

    def _get_recommendation(self, secure_wins: int, legacy_wins: int) -> str:
        """Generate recommendation based on results."""
        if secure_wins > legacy_wins:
            return """
The SECURE format is recommended for most use cases:

Advantages:
  ✓ Self-contained - no external manifest files needed
  ✓ Block-level cryptographic signatures for immutability  
  ✓ Built-in tamper detection with detailed error reporting
  ✓ Embedded provenance chain for full audit trail
  ✓ Merkle root for fast integrity verification

Trade-offs:
  • Higher file size overhead due to signatures (~256 bytes/block)
  • Slower creation time due to cryptographic operations
  
Use SECURE format when:
  • Security and integrity are critical
  • Audit trails are required
  • Files may be shared or stored externally
  • Tamper detection is important

Use LEGACY format when:
  • Maximum write performance is critical
  • File size is constrained  
  • Operating in a trusted environment
  • Backward compatibility is required
"""
        else:
            return """
The LEGACY format may be suitable for performance-critical applications,
but consider the security trade-offs carefully.
"""


def main():
    """Run format comparison benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="MAIF Format Comparison Benchmark")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark with fewer iterations"
    )

    args = parser.parse_args()

    benchmark = FormatComparisonBenchmark(args.output_dir)

    if args.quick:
        benchmark.text_sizes = [1000, 10000]
        benchmark.block_counts = [5, 20]
        benchmark.iterations = 3
        print("Running in quick mode with reduced iterations")

    try:
        _report = benchmark.run_all_benchmarks()  # noqa: F841
        return 0
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
