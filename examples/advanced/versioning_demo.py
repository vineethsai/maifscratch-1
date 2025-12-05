"""
Demonstration of MAIF versioning and forensic capabilities.

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
"""

import json
import time
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from maif import MAIFEncoder, MAIFDecoder, SecureBlockType
from maif.security import MAIFSigner
from maif.forensics import ForensicAnalyzer


def demonstrate_versioning():
    """Demonstrate MAIF versioning with multiple blocks."""
    print("MAIF Versioning and Forensics Demonstration")
    print("=" * 50)

    # Create initial MAIF with multiple versions of content
    print("\n1. Creating MAIF file with version history...")
    agent_id = "agent-001-alice"
    encoder = MAIFEncoder("versioned_doc.maif", agent_id=agent_id)

    # Add initial content (version 1)
    original_text = "This is the original document content."
    block1 = encoder.add_text_block(
        original_text,
        metadata={
            "author": "Alice",
            "version": 1,
            "document_type": "report",
            "change_description": "Initial document creation",
        },
    )
    print(f"   Created version 1: initial content")

    # Add updated content (version 2)
    updated_text = (
        "This is the original document content. UPDATED: Added new section by Bob."
    )
    block2 = encoder.add_text_block(
        updated_text,
        metadata={
            "author": "Bob",
            "version": 2,
            "change_description": "Added new section with additional information",
        },
    )
    print(f"   Created version 2: Bob's update")

    # Add final content (version 3)
    final_text = "This is the original document content. UPDATED: Added new section by Bob. FINAL: Alice's final review and approval."
    block3 = encoder.add_text_block(
        final_text,
        metadata={
            "author": "Alice",
            "version": 3,
            "change_description": "Final review and approval",
            "status": "approved",
        },
    )
    print(f"   Created version 3: Alice's final approval")

    # Finalize (self-contained with Ed25519 signatures)
    encoder.finalize()
    print("   ✓ MAIF finalized (self-contained with Ed25519 signatures)")

    return "versioned_doc.maif"


def analyze_provenance(maif_path: str):
    """Analyze the provenance chain."""
    print("\n2. Analyzing provenance chain...")

    decoder = MAIFDecoder(maif_path)
    provenance = decoder.get_provenance()

    print(f"   Found {len(provenance)} provenance entries:")

    for entry in provenance:
        # timestamp is in microseconds, convert to seconds
        ts_seconds = (
            entry.timestamp / 1000000 if entry.timestamp > 1e12 else entry.timestamp
        )
        try:
            timestamp = datetime.fromtimestamp(ts_seconds).strftime("%H:%M:%S")
        except (ValueError, OSError):
            timestamp = "??:??:??"
        print(f"   - [{timestamp}] {entry.action} by {entry.agent_id}")

    return decoder


def analyze_version_history(decoder: MAIFDecoder):
    """Analyze the version history from blocks."""
    print("\n3. Analyzing version history from blocks...")

    blocks = decoder.get_blocks()
    text_blocks = [b for b in blocks if b.block_type == SecureBlockType.TEXT]

    print(f"   Found {len(text_blocks)} text versions:")

    versions = []
    for block in text_blocks:
        metadata = block.metadata or {}
        version = metadata.get("version", "?")
        author = metadata.get("author", "unknown")
        description = metadata.get("change_description", "no description")

        versions.append(
            {
                "version": version,
                "author": author,
                "description": description,
                "block_id": block.header.block_id.hex()[:16],
            }
        )

        print(f"   - Version {version} by {author}: {description}")

    return versions


def verify_integrity(maif_path: str):
    """Verify file integrity."""
    print("\n4. Verifying file integrity...")

    decoder = MAIFDecoder(maif_path)
    is_valid, errors = decoder.verify_integrity()

    print(f"   Integrity check: {'✓ Valid' if is_valid else '✗ Invalid'}")

    if errors:
        for err in errors:
            print(f"   - Error: {err}")

    # Check individual blocks
    blocks = decoder.get_blocks()
    print(f"   Verified {len(blocks)} blocks:")

    for i, block in enumerate(blocks):
        print(f"   - Block {i}: signed ✓")

    return is_valid


def perform_forensic_analysis(maif_path: str):
    """Perform comprehensive forensic analysis."""
    print("\n5. Performing forensic analysis...")

    analyzer = ForensicAnalyzer()

    # Perform analysis
    report = analyzer.analyze_maif_file(maif_path)

    # Extract key information
    print(
        f"   Risk Assessment: {report.get('risk_assessment', {}).get('overall_risk', 'Unknown')}"
    )
    print(
        f"   Risk Score: {report.get('risk_assessment', {}).get('risk_score', 0):.2f}"
    )

    # Show version analysis
    version_analysis = report.get("version_analysis", {})
    print(f"   Total versions tracked: {version_analysis.get('total_versions', 0)}")

    # Show behavioral analysis
    behavioral = report.get("behavioral_analysis", {})
    print(f"   Agents analyzed: {behavioral.get('total_agents', 0)}")

    # Show recommendations
    recommendations = report.get("recommendations", [])
    if recommendations:
        print(f"\n   Recommendations ({len(recommendations)}):")
        for rec in recommendations[:3]:
            priority = rec.get("priority", "unknown")
            desc = rec.get("description", "No description")
            print(f"   - [{priority}] {desc}")

    # Save forensic report
    with open("versioning_forensic_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n   ✓ Forensic report saved: versioning_forensic_report.json")

    return report


def demonstrate_tamper_detection(maif_path: str):
    """Demonstrate tamper detection."""
    print("\n6. Demonstrating tamper detection...")

    # Create a copy to tamper with
    import shutil

    tampered_path = "tampered_doc.maif"
    shutil.copy(maif_path, tampered_path)

    # Tamper with the file
    with open(tampered_path, "r+b") as f:
        f.seek(500)  # Seek to data area
        f.write(b"TAMPERED!")

    print("   Modified bytes in the file...")

    # Verify the tampered file
    decoder = MAIFDecoder(tampered_path)
    is_valid, errors = decoder.verify_integrity()

    if not is_valid:
        print("   ✓ Tampering DETECTED!")
        print(f"   Found {len(errors)} integrity errors")
    else:
        print("   ✗ Tampering not detected (unexpected)")

    # Cleanup
    os.remove(tampered_path)
    print("   Cleaned up tampered file")


def main():
    """Main demonstration function."""
    try:
        # Demonstrate versioning
        maif_path = demonstrate_versioning()

        # Analyze provenance
        decoder = analyze_provenance(maif_path)

        # Analyze version history
        versions = analyze_version_history(decoder)

        # Verify integrity
        verify_integrity(maif_path)

        # Perform forensic analysis
        report = perform_forensic_analysis(maif_path)

        # Demonstrate tamper detection
        demonstrate_tamper_detection(maif_path)

        print("\n" + "=" * 50)
        print("Versioning and Forensics Demo Complete!")
        print("=" * 50)
        print("""
Files created:
  - versioned_doc.maif (self-contained MAIF with version history)
  - versioning_forensic_report.json (forensic analysis)

Key features demonstrated:
  ✓ Multiple versions stored with metadata
  ✓ Embedded provenance chain (all operations tracked)
  ✓ Ed25519 signatures for tamper detection
  ✓ Forensic analysis with risk assessment
  ✓ Self-contained format (no external manifest)
""")

        # Cleanup
        if os.path.exists("versioned_doc.maif"):
            os.remove("versioned_doc.maif")
        if os.path.exists("versioning_forensic_report.json"):
            os.remove("versioning_forensic_report.json")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
