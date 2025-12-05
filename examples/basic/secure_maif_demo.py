#!/usr/bin/env python3
"""
Secure MAIF Format Demo

Demonstrates the self-contained, immutable MAIF format with:
- Embedded security and provenance
- Block-level signatures (immutable on write)
- Tamper detection
"""

import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import json
import tempfile
import shutil
from maif import (
    MAIFEncoder as SecureMAIFWriter,
    MAIFDecoder as SecureMAIFReader,
    BlockType as SecureBlockType,
    verify_maif as verify_secure_maif,
)


def create_demo_file():
    """Create a demo secure MAIF file."""
    print("=" * 60)
    print("SECURE MAIF FORMAT DEMO")
    print("=" * 60)
    print()

    output_path = "demo_secure.maif"

    # Create writer
    print("Creating secure MAIF file...")
    writer = SecureMAIFWriter(
        file_path=output_path,
        agent_id="demo-agent-001",
        agent_did="did:maif:demo-agent-001",
    )

    # Add text blocks
    texts = [
        "Artificial intelligence is transforming how we work and live.",
        "Machine learning models require careful validation and testing.",
        "Trustworthy AI systems need transparency and accountability.",
    ]

    print("\nAdding text blocks (each signed immediately)...")
    for i, text in enumerate(texts):
        block_id = writer.add_text_block(
            text, metadata={"source": f"document_{i}.txt", "language": "en"}
        )
        print(f"Block {i + 1}: {block_id[:16]}... ({len(text)} bytes)")

    # Add embeddings
    print("\nAdding embeddings block...")
    # Fake embeddings for demo
    embeddings = [[0.1 * i + 0.01 * j for j in range(384)] for i in range(3)]
    emb_id = writer.add_embeddings_block(
        embeddings, metadata={"model": "all-MiniLM-L6-v2", "source": "demo"}
    )
    print(f"Embeddings: {emb_id[:16]}... ({len(embeddings)} vectors)")

    # Finalize (signs the entire file)
    print("\nFinalizing and signing file...")
    writer.finalize()

    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"File created: {output_path} ({file_size:,} bytes)")

    return output_path


def verify_file(file_path: str):
    """Verify a secure MAIF file."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    print(f"\nVerifying {file_path}...")

    reader = SecureMAIFReader(file_path)
    is_valid, errors = reader.verify_integrity()

    print(f"\nFile Information:")
    info = reader.get_file_info()
    print(f"Version: {info['version']}")
    print(f"File ID: {info['file_id'][:16]}...")
    print(f"Agent DID: {info['agent_did']}")
    print(f"Block Count: {info['block_count']}")
    print(f"Merkle Root: {info['merkle_root'][:32]}...")
    print(f"Signed: {'Yes' if info['is_signed'] else 'No'}")
    print(f"Finalized: {'Yes' if info['is_finalized'] else 'No'}")

    print(f"\nIntegrity Check:")
    if is_valid:
        print("File integrity VERIFIED - no tampering detected")
    else:
        print("VERIFICATION FAILED:")
        for error in errors:
            print(f"   - {error}")

    print(f"\nBlocks:")
    for i, block in enumerate(reader.get_blocks()):
        block_type = SecureBlockType(block.header.block_type).name
        status = "OK" if not (block.header.flags & 0x20) else "TAMPERED"
        print(f"[{i}] {block_type}: {len(block.data)} bytes - {status}")

    print(f"\nProvenance Chain ({len(reader.get_provenance())} entries):")
    for entry in reader.get_provenance():
        action = entry.action
        ts = entry.timestamp / 1000000  # Convert to seconds
        from datetime import datetime

        dt = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        print(f"[{entry.chain_position}] {dt} - {action}")

    return is_valid


def demonstrate_tamper_detection(original_path: str):
    """Demonstrate tamper detection by modifying a file."""
    print("\n" + "=" * 60)
    print("TAMPER DETECTION DEMO")
    print("=" * 60)

    # Create a copy to tamper with
    tampered_path = "tampered.maif"
    shutil.copy(original_path, tampered_path)

    print(f"\nTampering with {tampered_path}...")

    # Modify bytes in block data area (after file header 444 bytes + block header 372 bytes)
    # Block data starts at offset 444 + 372 = 816
    tamper_offset = 900  # Well into first block's data
    with open(tampered_path, "r+b") as f:
        f.seek(tamper_offset)
        original_bytes = f.read(10)
        f.seek(tamper_offset)
        # Write different bytes
        f.write(b"TAMPERED!!")

    print(f"Modified bytes at offset {tamper_offset} (block data area)")
    print(f"Original: {original_bytes}")
    print(f"Replaced with: b'TAMPERED!!'")

    print(f"\nVerifying tampered file...")

    reader = SecureMAIFReader(tampered_path)
    is_valid, errors = reader.verify_integrity()

    if is_valid:
        print("Unexpected: file passed verification")
    else:
        print("Tampering DETECTED!")
        print("Errors found:")
        for error in errors[:5]:  # Limit to first 5
            print(f"   - {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")

        if reader.get_tampered_blocks():
            print(f"\n   Tampered blocks: {reader.get_tampered_blocks()}")

    # Clean up
    os.remove(tampered_path)
    print(f"\n   Cleaned up {tampered_path}")


def show_embedded_metadata(file_path: str):
    """Display the metadata embedded in the self-contained MAIF file."""
    print("\n" + "=" * 60)
    print("EMBEDDED METADATA (extracted from .maif file)")
    print("=" * 60)

    reader = SecureMAIFReader(file_path)
    reader.load()

    # Get the embedded manifest data (no external file needed!)
    manifest = reader.export_manifest()

    print("\nFile is completely self-contained:")
    print(f"Format: {manifest['format']}")
    print(f"Version: {manifest['maif_version']}")
    print(f"Blocks: {len(manifest['blocks'])}")
    print(f"Provenance entries: {len(manifest['provenance'])}")
    print(f"Algorithm: {manifest['security']['key_algorithm']}")

    print("\n   All metadata, provenance, and security info is embedded")
    print("   in the .maif file itself - no external manifest needed!")


def main():
    """Run the complete demo."""
    try:
        # Create file
        file_path = create_demo_file()

        # Verify it
        verify_file(file_path)

        # Show tamper detection
        demonstrate_tamper_detection(file_path)

        # Show that all metadata is embedded (no external manifest!)
        show_embedded_metadata(file_path)

        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print(f"""
Files created:
  - {file_path} (self-contained secure MAIF - everything is inside!)

Key features demonstrated:
  - Self-contained format (NO external manifest files needed)
  - Block-level Ed25519 signatures (each block signed on write)
  - Immutability (signed blocks cannot be modified)
  - Tamper detection (modifications detected via signature verification)
  - Provenance chain (complete audit trail embedded in file)
  - Merkle root (fast integrity verification)
""")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
