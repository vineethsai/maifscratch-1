#!/usr/bin/env python3
"""
MAIF Basic Usage Example

Demonstrates the core functionality of the MAIF library:
- Creating MAIF files with text and embeddings
- Verifying file integrity
- Reading and inspecting MAIF files
- Provenance tracking
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from maif import MAIFEncoder, MAIFDecoder, MAIFParser, BlockType, verify_maif


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("MAIF Library Demonstration - Secure Format v3")
    print("=" * 60)

    output_file = "sample.maif"

    # =========================================================================
    # Create a MAIF file
    # =========================================================================
    print("\n📝 Creating MAIF file...")

    encoder = MAIFEncoder(output_file, agent_id="demo-agent-001")

    # Add text blocks
    encoder.add_text_block(
        "This is the first text block in the MAIF file.",
        metadata={"source": "demo", "language": "en"},
    )

    encoder.add_text_block(
        "MAIF provides secure, verifiable storage for AI-generated content.",
        metadata={"source": "demo", "topic": "description"},
    )

    encoder.add_text_block(
        "Each block is cryptographically signed with Ed25519 on creation.",
        metadata={"source": "demo", "topic": "security"},
    )

    # Add embeddings (example vectors)
    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100-dim vector
        [0.2, 0.3, 0.4, 0.5, 0.6] * 20,
        [0.3, 0.4, 0.5, 0.6, 0.7] * 20,
    ]
    encoder.add_embeddings_block(
        sample_embeddings, metadata={"model": "demo-embeddings", "dimensions": 100}
    )

    # Finalize (signs the file)
    encoder.finalize()

    file_size = os.path.getsize(output_file)
    print(f"✓ Created: {output_file} ({file_size:,} bytes)")

    # =========================================================================
    # Verify the file
    # =========================================================================
    print("\n🔐 Verifying file integrity...")

    is_valid, report = verify_maif(output_file)
    print(f"✓ File integrity: {'VALID ✅' if is_valid else 'INVALID ❌'}")
    if not is_valid:
        print(f"  Errors: {report.get('errors', [])}")

    # =========================================================================
    # Read and inspect the file
    # =========================================================================
    print("\n📖 Reading MAIF file...")

    decoder = MAIFDecoder(output_file)
    decoder.load()

    # File info
    file_info = decoder.get_file_info()
    print(f"\n📋 File Information:")
    print(f"  Version: {file_info['version']}")
    print(f"  Agent: {file_info['agent_did']}")
    print(f"  Blocks: {file_info['block_count']}")
    print(f"  Signed: {'Yes' if file_info['is_signed'] else 'No'}")
    print(f"  Finalized: {'Yes' if file_info['is_finalized'] else 'No'}")
    print(f"  Merkle Root: {file_info['merkle_root'][:32]}...")

    # List blocks
    print(f"\n📦 Blocks ({len(decoder.blocks)}):")
    for i, block in enumerate(decoder.blocks):
        type_name = BlockType(block.header.block_type).name
        size = block.header.size
        is_signed = bool(block.header.flags & 0x01)
        print(f"  [{i}] {type_name}: {size} bytes {'🔐' if is_signed else ''}")

    # Read text content
    print("\n📄 Text Content:")
    for i, block in enumerate(decoder.blocks):
        if block.header.block_type == BlockType.TEXT:
            text = decoder.get_text_content(i)
            if text:
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f'  [{i}] "{preview}"')

    # Provenance chain
    print(f"\n🔗 Provenance Chain ({len(decoder.provenance)} entries):")
    for entry in decoder.provenance:
        action = entry.action
        agent = entry.agent_id
        print(f"  • {action} by {agent}")

    # Security info
    security = decoder.get_security_info()
    print(f"\n🔒 Security:")
    print(f"  Algorithm: {security.get('key_algorithm', 'Ed25519')}")
    print(f"  Signer: {security.get('signer_id', 'N/A')}")

    # =========================================================================
    # Using the high-level parser
    # =========================================================================
    print("\n📊 Using MAIFParser (high-level API):")

    parser = MAIFParser(output_file)
    parser.load()

    print(f"  Loaded {len(parser.blocks)} blocks")
    print(f"  Provenance: {len(parser.provenance)} entries")
    is_valid, errors = parser.verify()
    print(f"  Integrity: {'Valid' if is_valid else 'Issues found'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("✅ Demonstration completed successfully!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  - {output_file}")
    print("\nKey features demonstrated:")
    print("  - Self-contained MAIF files (no manifest needed)")
    print("  - Ed25519 cryptographic signatures")
    print("  - Provenance tracking")
    print("  - Tamper detection")


if __name__ == "__main__":
    main()
