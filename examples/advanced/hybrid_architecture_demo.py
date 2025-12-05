"""
MAIF Hybrid Architecture Demo

This example demonstrates MAIF's architecture for different use cases:
1. Simple API - Easy to use high-level interface
2. Native SDK - Direct access for advanced use cases
3. Quick operations - One-liner functions

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain

Run this demo to see how each interface serves different use cases.
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maif_api import MAIF, load_maif, quick_text_maif, create_maif
from maif import MAIFEncoder, MAIFDecoder


def demo_simple_api():
    """Demonstrate the simple maif_api interface."""
    print("\n" + "=" * 60)
    print("1. SIMPLE API DEMO - Easy to Use Interface")
    print("=" * 60)

    # Create a MAIF instance with the simple API
    maif = MAIF("demo_agent")

    print(f"📝 Created MAIF instance for agent: demo_agent")

    # Add content
    print(f"📝 Adding text content...")
    start_time = time.time()

    for i in range(10):
        maif.add_text(f"High-performance content block {i}")

    write_time = time.time() - start_time
    print(f"✅ Added 10 text blocks in {write_time:.3f}s")

    # Save to file (self-contained, no manifest needed)
    test_file = Path("demo_simple.maif")
    print(f"💾 Saving to {test_file}...")
    maif.save(str(test_file))

    # Load and verify
    print(f"📖 Loading back from file...")
    loaded = MAIF.load(str(test_file))

    # Access texts directly
    text_count = len(loaded.texts)
    print(f"✅ Loaded {text_count} text blocks")

    # Verify integrity
    is_valid = loaded.verify()
    print(f"🔒 Integrity check: {'VALID' if is_valid else 'INVALID'}")

    # Show first few texts
    for i, text in enumerate(loaded.texts[:3]):
        print(f"   [{i}] {text[:50]}...")

    # Note: file is NOT cleaned up - stays in demo_output


def demo_native_encoder():
    """Demonstrate the native MAIFEncoder interface (secure format)."""
    print("\n" + "=" * 60)
    print("2. NATIVE ENCODER DEMO - Direct Access Interface")
    print("=" * 60)

    # Create an encoder directly (secure format with Ed25519)
    test_file = Path("demo_native.maif")
    encoder = MAIFEncoder(str(test_file), agent_id="native_agent")

    print(f"📝 Created MAIFEncoder for native access...")

    # Add various block types
    print(f"📝 Adding content blocks...")
    start_time = time.time()

    # Add text blocks
    for i in range(5):
        encoder.add_text_block(
            f"Native text block {i}: This is direct encoder access.",
            {"source": f"native_demo_{i}"},
        )

    # Add binary data
    binary_data = b"Binary content " * 100
    encoder.add_binary_block(binary_data, metadata={"type": "test_data"})

    write_time = time.time() - start_time
    print(f"✅ Added blocks in {write_time:.3f}s")

    # Finalize (self-contained, Ed25519 signed)
    print(f"💾 Finalizing MAIF file...")
    encoder.finalize()

    # Read back with decoder
    print(f"📖 Reading with MAIFDecoder...")
    decoder = MAIFDecoder(str(test_file))

    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    print(f"🔒 Integrity: {'VALID' if is_valid else 'INVALID'}")

    # List blocks
    blocks = decoder.get_blocks()
    print(f"✅ Read {len(blocks)} blocks:")

    for i, block in enumerate(blocks):
        block_id = block.header.block_id.hex()[:8]
        size = len(block.data) if block.data else 0
        print(f"   Block {block_id}: {size} bytes, signed ✓")

    # Note: file stays for demo_output collection


def demo_quick_operations():
    """Demonstrate quick one-liner operations."""
    print("\n" + "=" * 60)
    print("3. QUICK OPERATIONS DEMO - One-Liners")
    print("=" * 60)

    # Quick text MAIF creation
    print(f"⚡ Creating quick text MAIF...")

    result = quick_text_maif(
        "This is a quick one-liner to create a MAIF file with text content.",
        "quick_demo.maif",
    )

    print(f"✅ Created: {result}")

    # Load and verify
    loaded = load_maif("quick_demo.maif")
    print(f"📖 Loaded {len(loaded.texts)} text blocks")
    print(f"🔒 Integrity: {'VALID' if loaded.verify() else 'INVALID'}")

    # Note: file stays for demo_output collection


def demo_create_helper():
    """Demonstrate the create_maif helper."""
    print("\n" + "=" * 60)
    print("4. CREATE_MAIF HELPER DEMO")
    print("=" * 60)

    # Create MAIF with multiple texts in one call
    print(f"📝 Creating MAIF with multiple texts...")

    result = create_maif(
        output_path="helper_demo.maif",
        texts=[
            "First document for the helper demo.",
            "Second document with more content.",
            "Third document to show batch creation.",
        ],
        embeddings=[[0.1] * 64, [0.2] * 64],  # Sample embeddings
        agent_id="helper_agent",
    )

    print(f"✅ Created: {result}")

    # Load and verify
    loaded = load_maif(result)
    print(f"📖 Loaded {len(loaded.texts)} text blocks")
    print(f"🔒 Integrity: {'VALID' if loaded.verify() else 'INVALID'}")


def demo_use_case_recommendations():
    """Show use case recommendations."""
    print("\n" + "=" * 60)
    print("5. USE CASE RECOMMENDATIONS")
    print("=" * 60)

    use_cases = {
        "Chat Agent": "Use MAIF('agent_id') for session memory, call save() after each turn",
        "Document Processing": "Use MAIFEncoder for batch processing with finalize()",
        "Real-time Analytics": "Use MAIF() with streaming add_text() calls",
        "Multi-Agent System": "Each agent gets own MAIF instance via MAIF(agent_id)",
        "Audit Trail": "Use MAIFEncoder for detailed provenance tracking",
        "Quick Scripts": "Use quick_text_maif() for simple one-off operations",
    }

    for use_case, recommendation in use_cases.items():
        print(f"🎯 {use_case}:")
        print(f"   {recommendation}")


def main():
    """Run the complete demonstration."""
    print("MAIF Hybrid Architecture Demonstration")
    print("Simple API for easy use, Native SDK for advanced control")

    try:
        # Demo each interface
        demo_simple_api()
        demo_native_encoder()
        demo_quick_operations()
        demo_create_helper()
        demo_use_case_recommendations()

        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETE")
        print("=" * 60)
        print("""
Key takeaways:
  • Use MAIF class for most use cases (simple, intuitive)
  • Use MAIFEncoder/MAIFDecoder for advanced low-level access
  • Use quick_text_maif() for simple one-off operations
  • Use create_maif() for batch creation
  • All interfaces produce self-contained .maif files
  • Ed25519 signatures ensure tamper detection
""")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
