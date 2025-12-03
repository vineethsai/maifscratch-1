"""
MAIF Hybrid Architecture Demo

This example demonstrates MAIF's architecture for different use cases:
1. Simple API - Easy to use high-level interface
2. Native SDK - Direct access for advanced use cases
3. File-based operations - POSIX-style operations

Run this demo to see how each interface serves different use cases.
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from maif_api import create_maif, MAIF, load_maif, quick_text_maif
from maif.core import MAIFEncoder, MAIFDecoder


def demo_simple_api():
    """Demonstrate the simple maif_api interface."""
    print("\n" + "="*60)
    print("1. SIMPLE API DEMO - Easy to Use Interface")
    print("="*60)
    
    # Create a MAIF instance with the simple API
    maif = create_maif(agent_id="demo_agent", enable_privacy=True)
    
    print(f"📝 Created MAIF instance for agent: demo_agent")
    
    # Add content
    print(f"📝 Adding text content...")
    start_time = time.time()
    
    for i in range(10):
        maif.add_text(
            f"High-performance content block {i}",
            title=f"Block {i}"
        )
    
    write_time = time.time() - start_time
    print(f"✅ Added 10 text blocks in {write_time:.3f}s")
    
    # Save to file
    test_file = Path("demo_simple.maif")
    print(f"💾 Saving to {test_file}...")
    maif.save(str(test_file))
    
    # Load and verify
    print(f"📖 Loading back from file...")
    loaded = MAIF.load(str(test_file))
    content_list = loaded.get_content_list()
    print(f"✅ Loaded {len(content_list)} blocks")
    
    # Verify integrity
    is_valid = loaded.verify_integrity()
    print(f"🔒 Integrity check: {'VALID' if is_valid else 'INVALID'}")
    
    # Clean up
    if test_file.exists():
        test_file.unlink()
    manifest = Path(str(test_file) + ".manifest.json")
    if manifest.exists():
        manifest.unlink()


def demo_native_encoder():
    """Demonstrate the native MAIFEncoder interface."""
    print("\n" + "="*60)
    print("2. NATIVE ENCODER DEMO - Direct Access Interface")
    print("="*60)
    
    # Create an encoder directly
    encoder = MAIFEncoder(agent_id="native_agent")
    
    print(f"📝 Created MAIFEncoder for native access...")
    
    # Add various block types
    print(f"📝 Adding content blocks...")
    start_time = time.time()
    
    # Add text blocks
    for i in range(5):
        encoder.add_text_block(
            f"Native text block {i}: This is direct encoder access.",
            {"source": f"native_demo_{i}"}
        )
    
    # Add binary data
    binary_data = b"Binary content " * 100
    encoder.add_binary_block(binary_data, "data", {"type": "test_data"})
    
    write_time = time.time() - start_time
    print(f"✅ Added blocks in {write_time:.3f}s")
    
    # Save
    test_file = Path("demo_native.maif")
    manifest_file = Path("demo_native.manifest.json")
    
    print(f"💾 Building MAIF file...")
    encoder.build_maif(str(test_file), str(manifest_file))
    
    # Read back with decoder
    print(f"📖 Reading with MAIFDecoder...")
    decoder = MAIFDecoder(str(test_file), str(manifest_file))
    
    block_count = 0
    for block in decoder.blocks:
        block_count += 1
        data_size = len(block.data) if block.data else 0
        print(f"   Block {block.block_id[:8]}: {block.block_type}, {data_size} bytes")
    
    print(f"✅ Read {block_count} blocks")
    
    # Clean up
    if test_file.exists():
        test_file.unlink()
    if manifest_file.exists():
        manifest_file.unlink()


def demo_quick_operations():
    """Demonstrate quick one-liner operations."""
    print("\n" + "="*60)
    print("3. QUICK OPERATIONS DEMO - One-Liners")
    print("="*60)
    
    # Quick text MAIF creation
    print(f"⚡ Creating quick text MAIF...")
    
    quick_text_maif(
        "This is a quick one-liner to create a MAIF file with text content.",
        "quick_demo.maif",
        title="Quick Demo"
    )
    
    print(f"✅ Created quick_demo.maif")
    
    # Load and verify
    loaded = load_maif("quick_demo.maif")
    content = loaded.get_content_list()
    print(f"📖 Verified: {len(content)} blocks in file")
    
    # Clean up
    Path("quick_demo.maif").unlink()
    manifest = Path("quick_demo.maif.manifest.json")
    if manifest.exists():
        manifest.unlink()


def demo_multimodal():
    """Demonstrate multimodal content handling."""
    print("\n" + "="*60)
    print("4. MULTIMODAL DEMO - Mixed Content Types")
    print("="*60)
    
    maif = create_maif(agent_id="multimodal_agent", enable_privacy=True)
    
    print(f"📝 Adding multimodal content...")
    
    # Add text
    maif.add_text("This is a text description", title="Description")
    
    # Add multimodal content
    maif.add_multimodal({
        "text": "A multimodal entry with text and metadata",
        "metadata": {"type": "demo", "version": "1.0"},
        "tags": ["demo", "multimodal", "test"]
    }, title="Multimodal Entry")
    
    # Add embeddings (simulated)
    import random
    embeddings = [[random.random() for _ in range(384)] for _ in range(3)]
    maif.add_embeddings(embeddings, model_name="demo_model")
    
    print(f"✅ Added text, multimodal, and embedding content")
    
    # Save and search
    test_file = "demo_multimodal.maif"
    maif.save(test_file)
    
    # Reload to enable search
    loaded_maif = load_maif(test_file)
    
    # Search
    print(f"🔍 Searching for 'multimodal'...")
    results = loaded_maif.search("multimodal", top_k=3)
    print(f"   Found {len(results)} results")
    
    # Privacy report
    print(f"🔒 Privacy report:")
    report = maif.get_privacy_report()
    print(f"   Blocks: {report.get('total_blocks', 0)}")
    print(f"   Encrypted: {report.get('encrypted_blocks', 0)}")
    
    # Clean up
    Path(test_file).unlink()
    manifest = Path(test_file + ".manifest.json")
    if manifest.exists():
        manifest.unlink()


def demo_use_case_recommendations():
    """Show use case recommendations."""
    print("\n" + "="*60)
    print("5. USE CASE RECOMMENDATIONS")
    print("="*60)
    
    use_cases = {
        "Chat Agent": "Use create_maif() for session memory, save after each turn",
        "Document Processing": "Use MAIFEncoder for batch processing",
        "Real-time Analytics": "Use create_maif() with streaming add_text calls",
        "Multi-Agent System": "Each agent gets own MAIF instance via create_maif()",
        "Audit Trail": "Use add_text with timestamps, enable_privacy=True",
        "Quick Scripts": "Use quick_text_maif() for simple use cases"
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
        demo_multimodal()
        demo_use_case_recommendations()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE")
        print("="*60)
        print("Key takeaways:")
        print("• Use maif_api (create_maif, MAIF) for most use cases")
        print("• Use MAIFEncoder/MAIFDecoder for advanced low-level access")
        print("• Use quick_text_maif() for simple one-off operations")
        print("• All interfaces produce compatible .maif files")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
