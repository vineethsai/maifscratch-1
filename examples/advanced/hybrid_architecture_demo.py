"""
MAIF Hybrid Architecture Demo

This example demonstrates the three interfaces described in the decision memo:
1. Native SDK - High-performance direct access
2. FUSE Filesystem - POSIX interface for convenience
3. gRPC Daemon - Multi-writer service for distributed scenarios

Run this demo to see how each interface serves different use cases.
"""

import os
import sys
import time
import asyncio
import tempfile
import threading
from pathlib import Path

# Add the parent directory to the path so we can import maif_sdk
sys.path.insert(0, str(Path(__file__).parent.parent))

import maif_sdk
from maif_sdk import (
    create_client, create_artifact, 
    ContentType, SecurityLevel, CompressionLevel,
    FUSE_AVAILABLE, GRPC_AVAILABLE
)


def demo_native_sdk():
    """Demonstrate the high-performance native SDK interface."""
    print("\n" + "="*60)
    print("1. NATIVE SDK DEMO - High Performance 'Hot Path'")
    print("="*60)
    
    # Create a high-performance client with memory mapping
    with create_client(agent_id="performance_agent", enable_mmap=True) as client:
        
        # Create test file
        test_file = Path("demo_native.maif")
        
        print(f"📝 Writing content with native SDK...")
        start_time = time.time()
        
        # Write multiple content blocks efficiently
        block_ids = []
        for i in range(10):
            content = f"High-performance content block {i}".encode('utf-8')
            block_id = client.write_content(
                filepath=test_file,
                content=content,
                content_type=ContentType.TEXT,
                flush_immediately=(i == 9)  # Only flush on last write
            )
            block_ids.append(block_id)
        
        write_time = time.time() - start_time
        print(f"✅ Wrote 10 blocks in {write_time:.3f}s")
        
        # Read content back with memory mapping
        print(f"📖 Reading content with native SDK...")
        start_time = time.time()
        
        content_count = 0
        for content in client.read_content(test_file):
            content_count += 1
            print(f"   Block {content['block_id'][:8]}: {len(content['data'])} bytes")
        
        read_time = time.time() - start_time
        print(f"✅ Read {content_count} blocks in {read_time:.3f}s")
        
        # Show file info
        info = client.get_file_info(test_file)
        print(f"📊 File info: {info['total_blocks']} blocks, {info['file_size']} bytes")
        
        # Clean up
        if test_file.exists():
            test_file.unlink()


def demo_artifact_interface():
    """Demonstrate the high-level Artifact interface."""
    print("\n" + "="*60)
    print("2. ARTIFACT INTERFACE DEMO - High-Level API")
    print("="*60)
    
    # Create an artifact with different content types
    artifact = create_artifact(
        name="Multi-Modal Demo",
        security_level=SecurityLevel.INTERNAL,
        compression_level=CompressionLevel.BALANCED
    )
    
    print(f"📦 Created artifact: {artifact.name}")
    
    # Add different types of content
    print(f"📝 Adding text content...")
    artifact.add_text(
        "This is a sample text document for the demo.",
        title="Sample Text",
        description="Demonstration text content"
    )
    
    print(f"🖼️ Adding image content...")
    # Simulate image data
    fake_image_data = b'\x89PNG\r\n\x1a\n' + b'fake_image_data' * 100
    artifact.add_image(
        fake_image_data,
        title="Demo Image",
        format="png",
        description="Simulated image content"
    )
    
    print(f"📄 Adding document content...")
    fake_doc_data = b'%PDF-1.4' + b'fake_pdf_content' * 50
    artifact.add_document(
        fake_doc_data,
        title="Demo Document",
        format="pdf",
        description="Simulated PDF document"
    )
    
    print(f"📊 Artifact summary: {len(artifact)} items")
    
    # Save the artifact
    artifact_file = Path("demo_artifact.maif")
    print(f"💾 Saving artifact to {artifact_file}...")
    artifact.save(artifact_file)
    
    # Load it back
    print(f"📂 Loading artifact from file...")
    loaded_artifact = create_artifact().load(artifact_file)
    print(f"✅ Loaded artifact: {loaded_artifact}")
    
    # Show content
    print(f"📋 Content summary:")
    for content in loaded_artifact.get_content():
        print(f"   {content['content_type']}: {content['size']} bytes")
    
    # Clean up
    if artifact_file.exists():
        artifact_file.unlink()


def demo_fuse_filesystem():
    """Demonstrate the FUSE filesystem interface."""
    print("\n" + "="*60)
    print("3. FUSE FILESYSTEM DEMO - POSIX Interface")
    print("="*60)
    
    if not FUSE_AVAILABLE:
        print("❌ FUSE not available. Install with: pip install fusepy")
        print("   This interface provides POSIX semantics for legacy tools.")
        return
    
    print("🔧 FUSE is available but demo requires manual mounting.")
    print("   To test FUSE interface:")
    print("   1. Create a directory with MAIF files")
    print("   2. Run: python -m maif_sdk.fuse_fs /path/to/maif/files /mnt/maif")
    print("   3. Access files via: ls /mnt/maif, cat /mnt/maif/file/content.txt")
    print("   4. Unmount with: fusermount -u /mnt/maif (Linux) or umount /mnt/maif (macOS)")
    
    # Create a sample MAIF file for FUSE demo
    sample_file = Path("fuse_demo.maif")
    with create_client() as client:
        client.write_content(
            sample_file,
            b"This content can be accessed via FUSE filesystem",
            ContentType.TEXT
        )
        client.write_content(
            sample_file,
            b"Another block of content for FUSE demo",
            ContentType.TEXT
        )
    
    print(f"📁 Created sample file: {sample_file}")
    print(f"   This file would appear as a directory in FUSE with content files inside.")
    
    # Clean up
    if sample_file.exists():
        sample_file.unlink()


async def demo_grpc_service():
    """Demonstrate the gRPC service interface."""
    print("\n" + "="*60)
    print("4. gRPC SERVICE DEMO - Multi-Writer Interface")
    print("="*60)
    
    if not GRPC_AVAILABLE:
        print("❌ gRPC not available. Install with: pip install grpcio grpcio-tools")
        print("   This interface enables multi-writer concurrency and containerization.")
        return
    
    print("🔧 gRPC is available but requires protobuf generation.")
    print("   To test gRPC interface:")
    print("   1. Generate protobuf files:")
    print("      python -m maif_sdk.grpc_daemon --generate-proto")
    print("   2. Start the service:")
    print("      python -m maif_sdk.grpc_daemon --host localhost --port 50051")
    print("   3. Connect clients to localhost:50051")
    
    print("📡 gRPC service provides:")
    print("   - Safe multi-writer concurrency")
    print("   - Container-friendly access (no root FUSE)")
    print("   - Distributed MAIF operations")
    print("   - Session management and cleanup")


def demo_workload_recommendations():
    """Show workload-specific recommendations."""
    print("\n" + "="*60)
    print("5. WORKLOAD RECOMMENDATIONS")
    print("="*60)
    
    workloads = [
        "interactive",
        "edge_low", 
        "chat_medium",
        "high_tps",
        "data_exchange"
    ]
    
    for workload in workloads:
        recommendation = maif_sdk.get_recommended_interface(workload)
        print(f"🎯 {workload.upper()}: {recommendation}")


def demo_performance_comparison():
    """Compare performance of different interfaces."""
    print("\n" + "="*60)
    print("6. PERFORMANCE COMPARISON")
    print("="*60)
    
    # Test data
    test_data = b"Performance test data " * 1000  # ~22KB
    test_file = Path("perf_test.maif")
    
    print(f"📊 Testing with {len(test_data)} byte blocks...")
    
    # Native SDK performance
    print(f"\n🚀 Native SDK (direct mmap I/O):")
    with create_client(enable_mmap=True, buffer_size=64*1024) as client:
        start_time = time.time()
        
        for i in range(50):
            client.write_content(
                test_file,
                test_data,
                ContentType.DATA,
                flush_immediately=(i == 49)
            )
        
        sdk_time = time.time() - start_time
        print(f"   ✅ 50 writes: {sdk_time:.3f}s ({50/sdk_time:.1f} writes/sec)")
    
    # Quick operations
    print(f"\n⚡ Quick operations (convenience functions):")
    start_time = time.time()
    
    for i in range(10):
        maif_sdk.quick_write(
            f"quick_test_{i}.maif",
            test_data,
            ContentType.DATA
        )
    
    quick_time = time.time() - start_time
    print(f"   ✅ 10 quick writes: {quick_time:.3f}s ({10/quick_time:.1f} writes/sec)")
    
    # Cleanup
    for f in Path(".").glob("*.maif"):
        f.unlink()
    
    print(f"\n📈 Performance summary:")
    print(f"   - Native SDK: Optimized for high-throughput scenarios")
    print(f"   - FUSE: Adds 20-50µs per syscall but enables POSIX tools")
    print(f"   - gRPC: Network overhead but enables distributed access")


def main():
    """Run the complete hybrid architecture demonstration."""
    print("MAIF Hybrid Architecture Demonstration")
    print("Based on the decision memo recommendations")
    print(f"SDK Version: {maif_sdk.__version__}")
    print(f"FUSE Available: {FUSE_AVAILABLE}")
    print(f"gRPC Available: {GRPC_AVAILABLE}")
    
    try:
        # Demo each interface
        demo_native_sdk()
        demo_artifact_interface()
        demo_fuse_filesystem()
        
        # gRPC demo requires async
        asyncio.run(demo_grpc_service())
        
        # Show recommendations and performance
        demo_workload_recommendations()
        demo_performance_comparison()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETE")
        print("="*60)
        print("Key takeaways:")
        print("• Use Native SDK for performance-critical operations")
        print("• Use FUSE for human exploration and legacy tools")
        print("• Use gRPC for multi-writer and containerized scenarios")
        print("• Choose interface based on your workload pattern")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()