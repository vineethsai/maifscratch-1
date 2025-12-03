"""
Video functionality demonstration for MAIF.
Shows enhanced video storage, metadata extraction, and querying capabilities.
"""

import os
import sys
import tempfile
import struct

# Add the parent directory to the path so we can import maif
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif.core import MAIFEncoder, MAIFDecoder
from maif.privacy import PrivacyPolicy, PrivacyLevel, EncryptionMode


def create_sample_video_data(format_type: str = "mp4", duration: float = 10.0, 
                           width: int = 1920, height: int = 1080) -> bytes:
    """Create sample video data for demonstration."""
    if format_type == "mp4":
        return create_mock_mp4(duration, width, height)
    elif format_type == "avi":
        return create_mock_avi()
    else:
        # Generic video data
        return b'\x00' * (1024 * 1024)  # 1MB of dummy data


def create_mock_mp4(duration: float, width: int, height: int) -> bytes:
    """Create mock MP4 data with proper structure."""
    data = bytearray()
    
    # ftyp box
    ftyp_size = 32
    data.extend(struct.pack('>I', ftyp_size))
    data.extend(b'ftyp')
    data.extend(b'mp42')
    data.extend(struct.pack('>I', 0))
    data.extend(b'mp42isom')
    data.extend(b'\x00' * 12)
    
    # mvhd box
    mvhd_size = 108
    data.extend(struct.pack('>I', mvhd_size))
    data.extend(b'mvhd')
    data.extend(struct.pack('>I', 0))
    data.extend(struct.pack('>I', 0))
    data.extend(struct.pack('>I', 0))
    timescale = 1000
    data.extend(struct.pack('>I', timescale))
    data.extend(struct.pack('>I', int(duration * timescale)))
    data.extend(b'\x00' * 80)
    
    # tkhd box
    tkhd_size = 92
    data.extend(struct.pack('>I', tkhd_size))
    data.extend(b'tkhd')
    data.extend(struct.pack('>I', 0))
    data.extend(b'\x00' * 72)
    data.extend(struct.pack('>I', width << 16))
    data.extend(struct.pack('>I', height << 16))
    
    # Add dummy video data
    data.extend(b'\x00' * 2000)
    
    return bytes(data)


def create_mock_avi() -> bytes:
    """Create mock AVI data."""
    data = bytearray()
    data.extend(b'RIFF')
    data.extend(struct.pack('<I', 1500))
    data.extend(b'AVI ')
    data.extend(b'\x00' * 1500)
    return bytes(data)


def demonstrate_video_storage():
    """Demonstrate video storage with metadata extraction."""
    print("=== Video Storage Demonstration ===")
    
    encoder = MAIFEncoder(agent_id="video-demo-agent")
    
    # Create sample videos with different properties
    videos = [
        {
            "data": create_sample_video_data("mp4", 15.5, 1920, 1080),
            "metadata": {
                "title": "High Definition Demo",
                "description": "A sample 1080p video",
                "tags": ["demo", "hd", "test"],
                "creator": "MAIF Demo"
            }
        },
        {
            "data": create_sample_video_data("mp4", 30.0, 1280, 720),
            "metadata": {
                "title": "Standard Definition Demo",
                "description": "A sample 720p video",
                "tags": ["demo", "sd", "test"],
                "creator": "MAIF Demo"
            }
        },
        {
            "data": create_sample_video_data("avi"),
            "metadata": {
                "title": "AVI Format Demo",
                "description": "A sample AVI video",
                "tags": ["demo", "avi", "legacy"],
                "creator": "MAIF Demo"
            }
        }
    ]
    
    print(f"Adding {len(videos)} videos to MAIF...")
    
    for i, video in enumerate(videos):
        print(f"\nAdding video {i+1}: {video['metadata']['title']}")
        
        # Add video with automatic metadata extraction
        video_hash = encoder.add_video_block(
            video["data"],
            metadata=video["metadata"],
            extract_metadata=True
        )
        
        print(f"  Video hash: {video_hash[:16]}...")
        
        # Show extracted metadata
        video_block = encoder.blocks[-1]
        extracted = video_block.metadata
        
        print(f"  Format: {extracted.get('format', 'unknown')}")
        print(f"  Duration: {extracted.get('duration', 'unknown')} seconds")
        print(f"  Resolution: {extracted.get('resolution', 'unknown')}")
        print(f"  Size: {extracted.get('size_bytes', 0) / 1024:.1f} KB")
        print(f"  Semantic analysis: {extracted.get('has_semantic_analysis', False)}")
    
    return encoder


def demonstrate_video_querying(encoder: MAIFEncoder):
    """Demonstrate video querying capabilities."""
    print("\n=== Video Querying Demonstration ===")
    
    # Save MAIF file
    temp_dir = tempfile.mkdtemp()
    maif_path = os.path.join(temp_dir, "video_demo.maif")
    manifest_path = os.path.join(temp_dir, "video_demo.maif.manifest.json")
    
    encoder.build_maif(maif_path, manifest_path)
    print(f"MAIF file saved to: {maif_path}")
    
    # Load and query
    decoder = MAIFDecoder(maif_path, manifest_path)
    
    print("\n--- Basic Video Information ---")
    video_blocks = decoder.get_video_blocks()
    print(f"Total videos found: {len(video_blocks)}")
    
    for i, block in enumerate(video_blocks):
        metadata = block.metadata
        print(f"\nVideo {i+1}:")
        print(f"  Title: {metadata.get('title', 'Unknown')}")
        print(f"  Format: {metadata.get('format', 'unknown')}")
        print(f"  Duration: {metadata.get('duration', 'unknown')} seconds")
        print(f"  Resolution: {metadata.get('resolution', 'unknown')}")
    
    print("\n--- Query Examples ---")
    
    # Query by duration
    print("\n1. Videos longer than 20 seconds:")
    long_videos = decoder.query_videos(duration_range=(20.0, 60.0))
    for video in long_videos:
        print(f"  - {video['metadata']['title']} ({video['duration']}s)")
    
    # Query by resolution
    print("\n2. HD videos (1080p or higher):")
    hd_videos = decoder.query_videos(min_resolution="1080p")
    for video in hd_videos:
        print(f"  - {video['metadata']['title']} ({video['resolution']})")
    
    # Query by format
    print("\n3. MP4 videos:")
    mp4_videos = decoder.query_videos(format_filter="mp4")
    for video in mp4_videos:
        print(f"  - {video['metadata']['title']} (format: {video['format']})")
    
    # Query by size
    print("\n4. Videos smaller than 5MB:")
    small_videos = decoder.query_videos(max_size_mb=5.0)
    for video in small_videos:
        size_mb = video['size_bytes'] / (1024 * 1024)
        print(f"  - {video['metadata']['title']} ({size_mb:.1f}MB)")
    
    # Combined query
    print("\n5. Short HD videos (less than 20 seconds, 1080p+):")
    short_hd = decoder.query_videos(
        duration_range=(0.0, 20.0),
        min_resolution="1080p"
    )
    for video in short_hd:
        print(f"  - {video['metadata']['title']} ({video['duration']}s, {video['resolution']})")
    
    return decoder, temp_dir


def demonstrate_semantic_search(decoder: MAIFDecoder):
    """Demonstrate semantic video search."""
    print("\n=== Semantic Video Search ===")
    
    # Search for videos by content
    search_queries = [
        "high definition video",
        "demo content",
        "standard quality"
    ]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        results = decoder.search_videos_by_content(query, top_k=3)
        
        if results:
            for i, result in enumerate(results):
                print(f"  {i+1}. {result['metadata']['title']} "
                      f"(similarity: {result['similarity_score']:.3f})")
        else:
            print("  No semantic search results (semantic module may not be available)")


def demonstrate_video_statistics(decoder: MAIFDecoder):
    """Demonstrate video statistics and summary."""
    print("\n=== Video Statistics ===")
    
    summary = decoder.get_video_summary()
    
    print(f"Total videos: {summary.get('total_videos', 0)}")
    print(f"Total duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
    print(f"Total size: {summary.get('total_size_mb', 0):.1f} MB")
    print(f"Average duration: {summary.get('average_duration', 0):.1f} seconds")
    print(f"Videos with semantic analysis: {summary.get('videos_with_semantic_analysis', 0)}")
    
    print("\nFormat distribution:")
    for format_name, count in summary.get('formats', {}).items():
        print(f"  {format_name}: {count} videos")
    
    print("\nResolution distribution:")
    for resolution, count in summary.get('resolutions', {}).items():
        print(f"  {resolution}: {count} videos")


def demonstrate_privacy_features():
    """Demonstrate video storage with privacy controls."""
    print("\n=== Privacy-Enabled Video Storage ===")
    
    encoder = MAIFEncoder(enable_privacy=True, agent_id="privacy-demo-agent")
    
    # Create privacy policy for sensitive video content
    confidential_policy = PrivacyPolicy(
        privacy_level=PrivacyLevel.CONFIDENTIAL,
        encryption_mode=EncryptionMode.AES_GCM,
        anonymization_required=False,
        audit_required=True
    )
    
    # Add confidential video
    sensitive_video = create_sample_video_data("mp4", 25.0, 1920, 1080)
    
    video_hash = encoder.add_video_block(
        sensitive_video,
        metadata={
            "title": "Confidential Meeting Recording",
            "description": "Internal company meeting",
            "classification": "confidential"
        },
        privacy_policy=confidential_policy
    )
    
    print("Added confidential video with encryption")
    print(f"Video hash: {video_hash[:16]}...")
    
    # Check privacy metadata
    video_block = encoder.blocks[0]
    privacy_info = video_block.metadata.get('privacy_policy', {})
    print(f"Privacy level: {privacy_info.get('privacy_level', 'none')}")
    print(f"Encryption mode: {privacy_info.get('encryption_mode', 'none')}")
    print(f"Audit required: {privacy_info.get('audit_required', False)}")


def demonstrate_video_retrieval(decoder: MAIFDecoder, temp_dir: str):
    """Demonstrate video data retrieval."""
    print("\n=== Video Data Retrieval ===")
    
    video_blocks = decoder.get_video_blocks()
    
    if video_blocks:
        # Get the first video
        first_video = video_blocks[0]
        print(f"Retrieving video: {first_video.metadata.get('title', 'Unknown')}")
        
        # Get video data
        video_data = decoder.get_video_data(first_video.block_id)
        
        if video_data:
            print(f"Retrieved {len(video_data)} bytes of video data")
            
            # Save to file
            output_path = os.path.join(temp_dir, "retrieved_video.mp4")
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            print(f"Video saved to: {output_path}")
        else:
            print("Failed to retrieve video data")


def main():
    """Run the complete video functionality demonstration."""
    print("MAIF Enhanced Video Functionality Demo")
    print("=" * 50)
    
    try:
        # Demonstrate video storage
        encoder = demonstrate_video_storage()
        
        # Demonstrate querying
        decoder, temp_dir = demonstrate_video_querying(encoder)
        
        # Demonstrate semantic search
        demonstrate_semantic_search(decoder)
        
        # Demonstrate statistics
        demonstrate_video_statistics(decoder)
        
        # Demonstrate video retrieval
        demonstrate_video_retrieval(decoder, temp_dir)
        
        # Demonstrate privacy features
        demonstrate_privacy_features()
        
        print("\n" + "=" * 50)
        print("Video functionality demonstration completed!")
        print("\nKey features demonstrated:")
        print("✓ Automatic video metadata extraction")
        print("✓ Semantic embedding generation")
        print("✓ Advanced video querying by properties")
        print("✓ Semantic content search")
        print("✓ Video statistics and summaries")
        print("✓ Privacy-controlled video storage")
        print("✓ Video data retrieval")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()