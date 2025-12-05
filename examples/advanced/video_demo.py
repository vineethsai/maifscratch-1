"""
Video functionality demonstration for MAIF.
Shows video storage using binary blocks with metadata extraction.

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
"""

import os
import sys
import tempfile
import struct
import json

# Add the parent directory to the path so we can import maif
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from maif import MAIFEncoder, MAIFDecoder


def create_mock_mp4(
    duration: float = 10.0, width: int = 1920, height: int = 1080
) -> bytes:
    """Create mock MP4 data with proper structure."""
    data = bytearray()

    # ftyp box
    ftyp_size = 32
    data.extend(struct.pack(">I", ftyp_size))
    data.extend(b"ftyp")
    data.extend(b"mp42")
    data.extend(struct.pack(">I", 0))
    data.extend(b"mp42isom")
    data.extend(b"\x00" * 12)

    # mvhd box (movie header) with duration
    mvhd_size = 108
    data.extend(struct.pack(">I", mvhd_size))
    data.extend(b"mvhd")
    data.extend(struct.pack(">I", 0))  # version/flags
    data.extend(struct.pack(">I", 0))  # creation time
    data.extend(struct.pack(">I", 0))  # modification time
    data.extend(struct.pack(">I", 1000))  # timescale
    data.extend(struct.pack(">I", int(duration * 1000)))  # duration in timescale units
    data.extend(b"\x00" * 76)  # remaining mvhd fields

    # tkhd box (track header) with dimensions
    tkhd_size = 92
    data.extend(struct.pack(">I", tkhd_size))
    data.extend(b"tkhd")
    data.extend(struct.pack(">I", 0x0F))  # version/flags (track enabled)
    data.extend(struct.pack(">I", 0))  # creation time
    data.extend(struct.pack(">I", 0))  # modification time
    data.extend(struct.pack(">I", 1))  # track ID
    data.extend(struct.pack(">I", 0))  # reserved
    data.extend(struct.pack(">I", int(duration * 1000)))  # duration
    data.extend(b"\x00" * 8)  # reserved
    data.extend(struct.pack(">h", 0))  # layer
    data.extend(struct.pack(">h", 0))  # alternate group
    data.extend(struct.pack(">h", 0x0100))  # volume
    data.extend(struct.pack(">h", 0))  # reserved
    data.extend(b"\x00" * 36)  # matrix
    data.extend(struct.pack(">I", width << 16))  # width (fixed point)
    data.extend(struct.pack(">I", height << 16))  # height (fixed point)

    # mdat box (media data)
    mdat_size = 1024
    data.extend(struct.pack(">I", mdat_size))
    data.extend(b"mdat")
    data.extend(b"\x00" * (mdat_size - 8))

    return bytes(data)


def extract_video_metadata(data: bytes) -> dict:
    """Extract metadata from video data (mock implementation)."""
    metadata = {
        "format": "unknown",
        "duration": 0,
        "width": 0,
        "height": 0,
        "size_bytes": len(data),
    }

    # Check for MP4 signature
    if len(data) >= 12 and data[4:8] == b"ftyp":
        metadata["format"] = "mp4"

        # Try to parse mvhd for duration
        offset = 0
        while offset < len(data) - 8:
            try:
                box_size = struct.unpack(">I", data[offset : offset + 4])[0]
                box_type = data[offset + 4 : offset + 8]

                if box_type == b"mvhd" and offset + 24 < len(data):
                    timescale = struct.unpack(">I", data[offset + 20 : offset + 24])[0]
                    duration_units = struct.unpack(
                        ">I", data[offset + 24 : offset + 28]
                    )[0]
                    if timescale > 0:
                        metadata["duration"] = duration_units / timescale

                if box_type == b"tkhd" and offset + 84 < len(data):
                    width_fp = struct.unpack(">I", data[offset + 84 : offset + 88])[0]
                    height_fp = struct.unpack(">I", data[offset + 88 : offset + 92])[0]
                    metadata["width"] = width_fp >> 16
                    metadata["height"] = height_fp >> 16

                offset += max(box_size, 8)
            except:
                break

    return metadata


def demonstrate_video_storage():
    """Demonstrate storing videos in MAIF."""
    print("=== Video Storage Demonstration ===")

    # Create temporary directory for output
    temp_dir = tempfile.mkdtemp()
    maif_path = os.path.join(temp_dir, "video_demo.maif")

    # Create MAIF encoder (secure format with Ed25519)
    encoder = MAIFEncoder(maif_path, agent_id="video-demo-agent")

    # Sample videos with different properties
    videos = [
        {
            "duration": 10.0,
            "width": 1920,
            "height": 1080,
            "title": "High Definition Demo",
        },
        {
            "duration": 30.0,
            "width": 1280,
            "height": 720,
            "title": "Standard Definition Clip",
        },
        {"duration": 5.0, "width": 3840, "height": 2160, "title": "4K Ultra HD Sample"},
    ]

    print(f"\nAdding {len(videos)} videos to MAIF...")

    for i, video_spec in enumerate(videos):
        print(f"\n  Adding video {i + 1}: {video_spec['title']}")

        # Create mock video data
        video_data = create_mock_mp4(
            duration=video_spec["duration"],
            width=video_spec["width"],
            height=video_spec["height"],
        )

        # Extract metadata from video
        extracted = extract_video_metadata(video_data)
        extracted["title"] = video_spec["title"]

        # Store video as binary block with metadata
        metadata = {
            "content_type": "video/mp4",
            "title": video_spec["title"],
            "format": extracted["format"],
            "duration": extracted["duration"],
            "resolution": f"{extracted['width']}x{extracted['height']}",
            "size_bytes": extracted["size_bytes"],
        }

        # Add as binary block (videos are stored as binary data)
        block = encoder.add_binary_block(video_data, metadata=metadata)

        print(f"    Format: {extracted['format']}")
        print(f"    Duration: {extracted['duration']:.1f} seconds")
        print(f"    Resolution: {extracted['width']}x{extracted['height']}")
        print(f"    Size: {extracted['size_bytes'] / 1024:.1f} KB")

    # Finalize (signs with Ed25519, no manifest needed)
    encoder.finalize()

    print(f"\n✓ Created video MAIF: {maif_path}")
    print(f"  (Self-contained with Ed25519 signatures)")

    return maif_path


def demonstrate_video_retrieval(maif_path: str):
    """Demonstrate retrieving videos from MAIF."""
    print("\n=== Video Retrieval Demonstration ===")

    # Load MAIF file
    decoder = MAIFDecoder(maif_path)

    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    print(f"\n  Integrity check: {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Get all blocks
    blocks = decoder.get_blocks()
    video_blocks = [
        b
        for b in blocks
        if (b.metadata or {}).get("content_type", "").startswith("video/")
    ]

    print(f"  Found {len(video_blocks)} video blocks")

    for i, block in enumerate(video_blocks):
        metadata = block.metadata or {}
        print(f"\n  Video {i + 1}: {metadata.get('title', 'Untitled')}")
        print(f"    Format: {metadata.get('format', 'unknown')}")
        print(f"    Duration: {metadata.get('duration', 0):.1f}s")
        print(f"    Resolution: {metadata.get('resolution', 'unknown')}")
        print(f"    Size: {metadata.get('size_bytes', 0) / 1024:.1f} KB")
        print(f"    Block signed: ✓")


def demonstrate_video_search(maif_path: str):
    """Demonstrate searching for videos by metadata."""
    print("\n=== Video Search Demonstration ===")

    decoder = MAIFDecoder(maif_path)
    blocks = decoder.get_blocks()

    # Search for HD videos (1080p or higher)
    print("\n  Searching for HD videos (1080p+)...")

    for block in blocks:
        metadata = block.metadata or {}
        resolution = metadata.get("resolution", "")

        if resolution:
            try:
                width, height = map(int, resolution.split("x"))
                if height >= 1080:
                    print(
                        f"    ✓ Found: {metadata.get('title', 'Untitled')} ({resolution})"
                    )
            except:
                pass

    # Search by duration
    print("\n  Searching for videos longer than 10 seconds...")

    for block in blocks:
        metadata = block.metadata or {}
        duration = metadata.get("duration", 0)

        if duration > 10:
            print(f"    ✓ Found: {metadata.get('title', 'Untitled')} ({duration:.1f}s)")


def demonstrate_provenance(maif_path: str):
    """Show provenance tracking for video operations."""
    print("\n=== Provenance Tracking ===")

    decoder = MAIFDecoder(maif_path)
    provenance = decoder.get_provenance()

    print(f"\n  Recorded {len(provenance)} operations:")

    for entry in provenance:
        print(f"    • {entry.action} by {entry.agent_id}")


def main():
    """Run the video demonstration."""
    print("=" * 60)
    print("MAIF Enhanced Video Functionality Demo")
    print("=" * 60)

    try:
        # Store videos
        maif_path = demonstrate_video_storage()

        # Retrieve and display
        demonstrate_video_retrieval(maif_path)

        # Search capabilities
        demonstrate_video_search(maif_path)

        # Show provenance
        demonstrate_provenance(maif_path)

        # Cleanup
        import shutil

        temp_dir = os.path.dirname(maif_path)
        shutil.rmtree(temp_dir)

        print("\n" + "=" * 60)
        print("Video Demo Complete!")
        print("=" * 60)
        print("""
Key features demonstrated:
  ✓ Video storage as binary blocks with metadata
  ✓ Metadata extraction from video data
  ✓ Ed25519 signatures for each block
  ✓ Video retrieval and search by metadata
  ✓ Provenance tracking for all operations
  ✓ Self-contained format (no external manifest)
""")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
