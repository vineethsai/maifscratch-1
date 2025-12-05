#!/usr/bin/env python3
"""
Create sample MAIF files with various block types for testing viewers.

This creates:
1. sample_text.maif - Text content
2. sample_audio.maif - Audio content with metadata
3. sample_video.maif - Video content with metadata
4. sample_image.maif - Image content with metadata
5. sample_embeddings.maif - Embedding vectors
6. sample_multimodal.maif - All types combined
"""

import os
import sys
import struct
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif.secure_format import MAIFEncoder, SecureBlockType


def create_mock_audio_data(
    duration: float = 180.0, sample_rate: int = 44100
) -> tuple[bytes, dict]:
    """Create mock audio data with WAV-like header."""
    # Create a simple WAV-like header
    data = bytearray()

    # RIFF header
    data.extend(b"RIFF")
    data.extend(struct.pack("<I", 36 + 1024))  # file size - 8
    data.extend(b"WAVE")

    # fmt chunk
    data.extend(b"fmt ")
    data.extend(struct.pack("<I", 16))  # chunk size
    data.extend(struct.pack("<H", 1))  # audio format (PCM)
    data.extend(struct.pack("<H", 2))  # channels (stereo)
    data.extend(struct.pack("<I", sample_rate))  # sample rate
    data.extend(struct.pack("<I", sample_rate * 4))  # byte rate
    data.extend(struct.pack("<H", 4))  # block align
    data.extend(struct.pack("<H", 16))  # bits per sample

    # data chunk
    data.extend(b"data")
    data.extend(struct.pack("<I", 1024))
    data.extend(b"\x00" * 1024)  # mock audio samples

    metadata = {
        "format": "WAV",
        "codec": "PCM",
        "duration": duration,
        "sample_rate": sample_rate,
        "channels": 2,
        "bitrate": sample_rate * 4 * 8,
        "bit_depth": 16,
        "waveform_preview": True,
    }

    return bytes(data), metadata


def create_mock_video_data(
    duration: float = 150.0, width: int = 1920, height: int = 1080
) -> tuple[bytes, dict]:
    """Create mock MP4 video data."""
    data = bytearray()

    # ftyp box
    ftyp_size = 32
    data.extend(struct.pack(">I", ftyp_size))
    data.extend(b"ftyp")
    data.extend(b"mp42")
    data.extend(struct.pack(">I", 0))
    data.extend(b"mp42isom")
    data.extend(b"\x00" * 12)

    # mvhd box (movie header)
    mvhd_size = 108
    data.extend(struct.pack(">I", mvhd_size))
    data.extend(b"mvhd")
    data.extend(struct.pack(">I", 0))
    data.extend(struct.pack(">I", 0))
    data.extend(struct.pack(">I", 0))
    data.extend(struct.pack(">I", 1000))
    data.extend(struct.pack(">I", int(duration * 1000)))
    data.extend(b"\x00" * 76)

    # mdat box
    mdat_size = 2048
    data.extend(struct.pack(">I", mdat_size))
    data.extend(b"mdat")
    data.extend(b"\x00" * (mdat_size - 8))

    metadata = {
        "format": "MP4",
        "codec": "H.264",
        "width": width,
        "height": height,
        "duration": duration,
        "fps": 30,
        "frame_rate": 30,
        "bitrate": 8000000,
        "audio_codec": "AAC",
        "frame_count": int(duration * 30),
    }

    return bytes(data), metadata


def create_mock_image_data(width: int = 1024, height: int = 768) -> tuple[bytes, dict]:
    """Create mock PNG image data."""
    data = bytearray()

    # PNG signature
    data.extend(b"\x89PNG\r\n\x1a\n")

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    data.extend(struct.pack(">I", 13))  # chunk length
    data.extend(b"IHDR")
    data.extend(ihdr_data)
    data.extend(struct.pack(">I", 0))  # CRC placeholder

    # IDAT chunk (mock compressed data)
    idat_data = b"\x00" * 512
    data.extend(struct.pack(">I", len(idat_data)))
    data.extend(b"IDAT")
    data.extend(idat_data)
    data.extend(struct.pack(">I", 0))  # CRC placeholder

    # IEND chunk
    data.extend(struct.pack(">I", 0))
    data.extend(b"IEND")
    data.extend(struct.pack(">I", 0))  # CRC placeholder

    metadata = {
        "format": "PNG",
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "color_mode": "RGBA",
        "channels": 4,
        "bit_depth": 8,
        "dpi": 144,
    }

    return bytes(data), metadata


def create_embedding_data(dimensions: int = 384) -> tuple[bytes, dict]:
    """Create mock embedding vector."""
    import random

    random.seed(42)

    # Create normalized random embedding
    embedding = [random.gauss(0, 1) for _ in range(dimensions)]
    norm = sum(x * x for x in embedding) ** 0.5
    embedding = [x / norm for x in embedding]

    # Pack as float32
    data = struct.pack(f"{dimensions}f", *embedding)

    metadata = {
        "dimensions": dimensions,
        "dtype": "float32",
        "model": "all-MiniLM-L6-v2",
        "normalized": True,
    }

    return data, metadata


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "sample_maif_files")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Creating Sample MAIF Files for Viewer Testing")
    print("=" * 60)

    # 1. Text sample
    print("\n📝 Creating sample_text.maif...")
    text_path = os.path.join(output_dir, "sample_text.maif")
    encoder = MAIFEncoder(text_path, agent_id="sample-text-agent")
    encoder.add_text_block(
        "This is a sample text document stored in the MAIF format.\n\n"
        "MAIF (Model Artifact Interchange Format) is a secure, self-contained "
        "file format designed for AI applications. It supports:\n\n"
        "• Text content with full Unicode support\n"
        "• Embedding vectors for semantic search\n"
        "• Audio and video content\n"
        "• Images and binary data\n"
        "• Knowledge graphs\n"
        "• Cryptographic signatures and provenance chains\n\n"
        "All content is signed and hash-chained for integrity verification.",
        metadata={"source": "sample_document.txt", "language": "en"},
    )
    encoder.finalize()
    print(f"   ✅ Created: {text_path}")

    # 2. Audio sample
    print("\n🎵 Creating sample_audio.maif...")
    audio_path = os.path.join(output_dir, "sample_audio.maif")
    encoder = MAIFEncoder(audio_path, agent_id="sample-audio-agent")
    audio_data, audio_meta = create_mock_audio_data(duration=185.5, sample_rate=48000)
    encoder.add_binary_block(
        audio_data, block_type=SecureBlockType.AUDIO, metadata=audio_meta
    )
    encoder.finalize()
    print(f"   ✅ Created: {audio_path}")
    print(
        f"      Duration: {audio_meta['duration']:.1f}s, {audio_meta['sample_rate']}Hz, {audio_meta['channels']} channels"
    )

    # 3. Video sample
    print("\n🎬 Creating sample_video.maif...")
    video_path = os.path.join(output_dir, "sample_video.maif")
    encoder = MAIFEncoder(video_path, agent_id="sample-video-agent")
    video_data, video_meta = create_mock_video_data(
        duration=152.3, width=1920, height=1080
    )
    encoder.add_binary_block(
        video_data, block_type=SecureBlockType.VIDEO, metadata=video_meta
    )
    encoder.finalize()
    print(f"   ✅ Created: {video_path}")
    print(
        f"      Resolution: {video_meta['width']}×{video_meta['height']}, {video_meta['duration']:.1f}s @ {video_meta['fps']}fps"
    )

    # 4. Image sample
    print("\n🖼️ Creating sample_image.maif...")
    image_path = os.path.join(output_dir, "sample_image.maif")
    encoder = MAIFEncoder(image_path, agent_id="sample-image-agent")
    image_data, image_meta = create_mock_image_data(width=2048, height=1536)
    encoder.add_binary_block(
        image_data, block_type=SecureBlockType.IMAGE, metadata=image_meta
    )
    encoder.finalize()
    print(f"   ✅ Created: {image_path}")
    print(
        f"      Dimensions: {image_meta['width']}×{image_meta['height']}, {image_meta['color_mode']}"
    )

    # 5. Embeddings sample
    print("\n🧠 Creating sample_embeddings.maif...")
    emb_path = os.path.join(output_dir, "sample_embeddings.maif")
    encoder = MAIFEncoder(emb_path, agent_id="sample-embeddings-agent")
    emb_data, emb_meta = create_embedding_data(dimensions=384)
    encoder.add_binary_block(
        emb_data, block_type=SecureBlockType.EMBEDDINGS, metadata=emb_meta
    )
    encoder.finalize()
    print(f"   ✅ Created: {emb_path}")
    print(f"      Dimensions: {emb_meta['dimensions']}, Model: {emb_meta['model']}")

    # 6. Multimodal sample (all types)
    print("\n🌐 Creating sample_multimodal.maif...")
    multi_path = os.path.join(output_dir, "sample_multimodal.maif")
    encoder = MAIFEncoder(multi_path, agent_id="sample-multimodal-agent")

    # Add text
    encoder.add_text_block(
        "This is a multimodal MAIF file containing various content types:\n"
        "- Text content (this block)\n"
        "- Audio recording\n"
        "- Video clip\n"
        "- Image data\n"
        "- Embedding vectors\n",
        metadata={"type": "description", "language": "en"},
    )

    # Add audio
    audio_data, audio_meta = create_mock_audio_data(duration=30.0)
    encoder.add_binary_block(audio_data, SecureBlockType.AUDIO, audio_meta)

    # Add video
    video_data, video_meta = create_mock_video_data(
        duration=15.0, width=1280, height=720
    )
    encoder.add_binary_block(video_data, SecureBlockType.VIDEO, video_meta)

    # Add image
    image_data, image_meta = create_mock_image_data(width=800, height=600)
    encoder.add_binary_block(image_data, SecureBlockType.IMAGE, image_meta)

    # Add embeddings
    emb_data, emb_meta = create_embedding_data(dimensions=768)
    encoder.add_binary_block(emb_data, SecureBlockType.EMBEDDINGS, emb_meta)

    encoder.finalize()
    print(f"   ✅ Created: {multi_path}")
    print(f"      Contains: text, audio, video, image, embeddings")

    print("\n" + "=" * 60)
    print("✅ All sample files created successfully!")
    print("=" * 60)
    print(f"\n📁 Output directory: {output_dir}")
    print("\nTo test in VS Code extension:")
    print(f"   Open any .maif file from: {output_dir}")
    print("\nTo test in web viewer:")
    print(f"   1. Open tools/maif-explorer/index.html")
    print(f"   2. Drag and drop a .maif file")


if __name__ == "__main__":
    main()
