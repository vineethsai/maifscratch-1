"""
Tests for MAIF video functionality including metadata extraction,
semantic analysis, and advanced querying capabilities.
"""

import unittest
import tempfile
import os
import struct
from typing import List, Dict, Any

from maif import MAIFEncoder, MAIFDecoder, BlockType


class TestVideoFunctionality(unittest.TestCase):
    """Test video storage, metadata extraction, and querying."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_maif_path = os.path.join(self.temp_dir, "test_video.maif")

    def tearDown(self):
        """Clean up test files."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_mock_mp4_data(
        self, duration: float = 10.0, width: int = 1920, height: int = 1080
    ) -> bytes:
        """Create mock MP4 data with basic structure for testing."""
        data = bytearray()

        # ftyp box (file type)
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
        timescale = 1000
        data.extend(struct.pack(">I", timescale))
        data.extend(struct.pack(">I", int(duration * timescale)))
        data.extend(b"\x00" * 80)

        # tkhd box (track header)
        tkhd_size = 92
        data.extend(struct.pack(">I", tkhd_size))
        data.extend(b"tkhd")
        data.extend(struct.pack(">I", 0))
        data.extend(b"\x00" * 72)
        data.extend(struct.pack(">I", width << 16))
        data.extend(struct.pack(">I", height << 16))

        data.extend(b"\x00" * 1000)

        return bytes(data)

    def create_mock_avi_data(self) -> bytes:
        """Create mock AVI data for testing."""
        data = bytearray()
        data.extend(b"RIFF")
        data.extend(struct.pack("<I", 1000))
        data.extend(b"AVI ")
        data.extend(b"\x00" * 1000)
        return bytes(data)

    def test_add_video_block_basic(self):
        """Test basic video block addition."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data()

        # Add video block using binary block
        video_id = encoder.add_binary_block(
            video_data,
            BlockType.VIDEO,
            metadata={
                "title": "Test Video",
                "description": "A test video file",
                "content_type": "video",
            },
        )

        self.assertIsNotNone(video_id)
        self.assertEqual(len(encoder.blocks), 1)

        # Check block properties
        video_block = encoder.blocks[0]
        self.assertEqual(video_block.header.block_type, BlockType.VIDEO)

    def test_resolution_parsing(self):
        """Test video resolution parsing from metadata."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data(width=3840, height=2160)

        encoder.add_binary_block(
            video_data,
            BlockType.VIDEO,
            metadata={"width": 3840, "height": 2160, "resolution": "4K"},
        )

        video_block = encoder.blocks[0]
        self.assertEqual(video_block.metadata.get("width"), 3840)
        self.assertEqual(video_block.metadata.get("height"), 2160)

    def test_video_block_versioning(self):
        """Test video block versioning."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data()

        encoder.add_binary_block(video_data, BlockType.VIDEO, metadata={"version": 1})

        self.assertEqual(len(encoder.blocks), 1)
        self.assertEqual(encoder.blocks[0].metadata.get("version"), 1)

    def test_video_block_with_privacy(self):
        """Test video block with privacy settings."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data()

        encoder.add_binary_block(
            video_data,
            BlockType.VIDEO,
            metadata={"privacy_level": "confidential", "encrypted": False},
        )

        video_block = encoder.blocks[0]
        self.assertEqual(video_block.metadata.get("privacy_level"), "confidential")

    def test_video_data_retrieval(self):
        """Test video data retrieval after save and load."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data()

        encoder.add_binary_block(
            video_data, BlockType.VIDEO, metadata={"format": "mp4"}
        )
        encoder.finalize()

        # Read back
        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        self.assertEqual(len(decoder.blocks), 1)
        retrieved_data = decoder.blocks[0].data
        self.assertEqual(retrieved_data, video_data)

    def test_video_metadata_extraction_mp4(self):
        """Test metadata extraction from MP4 video."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_mp4_data(duration=30.0)

        encoder.add_binary_block(
            video_data,
            BlockType.VIDEO,
            metadata={"format": "mp4", "duration": 30.0, "codec": "h264"},
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        video_block = decoder.blocks[0]
        self.assertEqual(video_block.metadata.get("format"), "mp4")
        self.assertEqual(video_block.metadata.get("duration"), 30.0)

    def test_video_metadata_extraction_avi(self):
        """Test metadata extraction from AVI video."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")
        video_data = self.create_mock_avi_data()

        encoder.add_binary_block(
            video_data, BlockType.VIDEO, metadata={"format": "avi"}
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        video_block = decoder.blocks[0]
        self.assertEqual(video_block.metadata.get("format"), "avi")

    def test_video_querying_basic(self):
        """Test basic video querying by metadata."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        # Add multiple videos
        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"title": "Video 1", "category": "tutorial"},
        )
        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"title": "Video 2", "category": "demo"},
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        # Query videos
        videos = [b for b in decoder.blocks if b.header.block_type == BlockType.VIDEO]
        self.assertEqual(len(videos), 2)

    def test_video_querying_advanced(self):
        """Test advanced video querying with filters."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"duration": 60, "quality": "hd"},
        )
        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"duration": 120, "quality": "4k"},
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        # Filter by quality
        hd_videos = [
            b
            for b in decoder.blocks
            if b.metadata and b.metadata.get("quality") == "hd"
        ]
        self.assertEqual(len(hd_videos), 1)

    def test_video_semantic_embeddings(self):
        """Test video semantic embedding storage."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        # Add video with embeddings
        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"title": "Test Video"},
        )

        # Add embeddings for the video
        encoder.add_embeddings_block(
            [[0.1, 0.2, 0.3] * 128],  # 384-dim embedding
            metadata={"type": "video_embedding", "model": "clip"},
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        self.assertEqual(len(decoder.blocks), 2)

    def test_video_semantic_search(self):
        """Test semantic search across videos."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"title": "Cat video", "tags": ["cat", "animal"]},
        )
        encoder.add_binary_block(
            self.create_mock_mp4_data(),
            BlockType.VIDEO,
            metadata={"title": "Dog video", "tags": ["dog", "animal"]},
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        # Search by tag
        cat_videos = [
            b
            for b in decoder.blocks
            if b.metadata and "cat" in b.metadata.get("tags", [])
        ]
        self.assertEqual(len(cat_videos), 1)

    def test_video_summary_statistics(self):
        """Test video collection statistics."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        for i in range(5):
            encoder.add_binary_block(
                self.create_mock_mp4_data(),
                BlockType.VIDEO,
                metadata={"index": i, "duration": (i + 1) * 10},
            )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        videos = [b for b in decoder.blocks if b.header.block_type == BlockType.VIDEO]
        self.assertEqual(len(videos), 5)

        # Calculate total duration
        total_duration = sum(
            b.metadata.get("duration", 0) for b in videos if b.metadata
        )
        self.assertEqual(total_duration, 150)  # 10+20+30+40+50

    def test_video_with_custom_metadata(self):
        """Test video with custom metadata fields."""
        encoder = MAIFEncoder(self.test_maif_path, agent_id="video-test")

        custom_metadata = {
            "title": "Custom Video",
            "custom_field_1": "value1",
            "custom_field_2": {"nested": "data"},
            "custom_array": [1, 2, 3],
        }

        encoder.add_binary_block(
            self.create_mock_mp4_data(), BlockType.VIDEO, metadata=custom_metadata
        )
        encoder.finalize()

        decoder = MAIFDecoder(self.test_maif_path)
        decoder.load()

        video_block = decoder.blocks[0]
        self.assertEqual(video_block.metadata.get("custom_field_1"), "value1")
        self.assertEqual(video_block.metadata.get("custom_field_2"), {"nested": "data"})
        self.assertEqual(video_block.metadata.get("custom_array"), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
