"""
Test audio block creation in v3 format.
"""

import unittest
import tempfile
import os
import shutil
from maif import MAIFEncoder, MAIFDecoder, BlockType


class TestAudioFix(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_audio_block_creation(self):
        """Test creating audio block."""
        encoder = MAIFEncoder(self.maif_path, agent_id="test")

        # Create audio block
        block_id = encoder.add_binary_block(
            data=b"test audio data",
            block_type=BlockType.AUDIO,
            metadata={"format": "mp3"},
        )
        encoder.finalize()

        # Verify block type is correct
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        block = decoder.blocks[0]
        self.assertEqual(block.header.block_type, BlockType.AUDIO)

        print("SUCCESS: Audio block created correctly.")


if __name__ == "__main__":
    unittest.main()
