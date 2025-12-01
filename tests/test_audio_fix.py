
import unittest
from maif.core import MAIFEncoder
from maif.block_types import BlockType

class TestAudioFix(unittest.TestCase):
    def test_audio_block_creation(self):
        encoder = MAIFEncoder()
        
        # Test creating audio block with "audio" string
        block_id = encoder.add_binary_block(
            data=b"test audio data",
            block_type="audio",
            metadata={"format": "mp3"}
        )
        
        # Verify block type is correct
        block = encoder.blocks[0]
        self.assertEqual(block.block_type, BlockType.AUDIO_DATA.value)
        self.assertEqual(block.block_type, "AUDI")
        
        print("SUCCESS: Audio block created correctly with 'audio' string.")

if __name__ == "__main__":
    unittest.main()
