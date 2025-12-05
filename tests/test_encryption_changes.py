#!/usr/bin/env python3
"""
Test suite for verifying encryption functionality (AES-GCM).
"""

import pytest
import tempfile
import os
import shutil

from maif import MAIFEncoder, MAIFDecoder
from maif.privacy import PrivacyEngine, EncryptionMode


class TestEncryptionModeChanges:
    """Test suite for encryption mode changes."""

    def test_fernet_enum_removed(self):
        """Verify Fernet is no longer in EncryptionMode enum."""
        available_modes = [mode.value for mode in EncryptionMode]
        assert "fernet" not in available_modes
        print("✓ Fernet successfully removed from EncryptionMode enum")

    def test_aes_gcm_available(self):
        """Verify AES-GCM is available."""
        available_modes = [mode.value for mode in EncryptionMode]
        assert "aes_gcm" in available_modes
        print("✓ AES-GCM is available in EncryptionMode enum")

    def test_aes_gcm_encryption_decryption(self):
        """Verify AES-GCM encryption and decryption works."""
        privacy_engine = PrivacyEngine()
        test_data = b"This is test data for AES-GCM encryption"
        block_id = "test_block_001"

        # Encrypt data
        encrypted_data, metadata = privacy_engine.encrypt_data(
            test_data, block_id, EncryptionMode.AES_GCM
        )

        # Verify encryption worked
        assert encrypted_data != test_data
        assert metadata["algorithm"] == "AES-GCM"
        assert "iv" in metadata
        assert "tag" in metadata

        # Decrypt data
        decrypted_data = privacy_engine.decrypt_data(encrypted_data, block_id, metadata)

        # Verify decryption worked
        assert decrypted_data == test_data
        print("✓ AES-GCM encryption and decryption working correctly")

    def test_fernet_encryption_raises_error(self):
        """Verify attempting to use Fernet raises an error."""
        try:
            fernet_mode = EncryptionMode.FERNET
            # Should not get here
            assert False, "FERNET should not exist"
        except AttributeError:
            # Expected
            print("✓ Fernet enum access correctly raises AttributeError")

    def test_maif_encoder_with_aes_gcm(self):
        """Verify MAIFEncoder works with encrypted content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            maif_path = os.path.join(temp_dir, "encrypted.maif")

            # Create encoder
            encoder = MAIFEncoder(maif_path, agent_id="encryption-test")

            # Add encrypted content (encryption handled at privacy layer)
            encoder.add_text_block("Secure content", metadata={"encrypted": True})
            encoder.finalize()

            # Verify file created
            assert os.path.exists(maif_path)

            # Verify readable
            decoder = MAIFDecoder(maif_path)
            decoder.load()
            assert len(decoder.blocks) == 1

            print("✓ MAIFEncoder works with encrypted content")

    def test_default_encryption_is_aes_gcm(self):
        """Verify default encryption mode is AES-GCM."""
        privacy_engine = PrivacyEngine()
        test_data = b"Test data"
        block_id = "test"

        # Encrypt with default mode
        encrypted, metadata = privacy_engine.encrypt_data(test_data, block_id)

        # Verify AES-GCM is default
        assert metadata["algorithm"] == "AES-GCM"
        print("✓ Default encryption mode is AES-GCM")


class TestEncryptionIntegration:
    """Test encryption integration."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "test.maif")

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_encrypted_block_roundtrip(self):
        """Test encrypted data roundtrip."""
        privacy = PrivacyEngine()

        original_data = b"Sensitive information"
        block_id = "sensitive_block"

        # Encrypt
        encrypted, metadata = privacy.encrypt_data(original_data, block_id)

        # Decrypt
        decrypted = privacy.decrypt_data(encrypted, block_id, metadata)

        assert decrypted == original_data

    def test_multiple_blocks_encryption(self):
        """Test encrypting multiple blocks."""
        privacy = PrivacyEngine()

        blocks = [
            (b"Block 1", "block1"),
            (b"Block 2", "block2"),
            (b"Block 3", "block3"),
        ]

        encrypted_blocks = []
        for data, block_id in blocks:
            encrypted, metadata = privacy.encrypt_data(data, block_id)
            encrypted_blocks.append((encrypted, metadata, block_id))

        # Verify all can be decrypted
        for i, (encrypted, metadata, block_id) in enumerate(encrypted_blocks):
            decrypted = privacy.decrypt_data(encrypted, block_id, metadata)
            assert decrypted == blocks[i][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
