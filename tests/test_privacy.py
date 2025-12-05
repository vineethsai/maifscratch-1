"""
Comprehensive tests for MAIF privacy functionality.
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from maif.privacy import (
    PrivacyEngine,
    PrivacyPolicy,
    AccessRule,
    PrivacyLevel,
    EncryptionMode,
    DifferentialPrivacy,
    SecureMultipartyComputation,
    ZeroKnowledgeProof,
)


class TestPrivacyPolicy:
    """Test PrivacyPolicy data structure."""

    def test_privacy_policy_creation(self):
        """Test basic PrivacyPolicy creation."""
        policy = PrivacyPolicy(
            level=PrivacyLevel.HIGH,
            encryption_mode=EncryptionMode.AES_GCM,
            anonymize=True,
            retention_days=365,
            access_conditions={"role": "admin"},
        )

        assert policy.level == PrivacyLevel.HIGH
        assert policy.encryption_mode == EncryptionMode.AES_GCM
        assert policy.anonymize is True
        assert policy.retention_days == 365
        assert policy.access_conditions["role"] == "admin"

    def test_privacy_policy_defaults(self):
        """Test PrivacyPolicy default values."""
        policy = PrivacyPolicy()

        assert policy.level == PrivacyLevel.MEDIUM
        assert policy.encryption_mode == EncryptionMode.AES_GCM
        assert policy.anonymize is False
        assert policy.retention_days == 30
        assert policy.access_conditions == {}

    def test_privacy_policy_validation(self):
        """Test PrivacyPolicy validation in __post_init__."""
        # Valid retention days
        policy = PrivacyPolicy(retention_days=365)
        assert policy.retention_days == 365

        # Invalid retention days should be corrected
        policy = PrivacyPolicy(retention_days=-1)
        assert policy.retention_days == 30  # Should default to 30


class TestAccessRule:
    """Test AccessRule data structure."""

    def test_access_rule_creation(self):
        """Test basic AccessRule creation."""
        rule = AccessRule(
            subject="user123",
            resource="block_*",
            permissions=["read", "write"],
            conditions={"time_limit": "2024-12-31"},
        )

        assert rule.subject == "user123"
        assert rule.resource == "block_*"
        assert rule.permissions == ["read", "write"]
        assert rule.conditions["time_limit"] == "2024-12-31"


class TestPrivacyEngine:
    """Test PrivacyEngine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.privacy_engine = PrivacyEngine()

    def test_privacy_engine_initialization(self):
        """Test PrivacyEngine initialization."""
        assert self.privacy_engine.master_key is not None
        assert len(self.privacy_engine.master_key) == 32  # 256 bits
        assert self.privacy_engine.access_rules == []
        assert self.privacy_engine.retention_policies == {}

    def test_derive_key(self):
        """Test key derivation."""
        context = "test_context"

        # Derive key without salt
        key1 = self.privacy_engine.derive_key(context)
        assert len(key1) == 32  # 256 bits

        # Derive key with salt
        salt = b"test_salt_123456"
        key2 = self.privacy_engine.derive_key(context, salt)
        assert len(key2) == 32

        # Same context should produce same key
        key3 = self.privacy_engine.derive_key(context)
        assert key1 == key3

        # Different context should produce different key
        key4 = self.privacy_engine.derive_key("different_context")
        assert key1 != key4

    def test_encrypt_decrypt_aes_gcm(self):
        """Test AES-GCM encryption and decryption."""
        test_data = b"Sensitive information that needs encryption"
        block_id = "test_block_123"

        # Encrypt data
        encrypted_data, metadata = self.privacy_engine.encrypt_data(
            data=test_data, block_id=block_id, encryption_mode=EncryptionMode.AES_GCM
        )

        assert encrypted_data != test_data
        assert metadata["algorithm"] == "AES-GCM"
        assert "iv" in metadata
        assert "tag" in metadata

        # Decrypt data
        decrypted_data = self.privacy_engine.decrypt_data(
            encrypted_data=encrypted_data,
            block_id=block_id,
            encryption_metadata=metadata,
        )

        assert decrypted_data == test_data

    def test_encrypt_decrypt_chacha20(self):
        """Test ChaCha20 encryption and decryption."""
        test_data = b"Another sensitive piece of information"
        block_id = "test_block_456"

        # Encrypt data
        encrypted_data, metadata = self.privacy_engine.encrypt_data(
            data=test_data,
            block_id=block_id,
            encryption_mode=EncryptionMode.CHACHA20_POLY1305,
        )

        assert encrypted_data != test_data
        assert metadata["algorithm"] == "ChaCha20-Poly1305"
        assert "nonce" in metadata

        # Decrypt data
        decrypted_data = self.privacy_engine.decrypt_data(
            encrypted_data=encrypted_data,
            block_id=block_id,
            encryption_metadata=metadata,
        )

        assert decrypted_data == test_data

    def test_encryption_with_different_block_ids(self):
        """Test that different block IDs produce different encrypted data."""
        test_data = b"Same data, different blocks"

        encrypted1, _ = self.privacy_engine.encrypt_data(
            data=test_data, block_id="block_1", encryption_mode=EncryptionMode.AES_GCM
        )

        encrypted2, _ = self.privacy_engine.encrypt_data(
            data=test_data, block_id="block_2", encryption_mode=EncryptionMode.AES_GCM
        )

        # Same data with different block IDs should produce different ciphertext
        assert encrypted1 != encrypted2

    def test_anonymize_data(self):
        """Test data anonymization."""
        test_cases = [
            ("John Smith works at ACME Corp", ["John Smith", "ACME Corp"]),
            ("Call me at 555-123-4567", ["555-123-4567"]),
            ("My email is john@example.com", ["john@example.com"]),
            ("SSN: 123-45-6789", ["123-45-6789"]),
            ("Credit card: 4532-1234-5678-9012", ["4532-1234-5678-9012"]),
        ]

        for original_text, sensitive_items in test_cases:
            anonymized = self.privacy_engine.anonymize_data(original_text, "general")

            # Check that sensitive items are removed/replaced
            for item in sensitive_items:
                assert item not in anonymized, f"Failed to anonymize: {item}"

            # Check that some content remains
            assert len(anonymized) > 0
            assert anonymized != original_text

    def test_anonymize_data_contexts(self):
        """Test anonymization with different contexts."""
        text = "Patient John Doe, age 45, diagnosed with condition X"

        # Medical context should be more aggressive
        medical_anonymized = self.privacy_engine.anonymize_data(text, "medical")

        # General context
        general_anonymized = self.privacy_engine.anonymize_data(text, "general")

        # Both should anonymize the name
        assert "John Doe" not in medical_anonymized
        assert "John Doe" not in general_anonymized

        # Results might differ based on context
        assert medical_anonymized is not None
        assert general_anonymized is not None

    def test_check_access_with_rules(self):
        """Test access checking with defined rules."""
        # Add access rules
        self.privacy_engine.access_rules = [
            AccessRule(
                subject="user123",
                resource="block_*",
                permissions=["read"],
                conditions={},
            ),
            AccessRule(
                subject="admin",
                resource="*",
                permissions=["read", "write", "delete"],
                conditions={},
            ),
        ]

        # Test user access
        assert self.privacy_engine.check_access("user123", "block_001", "read") is True
        assert (
            self.privacy_engine.check_access("user123", "block_001", "write") is False
        )
        assert (
            self.privacy_engine.check_access("user123", "other_resource", "read")
            is False
        )

        # Test admin access
        assert self.privacy_engine.check_access("admin", "any_resource", "read") is True
        assert (
            self.privacy_engine.check_access("admin", "any_resource", "write") is True
        )
        assert (
            self.privacy_engine.check_access("admin", "any_resource", "delete") is True
        )

        # Test unauthorized user
        assert (
            self.privacy_engine.check_access("unauthorized", "block_001", "read")
            is False
        )

    def test_pattern_matching(self):
        """Test resource pattern matching."""
        test_cases = [
            ("block_123", "block_*", True),
            ("block_456", "block_*", True),
            ("other_123", "block_*", False),
            ("any_resource", "*", True),
            ("specific_resource", "specific_resource", True),
            ("specific_resource", "different_resource", False),
        ]

        for resource, pattern, expected in test_cases:
            result = self.privacy_engine._matches_pattern(resource, pattern)
            assert result == expected, (
                f"Pattern matching failed: {resource} vs {pattern}"
            )

    def test_retention_policy_enforcement(self):
        """Test retention policy enforcement."""
        import time

        # Set retention policy
        self.privacy_engine.retention_policies["test_block"] = {
            "created_at": time.time() - (40 * 24 * 3600),  # 40 days ago
            "retention_days": 30,
        }

        # Mock the deletion function
        deleted_blocks = []

        def mock_delete_block(block_id):
            deleted_blocks.append(block_id)

        with patch.object(
            self.privacy_engine, "_delete_expired_block", mock_delete_block
        ):
            self.privacy_engine.enforce_retention_policy()

        # Block should be marked for deletion
        assert "test_block" in deleted_blocks

    def test_privacy_report_generation(self):
        """Test privacy report generation."""
        # Add some test data
        self.privacy_engine.access_rules = [
            AccessRule("user1", "resource1", ["read"], {}),
            AccessRule("user2", "resource2", ["read", "write"], {}),
        ]

        self.privacy_engine.retention_policies = {
            "block1": {"created_at": 1234567890, "retention_days": 30},
            "block2": {"created_at": 1234567890, "retention_days": 365},
        }

        report = self.privacy_engine.generate_privacy_report()

        assert "access_rules_count" in report
        assert "retention_policies_count" in report
        assert "encryption_enabled" in report
        assert report["access_rules_count"] == 2
        assert report["retention_policies_count"] == 2
        assert report["encryption_enabled"] is True


class TestDifferentialPrivacy:
    """Test DifferentialPrivacy functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dp = DifferentialPrivacy(epsilon=1.0)

    def test_differential_privacy_initialization(self):
        """Test DifferentialPrivacy initialization."""
        assert self.dp.epsilon == 1.0

    def test_add_noise(self):
        """Test noise addition for differential privacy."""
        original_value = 100.0

        # Add noise multiple times
        noisy_values = []
        for _ in range(100):
            noisy_value = self.dp.add_noise(original_value)
            noisy_values.append(noisy_value)

        # Check that noise was added (values should vary)
        unique_values = set(noisy_values)
        assert len(unique_values) > 1  # Should have different values due to noise

        # Check that values are reasonably close to original
        avg_value = sum(noisy_values) / len(noisy_values)
        assert abs(avg_value - original_value) < 10.0  # Should be close on average

    def test_noise_sensitivity(self):
        """Test noise scaling with sensitivity parameter."""
        value = 50.0

        # Low sensitivity should add less noise
        low_sensitivity_noise = []
        for _ in range(50):
            noisy = self.dp.add_noise(value, sensitivity=0.1)
            low_sensitivity_noise.append(abs(noisy - value))

        # High sensitivity should add more noise
        high_sensitivity_noise = []
        for _ in range(50):
            noisy = self.dp.add_noise(value, sensitivity=10.0)
            high_sensitivity_noise.append(abs(noisy - value))

        avg_low_noise = sum(low_sensitivity_noise) / len(low_sensitivity_noise)
        avg_high_noise = sum(high_sensitivity_noise) / len(high_sensitivity_noise)

        # High sensitivity should generally produce more noise
        assert avg_high_noise > avg_low_noise


class TestSecureMultipartyComputation:
    """Test SecureMultipartyComputation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.smc = SecureMultipartyComputation()

    def test_secret_sharing(self):
        """Test secret sharing functionality."""
        secret_value = 12345
        num_parties = 3

        shares = self.smc.secret_share(secret_value, num_parties)

        assert len(shares) == num_parties
        assert all(isinstance(share, int) for share in shares)

        # Shares should be different
        assert len(set(shares)) > 1

        # Sum of shares should equal original value (simple additive sharing)
        assert sum(shares) == secret_value

    def test_secret_sharing_different_parties(self):
        """Test secret sharing with different numbers of parties."""
        secret_value = 98765

        for num_parties in [2, 3, 5, 10]:
            shares = self.smc.secret_share(secret_value, num_parties)

            assert len(shares) == num_parties
            assert sum(shares) == secret_value


class TestZeroKnowledgeProof:
    """Test ZeroKnowledgeProof functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.zkp = ZeroKnowledgeProof()

    def test_commitment_and_verification(self):
        """Test commitment creation and verification."""
        secret_value = b"secret_data_12345"

        # Create commitment
        commitment = self.zkp.commit(secret_value)

        assert commitment is not None
        assert len(commitment) > 0
        assert commitment != secret_value  # Should be different from original

    def test_commitment_with_nonce(self):
        """Test commitment with explicit nonce."""
        secret_value = b"secret_data_67890"
        nonce = b"explicit_nonce_123"

        # Create commitment with nonce
        commitment = self.zkp.commit(secret_value, nonce)

        # Verify commitment
        is_valid = self.zkp.verify_commitment(commitment, secret_value, nonce)
        assert is_valid is True

        # Verification with wrong value should fail
        wrong_value = b"wrong_secret_data"
        is_valid = self.zkp.verify_commitment(commitment, wrong_value, nonce)
        assert is_valid is False

        # Verification with wrong nonce should fail
        wrong_nonce = b"wrong_nonce_456"
        is_valid = self.zkp.verify_commitment(commitment, secret_value, wrong_nonce)
        assert is_valid is False

    def test_commitment_deterministic(self):
        """Test that commitments are deterministic with same inputs."""
        secret_value = b"deterministic_test"
        nonce = b"fixed_nonce_789"

        # Create multiple commitments with same inputs
        commitment1 = self.zkp.commit(secret_value, nonce)
        commitment2 = self.zkp.commit(secret_value, nonce)

        # Should be identical
        assert commitment1 == commitment2

    def test_commitment_different_with_different_inputs(self):
        """Test that commitments differ with different inputs."""
        nonce = b"same_nonce_123"

        # Different values should produce different commitments
        commitment1 = self.zkp.commit(b"value1", nonce)
        commitment2 = self.zkp.commit(b"value2", nonce)

        assert commitment1 != commitment2

        # Same value with different nonces should produce different commitments
        value = b"same_value"
        commitment3 = self.zkp.commit(value, b"nonce1")
        commitment4 = self.zkp.commit(value, b"nonce2")

        assert commitment3 != commitment4


class TestPrivacyIntegration:
    """Test privacy integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.privacy_engine = PrivacyEngine()

    def test_end_to_end_privacy_workflow(self):
        """Test complete privacy workflow."""
        # 1. Define privacy policy
        policy = PrivacyPolicy(
            level=PrivacyLevel.HIGH,
            encryption_mode=EncryptionMode.AES_GCM,
            anonymize=True,
            retention_days=90,
        )

        # 2. Original sensitive data
        original_data = "Patient John Doe, SSN: 123-45-6789, diagnosed with condition X"

        # 3. Anonymize data
        anonymized_data = self.privacy_engine.anonymize_data(original_data, "medical")
        assert "John Doe" not in anonymized_data
        assert "123-45-6789" not in anonymized_data

        # 4. Encrypt anonymized data
        encrypted_data, metadata = self.privacy_engine.encrypt_data(
            data=anonymized_data.encode("utf-8"),
            block_id="patient_record_001",
            encryption_mode=policy.encryption_mode,
        )

        # 5. Verify encryption
        assert encrypted_data != anonymized_data.encode("utf-8")
        assert metadata["algorithm"] == "AES-GCM"

        # 6. Decrypt and verify
        decrypted_data = self.privacy_engine.decrypt_data(
            encrypted_data=encrypted_data,
            block_id="patient_record_001",
            encryption_metadata=metadata,
        )

        assert decrypted_data.decode("utf-8") == anonymized_data

    def test_multi_level_privacy(self):
        """Test different privacy levels."""
        test_data = "Sensitive corporate information about Project Alpha"

        privacy_levels = [
            (PrivacyLevel.LOW, False),
            (PrivacyLevel.MEDIUM, True),
            (PrivacyLevel.HIGH, True),
        ]

        for level, should_anonymize in privacy_levels:
            policy = PrivacyPolicy(level=level, anonymize=should_anonymize)

            if should_anonymize:
                processed_data = self.privacy_engine.anonymize_data(
                    test_data, "corporate"
                )
                # Some anonymization should occur
                assert processed_data != test_data
            else:
                processed_data = test_data

            # All levels should support encryption
            encrypted_data, metadata = self.privacy_engine.encrypt_data(
                data=processed_data.encode("utf-8"),
                block_id=f"test_block_{level.value}",
                encryption_mode=policy.encryption_mode,
            )

            assert encrypted_data != processed_data.encode("utf-8")
            assert metadata["algorithm"] in ["AES-GCM", "ChaCha20-Poly1305"]


class TestPrivacyErrorHandling:
    """Test privacy error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.privacy_engine = PrivacyEngine()

    def test_empty_data_encryption(self):
        """Test encryption of empty data."""
        empty_data = b""

        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="Cannot encrypt empty data"):
            self.privacy_engine.encrypt_data(
                data=empty_data,
                block_id="empty_block",
                encryption_mode=EncryptionMode.AES_GCM,
            )

    def test_invalid_encryption_mode(self):
        """Test handling of invalid encryption modes."""
        test_data = b"test data"

        # This should raise an exception or handle gracefully
        with pytest.raises((ValueError, AttributeError)):
            self.privacy_engine.encrypt_data(
                data=test_data, block_id="test_block", encryption_mode="INVALID_MODE"
            )

    def test_corrupted_encryption_metadata(self):
        """Test handling of corrupted encryption metadata."""
        test_data = b"test data for corruption test"

        # Encrypt normally
        encrypted_data, metadata = self.privacy_engine.encrypt_data(
            data=test_data,
            block_id="corruption_test",
            encryption_mode=EncryptionMode.AES_GCM,
        )

        # Corrupt metadata
        corrupted_metadata = metadata.copy()
        corrupted_metadata["iv"] = "corrupted_iv"

        # Decryption should fail gracefully
        with pytest.raises(Exception):
            self.privacy_engine.decrypt_data(
                encrypted_data=encrypted_data,
                block_id="corruption_test",
                encryption_metadata=corrupted_metadata,
            )

    def test_anonymization_edge_cases(self):
        """Test anonymization with edge cases."""
        edge_cases = [
            "",  # Empty string
            "   ",  # Whitespace only
            "No sensitive data here",  # No patterns to anonymize
            "123",  # Numbers only
            "!@#$%^&*()",  # Special characters only
        ]

        for test_case in edge_cases:
            result = self.privacy_engine.anonymize_data(test_case, "general")

            # Should not crash and should return a string
            assert isinstance(result, str)
            assert len(result) >= 0


class TestPrivacyPerformance:
    """Test privacy performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.privacy_engine = PrivacyEngine()

    def test_encryption_performance(self):
        """Test encryption performance with various data sizes."""
        import time

        data_sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB

        for size in data_sizes:
            test_data = b"x" * size

            start_time = time.time()
            encrypted_data, metadata = self.privacy_engine.encrypt_data(
                data=test_data,
                block_id=f"perf_test_{size}",
                encryption_mode=EncryptionMode.AES_GCM,
            )
            end_time = time.time()

            duration = end_time - start_time

            assert encrypted_data is not None
            assert duration < 2.0  # Should complete in under 2 seconds

    def test_anonymization_performance(self):
        """Test anonymization performance."""
        import time

        # Create text with multiple sensitive patterns
        test_text = (
            """
        John Smith works at ACME Corp. His email is john.smith@acme.com.
        Phone: 555-123-4567. SSN: 123-45-6789.
        Jane Doe also works there. Her email is jane.doe@acme.com.
        Phone: 555-987-6543. SSN: 987-65-4321.
        """
            * 100
        )  # Repeat to make it larger

        start_time = time.time()
        anonymized = self.privacy_engine.anonymize_data(test_text, "general")
        end_time = time.time()

        duration = end_time - start_time

        assert anonymized is not None
        assert duration < 5.0  # Should complete in under 5 seconds

        # Verify anonymization occurred
        assert "John Smith" not in anonymized
        assert "Jane Doe" not in anonymized
        assert "555-123-4567" not in anonymized

    def test_access_control_performance(self):
        """Test access control performance with many rules."""
        import time

        # Create many access rules
        for i in range(1000):
            self.privacy_engine.access_rules.append(
                AccessRule(
                    subject=f"user_{i}",
                    resource=f"resource_{i}",
                    permissions=["read"],
                    conditions={},
                )
            )

        # Test access checking performance
        start_time = time.time()

        for i in range(100):
            result = self.privacy_engine.check_access(
                f"user_{i}", f"resource_{i}", "read"
            )
            assert result is True

        end_time = time.time()
        duration = end_time - start_time

        # Should handle many rules efficiently
        assert duration < 1.0  # Should complete in under 1 second


if __name__ == "__main__":
    pytest.main([__file__])
