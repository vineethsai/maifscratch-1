"""
Comprehensive tests for MAIF security functionality (Ed25519).
"""

import pytest
import tempfile
import os
import json
import time
from unittest.mock import Mock, patch, MagicMock
from cryptography.hazmat.primitives.asymmetric import ed25519

from maif.security import (
    MAIFSigner,
    MAIFVerifier,
    ProvenanceEntry,
    AccessController,
    SecurityManager,
    generate_key_pair,
    sign_data,
    verify_signature,
    hash_data,
)


class TestProvenanceEntry:
    """Test ProvenanceEntry data structure."""

    def test_provenance_entry_creation(self):
        """Test basic ProvenanceEntry creation."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="abc123def456",
            previous_hash="def456ghi789",
        )

        assert entry.timestamp == 1234567890.0
        assert entry.agent_id == "test_agent"
        assert entry.action == "create_block"
        assert entry.block_hash == "abc123def456"
        assert entry.previous_hash == "def456ghi789"

    def test_provenance_entry_to_dict(self):
        """Test ProvenanceEntry serialization."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="abc123def456",
            previous_hash="def456ghi789",
        )

        entry_dict = entry.to_dict()

        assert entry_dict["timestamp"] == 1234567890.0
        assert entry_dict["agent_id"] == "test_agent"
        assert entry_dict["action"] == "create_block"
        assert entry_dict["block_hash"] == "abc123def456"
        assert entry_dict["previous_hash"] == "def456ghi789"

    def test_provenance_entry_hash_calculation(self):
        """Test entry hash calculation."""
        entry = ProvenanceEntry(
            timestamp=1234567890.0,
            agent_id="test_agent",
            action="create_block",
            block_hash="abc123",
        )

        hash1 = entry.calculate_entry_hash()
        assert hash1 is not None
        assert len(hash1) == 64  # SHA-256 hex

        # Hash should be deterministic
        hash2 = entry.calculate_entry_hash()
        assert hash1 == hash2

    def test_provenance_entry_from_dict(self):
        """Test ProvenanceEntry deserialization."""
        data = {
            "timestamp": 1234567890.0,
            "agent_id": "test_agent",
            "action": "create_block",
            "block_hash": "abc123",
        }

        entry = ProvenanceEntry.from_dict(data)
        assert entry.timestamp == 1234567890.0
        assert entry.agent_id == "test_agent"


class TestMAIFSigner:
    """Test MAIFSigner functionality with Ed25519."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.signer = MAIFSigner(agent_id="test_agent")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_signer_initialization(self):
        """Test MAIFSigner initialization."""
        assert self.signer.agent_id == "test_agent"
        assert self.signer.private_key is not None
        # Should have genesis block
        assert len(self.signer.provenance_chain) == 1
        assert self.signer.provenance_chain[0].action == "genesis"

    def test_signer_with_existing_key(self):
        """Test MAIFSigner with existing Ed25519 private key."""
        private_key = ed25519.Ed25519PrivateKey.generate()
        signer = MAIFSigner(private_key=private_key, agent_id="test")

        assert signer.private_key == private_key
        assert signer.public_key is not None

    def test_get_public_key_bytes(self):
        """Test getting public key as bytes."""
        pub_key = self.signer.get_public_key_bytes()
        assert isinstance(pub_key, bytes)
        assert len(pub_key) == 32  # Ed25519 public key is 32 bytes

    def test_get_public_key_hex(self):
        """Test getting public key as hex string."""
        pub_key_hex = self.signer.get_public_key_hex()
        assert isinstance(pub_key_hex, str)
        assert len(pub_key_hex) == 64  # 32 bytes = 64 hex chars

    def test_sign_data(self):
        """Test data signing."""
        data = b"Hello, MAIF!"
        signature = self.signer.sign_data(data)

        assert signature is not None
        assert isinstance(signature, bytes)
        assert len(signature) == 64  # Ed25519 signature is 64 bytes

    def test_sign_data_base64(self):
        """Test data signing with base64 output."""
        data = b"Hello, MAIF!"
        signature = self.signer.sign_data_base64(data)

        assert isinstance(signature, str)
        import base64

        decoded = base64.b64decode(signature)
        assert len(decoded) == 64

    def test_verify_signature(self):
        """Test signature verification."""
        data = b"Test data"
        signature = self.signer.sign_data(data)

        is_valid = self.signer.verify_signature(data, signature)
        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test invalid signature detection."""
        data = b"Test data"
        signature = self.signer.sign_data(data)

        # Tamper with signature
        tampered = bytearray(signature)
        tampered[0] ^= 0xFF

        is_valid = self.signer.verify_signature(data, bytes(tampered))
        assert is_valid is False

    def test_add_provenance_entry(self):
        """Test adding provenance entries."""
        initial_count = len(self.signer.provenance_chain)

        entry = self.signer.add_provenance_entry(
            action="test_action", block_hash="abc123"
        )

        assert len(self.signer.provenance_chain) == initial_count + 1
        assert entry.action == "test_action"
        assert entry.agent_id == "test_agent"

    def test_provenance_chain_linking(self):
        """Test that provenance entries are properly linked."""
        self.signer.add_provenance_entry("action1", "hash1")
        self.signer.add_provenance_entry("action2", "hash2")

        chain = self.signer.provenance_chain
        assert len(chain) >= 3  # genesis + 2 entries

        # Check linking
        for i in range(1, len(chain)):
            assert chain[i].previous_hash == chain[i - 1].entry_hash

    def test_sign_manifest(self):
        """Test manifest signing."""
        manifest = {"version": "1.0", "blocks": ["block1", "block2"]}

        signed = self.signer.sign_manifest(manifest)

        assert "signature" in signed
        assert "public_key" in signed
        assert "signature_metadata" in signed

    def test_verify_manifest(self):
        """Test manifest verification."""
        manifest = {"test": "data"}
        signed = self.signer.sign_manifest(manifest)

        is_valid, msg = self.signer.verify_manifest(signed)
        assert is_valid is True

    def test_export_key_pair(self):
        """Test key pair export."""
        exported = self.signer.export_key_pair()

        assert "private_key" in exported
        assert "public_key" in exported
        assert exported["algorithm"] == "Ed25519"

    def test_from_private_key_hex(self):
        """Test creating signer from hex key."""
        exported = self.signer.export_key_pair()

        restored = MAIFSigner.from_private_key_hex(
            exported["private_key"], agent_id="restored"
        )

        # Signatures should match
        data = b"test"
        sig1 = self.signer.sign_data(data)
        sig2 = restored.sign_data(data)
        assert sig1 == sig2


class TestMAIFVerifier:
    """Test MAIFVerifier functionality."""

    def test_verifier_from_hex(self):
        """Test creating verifier from hex public key."""
        signer = MAIFSigner(agent_id="test")
        pub_key_hex = signer.get_public_key_hex()

        verifier = MAIFVerifier(pub_key_hex)

        data = b"test data"
        sig = signer.sign_data(data)

        assert verifier.verify(data, sig) is True

    def test_verifier_from_bytes(self):
        """Test creating verifier from bytes public key."""
        signer = MAIFSigner(agent_id="test")
        pub_key_bytes = signer.get_public_key_bytes()

        verifier = MAIFVerifier(pub_key_bytes)

        data = b"test data"
        sig = signer.sign_data(data)

        assert verifier.verify(data, sig) is True

    def test_verifier_detect_tampering(self):
        """Test that verifier detects tampered data."""
        signer = MAIFSigner(agent_id="test")
        verifier = MAIFVerifier(signer.get_public_key_hex())

        data = b"original data"
        sig = signer.sign_data(data)

        tampered_data = b"tampered data"
        assert verifier.verify(tampered_data, sig) is False

    def test_verify_manifest(self):
        """Test manifest verification."""
        signer = MAIFSigner(agent_id="test")
        verifier = MAIFVerifier(signer.get_public_key_hex())

        manifest = {"data": "test"}
        signed = signer.sign_manifest(manifest)

        is_valid, msg = verifier.verify_manifest(signed)
        assert is_valid is True


class TestAccessController:
    """Test AccessController functionality."""

    def test_grant_access(self):
        """Test granting access."""
        controller = AccessController()
        controller.grant_access("agent1", "resource1", ["read", "write"])

        assert controller.check_access("agent1", "resource1", "read") is True
        assert controller.check_access("agent1", "resource1", "write") is True

    def test_check_permission_denied(self):
        """Test permission denial."""
        controller = AccessController()
        controller.grant_access("agent1", "resource1", ["read"])

        assert controller.check_access("agent1", "resource1", "write") is False
        assert controller.check_access("agent2", "resource1", "read") is False

    def test_revoke_access(self):
        """Test revoking access."""
        controller = AccessController()
        controller.grant_access("agent1", "resource1", ["read"])
        controller.revoke_access("agent1", "resource1")

        assert controller.check_access("agent1", "resource1", "read") is False


class TestSecurityManager:
    """Test SecurityManager legacy compatibility."""

    def test_initialization(self):
        """Test SecurityManager initialization."""
        manager = SecurityManager(agent_id="test")
        assert manager.signer is not None
        assert manager.access_controller is not None

    def test_sign_and_verify(self):
        """Test signing and verification."""
        manager = SecurityManager(agent_id="test")
        data = b"test data"

        sig = manager.sign_data(data)
        assert manager.verify_signature(data, sig) is True

    def test_get_public_key(self):
        """Test getting public key."""
        manager = SecurityManager(agent_id="test")
        pub_key = manager.get_public_key()

        assert isinstance(pub_key, str)
        assert len(pub_key) == 64  # Hex string


class TestUtilityFunctions:
    """Test utility functions."""

    def test_generate_key_pair(self):
        """Test key pair generation."""
        private, public = generate_key_pair()

        assert private is not None
        assert public is not None

    def test_sign_data_function(self):
        """Test sign_data function."""
        private, _ = generate_key_pair()
        data = b"test"

        sig = sign_data(data, private)
        assert len(sig) == 64

    def test_verify_signature_function(self):
        """Test verify_signature function."""
        private, public = generate_key_pair()
        data = b"test"
        sig = sign_data(data, private)

        assert verify_signature(data, sig, public) is True

    def test_hash_data_function(self):
        """Test hash_data function."""
        data = b"test"
        hash_hex = hash_data(data)

        assert len(hash_hex) == 64  # SHA-256 hex


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
