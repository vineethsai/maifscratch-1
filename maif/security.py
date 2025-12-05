"""
Security and cryptographic functionality for MAIF.

This module provides comprehensive security features including:
- Ed25519 digital signatures (fast and secure)
- Provenance tracking
- Signature verification
- Audit logging for security events

Note: MAIF v3 files have built-in Ed25519 signing via MAIFEncoder.
This module provides standalone signing utilities and compatibility.
"""

import hashlib
import json
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import uuid
import os
import base64

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceEntry:
    """
    Represents a single provenance entry in an immutable chain.

    Each entry contains cryptographic links to previous entries,
    creating a tamper-evident chain of custody and operations.
    """

    timestamp: float
    agent_id: str
    action: str
    block_hash: str
    signature: str = ""
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    agent_did: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    verification_status: str = "unverified"
    chain_position: int = 0

    def __post_init__(self):
        """Calculate entry hash if not provided."""
        if self.entry_hash is None:
            self.calculate_entry_hash()

    def calculate_entry_hash(self) -> str:
        """Calculate cryptographic hash of this entry."""
        try:
            hash_dict = {
                "timestamp": self.timestamp,
                "agent_id": self.agent_id,
                "agent_did": self.agent_did,
                "action": self.action,
                "block_hash": self.block_hash,
                "previous_hash": self.previous_hash,
                "chain_position": self.chain_position,
                "metadata": self.metadata,
            }

            canonical_json = json.dumps(hash_dict, sort_keys=True).encode()
            self.entry_hash = hashlib.sha256(canonical_json).hexdigest()
            return self.entry_hash
        except Exception as e:
            logger.error(f"Error calculating entry hash: {e}")
            raise ValueError(f"Failed to calculate entry hash: {e}")

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_did": self.agent_did,
            "action": self.action,
            "block_hash": self.block_hash,
            "signature": self.signature,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "chain_position": self.chain_position,
            "metadata": self.metadata,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProvenanceEntry":
        """Create from dictionary."""
        return cls(
            timestamp=data.get("timestamp", 0),
            agent_id=data.get("agent_id", ""),
            action=data.get("action", ""),
            block_hash=data.get("block_hash", ""),
            signature=data.get("signature", ""),
            previous_hash=data.get("previous_hash"),
            entry_hash=data.get("entry_hash"),
            agent_did=data.get("agent_did"),
            metadata=data.get("metadata", {}),
            verification_status=data.get("verification_status", "unverified"),
            chain_position=data.get("chain_position", 0),
        )


class MAIFSigner:
    """
    Handles digital signing and provenance for MAIF files using Ed25519.

    Ed25519 provides:
    - Fast key generation (~0.1ms vs ~50-100ms for RSA-2048)
    - Fast signing (~0.05ms vs ~1-5ms for RSA-2048)
    - Small signatures (64 bytes vs 256 bytes for RSA-2048)
    - Same security level as RSA-3072

    Note: MAIFEncoder already includes Ed25519 signing. This class is for
    standalone signing needs or manual provenance management.

    Usage:
        signer = MAIFSigner(agent_id="my-agent")
        signature = signer.sign_data(b"data to sign")
        is_valid = signer.verify_signature(b"data to sign", signature)
    """

    def __init__(
        self,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        agent_id: Optional[str] = None,
    ):
        """
        Initialize signer with Ed25519 key.

        Args:
            private_key: Optional Ed25519 private key. Generated if not provided.
            agent_id: Optional agent identifier. Generated if not provided.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_did = f"did:maif:{self.agent_id}"
        self.provenance_chain: List[ProvenanceEntry] = []
        self.chain_root_hash: Optional[str] = None
        self._lock = threading.RLock()

        # Generate or use provided Ed25519 key
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = ed25519.Ed25519PrivateKey.generate()

        self.public_key = self.private_key.public_key()

        # Initialize the chain with a genesis entry
        self._create_genesis_entry()

    def _create_genesis_entry(self):
        """Create the genesis entry for the provenance chain."""
        genesis = ProvenanceEntry(
            timestamp=time.time(),
            agent_id=self.agent_id,
            agent_did=self.agent_did,
            action="genesis",
            block_hash=hashlib.sha256(f"genesis:{self.agent_id}".encode()).hexdigest(),
            chain_position=0,
            metadata={"key_algorithm": "Ed25519", "created": time.time()},
        )
        genesis.calculate_entry_hash()
        genesis.signature = self.sign_data(genesis.entry_hash.encode()).hex()
        genesis.verification_status = "verified"

        self.provenance_chain.append(genesis)
        self.chain_root_hash = genesis.entry_hash

    def get_public_key_bytes(self) -> bytes:
        """Get the raw Ed25519 public key (32 bytes)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )

    def get_public_key_hex(self) -> str:
        """Get the public key as hex string."""
        return self.get_public_key_bytes().hex()

    def sign_data(self, data: bytes) -> bytes:
        """
        Sign data with Ed25519.

        Args:
            data: Data to sign

        Returns:
            64-byte Ed25519 signature
        """
        if not data:
            raise ValueError("Cannot sign empty data")
        return self.private_key.sign(data)

    def sign_data_base64(self, data: bytes) -> str:
        """Sign data and return base64-encoded signature."""
        return base64.b64encode(self.sign_data(data)).decode("ascii")

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """
        Verify an Ed25519 signature.

        Args:
            data: Original data that was signed
            signature: 64-byte Ed25519 signature

        Returns:
            True if signature is valid
        """
        try:
            self.public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False

    def verify_signature_base64(self, data: bytes, signature_b64: str) -> bool:
        """Verify a base64-encoded signature."""
        try:
            signature = base64.b64decode(signature_b64)
            return self.verify_signature(data, signature)
        except Exception:
            return False

    def add_provenance_entry(
        self, action: str, block_hash: str, metadata: Optional[Dict] = None
    ) -> ProvenanceEntry:
        """
        Add a new entry to the provenance chain.

        Args:
            action: Action being recorded (e.g., "add_block", "update")
            block_hash: Hash of the block/data being operated on
            metadata: Optional additional metadata

        Returns:
            The created ProvenanceEntry
        """
        with self._lock:
            previous_entry = (
                self.provenance_chain[-1] if self.provenance_chain else None
            )

            entry = ProvenanceEntry(
                timestamp=time.time(),
                agent_id=self.agent_id,
                agent_did=self.agent_did,
                action=action,
                block_hash=block_hash,
                previous_hash=previous_entry.entry_hash if previous_entry else None,
                chain_position=len(self.provenance_chain),
                metadata=metadata or {},
            )

            entry.calculate_entry_hash()
            entry.signature = self.sign_data(entry.entry_hash.encode()).hex()
            entry.verification_status = "verified"

            self.provenance_chain.append(entry)
            return entry

    def sign_manifest(self, manifest: Dict) -> Dict:
        """
        Sign a manifest dictionary.

        Args:
            manifest: The manifest to sign

        Returns:
            Manifest with signature and public key added
        """
        # Create canonical representation
        manifest_copy = {
            k: v
            for k, v in manifest.items()
            if k not in ("signature", "public_key", "signature_metadata")
        }
        canonical = json.dumps(manifest_copy, sort_keys=True).encode()

        # Sign
        signature = self.sign_data(canonical)

        # Add signature info
        manifest["signature"] = signature.hex()
        manifest["public_key"] = self.get_public_key_hex()
        manifest["signature_metadata"] = {
            "signer_id": self.agent_id,
            "signer_did": self.agent_did,
            "timestamp": time.time(),
            "algorithm": "Ed25519",
            "provenance_chain": [e.to_dict() for e in self.provenance_chain],
        }

        return manifest

    def verify_manifest(self, manifest: Dict) -> Tuple[bool, str]:
        """
        Verify a signed manifest.

        Args:
            manifest: Manifest with signature

        Returns:
            (is_valid, message)
        """
        if "signature" not in manifest or "public_key" not in manifest:
            return False, "Missing signature or public key"

        try:
            # Get signature and public key
            signature = bytes.fromhex(manifest["signature"])
            pub_key_bytes = bytes.fromhex(manifest["public_key"])

            # Load public key
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_key_bytes)

            # Recreate canonical form
            manifest_copy = {
                k: v
                for k, v in manifest.items()
                if k not in ("signature", "public_key", "signature_metadata")
            }
            canonical = json.dumps(manifest_copy, sort_keys=True).encode()

            # Verify
            public_key.verify(signature, canonical)
            return True, "Signature valid"

        except InvalidSignature:
            return False, "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {e}"

    def get_provenance_chain(self) -> List[Dict]:
        """Get the provenance chain as a list of dictionaries."""
        return [entry.to_dict() for entry in self.provenance_chain]

    def export_key_pair(self) -> Dict[str, str]:
        """
        Export the key pair for storage.

        Returns:
            Dict with 'private_key' and 'public_key' as hex strings
        """
        private_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return {
            "private_key": private_bytes.hex(),
            "public_key": self.get_public_key_hex(),
            "algorithm": "Ed25519",
        }

    @classmethod
    def from_private_key_hex(
        cls, private_key_hex: str, agent_id: Optional[str] = None
    ) -> "MAIFSigner":
        """
        Create a signer from a hex-encoded private key.

        Args:
            private_key_hex: 32-byte Ed25519 private key as hex
            agent_id: Optional agent ID

        Returns:
            MAIFSigner instance
        """
        private_bytes = bytes.fromhex(private_key_hex)
        private_key = ed25519.Ed25519PrivateKey.from_private_bytes(private_bytes)
        return cls(private_key=private_key, agent_id=agent_id)


class MAIFVerifier:
    """
    Verifies Ed25519 signatures without access to private key.

    Usage:
        verifier = MAIFVerifier(public_key_hex)
        is_valid = verifier.verify(data, signature)
    """

    def __init__(self, public_key: Union[str, bytes, ed25519.Ed25519PublicKey]):
        """
        Initialize verifier with public key.

        Args:
            public_key: Ed25519 public key as hex string, bytes, or key object
        """
        if isinstance(public_key, ed25519.Ed25519PublicKey):
            self.public_key = public_key
        elif isinstance(public_key, str):
            pub_bytes = bytes.fromhex(public_key)
            self.public_key = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        elif isinstance(public_key, bytes):
            self.public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        else:
            raise ValueError("Invalid public key format")

    def verify(self, data: bytes, signature: bytes) -> bool:
        """
        Verify a signature.

        Args:
            data: Original data
            signature: 64-byte Ed25519 signature

        Returns:
            True if valid
        """
        try:
            self.public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False
        except Exception:
            return False

    def verify_hex(self, data: bytes, signature_hex: str) -> bool:
        """Verify with hex-encoded signature."""
        try:
            signature = bytes.fromhex(signature_hex)
            return self.verify(data, signature)
        except Exception:
            return False

    def verify_base64(self, data: bytes, signature_b64: str) -> bool:
        """Verify with base64-encoded signature."""
        try:
            signature = base64.b64decode(signature_b64)
            return self.verify(data, signature)
        except Exception:
            return False

    def verify_manifest(self, manifest: Dict) -> Tuple[bool, str]:
        """
        Verify a signed manifest.

        Args:
            manifest: Manifest dictionary with 'signature' field

        Returns:
            (is_valid, message)
        """
        if "signature" not in manifest:
            return False, "No signature in manifest"

        try:
            signature = bytes.fromhex(manifest["signature"])
            manifest_copy = {
                k: v
                for k, v in manifest.items()
                if k not in ("signature", "public_key", "signature_metadata")
            }
            canonical = json.dumps(manifest_copy, sort_keys=True).encode()

            if self.verify(canonical, signature):
                return True, "Signature valid"
            return False, "Invalid signature"
        except Exception as e:
            return False, f"Verification error: {e}"


# =============================================================================
# Utility Functions
# =============================================================================


def generate_key_pair() -> Tuple[ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey]:
    """Generate a new Ed25519 key pair."""
    private_key = ed25519.Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def sign_data(data: bytes, private_key: ed25519.Ed25519PrivateKey) -> bytes:
    """Sign data with Ed25519 private key."""
    return private_key.sign(data)


def verify_signature(
    data: bytes, signature: bytes, public_key: ed25519.Ed25519PublicKey
) -> bool:
    """Verify Ed25519 signature."""
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def hash_data(data: bytes) -> str:
    """Calculate SHA-256 hash of data."""
    return hashlib.sha256(data).hexdigest()


# =============================================================================
# Legacy Compatibility Classes
# =============================================================================


class AccessController:
    """Legacy access control - use PrivacyEngine for new code."""

    def __init__(self):
        self.access_rules = {}

    def grant_access(self, agent_id: str, resource: str, permissions: List[str]):
        self.access_rules[f"{agent_id}:{resource}"] = permissions

    def check_access(self, agent_id: str, resource: str, permission: str) -> bool:
        key = f"{agent_id}:{resource}"
        return permission in self.access_rules.get(key, [])

    def revoke_access(self, agent_id: str, resource: str):
        key = f"{agent_id}:{resource}"
        self.access_rules.pop(key, None)


class SecurityManager:
    """Legacy security manager - use MAIFSigner for new code."""

    def __init__(self, agent_id: str = None):
        self.signer = MAIFSigner(agent_id=agent_id) if agent_id else MAIFSigner()
        self.access_controller = AccessController()

    def sign_data(self, data: bytes) -> bytes:
        return self.signer.sign_data(data)

    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        return self.signer.verify_signature(data, signature)

    def get_public_key(self) -> str:
        return self.signer.get_public_key_hex()


__all__ = [
    "MAIFSigner",
    "MAIFVerifier",
    "ProvenanceEntry",
    "AccessController",
    "SecurityManager",
    "generate_key_pair",
    "sign_data",
    "verify_signature",
    "hash_data",
]
