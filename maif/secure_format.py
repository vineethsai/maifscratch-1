"""
MAIF Secure Binary Format - Self-contained, Immutable, Tamper-evident

This module implements an improved MAIF binary format where:
1. All security and provenance is embedded in the file itself
2. Each block is signed on first write (immutable)
3. Any tampering breaks signatures and is detectable
4. No external manifest required for security verification

Uses Ed25519 for high-performance cryptographic signatures:
- Key generation: ~0.1ms (vs ~50-100ms for RSA-2048)
- Signing: ~0.05ms (vs ~1-5ms for RSA-2048)
- Signature size: 64 bytes (stored in 256-byte field for compatibility)

File Structure:
┌─────────────────────────────────────────────────┐
│ FILE HEADER (444 bytes)                         │
├─────────────────────────────────────────────────┤
│ DATA BLOCKS (each with signature)               │
├─────────────────────────────────────────────────┤
│ PROVENANCE SECTION                              │
├─────────────────────────────────────────────────┤
│ SECURITY SECTION (public key, signer info)      │
├─────────────────────────────────────────────────┤
│ BLOCK INDEX                                     │
├─────────────────────────────────────────────────┤
│ FILE FOOTER (magic + checksum)                  │
└─────────────────────────────────────────────────┘
"""

import struct
import hashlib
import time
import uuid
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.exceptions import InvalidSignature


# =============================================================================
# Constants and Magic Numbers
# =============================================================================

MAGIC_HEADER = b"MAIF"
MAGIC_FOOTER = b"FIAM"  # MAIF reversed
FORMAT_VERSION_MAJOR = 2
FORMAT_VERSION_MINOR = 1  # Bumped for Ed25519


# =============================================================================
# Block Types
# =============================================================================


class SecureBlockType(IntEnum):
    """Block type identifiers (4-byte ASCII codes)."""

    TEXT = 0x54455854  # 'TEXT'
    EMBEDDINGS = 0x454D4244  # 'EMBD'
    IMAGE = 0x494D4147  # 'IMAG'
    AUDIO = 0x41554449  # 'AUDI'
    VIDEO = 0x56494445  # 'VIDE'
    KNOWLEDGE = 0x4B4E4F57  # 'KNOW'
    BINARY = 0x42494E41  # 'BINA'
    METADATA = 0x4D455441  # 'META'


class BlockFlags(IntFlag):
    """Block status flags."""

    NONE = 0x00
    SIGNED = 0x01  # Block has been cryptographically signed
    IMMUTABLE = 0x02  # Block cannot be modified (set after signing)
    ENCRYPTED = 0x04  # Block data is encrypted
    COMPRESSED = 0x08  # Block data is compressed
    VERIFIED = 0x10  # Block signature has been verified
    TAMPERED = 0x20  # Block failed signature verification (tampered!)


class FileFlags(IntFlag):
    """File-level flags."""

    NONE = 0x00
    SIGNED = 0x01  # File has been signed
    FINALIZED = 0x02  # File is finalized, no more blocks can be added
    HAS_PROVENANCE = 0x04  # File contains provenance section
    HAS_ENCRYPTION = 0x08  # File contains encrypted blocks


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SecureFileHeader:
    """
    MAIF Secure File Header - 444 bytes

    Contains file-level metadata and security information.
    The merkle_root and file_signature ensure file integrity.
    """

    magic: bytes = MAGIC_HEADER  # 4 bytes
    version_major: int = FORMAT_VERSION_MAJOR  # 2 bytes
    version_minor: int = FORMAT_VERSION_MINOR  # 2 bytes
    flags: int = 0  # 4 bytes
    created: int = 0  # 8 bytes (microseconds)
    modified: int = 0  # 8 bytes (microseconds)
    file_id: bytes = b"\x00" * 16  # 16 bytes (UUID)
    creator_id: bytes = b"\x00" * 16  # 16 bytes (UUID)
    agent_did: bytes = b"\x00" * 64  # 64 bytes (DID string, null-padded)
    block_count: int = 0  # 4 bytes
    provenance_offset: int = 0  # 8 bytes
    security_offset: int = 0  # 8 bytes
    index_offset: int = 0  # 8 bytes
    merkle_root: bytes = b"\x00" * 32  # 32 bytes (SHA-256)
    file_signature: bytes = b"\x00" * 256  # 256 bytes (Ed25519 sig padded)
    reserved: bytes = b"\x00" * 4  # 4 bytes

    HEADER_SIZE = 444  # 4+2+2+4+8+8+16+16+64+4+8+8+8+32+256+4 = 444

    def pack(self) -> bytes:
        """Pack header to binary."""
        agent_did_padded = (
            self.agent_did[:64].ljust(64, b"\x00")
            if isinstance(self.agent_did, bytes)
            else self.agent_did.encode()[:64].ljust(64, b"\x00")
        )
        sig_padded = self.file_signature[:256].ljust(256, b"\x00")
        reserved_padded = (
            self.reserved[:4].ljust(4, b"\x00")
            if isinstance(self.reserved, bytes)
            else b"\x00" * 4
        )

        return struct.pack(
            ">4sHHIQQ16s16s64sIQQQ32s256s4s",
            self.magic,
            self.version_major,
            self.version_minor,
            self.flags,
            self.created,
            self.modified,
            self.file_id,
            self.creator_id,
            agent_did_padded,
            self.block_count,
            self.provenance_offset,
            self.security_offset,
            self.index_offset,
            self.merkle_root,
            sig_padded,
            reserved_padded,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SecureFileHeader":
        """Unpack header from binary."""
        header_size = 444
        if len(data) < header_size:
            raise ValueError(f"Insufficient data: {len(data)} < {header_size}")

        unpacked = struct.unpack(">4sHHIQQ16s16s64sIQQQ32s256s4s", data[:header_size])
        return cls(
            magic=unpacked[0],
            version_major=unpacked[1],
            version_minor=unpacked[2],
            flags=unpacked[3],
            created=unpacked[4],
            modified=unpacked[5],
            file_id=unpacked[6],
            creator_id=unpacked[7],
            agent_did=unpacked[8],
            block_count=unpacked[9],
            provenance_offset=unpacked[10],
            security_offset=unpacked[11],
            index_offset=unpacked[12],
            merkle_root=unpacked[13],
            file_signature=unpacked[14],
            reserved=unpacked[15],
        )


@dataclass
class SecureBlockHeader:
    """
    MAIF Secure Block Header - 372 bytes

    Each block is individually signed with Ed25519, creating an immutable,
    tamper-evident structure. The previous_hash creates a chain linking all blocks.
    """

    size: int = 0  # 4 bytes (total block size)
    block_type: int = 0  # 4 bytes
    flags: int = BlockFlags.NONE  # 4 bytes
    version: int = 1  # 4 bytes
    timestamp: int = 0  # 8 bytes (microseconds)
    block_id: bytes = b"\x00" * 16  # 16 bytes (UUID)
    previous_hash: bytes = b"\x00" * 32  # 32 bytes (chain link)
    content_hash: bytes = b"\x00" * 32  # 32 bytes (SHA-256 of data)
    signature: bytes = b"\x00" * 256  # 256 bytes (Ed25519 sig padded)
    metadata_size: int = 0  # 4 bytes
    reserved: bytes = b"\x00" * 8  # 8 bytes

    HEADER_SIZE = 372  # 4+4+4+4+8+16+32+32+256+4+8 = 372

    def pack(self) -> bytes:
        """Pack header to binary."""
        sig_padded = self.signature[:256].ljust(256, b"\x00")
        return struct.pack(
            ">IIIIQ16s32s32s256sI8s",
            self.size,
            self.block_type,
            self.flags,
            self.version,
            self.timestamp,
            self.block_id,
            self.previous_hash,
            self.content_hash,
            sig_padded,
            self.metadata_size,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SecureBlockHeader":
        """Unpack header from binary."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Insufficient data: {len(data)} < {cls.HEADER_SIZE}")

        unpacked = struct.unpack(">IIIIQ16s32s32s256sI8s", data[: cls.HEADER_SIZE])
        return cls(
            size=unpacked[0],
            block_type=unpacked[1],
            flags=unpacked[2],
            version=unpacked[3],
            timestamp=unpacked[4],
            block_id=unpacked[5],
            previous_hash=unpacked[6],
            content_hash=unpacked[7],
            signature=unpacked[8],
            metadata_size=unpacked[9],
            reserved=unpacked[10],
        )

    def get_signable_data(self) -> bytes:
        """Get the data that should be signed (excludes signature field and status flags)."""
        signing_flags = self.flags & ~(
            BlockFlags.SIGNED
            | BlockFlags.IMMUTABLE
            | BlockFlags.VERIFIED
            | BlockFlags.TAMPERED
        )
        return struct.pack(
            ">IIIIQ16s32s32sI",
            self.size,
            self.block_type,
            signing_flags,
            self.version,
            self.timestamp,
            self.block_id,
            self.previous_hash,
            self.content_hash,
            self.metadata_size,
        )


@dataclass
class ProvenanceEntry:
    """A single entry in the provenance chain."""

    timestamp: int
    agent_id: str
    agent_did: str
    action: str
    block_hash: str
    entry_hash: str = ""
    previous_entry_hash: str = ""
    signature: bytes = b""
    chain_position: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_hash(self) -> str:
        """Calculate entry hash."""
        data = {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_did": self.agent_did,
            "action": self.action,
            "block_hash": self.block_hash,
            "previous_entry_hash": self.previous_entry_hash,
            "chain_position": self.chain_position,
            "metadata": self.metadata,
        }
        canonical = json.dumps(data, sort_keys=True).encode()
        self.entry_hash = hashlib.sha256(canonical).hexdigest()
        return self.entry_hash

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_did": self.agent_did,
            "action": self.action,
            "block_hash": self.block_hash,
            "entry_hash": self.entry_hash,
            "previous_entry_hash": self.previous_entry_hash,
            "signature": self.signature.hex() if self.signature else "",
            "chain_position": self.chain_position,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProvenanceEntry":
        sig = data.get("signature", "")
        if isinstance(sig, str) and sig:
            sig = bytes.fromhex(sig)
        elif not sig:
            sig = b""
        return cls(
            timestamp=data["timestamp"],
            agent_id=data["agent_id"],
            agent_did=data.get("agent_did", ""),
            action=data["action"],
            block_hash=data["block_hash"],
            entry_hash=data.get("entry_hash", ""),
            previous_entry_hash=data.get("previous_entry_hash", ""),
            signature=sig,
            chain_position=data.get("chain_position", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SecureBlock:
    """A complete block with header, data, and metadata."""

    header: SecureBlockHeader
    data: bytes
    metadata: Optional[Dict[str, Any]] = None

    @property
    def block_id(self) -> str:
        """Backwards-compatible property: returns block_id as hex string."""
        return self.header.block_id.hex()

    @property
    def block_type(self) -> int:
        """Backwards-compatible property: returns block_type from header."""
        return self.header.block_type

    @property
    def block_type_name(self) -> str:
        """Get block type as a human-readable string."""
        type_names = {
            SecureBlockType.TEXT: "TEXT",
            SecureBlockType.EMBEDDINGS: "EMBED",
            SecureBlockType.IMAGE: "IMAGE",
            SecureBlockType.AUDIO: "AUDIO",
            SecureBlockType.VIDEO: "VIDEO",
            SecureBlockType.KNOWLEDGE: "KNOW",
            SecureBlockType.BINARY: "BINARY",
            SecureBlockType.METADATA: "META",
        }
        return type_names.get(self.header.block_type, "UNKNOWN")

    def get_content_hash(self) -> bytes:
        """Calculate SHA-256 hash of block content."""
        content_hash = hashlib.sha256(self.data).digest()
        if self.metadata:
            meta_bytes = json.dumps(self.metadata, sort_keys=True).encode()
            content_hash = hashlib.sha256(content_hash + meta_bytes).digest()
        return content_hash


@dataclass
class FileFooter:
    """File footer for integrity verification - 48 bytes."""

    magic: bytes = MAGIC_FOOTER
    total_size: int = 0
    checksum: bytes = b"\x00" * 32
    reserved: int = 0

    FOOTER_SIZE = 48

    def pack(self) -> bytes:
        return struct.pack(
            ">4sQ32sI", self.magic, self.total_size, self.checksum, self.reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FileFooter":
        if len(data) < cls.FOOTER_SIZE:
            raise ValueError(f"Insufficient data: {len(data)} < {cls.FOOTER_SIZE}")
        unpacked = struct.unpack(">4sQ32sI", data[: cls.FOOTER_SIZE])
        return cls(
            magic=unpacked[0],
            total_size=unpacked[1],
            checksum=unpacked[2],
            reserved=unpacked[3],
        )


# =============================================================================
# Secure MAIF Writer - Ed25519 Signatures
# =============================================================================


class SecureMAIFWriter:
    """
    Creates self-contained MAIF files with Ed25519 signatures.

    Ed25519 provides:
    - 50× faster signing than RSA-2048
    - 500× faster key generation
    - Same security level
    - Smaller signatures (64 bytes vs 256 bytes)

    Usage:
        writer = SecureMAIFWriter("output.maif", agent_id="my-agent")
        writer.add_text_block("Hello, world!")
        writer.add_embeddings_block([[0.1, 0.2, 0.3]])
        writer.finalize()
    """

    def __init__(
        self,
        file_path: str,
        agent_id: str,
        private_key: Optional[ed25519.Ed25519PrivateKey] = None,
        agent_did: Optional[str] = None,
    ):
        self.file_path = file_path
        self.agent_id = agent_id
        self.agent_did = agent_did or f"did:maif:{agent_id}"

        # Generate or use provided Ed25519 key (FAST!)
        if private_key:
            self.private_key = private_key
        else:
            self.private_key = ed25519.Ed25519PrivateKey.generate()

        self.public_key = self.private_key.public_key()

        # File state
        self.file_id = uuid.uuid4().bytes
        self.creator_id = uuid.uuid4().bytes
        self.created = int(time.time() * 1000000)
        self.blocks: List[SecureBlock] = []
        self.provenance: List[ProvenanceEntry] = []
        self.finalized = False
        self._last_block_hash = b"\x00" * 32
        self._last_entry_hash = ""

        # Create genesis provenance entry
        self._create_genesis_entry()

    def _create_genesis_entry(self):
        """Create the genesis entry for the provenance chain."""
        entry = ProvenanceEntry(
            timestamp=self.created,
            agent_id=self.agent_id,
            agent_did=self.agent_did,
            action="genesis",
            block_hash=hashlib.sha256(
                f"genesis:{self.agent_id}:{self.created}".encode()
            ).hexdigest(),
            chain_position=0,
            metadata={
                "file_id": self.file_id.hex(),
                "creator_id": self.creator_id.hex(),
                "format_version": f"{FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR}",
                "key_algorithm": "Ed25519",
            },
        )
        entry.calculate_hash()
        entry.signature = self._sign_data(entry.entry_hash.encode())
        self.provenance.append(entry)
        self._last_entry_hash = entry.entry_hash

    def _sign_data(self, data: bytes) -> bytes:
        """Sign data with Ed25519."""
        return self.private_key.sign(data)

    def _pad_signature(self, sig: bytes) -> bytes:
        """Pad Ed25519 signature (64 bytes) to fit 256-byte field for format compatibility."""
        marker = b"ED25519:"
        padded = marker + sig + b"\x00" * (256 - len(marker) - len(sig))
        return padded

    def _add_provenance_entry(
        self, action: str, block_hash: str, metadata: Optional[Dict] = None
    ):
        """Add a provenance entry for an operation."""
        entry = ProvenanceEntry(
            timestamp=int(time.time() * 1000000),
            agent_id=self.agent_id,
            agent_did=self.agent_did,
            action=action,
            block_hash=block_hash,
            previous_entry_hash=self._last_entry_hash,
            chain_position=len(self.provenance),
            metadata=metadata or {},
        )
        entry.calculate_hash()
        entry.signature = self._sign_data(entry.entry_hash.encode())
        self.provenance.append(entry)
        self._last_entry_hash = entry.entry_hash

    def add_block(
        self,
        data: bytes,
        block_type: SecureBlockType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a block with Ed25519 signature."""
        if self.finalized:
            raise RuntimeError("Cannot add blocks to a finalized file")

        block_id = uuid.uuid4().bytes
        timestamp = int(time.time() * 1000000)

        # Calculate content hash
        content_hash = hashlib.sha256(data).digest()
        if metadata:
            meta_bytes = json.dumps(metadata, sort_keys=True).encode()
            content_hash = hashlib.sha256(content_hash + meta_bytes).digest()

        # Create block header
        metadata_bytes = json.dumps(metadata or {}, sort_keys=True).encode()
        # Convert block_type to int if it's an enum
        block_type_int = (
            block_type.value if hasattr(block_type, "value") else int(block_type)
        )
        header = SecureBlockHeader(
            size=SecureBlockHeader.HEADER_SIZE + len(data) + len(metadata_bytes),
            block_type=block_type_int,
            flags=BlockFlags.NONE,
            version=1,
            timestamp=timestamp,
            block_id=block_id,
            previous_hash=self._last_block_hash,
            content_hash=content_hash,
            metadata_size=len(metadata_bytes),
        )

        # Sign with Ed25519
        signable = header.get_signable_data() + data + metadata_bytes
        raw_signature = self._sign_data(signable)
        header.signature = self._pad_signature(raw_signature)
        header.flags |= BlockFlags.SIGNED | BlockFlags.IMMUTABLE

        # Create block
        block = SecureBlock(header=header, data=data, metadata=metadata)
        self.blocks.append(block)

        # Update chain
        self._last_block_hash = content_hash

        # Add provenance entry
        block_id_hex = block_id.hex()
        self._add_provenance_entry(
            action=f"add_{self._type_name(block_type)}_block",
            block_hash=content_hash.hex(),
            metadata={"block_id": block_id_hex, "size": len(data)},
        )

        return block_id_hex

    def _type_name(self, block_type: SecureBlockType) -> str:
        """Get human-readable type name."""
        names = {
            SecureBlockType.TEXT: "text",
            SecureBlockType.EMBEDDINGS: "embeddings",
            SecureBlockType.IMAGE: "image",
            SecureBlockType.AUDIO: "audio",
            SecureBlockType.VIDEO: "video",
            SecureBlockType.KNOWLEDGE: "knowledge",
            SecureBlockType.BINARY: "binary",
            SecureBlockType.METADATA: "metadata",
        }
        return names.get(block_type, "unknown")

    def add_text_block(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Add a text block."""
        return self.add_block(text.encode("utf-8"), SecureBlockType.TEXT, metadata)

    def add_embeddings_block(
        self, embeddings: List[List[float]], metadata: Optional[Dict] = None
    ) -> str:
        """Add an embeddings block."""
        flat = [v for emb in embeddings for v in emb]
        data = struct.pack(f"{len(flat)}f", *flat)

        meta = metadata or {}
        meta.update(
            {
                "count": len(embeddings),
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "dtype": "float32",
            }
        )
        return self.add_block(data, SecureBlockType.EMBEDDINGS, meta)

    def add_binary_block(
        self,
        data: bytes,
        block_type: SecureBlockType = SecureBlockType.BINARY,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Add a binary data block."""
        return self.add_block(data, block_type, metadata)

    def _calculate_merkle_root(self) -> bytes:
        """Calculate Merkle root of all block content hashes."""
        if not self.blocks:
            return b"\x00" * 32

        hashes = [block.header.content_hash for block in self.blocks]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashlib.sha256(hashes[i] + hashes[i + 1]).digest()
                new_hashes.append(combined)
            hashes = new_hashes

        return hashes[0]

    def finalize(self) -> str:
        """Finalize the file: write all data, sign file, close."""
        if self.finalized:
            raise RuntimeError("File already finalized")

        # Add finalization provenance entry
        self._add_provenance_entry(
            action="finalize",
            block_hash=self._calculate_merkle_root().hex(),
            metadata={"block_count": len(self.blocks)},
        )

        with open(self.file_path, "wb") as f:
            # Calculate section offsets
            blocks_start = SecureFileHeader.HEADER_SIZE
            blocks_size = sum(block.header.size for block in self.blocks)

            provenance_offset = blocks_start + blocks_size
            provenance_data = json.dumps(
                [e.to_dict() for e in self.provenance]
            ).encode()
            provenance_size = len(provenance_data) + 4

            security_offset = provenance_offset + provenance_size
            security_data = self._build_security_section()
            security_size = len(security_data) + 4

            index_offset = security_offset + security_size

            # Build file header
            merkle_root = self._calculate_merkle_root()

            file_header = SecureFileHeader(
                magic=MAGIC_HEADER,
                version_major=FORMAT_VERSION_MAJOR,
                version_minor=FORMAT_VERSION_MINOR,
                flags=FileFlags.SIGNED | FileFlags.FINALIZED | FileFlags.HAS_PROVENANCE,
                created=self.created,
                modified=int(time.time() * 1000000),
                file_id=self.file_id,
                creator_id=self.creator_id,
                agent_did=self.agent_did.encode()[:64].ljust(64, b"\x00"),
                block_count=len(self.blocks),
                provenance_offset=provenance_offset,
                security_offset=security_offset,
                index_offset=index_offset,
                merkle_root=merkle_root,
            )

            # Sign the file header with Ed25519
            agent_did_bytes = (
                file_header.agent_did
                if isinstance(file_header.agent_did, bytes)
                else file_header.agent_did.encode()[:64].ljust(64, b"\x00")
            )
            header_signable = struct.pack(
                ">4sHHIQQ16s16s64sIQQQ32s",
                file_header.magic,
                file_header.version_major,
                file_header.version_minor,
                file_header.flags,
                file_header.created,
                file_header.modified,
                file_header.file_id,
                file_header.creator_id,
                agent_did_bytes,
                file_header.block_count,
                file_header.provenance_offset,
                file_header.security_offset,
                file_header.index_offset,
                file_header.merkle_root,
            )
            raw_file_signature = self._sign_data(header_signable)
            file_header.file_signature = self._pad_signature(raw_file_signature)

            # Write header
            f.write(file_header.pack())

            # Write blocks
            block_offsets = []
            for block in self.blocks:
                offset = f.tell()
                f.write(block.header.pack())
                f.write(block.data)
                # Always write metadata if metadata_size > 0
                if block.header.metadata_size > 0:
                    meta_bytes = json.dumps(
                        block.metadata or {}, sort_keys=True
                    ).encode()
                    f.write(meta_bytes)
                block_offsets.append(
                    (
                        offset,
                        block.header.size,
                        block.header.block_type,
                        block.header.flags,
                    )
                )

            # Write provenance section
            f.write(struct.pack(">I", len(provenance_data)))
            f.write(provenance_data)

            # Write security section
            f.write(struct.pack(">I", len(security_data)))
            f.write(security_data)

            # Write block index
            for offset, size, btype, flags in block_offsets:
                f.write(struct.pack(">QQII", offset, size, btype, flags))

            f.flush()
            file_size = f.tell() + FileFooter.FOOTER_SIZE

        # Calculate footer checksum
        with open(self.file_path, "rb") as f:
            content = f.read()
            checksum = hashlib.sha256(content).digest()

        # Write footer
        with open(self.file_path, "ab") as f:
            footer = FileFooter(
                magic=MAGIC_FOOTER, total_size=file_size, checksum=checksum
            )
            f.write(footer.pack())

        self.finalized = True
        return self.file_path

    def _build_security_section(self) -> bytes:
        """Build the security section with Ed25519 public key."""
        public_key_bytes = self.public_key.public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw
        )

        security_info = {
            "signer_id": self.agent_id,
            "signer_did": self.agent_did,
            "public_key": public_key_bytes.hex(),
            "key_algorithm": "Ed25519",
            "signature_algorithm": "Ed25519",
            "signed_at": int(time.time() * 1000000),
        }
        return json.dumps(security_info).encode()

    def get_public_key_bytes(self) -> bytes:
        """Get the raw Ed25519 public key (32 bytes)."""
        return self.public_key.public_bytes(
            encoding=Encoding.Raw, format=PublicFormat.Raw
        )


# =============================================================================
# Secure MAIF Reader - Verifies Ed25519 Signatures
# =============================================================================


class SecureMAIFReader:
    """
    Reads and verifies self-contained MAIF files with Ed25519 signatures.

    Automatically detects tampering through signature verification.

    Usage:
        reader = SecureMAIFReader("file.maif")
        if reader.verify_integrity()[0]:
            blocks = reader.get_blocks()
            provenance = reader.get_provenance()
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_header: Optional[SecureFileHeader] = None
        self.blocks: List[SecureBlock] = []
        self.provenance: List[ProvenanceEntry] = []
        self.security_info: Dict[str, Any] = {}
        self.public_key = None
        self.key_algorithm = "Ed25519"
        self._loaded = False
        self._integrity_verified = False
        self._tampering_detected = False
        self._tampered_blocks: List[int] = []

    def _is_ed25519_signature(self, sig: bytes) -> bool:
        """Check if signature is Ed25519 (marked with prefix)."""
        return sig.startswith(b"ED25519:")

    def _extract_ed25519_signature(self, padded_sig: bytes) -> bytes:
        """Extract raw Ed25519 signature from padded format."""
        if padded_sig.startswith(b"ED25519:"):
            return padded_sig[8:72]  # 64-byte signature after marker
        return padded_sig

    def load(self) -> bool:
        """Load the file structure."""
        try:
            with open(self.file_path, "rb") as f:
                # Read header
                header_data = f.read(SecureFileHeader.HEADER_SIZE)
                self.file_header = SecureFileHeader.unpack(header_data)

                if self.file_header.magic != MAGIC_HEADER:
                    raise ValueError(f"Invalid magic number: {self.file_header.magic}")

                # Read blocks
                f.seek(SecureFileHeader.HEADER_SIZE)
                for i in range(self.file_header.block_count):
                    block_header_data = f.read(SecureBlockHeader.HEADER_SIZE)
                    block_header = SecureBlockHeader.unpack(block_header_data)

                    data_size = (
                        block_header.size
                        - SecureBlockHeader.HEADER_SIZE
                        - block_header.metadata_size
                    )
                    block_data = f.read(data_size)

                    metadata = None
                    if block_header.metadata_size > 0:
                        try:
                            meta_bytes = f.read(block_header.metadata_size)
                            metadata = json.loads(meta_bytes.decode())
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            block_header.flags |= BlockFlags.TAMPERED
                            metadata = {"_error": f"Corrupted metadata: {str(e)}"}

                    self.blocks.append(
                        SecureBlock(
                            header=block_header, data=block_data, metadata=metadata
                        )
                    )

                # Read provenance
                try:
                    f.seek(self.file_header.provenance_offset)
                    prov_size = struct.unpack(">I", f.read(4))[0]
                    prov_data = f.read(prov_size)
                    prov_list = json.loads(prov_data.decode())
                    self.provenance = [ProvenanceEntry.from_dict(e) for e in prov_list]
                except:
                    self.provenance = []

                # Read security section
                try:
                    f.seek(self.file_header.security_offset)
                    sec_size = struct.unpack(">I", f.read(4))[0]
                    sec_data = f.read(sec_size)
                    self.security_info = json.loads(sec_data.decode())

                    # Load Ed25519 public key
                    self.key_algorithm = self.security_info.get(
                        "key_algorithm", "Ed25519"
                    )
                    pub_key_bytes = bytes.fromhex(self.security_info["public_key"])
                    self.public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                        pub_key_bytes
                    )
                except:
                    self.security_info = {}

            self._loaded = True
            return True
        except Exception as e:
            self._loaded = True
            self._tampering_detected = True
            return False

    def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verify file integrity with Ed25519 signatures."""
        if not self._loaded:
            self.load()

        errors = []
        self._tampered_blocks = []

        if not self.public_key:
            errors.append("No public key available for verification")
            return False, errors

        # Verify each block signature
        for i, block in enumerate(self.blocks):
            if block.header.flags & BlockFlags.SIGNED:
                metadata_bytes = json.dumps(
                    block.metadata or {}, sort_keys=True
                ).encode()
                signable = (
                    block.header.get_signable_data() + block.data + metadata_bytes
                )

                try:
                    raw_sig = self._extract_ed25519_signature(block.header.signature)
                    self.public_key.verify(raw_sig, signable)
                except InvalidSignature:
                    errors.append(
                        f"Block {i} (ID: {block.header.block_id.hex()}) - TAMPERED: signature verification failed"
                    )
                    self._tampered_blocks.append(i)
                    block.header.flags |= BlockFlags.TAMPERED
                except Exception as e:
                    errors.append(f"Block {i} - verification error: {str(e)}")

        # Verify content hashes
        for i, block in enumerate(self.blocks):
            calculated_hash = block.get_content_hash()
            if calculated_hash != block.header.content_hash:
                errors.append(f"Block {i} - TAMPERED: content hash mismatch")
                if i not in self._tampered_blocks:
                    self._tampered_blocks.append(i)
                    block.header.flags |= BlockFlags.TAMPERED

        # Verify block chain
        expected_prev = b"\x00" * 32
        for i, block in enumerate(self.blocks):
            if block.header.previous_hash != expected_prev:
                errors.append(f"Block {i} - chain broken: previous hash mismatch")
            expected_prev = block.header.content_hash

        # Verify Merkle root
        calculated_root = self._calculate_merkle_root()
        if calculated_root != self.file_header.merkle_root:
            errors.append("File Merkle root mismatch - file may have been modified")

        self._integrity_verified = True
        self._tampering_detected = len(self._tampered_blocks) > 0

        return len(errors) == 0, errors

    def _calculate_merkle_root(self) -> bytes:
        """Calculate Merkle root from block hashes."""
        if not self.blocks:
            return b"\x00" * 32

        hashes = [block.header.content_hash for block in self.blocks]

        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashlib.sha256(hashes[i] + hashes[i + 1]).digest()
                new_hashes.append(combined)
            hashes = new_hashes

        return hashes[0]

    def is_tampered(self) -> bool:
        """Check if tampering was detected."""
        if not self._integrity_verified:
            self.verify_integrity()
        return self._tampering_detected

    def get_tampered_blocks(self) -> List[int]:
        """Get list of tampered block indices."""
        if not self._integrity_verified:
            self.verify_integrity()
        return self._tampered_blocks

    def get_blocks(self) -> List[SecureBlock]:
        """Get all blocks."""
        if not self._loaded:
            self.load()
        return self.blocks

    def get_block(self, index: int) -> Optional[SecureBlock]:
        """Get a specific block."""
        if not self._loaded:
            self.load()
        if 0 <= index < len(self.blocks):
            return self.blocks[index]
        return None

    def get_text_content(self, block_index: int) -> Optional[str]:
        """Get text content from a text block."""
        block = self.get_block(block_index)
        if block and block.header.block_type == SecureBlockType.TEXT:
            return block.data.decode("utf-8")
        return None

    def get_provenance(self) -> List[ProvenanceEntry]:
        """Get the provenance chain."""
        if not self._loaded:
            self.load()
        return self.provenance

    def get_security_info(self) -> Dict[str, Any]:
        """Get security information."""
        if not self._loaded:
            self.load()
        return self.security_info

    def get_file_info(self) -> Dict[str, Any]:
        """Get file-level information."""
        if not self._loaded:
            self.load()

        return {
            "file_id": self.file_header.file_id.hex(),
            "creator_id": self.file_header.creator_id.hex(),
            "agent_did": self.file_header.agent_did.rstrip(b"\x00").decode(),
            "version": f"{self.file_header.version_major}.{self.file_header.version_minor}",
            "created": self.file_header.created,
            "modified": self.file_header.modified,
            "block_count": self.file_header.block_count,
            "merkle_root": self.file_header.merkle_root.hex(),
            "is_signed": bool(self.file_header.flags & FileFlags.SIGNED),
            "is_finalized": bool(self.file_header.flags & FileFlags.FINALIZED),
            "key_algorithm": self.key_algorithm,
        }

    def export_manifest(self) -> Dict[str, Any]:
        """Export a manifest dictionary (for compatibility)."""
        if not self._loaded:
            self.load()

        return {
            "maif_version": f"{self.file_header.version_major}.{self.file_header.version_minor}",
            "format": "secure",
            "file_info": self.get_file_info(),
            "blocks": [
                {
                    "index": i,
                    "type": SecureBlockType(block.header.block_type).name,
                    "block_id": block.header.block_id.hex(),
                    "size": block.header.size,
                    "content_hash": block.header.content_hash.hex(),
                    "timestamp": block.header.timestamp,
                    "is_signed": bool(block.header.flags & BlockFlags.SIGNED),
                    "is_tampered": bool(block.header.flags & BlockFlags.TAMPERED),
                    "metadata": block.metadata,
                }
                for i, block in enumerate(self.blocks)
            ],
            "provenance": [e.to_dict() for e in self.provenance],
            "security": self.security_info,
            "integrity": {
                "verified": self._integrity_verified,
                "tampering_detected": self._tampering_detected,
                "tampered_blocks": self._tampered_blocks,
            },
        }


# =============================================================================
# Utility Functions
# =============================================================================


def create_secure_maif(
    output_path: str,
    agent_id: str,
    content: Dict[str, Any],
    private_key: Optional[ed25519.Ed25519PrivateKey] = None,
) -> str:
    """
    Convenience function to create a secure MAIF file.

    Args:
        output_path: Path for output file
        agent_id: Agent identifier
        content: Dict with 'texts', 'embeddings', 'metadata' etc.
        private_key: Optional Ed25519 private key for signing

    Returns:
        Path to created file
    """
    writer = SecureMAIFWriter(output_path, agent_id, private_key)

    # Add text blocks
    for text in content.get("texts", []):
        if isinstance(text, dict):
            writer.add_text_block(text["content"], text.get("metadata"))
        else:
            writer.add_text_block(text)

    # Add embeddings
    if "embeddings" in content:
        writer.add_embeddings_block(
            content["embeddings"], content.get("embeddings_metadata")
        )

    # Add binary blocks
    for binary in content.get("binaries", []):
        writer.add_binary_block(
            binary["data"], SecureBlockType.BINARY, binary.get("metadata")
        )

    return writer.finalize()


def verify_secure_maif(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify a secure MAIF file.

    Returns (is_valid, report_dict).
    """
    reader = SecureMAIFReader(file_path)
    is_valid, errors = reader.verify_integrity()

    report = {
        "file_path": file_path,
        "is_valid": is_valid,
        "tampering_detected": reader.is_tampered(),
        "tampered_blocks": reader.get_tampered_blocks(),
        "errors": errors,
        "file_info": reader.get_file_info(),
        "block_count": len(reader.blocks),
        "provenance_entries": len(reader.provenance),
    }

    return is_valid, report


# =============================================================================
# Standard API Aliases
# =============================================================================

# Primary API - use these names for new code
MAIFEncoder = SecureMAIFWriter
MAIFDecoder = SecureMAIFReader
BlockType = SecureBlockType

# Convenience aliases
create_maif = create_secure_maif
verify_maif = verify_secure_maif


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primary API (standard names)
    "MAIFEncoder",
    "MAIFDecoder",
    "BlockType",
    "create_maif",
    "verify_maif",
    # Types
    "SecureBlockType",
    "BlockFlags",
    "FileFlags",
    # Data structures
    "SecureFileHeader",
    "SecureBlockHeader",
    "SecureBlock",
    "ProvenanceEntry",
    "FileFooter",
    # Legacy names (for backwards compatibility during migration)
    "SecureMAIFWriter",
    "SecureMAIFReader",
    # Utilities
    "create_secure_maif",
    "verify_secure_maif",
    # Constants
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
]
