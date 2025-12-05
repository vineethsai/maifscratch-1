"""
Unified Block Format for MAIF
=============================

This module provides a standardized block format that ensures parity between
local file storage and AWS S3 backend implementations.
"""

import struct
import json
import uuid
import time
import hashlib
from typing import Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BlockType(Enum):
    """Unified block type constants using FourCC identifiers."""

    HEADER = "HDER"
    TEXT_DATA = "TEXT"
    EMBEDDING = "EMBD"
    KNOWLEDGE_GRAPH = "KGRF"
    SECURITY = "SECU"
    BINARY_DATA = "BDAT"
    IMAGE_DATA = "IMAG"
    AUDIO_DATA = "AUDI"
    VIDEO_DATA = "VIDE"
    AI_MODEL = "AIMD"
    PROVENANCE = "PROV"
    ACCESS_CONTROL = "ACCS"
    LIFECYCLE = "LIFE"
    CROSS_MODAL = "XMOD"
    SEMANTIC_BINDING = "SBND"
    COMPRESSED_EMBEDDINGS = "CEMB"

    @classmethod
    def from_string(cls, value: str) -> "BlockType":
        """Convert string to BlockType."""
        for block_type in cls:
            if block_type.value == value:
                return block_type
        raise ValueError(f"Unknown block type: {value}")

    @classmethod
    def from_int(cls, value: int) -> "BlockType":
        """Convert legacy integer representation to BlockType."""
        # Map legacy integer values to new string values
        legacy_map = {
            0x4D414946: cls.HEADER,  # 'MAIF'
            0x54455854: cls.TEXT_DATA,  # 'TEXT'
            0x494D4147: cls.IMAGE_DATA,  # 'IMAG'
            0x41554449: cls.AUDIO_DATA,  # 'AUDI'
            0x56494445: cls.VIDEO_DATA,  # 'VIDE'
            0x454D4244: cls.EMBEDDING,  # 'EMBD'
            0x4B4E4F57: cls.KNOWLEDGE_GRAPH,  # 'KNOW'
            0x53454355: cls.SECURITY,  # 'SECU'
            0x4C494645: cls.LIFECYCLE,  # 'LIFE'
        }
        return legacy_map.get(value, cls.BINARY_DATA)


@dataclass
class UnifiedBlockHeader:
    """
    Unified block header structure for both local and AWS storage.

    This header format ensures compatibility between different storage backends
    while maintaining all necessary metadata.
    """

    # Core fields (always present)
    magic: bytes = b"MAIF"  # 4 bytes - identifies MAIF block
    version: int = 1  # 4 bytes - header version
    size: int = 0  # 8 bytes - data size (excluding header)
    block_type: str = "BDAT"  # 4 bytes - FourCC type identifier
    uuid: str = ""  # 36 bytes - UUID string
    timestamp: float = 0.0  # 8 bytes - creation timestamp

    # Chain fields (for versioning)
    previous_hash: Optional[str] = None  # 64 bytes - hex hash of previous version
    block_hash: Optional[str] = None  # 64 bytes - hex hash of this block's data

    # Metadata fields
    flags: int = 0  # 4 bytes - block flags
    metadata_size: int = 0  # 4 bytes - size of JSON metadata
    reserved: bytes = b"\x00" * 28  # 28 bytes - reserved for future use

    HEADER_SIZE = 224  # Total fixed header size

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.uuid:
            self.uuid = str(uuid.uuid4())
        if self.timestamp == 0.0:
            self.timestamp = time.time()

        # Ensure string fields have correct length
        if len(self.uuid) != 36:
            self.uuid = str(uuid.UUID(self.uuid))  # Normalize UUID format

        # Ensure block_type is exactly 4 characters
        if len(self.block_type) > 4:
            self.block_type = self.block_type[:4]
        elif len(self.block_type) < 4:
            self.block_type = self.block_type.ljust(4, " ")

    def to_bytes(self) -> bytes:
        """Serialize header to bytes for local storage."""
        # Pack fixed-size fields
        uuid_bytes = self.uuid.encode("ascii")[:36].ljust(36, b"\x00")
        type_bytes = self.block_type.encode("ascii")[:4].ljust(4, b"\x00")

        # Handle optional fields
        prev_hash_bytes = (
            (self.previous_hash or "").encode("ascii")[:64].ljust(64, b"\x00")
        )
        block_hash_bytes = (
            (self.block_hash or "").encode("ascii")[:64].ljust(64, b"\x00")
        )

        # Pack header using big-endian byte order for consistency
        return struct.pack(
            ">4sIQ4s36sQ64s64sII28s",
            self.magic,
            self.version,
            self.size,
            type_bytes,
            uuid_bytes,
            int(self.timestamp * 1000000),  # microseconds
            prev_hash_bytes,
            block_hash_bytes,
            self.flags,
            self.metadata_size,
            self.reserved,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "UnifiedBlockHeader":
        """Deserialize header from bytes."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Insufficient data for header: {len(data)} < {cls.HEADER_SIZE}"
            )

        # Unpack header
        unpacked = struct.unpack(">4sIQ4s36sQ64s64sII28s", data[: cls.HEADER_SIZE])

        magic = unpacked[0]
        if magic != b"MAIF":
            raise ValueError(f"Invalid magic number: {magic}")

        return cls(
            magic=magic,
            version=unpacked[1],
            size=unpacked[2],
            block_type=unpacked[3].decode("ascii").rstrip("\x00"),
            uuid=unpacked[4].decode("ascii").rstrip("\x00"),
            timestamp=unpacked[5] / 1000000.0,  # Convert from microseconds
            previous_hash=unpacked[6].decode("ascii").rstrip("\x00") or None,
            block_hash=unpacked[7].decode("ascii").rstrip("\x00") or None,
            flags=unpacked[8],
            metadata_size=unpacked[9],
            reserved=unpacked[10],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert header to dictionary for JSON serialization (AWS storage)."""
        return {
            "magic": self.magic.decode("ascii"),
            "version": self.version,
            "size": self.size,
            "block_type": self.block_type,
            "uuid": self.uuid,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "block_hash": self.block_hash,
            "flags": self.flags,
            "metadata_size": self.metadata_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedBlockHeader":
        """Create header from dictionary (AWS metadata)."""
        return cls(
            magic=data.get("magic", "MAIF").encode("ascii"),
            version=int(data.get("version", 1)),
            size=int(data.get("size", 0)),
            block_type=data.get("block_type", "BDAT"),
            uuid=data.get("uuid", ""),
            timestamp=float(data.get("timestamp", 0.0)),
            previous_hash=data.get("previous_hash"),
            block_hash=data.get("block_hash"),
            flags=int(data.get("flags", 0)),
            metadata_size=int(data.get("metadata_size", 0)),
        )


class UnifiedBlockFlags(Enum):
    """Unified block flag constants."""

    COMPRESSED = 0x01
    ENCRYPTED = 0x02
    SIGNED = 0x04
    DELETED = 0x08
    DELTA_COMPRESSED = 0x10
    SEMANTIC_INDEXED = 0x20
    WATERMARKED = 0x40
    HAS_METADATA = 0x80


@dataclass
class UnifiedBlock:
    """
    Unified block structure combining header, data, and metadata.
    """

    header: UnifiedBlockHeader
    data: bytes
    metadata: Dict[str, Any]
    signature: Optional[Dict[str, Any]] = None

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of block data."""
        return hashlib.sha256(self.data).hexdigest()

    def to_bytes(self) -> bytes:
        """Serialize entire block to bytes."""
        # Serialize metadata to JSON
        metadata_json = json.dumps(self.metadata, separators=(",", ":")).encode("utf-8")

        # Update header with metadata size
        self.header.metadata_size = len(metadata_json)
        self.header.size = len(self.data)
        self.header.block_hash = self.calculate_hash()

        # Set metadata flag if metadata exists
        if metadata_json:
            self.header.flags |= UnifiedBlockFlags.HAS_METADATA.value

        # Combine header, metadata, and data
        return self.header.to_bytes() + metadata_json + self.data

    @classmethod
    def from_bytes(cls, data: bytes) -> "UnifiedBlock":
        """Deserialize block from bytes."""
        # Parse header
        header = UnifiedBlockHeader.from_bytes(data)

        # Extract metadata if present
        offset = UnifiedBlockHeader.HEADER_SIZE
        metadata = {}

        if header.metadata_size > 0:
            metadata_json = data[offset : offset + header.metadata_size]
            metadata = json.loads(metadata_json.decode("utf-8"))
            offset += header.metadata_size

        # Extract data
        block_data = data[offset : offset + header.size]

        return cls(header=header, data=block_data, metadata=metadata)

    def to_aws_format(self) -> Tuple[bytes, Dict[str, str]]:
        """
        Convert block to AWS S3 format.

        Returns:
            Tuple of (data, s3_metadata)
        """
        # Prepare S3 metadata
        s3_metadata = {
            "maif_magic": "MAIF",
            "maif_version": str(self.header.version),
            "maif_block_type": self.header.block_type,
            "maif_uuid": self.header.uuid,
            "maif_timestamp": str(self.header.timestamp),
            "maif_size": str(len(self.data)),
            "maif_flags": str(self.header.flags),
            "maif_block_hash": self.header.block_hash or self.calculate_hash(),
        }

        # Add optional fields
        if self.header.previous_hash:
            s3_metadata["maif_previous_hash"] = self.header.previous_hash

        # Add user metadata with prefix
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                s3_metadata[f"maif_meta_{key}"] = str(value)

        # Add signature if present
        if self.signature:
            s3_metadata["maif_signature"] = json.dumps(self.signature)

        return self.data, s3_metadata

    @classmethod
    def from_aws_format(
        cls, data: bytes, s3_metadata: Dict[str, str]
    ) -> "UnifiedBlock":
        """
        Create block from AWS S3 format.

        Args:
            data: Block data from S3 object
            s3_metadata: S3 object metadata

        Returns:
            UnifiedBlock instance
        """
        # Extract header fields from S3 metadata
        header = UnifiedBlockHeader(
            magic=b"MAIF",
            version=int(s3_metadata.get("maif_version", "1")),
            size=len(data),
            block_type=s3_metadata.get("maif_block_type", "BDAT"),
            uuid=s3_metadata.get("maif_uuid", str(uuid.uuid4())),
            timestamp=float(s3_metadata.get("maif_timestamp", "0")),
            previous_hash=s3_metadata.get("maif_previous_hash"),
            block_hash=s3_metadata.get("maif_block_hash"),
            flags=int(s3_metadata.get("maif_flags", "0")),
        )

        # Extract user metadata
        metadata = {}
        for key, value in s3_metadata.items():
            if key.startswith("maif_meta_"):
                metadata[key[10:]] = value

        # Extract signature if present
        signature = None
        if "maif_signature" in s3_metadata:
            try:
                signature = json.loads(s3_metadata["maif_signature"])
            except json.JSONDecodeError:
                logger.warning("Failed to parse signature metadata")

        return cls(header=header, data=data, metadata=metadata, signature=signature)


class BlockFormatConverter:
    """Utilities for converting between different block formats."""

    @staticmethod
    def legacy_to_unified(
        block_type: Union[str, int], data: bytes, metadata: Optional[Dict] = None
    ) -> UnifiedBlock:
        """Convert legacy block format to unified format."""
        # Handle block type conversion
        if isinstance(block_type, int):
            block_type_enum = BlockType.from_int(block_type)
            block_type_str = block_type_enum.value
        else:
            block_type_str = block_type

        # Create unified header
        header = UnifiedBlockHeader(block_type=block_type_str, size=len(data))

        # Create unified block
        return UnifiedBlock(header=header, data=data, metadata=metadata or {})

    @staticmethod
    def unified_to_legacy(block: UnifiedBlock) -> Tuple[str, bytes, Dict]:
        """Convert unified block to legacy format."""
        return block.header.block_type, block.data, block.metadata
