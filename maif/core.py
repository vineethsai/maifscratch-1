"""
MAIF Core Module - Encoding, Decoding, and Parsing

This module provides the primary interface for creating and reading MAIF files.
MAIF v2.x uses a self-contained binary format with embedded security (Ed25519 signatures)
and provenance tracking. No external manifest files are required.

Key Features:
- Ed25519 cryptographic signatures for each block
- Immutable blocks - signed on first write
- Tamper detection through signature verification
- Embedded provenance chain
- Merkle tree for file integrity

Usage:
    # Creating a MAIF file
    encoder = MAIFEncoder("output.maif", agent_id="my-agent")
    encoder.add_text_block("Hello, world!")
    encoder.add_embeddings_block([[0.1, 0.2, 0.3]])
    encoder.finalize()

    # Reading a MAIF file
    decoder = MAIFDecoder("output.maif")
    if decoder.verify_integrity()[0]:
        blocks = decoder.get_blocks()
        provenance = decoder.get_provenance()
"""

import json
import hashlib
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import the secure format implementation (Ed25519-based)
from .secure_format import (
    # Primary API
    SecureMAIFWriter as MAIFEncoder,
    SecureMAIFReader as MAIFDecoder,
    SecureBlockType as BlockType,
    create_secure_maif as create_maif,
    verify_secure_maif as verify_maif,
    # Data structures
    SecureBlock,
    SecureBlockHeader,
    SecureFileHeader,
    ProvenanceEntry,
    FileFooter,
    BlockFlags,
    FileFlags,
    # Constants
    MAGIC_HEADER,
    MAGIC_FOOTER,
    FORMAT_VERSION_MAJOR,
    FORMAT_VERSION_MINOR,
)


# =============================================================================
# Data Classes (kept for compatibility with existing code)
# =============================================================================


@dataclass
class MAIFBlock:
    """
    Represents a MAIF block with metadata.

    This class is maintained for backwards compatibility. New code should
    use the SecureBlock class directly for full functionality.
    """

    block_type: str
    offset: int = 0
    size: int = 0
    hash_value: str = ""
    version: int = 1
    previous_hash: Optional[str] = None
    block_id: Optional[str] = None
    metadata: Optional[Dict] = None
    data: Optional[bytes] = None

    def __post_init__(self):
        if self.block_id is None:
            self.block_id = str(uuid.uuid4())
        if self.data is not None and not self.hash_value:
            self.hash_value = hashlib.sha256(self.data).hexdigest()

    @property
    def hash(self) -> str:
        """Return the hash value."""
        return self.hash_value

    def to_dict(self) -> Dict:
        return {
            "type": self.block_type,
            "block_type": self.block_type,
            "offset": self.offset,
            "size": self.size,
            "hash": self.hash_value,
            "version": self.version,
            "previous_hash": self.previous_hash,
            "block_id": self.block_id,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_secure_block(cls, block: SecureBlock, index: int = 0) -> "MAIFBlock":
        """Create MAIFBlock from SecureBlock."""
        type_names = {
            BlockType.TEXT: "TEXT",
            BlockType.EMBEDDINGS: "EMBD",
            BlockType.IMAGE: "IMAG",
            BlockType.AUDIO: "AUDI",
            BlockType.VIDEO: "VIDE",
            BlockType.KNOWLEDGE: "KNOW",
            BlockType.BINARY: "BINA",
            BlockType.METADATA: "META",
        }
        block_type = type_names.get(block.header.block_type, "UNKNOWN")

        return cls(
            block_type=block_type,
            offset=0,  # Would need to track during reading
            size=block.header.size,
            hash_value=block.header.content_hash.hex(),
            version=block.header.version,
            previous_hash=block.header.previous_hash.hex()
            if block.header.previous_hash != b"\x00" * 32
            else None,
            block_id=block.header.block_id.hex(),
            metadata=block.metadata,
            data=block.data,
        )


@dataclass
class MAIFVersion:
    """Represents a version entry in the version history."""

    version: int
    timestamp: float
    agent_id: str
    operation: str  # "create", "update", "delete"
    block_hash: str
    block_id: Optional[str] = None
    previous_hash: Optional[str] = None
    change_description: Optional[str] = None

    @property
    def version_number(self) -> int:
        return self.version

    @property
    def current_hash(self) -> str:
        return self.block_hash

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "operation": self.operation,
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "current_hash": self.block_hash,
            "block_hash": self.block_hash,
            "change_description": self.change_description,
        }


@dataclass
class MAIFHeader:
    """MAIF file header information (simplified view)."""

    version: str = "2.1"
    created_timestamp: float = None
    creator_id: Optional[str] = None
    root_hash: Optional[str] = None
    agent_id: Optional[str] = None

    def __post_init__(self):
        if self.created_timestamp is None:
            self.created_timestamp = time.time()


# =============================================================================
# High-Level Parser (wraps MAIFDecoder for convenience)
# =============================================================================


class MAIFParser:
    """
    High-level MAIF parsing interface.

    Provides a convenient way to load and inspect MAIF files.

    Usage:
        parser = MAIFParser("file.maif")
        parser.load()

        for block in parser.blocks:
            print(f"Block: {block.block_type}, Size: {block.size}")

        for entry in parser.provenance:
            print(f"{entry.action} by {entry.agent_id}")
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._decoder = MAIFDecoder(file_path)
        self._loaded = False
        self.blocks: List[MAIFBlock] = []
        self.provenance: List[ProvenanceEntry] = []
        self.header: Optional[MAIFHeader] = None

    def load(self) -> bool:
        """Load and parse the MAIF file."""
        if self._decoder.load():
            self._loaded = True

            # Convert blocks
            self.blocks = [
                MAIFBlock.from_secure_block(block, i)
                for i, block in enumerate(self._decoder.blocks)
            ]

            # Copy provenance
            self.provenance = self._decoder.provenance

            # Build header info
            file_info = self._decoder.get_file_info()
            self.header = MAIFHeader(
                version=file_info.get("version", "2.1"),
                created_timestamp=file_info.get("created", 0)
                / 1000000,  # Convert from microseconds
                creator_id=file_info.get("creator_id"),
                root_hash=file_info.get("merkle_root"),
                agent_id=file_info.get("agent_did"),
            )

            return True
        return False

    def verify(self) -> Tuple[bool, List[str]]:
        """Verify file integrity."""
        return self._decoder.verify_integrity()

    def is_tampered(self) -> bool:
        """Check if file has been tampered with."""
        return self._decoder.is_tampered()

    def get_block_data(self, index: int) -> Optional[bytes]:
        """Get raw data for a block."""
        block = self._decoder.get_block(index)
        if block:
            return block.data
        return None

    def get_text_content(self, index: int) -> Optional[str]:
        """Get text content from a text block."""
        return self._decoder.get_text_content(index)

    def export_manifest(self) -> Dict[str, Any]:
        """Export a manifest dictionary (for compatibility)."""
        return self._decoder.export_manifest()

    @property
    def file_info(self) -> Dict[str, Any]:
        """Get file information."""
        if not self._loaded:
            self.load()
        return self._decoder.get_file_info()

    @property
    def security_info(self) -> Dict[str, Any]:
        """Get security information."""
        if not self._loaded:
            self.load()
        return self._decoder.get_security_info()


# =============================================================================
# Convenience Functions
# =============================================================================


def quick_create(
    output_path: str,
    texts: List[str] = None,
    embeddings: List[List[float]] = None,
    agent_id: str = "default-agent",
    metadata: Dict = None,
) -> str:
    """
    Quick way to create a MAIF file.

    Args:
        output_path: Path for output file
        texts: List of text strings to add
        embeddings: List of embedding vectors
        agent_id: Agent identifier
        metadata: Optional metadata for the file

    Returns:
        Path to created file
    """
    encoder = MAIFEncoder(output_path, agent_id)

    if texts:
        for i, text in enumerate(texts):
            encoder.add_text_block(text, {"index": i, **(metadata or {})})

    if embeddings:
        encoder.add_embeddings_block(embeddings, metadata)

    return encoder.finalize()


def quick_verify(file_path: str) -> bool:
    """
    Quick verification of a MAIF file.

    Returns True if file is valid and untampered.
    """
    is_valid, _ = verify_maif(file_path)
    return is_valid


def quick_read(file_path: str) -> Dict[str, Any]:
    """
    Quick read of a MAIF file.

    Returns a dict with file info, blocks, and provenance.
    """
    decoder = MAIFDecoder(file_path)
    decoder.load()
    return decoder.export_manifest()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Primary API
    "MAIFEncoder",
    "MAIFDecoder",
    "MAIFParser",
    "BlockType",
    # Data classes
    "MAIFBlock",
    "MAIFVersion",
    "MAIFHeader",
    # Secure format types
    "SecureBlock",
    "SecureBlockHeader",
    "SecureFileHeader",
    "ProvenanceEntry",
    "FileFooter",
    "BlockFlags",
    "FileFlags",
    # Convenience functions
    "create_maif",
    "verify_maif",
    "quick_create",
    "quick_verify",
    "quick_read",
    # Constants
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
]
