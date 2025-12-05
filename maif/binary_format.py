"""
Binary format specification and low-level parsers for MAIF.
"""

import struct
import io
from typing import Dict, List, Optional, Tuple, BinaryIO
from dataclasses import dataclass
from enum import IntEnum


class BlockType(IntEnum):
    """Standard MAIF block types."""

    HEADER = 0x4D414946  # 'MAIF'
    TEXT_DATA = 0x54455854  # 'TEXT'
    IMAGE_DATA = 0x494D4147  # 'IMAG'
    AUDIO_DATA = 0x41554449  # 'AUDI'
    VIDEO_DATA = 0x56494445  # 'VIDE'
    EMBEDDINGS = 0x454D4244  # 'EMBD'
    KNOWLEDGE_GRAPH = 0x4B4E4F57  # 'KNOW'
    SECURITY_META = 0x53454355  # 'SECU'
    LIFECYCLE_META = 0x4C494645  # 'LIFE'
    COMPRESSED = 0x434F4D50  # 'COMP'
    ENCRYPTED = 0x454E4352  # 'ENCR'


@dataclass
class BlockHeader:
    """MAIF block header structure."""

    size: int  # 4 bytes - total block size including header
    block_type: int  # 4 bytes - block type identifier
    version: int  # 4 bytes - block format version
    flags: int  # 4 bytes - block flags
    timestamp: int  # 8 bytes - creation timestamp (microseconds since epoch)
    block_id: bytes  # 16 bytes - UUID as bytes
    previous_hash: bytes  # 32 bytes - SHA-256 of previous version
    reserved: bytes  # 8 bytes - reserved for future use

    HEADER_SIZE = 80  # Total header size in bytes

    def pack(self) -> bytes:
        """Pack header into binary format."""
        return struct.pack(
            ">IIIIQ16s32s8s",
            self.size,
            self.block_type,
            self.version,
            self.flags,
            self.timestamp,
            self.block_id,
            self.previous_hash,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "BlockHeader":
        """Unpack header from binary format."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Insufficient data for header: {len(data)} < {cls.HEADER_SIZE}"
            )

        unpacked = struct.unpack(">IIIIQ16s32s8s", data[: cls.HEADER_SIZE])
        return cls(*unpacked)


class BlockFlags(IntEnum):
    """Block flag constants."""

    COMPRESSED = 0x01
    ENCRYPTED = 0x02
    SIGNED = 0x04
    DELETED = 0x08
    DELTA_COMPRESSED = 0x10
    SEMANTIC_INDEXED = 0x20
    WATERMARKED = 0x40
    RESERVED = 0x80


@dataclass
class MAIFFileHeader:
    """MAIF file header structure."""

    magic: bytes  # 4 bytes - 'MAIF'
    version_major: int  # 2 bytes - major version
    version_minor: int  # 2 bytes - minor version
    flags: int  # 4 bytes - file-level flags
    created: int  # 8 bytes - creation timestamp
    modified: int  # 8 bytes - last modification timestamp
    creator_id: bytes  # 16 bytes - creator UUID
    file_id: bytes  # 16 bytes - unique file ID
    block_count: int  # 4 bytes - number of blocks
    index_offset: int  # 8 bytes - offset to block index
    metadata_offset: int  # 8 bytes - offset to metadata section
    checksum: bytes  # 32 bytes - SHA-256 of entire file
    reserved: bytes  # 32 bytes - reserved for future use

    HEADER_SIZE = 144  # Total file header size

    def pack(self) -> bytes:
        """Pack file header into binary format."""
        return struct.pack(
            ">4sHHIQQ16s16sIQQ32s32s",
            self.magic,
            self.version_major,
            self.version_minor,
            self.flags,
            self.created,
            self.modified,
            self.creator_id,
            self.file_id,
            self.block_count,
            self.index_offset,
            self.metadata_offset,
            self.checksum,
            self.reserved,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "MAIFFileHeader":
        """Unpack file header from binary format."""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(
                f"Insufficient data for file header: {len(data)} < {cls.HEADER_SIZE}"
            )

        unpacked = struct.unpack(">4sHHIQQ16s16sIQQ32s32s", data[: cls.HEADER_SIZE])
        return cls(*unpacked)


class MAIFBinaryParser:
    """Low-level binary parser for MAIF files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_header: Optional[MAIFFileHeader] = None
        self.block_index: List[Tuple[int, int]] = []  # (offset, size) pairs

    def parse_file_header(self) -> MAIFFileHeader:
        """Parse the MAIF file header."""
        with open(self.file_path, "rb") as f:
            header_data = f.read(MAIFFileHeader.HEADER_SIZE)
            self.file_header = MAIFFileHeader.unpack(header_data)

            # Verify magic number
            if self.file_header.magic != b"MAIF":
                raise ValueError(f"Invalid MAIF magic number: {self.file_header.magic}")

            return self.file_header

    def parse_block_index(self) -> List[Tuple[int, int]]:
        """Parse the block index."""
        if not self.file_header:
            self.parse_file_header()

        with open(self.file_path, "rb") as f:
            f.seek(self.file_header.index_offset)

            self.block_index = []
            for _ in range(self.file_header.block_count):
                offset, size = struct.unpack(">QQ", f.read(16))
                self.block_index.append((offset, size))

        return self.block_index

    def read_block_header(self, block_index: int) -> BlockHeader:
        """Read a specific block header."""
        if not self.block_index:
            self.parse_block_index()

        if block_index >= len(self.block_index):
            raise IndexError(f"Block index {block_index} out of range")

        offset, _ = self.block_index[block_index]

        with open(self.file_path, "rb") as f:
            f.seek(offset)
            header_data = f.read(BlockHeader.HEADER_SIZE)
            return BlockHeader.unpack(header_data)

    def read_block_data(self, block_index: int) -> bytes:
        """Read block data (excluding header)."""
        if not self.block_index:
            self.parse_block_index()

        if block_index >= len(self.block_index):
            raise IndexError(f"Block index {block_index} out of range")

        offset, size = self.block_index[block_index]

        with open(self.file_path, "rb") as f:
            f.seek(offset + BlockHeader.HEADER_SIZE)
            data_size = size - BlockHeader.HEADER_SIZE
            return f.read(data_size)

    def read_full_block(self, block_index: int) -> Tuple[BlockHeader, bytes]:
        """Read complete block (header + data)."""
        header = self.read_block_header(block_index)
        data = self.read_block_data(block_index)
        return header, data

    def find_blocks_by_type(self, block_type: BlockType) -> List[int]:
        """Find all blocks of a specific type."""
        if not self.block_index:
            self.parse_block_index()

        matching_blocks = []
        for i in range(len(self.block_index)):
            header = self.read_block_header(i)
            if header.block_type == block_type:
                matching_blocks.append(i)

        return matching_blocks

    def verify_file_integrity(self) -> bool:
        """Verify file integrity using checksums."""
        if not self.file_header:
            self.parse_file_header()

        # Calculate actual file checksum (excluding the checksum field itself)
        import hashlib

        with open(self.file_path, "rb") as f:
            # Read everything except the checksum field
            data = f.read()

            # Zero out the checksum field for calculation
            header_data = bytearray(data[: MAIFFileHeader.HEADER_SIZE])
            # Checksum is at offset 96, 32 bytes
            header_data[96:128] = b"\x00" * 32

            # Reconstruct data with zeroed checksum
            modified_data = bytes(header_data) + data[MAIFFileHeader.HEADER_SIZE :]

            calculated_checksum = hashlib.sha256(modified_data).digest()
            return calculated_checksum == self.file_header.checksum


class MAIFBinaryWriter:
    """Low-level binary writer for MAIF files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.blocks: List[Tuple[BlockHeader, bytes]] = []
        self.file_header: Optional[MAIFFileHeader] = None

    def add_block(
        self,
        block_type: BlockType,
        data: bytes,
        flags: int = 0,
        block_id: bytes = None,
        previous_hash: bytes = None,
    ) -> int:
        """Add a block to the file."""
        import time
        import uuid

        if block_id is None:
            block_id = uuid.uuid4().bytes

        if previous_hash is None:
            previous_hash = b"\x00" * 32

        header = BlockHeader(
            size=BlockHeader.HEADER_SIZE + len(data),
            block_type=block_type,
            version=1,
            flags=flags,
            timestamp=int(time.time() * 1000000),  # microseconds
            block_id=block_id,
            previous_hash=previous_hash,
            reserved=b"\x00" * 8,
        )

        self.blocks.append((header, data))
        return len(self.blocks) - 1

    def write_file(self, creator_id: bytes = None, file_id: bytes = None):
        """Write the complete MAIF file."""
        import time
        import uuid
        import hashlib

        if creator_id is None:
            creator_id = uuid.uuid4().bytes

        if file_id is None:
            file_id = uuid.uuid4().bytes

        # Calculate offsets
        index_offset = MAIFFileHeader.HEADER_SIZE
        for header, data in self.blocks:
            index_offset += header.size

        metadata_offset = index_offset + (
            len(self.blocks) * 16
        )  # 16 bytes per index entry

        # Create file header (checksum will be calculated later)
        self.file_header = MAIFFileHeader(
            magic=b"MAIF",
            version_major=1,
            version_minor=0,
            flags=0,
            created=int(time.time() * 1000000),
            modified=int(time.time() * 1000000),
            creator_id=creator_id,
            file_id=file_id,
            block_count=len(self.blocks),
            index_offset=index_offset,
            metadata_offset=metadata_offset,
            checksum=hashlib.sha256(
                b"".join([header.pack() + data for header, data in self.blocks])
            ).digest(),
            reserved=b"\x00" * 32,
        )

        # Write file
        with open(self.file_path, "wb") as f:
            # Write placeholder header
            f.write(self.file_header.pack())

            # Write blocks and build index
            block_index = []
            for header, data in self.blocks:
                offset = f.tell()
                f.write(header.pack())
                f.write(data)
                block_index.append((offset, header.size))

            # Write block index
            for offset, size in block_index:
                f.write(struct.pack(">QQ", offset, size))

        # Read file back to calculate checksum
        with open(self.file_path, "rb") as f:
            file_data = f.read()

            # Zero out checksum field
            file_data_array = bytearray(file_data)
            file_data_array[96:128] = b"\x00" * 32

            checksum = hashlib.sha256(bytes(file_data_array)).digest()

        # Update header with real checksum
        self.file_header.checksum = checksum
        with open(self.file_path, "r+b") as f:
            f.seek(0)
            f.write(self.file_header.pack())


class MAIFStreamParser:
    """Streaming parser for large MAIF files."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_handle: Optional[BinaryIO] = None

    def __enter__(self):
        self.file_handle = open(self.file_path, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()

    def stream_blocks(self):
        """Generator that yields blocks one at a time."""
        if not self.file_handle:
            raise ValueError("Parser not opened - use with statement")

        # Parse file header
        self.file_handle.seek(0)
        file_header = MAIFFileHeader.unpack(
            self.file_handle.read(MAIFFileHeader.HEADER_SIZE)
        )

        # Stream through blocks
        current_offset = MAIFFileHeader.HEADER_SIZE

        for _ in range(file_header.block_count):
            self.file_handle.seek(current_offset)

            # Read block header
            header_data = self.file_handle.read(BlockHeader.HEADER_SIZE)
            if len(header_data) < BlockHeader.HEADER_SIZE:
                break

            header = BlockHeader.unpack(header_data)

            # Read block data
            data_size = header.size - BlockHeader.HEADER_SIZE
            data = self.file_handle.read(data_size)

            yield header, data

            current_offset += header.size
