"""
Unified Storage Abstraction for MAIF
====================================

This module provides a unified storage interface that maintains parity between
local file storage and AWS S3 backend, using the unified block format.
"""

import os
import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

from .unified_block_format import (
    UnifiedBlock,
    UnifiedBlockHeader,
    BlockType,
    BlockFormatConverter,
    UnifiedBlockFlags,
)
from .signature_verification import (
    SignatureVerifier,
    create_default_verifier,
    sign_block_data,
    verify_block_signature,
)

logger = logging.getLogger(__name__)


class UnifiedStorageBackend(ABC):
    """Abstract base class for unified storage backends."""

    @abstractmethod
    def store_block(self, block: UnifiedBlock) -> str:
        """Store a block and return its UUID."""
        pass

    @abstractmethod
    def retrieve_block(self, block_uuid: str) -> Optional[UnifiedBlock]:
        """Retrieve a block by UUID."""
        pass

    @abstractmethod
    def list_blocks(self) -> List[UnifiedBlockHeader]:
        """List all block headers."""
        pass

    @abstractmethod
    def delete_block(self, block_uuid: str) -> bool:
        """Delete a block by UUID."""
        pass

    @abstractmethod
    def update_block(self, block_uuid: str, block: UnifiedBlock) -> bool:
        """Update an existing block."""
        pass

    @abstractmethod
    def close(self):
        """Close the storage backend."""
        pass


class LocalFileBackend(UnifiedStorageBackend):
    """Local file storage backend using unified block format."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.blocks: Dict[
            str, Tuple[int, UnifiedBlockHeader]
        ] = {}  # uuid -> (offset, header)
        self._lock = threading.RLock()
        self._file_handle = None
        self._load_index()

    def _load_index(self):
        """Load block index from file."""
        if not os.path.exists(self.file_path):
            # Create new file with MAIF header
            self._create_new_file()
            return

        try:
            with open(self.file_path, "rb") as f:
                # Read file header (first block should be HDER type)
                header_data = f.read(UnifiedBlockHeader.HEADER_SIZE)
                if len(header_data) < UnifiedBlockHeader.HEADER_SIZE:
                    logger.warning("File too small, creating new file")
                    self._create_new_file()
                    return

                # Parse blocks
                f.seek(0)
                offset = 0

                while True:
                    # Try to read header
                    header_data = f.read(UnifiedBlockHeader.HEADER_SIZE)
                    if len(header_data) < UnifiedBlockHeader.HEADER_SIZE:
                        break

                    try:
                        header = UnifiedBlockHeader.from_bytes(header_data)

                        # Skip metadata and data
                        if header.metadata_size > 0:
                            f.seek(header.metadata_size, 1)
                        if header.size > 0:
                            f.seek(header.size, 1)

                        # Store in index
                        self.blocks[header.uuid] = (offset, header)

                        # Update offset
                        offset = f.tell()

                    except Exception as e:
                        logger.error(f"Error parsing block at offset {offset}: {e}")
                        break

                logger.info(f"Loaded {len(self.blocks)} blocks from {self.file_path}")

        except Exception as e:
            logger.error(f"Error loading file index: {e}")
            self._create_new_file()

    def _create_new_file(self):
        """Create a new MAIF file with header block."""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

                # Create header block
                header_block = UnifiedBlock(
                    header=UnifiedBlockHeader(
                        block_type=BlockType.HEADER.value, size=0
                    ),
                    data=b"",
                    metadata={
                        "format_version": "2.0",
                        "created_by": "unified_storage",
                        "features": ["unified_format", "aws_compatible"],
                    },
                )

                # Write to file
                with open(self.file_path, "wb") as f:
                    f.write(header_block.to_bytes())

                # Update index
                self.blocks[header_block.header.uuid] = (0, header_block.header)

                logger.info(f"Created new MAIF file: {self.file_path}")

            except Exception as e:
                logger.error(f"Error creating new file: {e}")
                raise

    def store_block(self, block: UnifiedBlock) -> str:
        """Store a block in the file."""
        with self._lock:
            try:
                # Calculate block hash
                block.header.block_hash = block.calculate_hash()

                # Open file for append
                with open(self.file_path, "ab") as f:
                    offset = f.tell()

                    # Write block
                    block_data = block.to_bytes()
                    f.write(block_data)
                    f.flush()

                    # Sync to disk
                    os.fsync(f.fileno())

                # Update index
                self.blocks[block.header.uuid] = (offset, block.header)

                logger.debug(f"Stored block {block.header.uuid} at offset {offset}")
                return block.header.uuid

            except Exception as e:
                logger.error(f"Error storing block: {e}")
                raise

    def retrieve_block(self, block_uuid: str) -> Optional[UnifiedBlock]:
        """Retrieve a block from the file."""
        with self._lock:
            if block_uuid not in self.blocks:
                return None

            offset, header = self.blocks[block_uuid]

            try:
                with open(self.file_path, "rb") as f:
                    f.seek(offset)

                    # Read full block
                    total_size = (
                        UnifiedBlockHeader.HEADER_SIZE
                        + header.metadata_size
                        + header.size
                    )

                    block_data = f.read(total_size)

                    # Parse block
                    return UnifiedBlock.from_bytes(block_data)

            except Exception as e:
                logger.error(f"Error retrieving block {block_uuid}: {e}")
                return None

    def list_blocks(self) -> List[UnifiedBlockHeader]:
        """List all block headers."""
        with self._lock:
            return [header for _, header in self.blocks.values()]

    def delete_block(self, block_uuid: str) -> bool:
        """Mark a block as deleted (soft delete)."""
        with self._lock:
            if block_uuid not in self.blocks:
                return False

            try:
                # Retrieve block
                block = self.retrieve_block(block_uuid)
                if not block:
                    return False

                # Set deleted flag
                block.header.flags |= UnifiedBlockFlags.DELETED.value

                # Create deletion record
                deletion_block = UnifiedBlock(
                    header=UnifiedBlockHeader(
                        block_type=BlockType.LIFECYCLE.value,
                        previous_hash=block.header.block_hash,
                    ),
                    data=b"",
                    metadata={
                        "action": "delete",
                        "target_uuid": block_uuid,
                        "deleted_at": block.header.timestamp,
                    },
                )

                # Store deletion record
                self.store_block(deletion_block)

                logger.info(f"Soft deleted block {block_uuid}")
                return True

            except Exception as e:
                logger.error(f"Error deleting block {block_uuid}: {e}")
                return False

    def update_block(self, block_uuid: str, block: UnifiedBlock) -> bool:
        """Update a block by creating a new version."""
        with self._lock:
            if block_uuid not in self.blocks:
                return False

            try:
                # Get previous block
                old_block = self.retrieve_block(block_uuid)
                if not old_block:
                    return False

                # Set version chain
                block.header.previous_hash = old_block.header.block_hash
                block.header.version = old_block.header.version + 1

                # Store new version
                new_uuid = self.store_block(block)

                logger.info(f"Updated block {block_uuid} -> {new_uuid}")
                return True

            except Exception as e:
                logger.error(f"Error updating block {block_uuid}: {e}")
                return False

    def close(self):
        """Close the file backend."""
        # Nothing to close for file-based access
        pass


class AWSS3Backend(UnifiedStorageBackend):
    """AWS S3 storage backend using unified block format."""

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "maif/",
        s3_client=None,
        enable_cache: bool = True,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.s3_client = s3_client
        self._lock = threading.RLock()
        self.enable_cache = enable_cache
        self._cache: Dict[str, UnifiedBlock] = {}
        self._index: Dict[str, UnifiedBlockHeader] = {}
        self._load_index()

    def _get_s3_client(self):
        """Get or create S3 client."""
        if self.s3_client is None:
            from .aws_s3_integration import S3Client

            self.s3_client = S3Client()
        return self.s3_client

    def _load_index(self):
        """Load block index from S3."""
        try:
            client = self._get_s3_client()
            index_key = f"{self.prefix}unified_index.json"

            response = client.get_object(bucket_name=self.bucket_name, key=index_key)

            if response and "Body" in response:
                index_data = json.loads(response["Body"].read().decode("utf-8"))

                # Rebuild index
                for block_data in index_data.get("blocks", []):
                    header = UnifiedBlockHeader.from_dict(block_data)
                    self._index[header.uuid] = header

                logger.info(f"Loaded {len(self._index)} blocks from S3 index")

        except Exception as e:
            logger.warning(f"Could not load S3 index: {e}")
            # Index will be rebuilt on demand

    def _save_index(self):
        """Save block index to S3."""
        try:
            # Prepare index data
            index_data = {
                "format_version": "2.0",
                "blocks": [header.to_dict() for header in self._index.values()],
                "updated_at": UnifiedBlockHeader().timestamp,
            }

            # Save to S3
            client = self._get_s3_client()
            index_key = f"{self.prefix}unified_index.json"

            client.put_object(
                bucket_name=self.bucket_name,
                key=index_key,
                data=json.dumps(index_data).encode("utf-8"),
                metadata={"content_type": "application/json"},
            )

            logger.debug("Saved index to S3")

        except Exception as e:
            logger.error(f"Error saving S3 index: {e}")

    def store_block(self, block: UnifiedBlock) -> str:
        """Store a block in S3."""
        with self._lock:
            try:
                # Calculate block hash
                block.header.block_hash = block.calculate_hash()

                # Convert to AWS format
                data, s3_metadata = block.to_aws_format()

                # Store in S3
                client = self._get_s3_client()
                s3_key = f"{self.prefix}blocks/{block.header.uuid}"

                client.put_object(
                    bucket_name=self.bucket_name,
                    key=s3_key,
                    data=data,
                    metadata=s3_metadata,
                )

                # Update index
                self._index[block.header.uuid] = block.header
                self._save_index()

                # Update cache
                if self.enable_cache:
                    self._cache[block.header.uuid] = block

                logger.debug(f"Stored block {block.header.uuid} in S3")
                return block.header.uuid

            except Exception as e:
                logger.error(f"Error storing block in S3: {e}")
                raise

    def retrieve_block(self, block_uuid: str) -> Optional[UnifiedBlock]:
        """Retrieve a block from S3."""
        with self._lock:
            # Check cache first
            if self.enable_cache and block_uuid in self._cache:
                return self._cache[block_uuid]

            try:
                # Retrieve from S3
                client = self._get_s3_client()
                s3_key = f"{self.prefix}blocks/{block_uuid}"

                response = client.get_object(bucket_name=self.bucket_name, key=s3_key)

                if not response or "Body" not in response:
                    return None

                # Get data and metadata
                data = response["Body"].read()
                s3_metadata = response.get("Metadata", {})

                # Convert from AWS format
                block = UnifiedBlock.from_aws_format(data, s3_metadata)

                # Update cache
                if self.enable_cache:
                    self._cache[block_uuid] = block

                return block

            except Exception as e:
                logger.error(f"Error retrieving block {block_uuid} from S3: {e}")
                return None

    def list_blocks(self) -> List[UnifiedBlockHeader]:
        """List all block headers."""
        with self._lock:
            return list(self._index.values())

    def delete_block(self, block_uuid: str) -> bool:
        """Delete a block from S3."""
        with self._lock:
            try:
                # Create deletion marker
                block = self.retrieve_block(block_uuid)
                if not block:
                    return False

                # Set deleted flag
                block.header.flags |= UnifiedBlockFlags.DELETED.value

                # Create deletion record
                deletion_block = UnifiedBlock(
                    header=UnifiedBlockHeader(
                        block_type=BlockType.LIFECYCLE.value,
                        previous_hash=block.header.block_hash,
                    ),
                    data=b"",
                    metadata={
                        "action": "delete",
                        "target_uuid": block_uuid,
                        "deleted_at": block.header.timestamp,
                    },
                )

                # Store deletion record
                self.store_block(deletion_block)

                # Remove from cache
                if block_uuid in self._cache:
                    del self._cache[block_uuid]

                logger.info(f"Soft deleted block {block_uuid} in S3")
                return True

            except Exception as e:
                logger.error(f"Error deleting block {block_uuid} from S3: {e}")
                return False

    def update_block(self, block_uuid: str, block: UnifiedBlock) -> bool:
        """Update a block by creating a new version."""
        with self._lock:
            try:
                # Get previous block
                old_block = self.retrieve_block(block_uuid)
                if not old_block:
                    return False

                # Set version chain
                block.header.previous_hash = old_block.header.block_hash
                block.header.version = old_block.header.version + 1

                # Store new version
                new_uuid = self.store_block(block)

                logger.info(f"Updated block {block_uuid} -> {new_uuid} in S3")
                return True

            except Exception as e:
                logger.error(f"Error updating block {block_uuid} in S3: {e}")
                return False

    def close(self):
        """Close the S3 backend."""
        # Clear cache
        self._cache.clear()


class UnifiedStorage:
    """
    High-level unified storage interface that automatically selects
    the appropriate backend and maintains format consistency.
    """

    def __init__(
        self,
        storage_path: str,
        use_aws: bool = False,
        aws_bucket: Optional[str] = None,
        aws_prefix: str = "maif/",
        verify_signatures: bool = True,
    ):
        """
        Initialize unified storage.

        Args:
            storage_path: Local file path or identifier
            use_aws: Whether to use AWS S3 backend
            aws_bucket: S3 bucket name (required if use_aws=True)
            aws_prefix: S3 key prefix
            verify_signatures: Whether to verify block signatures
        """
        self.storage_path = storage_path
        self.use_aws = use_aws
        self.verify_signatures = verify_signatures

        # Initialize backend
        if use_aws:
            if not aws_bucket:
                raise ValueError("AWS bucket name required when use_aws=True")
            self.backend = AWSS3Backend(bucket_name=aws_bucket, prefix=aws_prefix)
        else:
            self.backend = LocalFileBackend(storage_path)

        # Initialize signature verifier
        self.signature_verifier = (
            create_default_verifier() if verify_signatures else None
        )

        # Statistics
        self.stats = {
            "blocks_stored": 0,
            "blocks_retrieved": 0,
            "blocks_updated": 0,
            "blocks_deleted": 0,
        }

    def add_block(
        self,
        block_type: Union[str, BlockType],
        data: bytes,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Add a new block to storage.

        Args:
            block_type: Block type (string or BlockType enum)
            data: Block data
            metadata: Optional metadata

        Returns:
            Block UUID
        """
        # Handle block type
        if isinstance(block_type, BlockType):
            block_type_str = block_type.value
        else:
            block_type_str = block_type

        # Create unified block
        block = UnifiedBlock(
            header=UnifiedBlockHeader(block_type=block_type_str, size=len(data)),
            data=data,
            metadata=metadata or {},
        )

        # Sign block if enabled
        if self.verify_signatures and self.signature_verifier:
            try:
                signature = sign_block_data(
                    self.signature_verifier, data, key_id="default"
                )
                block.signature = signature
            except Exception as e:
                logger.warning(f"Failed to sign block: {e}")

        # Store block
        block_uuid = self.backend.store_block(block)
        self.stats["blocks_stored"] += 1

        return block_uuid

    def get_block(self, block_uuid: str) -> Optional[Tuple[str, bytes, Dict]]:
        """
        Retrieve a block by UUID.

        Args:
            block_uuid: Block UUID

        Returns:
            Tuple of (block_type, data, metadata) or None
        """
        block = self.backend.retrieve_block(block_uuid)
        if not block:
            return None

        self.stats["blocks_retrieved"] += 1

        # Verify signature if enabled
        if self.verify_signatures and self.signature_verifier and block.signature:
            try:
                is_valid = verify_block_signature(
                    self.signature_verifier, block.data, block.signature
                )
                if not is_valid:
                    logger.warning(f"Invalid signature for block {block_uuid}")
                    block.metadata["signature_valid"] = False
                else:
                    block.metadata["signature_valid"] = True
            except Exception as e:
                logger.warning(f"Error verifying signature: {e}")
                block.metadata["signature_valid"] = False

        return block.header.block_type, block.data, block.metadata

    def update_block(
        self, block_uuid: str, data: bytes, metadata: Optional[Dict] = None
    ) -> bool:
        """Update an existing block."""
        # Get current block
        current = self.backend.retrieve_block(block_uuid)
        if not current:
            return False

        # Create updated block
        updated_block = UnifiedBlock(
            header=UnifiedBlockHeader(
                block_type=current.header.block_type,
                uuid=block_uuid,  # Keep same UUID for update
            ),
            data=data,
            metadata=metadata or current.metadata,
        )

        # Update
        success = self.backend.update_block(block_uuid, updated_block)
        if success:
            self.stats["blocks_updated"] += 1

        return success

    def delete_block(self, block_uuid: str) -> bool:
        """Delete a block."""
        success = self.backend.delete_block(block_uuid)
        if success:
            self.stats["blocks_deleted"] += 1
        return success

    def list_blocks(self) -> List[Dict[str, Any]]:
        """List all blocks."""
        headers = self.backend.list_blocks()
        return [header.to_dict() for header in headers]

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "backend": "aws_s3" if self.use_aws else "local_file",
            "total_blocks": len(self.backend.list_blocks()),
            **self.stats,
        }

    def close(self):
        """Close the storage."""
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
