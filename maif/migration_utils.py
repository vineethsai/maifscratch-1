"""
MAIF Migration Utilities
========================

Utilities for migrating between different MAIF format versions and storage backends.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .core import MAIFDecoder, MAIFEncoder, MAIFParser
from .unified_storage import UnifiedStorage
from .unified_block_format import UnifiedBlock, UnifiedBlockHeader, BlockFormatConverter
from .block_storage import BlockStorage
from .aws_s3_block_storage import S3BlockStorage

logger = logging.getLogger(__name__)


class MAIFMigrationError(Exception):
    """Base exception for MAIF migration errors."""

    pass


class MAIFMigrator:
    """Handles migration between different MAIF formats and storage backends."""

    def __init__(self):
        self.converter = BlockFormatConverter()
        self.stats = {"blocks_migrated": 0, "errors": 0, "warnings": 0}

    def migrate_file_to_unified(
        self,
        source_maif_path: str,
        target_path: str,
        use_aws: bool = False,
        aws_bucket: Optional[str] = None,
        aws_prefix: str = "maif/",
    ) -> Dict[str, any]:
        """
        Migrate a MAIF file to unified format (v3).

        Args:
            source_maif_path: Path to source MAIF file
            target_path: Target path (file path or S3 prefix)
            use_aws: Whether to use AWS S3 storage
            aws_bucket: S3 bucket name (required if use_aws=True)
            aws_prefix: S3 key prefix

        Returns:
            Migration statistics
        """
        logger.info(f"Starting migration from {source_maif_path} to unified format")

        try:
            # Read source MAIF file (v3 format - self-contained)
            decoder = MAIFDecoder(source_maif_path)
            decoder.load()

            # Create unified storage
            unified_storage = UnifiedStorage(
                storage_path=target_path,
                use_aws=use_aws,
                aws_bucket=aws_bucket,
                aws_prefix=aws_prefix,
                verify_signatures=True,
            )

            # Migrate each block
            for block in decoder.blocks:
                try:
                    # Get block data
                    block_data = decoder._read_block_data(block)

                    # Create unified block
                    unified_block = UnifiedBlock(
                        header=UnifiedBlockHeader(
                            magic=b"MAIF",
                            version=1,
                            size=block.size,
                            block_type=block.block_type,
                            uuid=block.block_id,
                            timestamp=decoder.manifest.get("created", 0),
                            previous_hash=block.previous_hash,
                            block_hash=block.hash_value,
                            flags=0,
                            metadata_size=len(json.dumps(block.metadata or {})),
                            reserved=b"\x00" * 28,
                        ),
                        data=block_data,
                        metadata=block.metadata,
                    )

                    # Store in unified storage
                    unified_storage.store_block(unified_block)
                    self.stats["blocks_migrated"] += 1

                    logger.debug(
                        f"Migrated block {block.block_id} ({block.block_type})"
                    )

                except Exception as e:
                    logger.error(f"Failed to migrate block {block.block_id}: {e}")
                    self.stats["errors"] += 1

            # Store manifest
            manifest = decoder.manifest.copy()
            manifest["format_version"] = "unified-1.0"
            manifest["migration_info"] = {
                "source_path": source_maif_path,
                "migration_time": time.time(),
                "blocks_migrated": self.stats["blocks_migrated"],
            }

            unified_storage.store_metadata("manifest", manifest)

            # Store version history if available
            if hasattr(decoder, "version_history") and decoder.version_history:
                unified_storage.store_metadata(
                    "version_history", decoder.version_history
                )

            logger.info(
                f"Migration completed: {self.stats['blocks_migrated']} blocks migrated"
            )

            return self.stats

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise MAIFMigrationError(f"Failed to migrate MAIF file: {e}")

    def migrate_between_backends(
        self,
        source_path: str,
        target_path: str,
        source_is_aws: bool = False,
        target_is_aws: bool = False,
        source_bucket: Optional[str] = None,
        target_bucket: Optional[str] = None,
        source_prefix: str = "maif/",
        target_prefix: str = "maif/",
    ) -> Dict[str, any]:
        """
        Migrate MAIF data between different storage backends.

        Args:
            source_path: Source path (file path or S3 prefix)
            target_path: Target path (file path or S3 prefix)
            source_is_aws: Whether source is AWS S3
            target_is_aws: Whether target is AWS S3
            source_bucket: Source S3 bucket (required if source_is_aws=True)
            target_bucket: Target S3 bucket (required if target_is_aws=True)
            source_prefix: Source S3 prefix
            target_prefix: Target S3 prefix

        Returns:
            Migration statistics
        """
        logger.info(
            f"Migrating between backends: {'AWS' if source_is_aws else 'Local'} -> {'AWS' if target_is_aws else 'Local'}"
        )

        try:
            # Create source storage
            source_storage = UnifiedStorage(
                storage_path=source_path,
                use_aws=source_is_aws,
                aws_bucket=source_bucket,
                aws_prefix=source_prefix,
                verify_signatures=True,
            )

            # Create target storage
            target_storage = UnifiedStorage(
                storage_path=target_path,
                use_aws=target_is_aws,
                aws_bucket=target_bucket,
                aws_prefix=target_prefix,
                verify_signatures=True,
            )

            # List all blocks in source
            block_ids = source_storage.list_blocks()

            # Migrate each block
            for block_id in block_ids:
                try:
                    # Retrieve from source
                    block = source_storage.retrieve_block(block_id)

                    if block:
                        # Store in target
                        target_storage.store_block(block)
                        self.stats["blocks_migrated"] += 1
                        logger.debug(f"Migrated block {block_id}")
                    else:
                        logger.warning(f"Could not retrieve block {block_id}")
                        self.stats["warnings"] += 1

                except Exception as e:
                    logger.error(f"Failed to migrate block {block_id}: {e}")
                    self.stats["errors"] += 1

            # Migrate metadata
            try:
                # Manifest
                manifest = source_storage.get_metadata("manifest")
                if manifest:
                    manifest["backend_migration"] = {
                        "from": "aws" if source_is_aws else "local",
                        "to": "aws" if target_is_aws else "local",
                        "migration_time": time.time(),
                    }
                    target_storage.store_metadata("manifest", manifest)

                # Version history
                version_history = source_storage.get_metadata("version_history")
                if version_history:
                    target_storage.store_metadata("version_history", version_history)

            except Exception as e:
                logger.warning(f"Failed to migrate metadata: {e}")
                self.stats["warnings"] += 1

            logger.info(
                f"Backend migration completed: {self.stats['blocks_migrated']} blocks migrated"
            )

            return self.stats

        except Exception as e:
            logger.error(f"Backend migration failed: {e}")
            raise MAIFMigrationError(f"Failed to migrate between backends: {e}")

    def verify_migration(
        self,
        source_path: str,
        target_path: str,
        source_is_aws: bool = False,
        target_is_aws: bool = False,
        source_bucket: Optional[str] = None,
        target_bucket: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Verify that migration was successful by comparing source and target.

        Returns:
            Verification results
        """
        logger.info("Verifying migration integrity")

        results = {
            "success": True,
            "blocks_matched": 0,
            "blocks_mismatched": 0,
            "missing_blocks": [],
            "hash_mismatches": [],
        }

        try:
            # For MAIF source files (v3 format)
            if not source_is_aws and os.path.exists(source_path):
                decoder = MAIFDecoder(source_path)
                decoder.load()
                source_blocks = {
                    block.header.block_id: block for block in decoder.blocks
                }
            else:
                # Unified source
                source_storage = UnifiedStorage(
                    storage_path=source_path,
                    use_aws=source_is_aws,
                    aws_bucket=source_bucket,
                    verify_signatures=True,
                )
                source_blocks = {}
                for block_id in source_storage.list_blocks():
                    block = source_storage.retrieve_block(block_id)
                    if block:
                        source_blocks[block_id] = block

            # Target storage (always unified)
            target_storage = UnifiedStorage(
                storage_path=target_path,
                use_aws=target_is_aws,
                aws_bucket=target_bucket,
                verify_signatures=True,
            )

            # Compare blocks
            for block_id, source_block in source_blocks.items():
                target_block = target_storage.retrieve_block(block_id)

                if not target_block:
                    results["missing_blocks"].append(block_id)
                    results["success"] = False
                else:
                    # Compare hashes
                    source_hash = (
                        source_block.hash_value
                        if hasattr(source_block, "hash_value")
                        else source_block.header.block_hash
                    )
                    target_hash = target_block.header.block_hash

                    if source_hash == target_hash:
                        results["blocks_matched"] += 1
                    else:
                        results["blocks_mismatched"] += 1
                        results["hash_mismatches"].append(
                            {
                                "block_id": block_id,
                                "source_hash": source_hash,
                                "target_hash": target_hash,
                            }
                        )
                        results["success"] = False

            logger.info(
                f"Verification completed: {results['blocks_matched']} blocks matched, "
                f"{results['blocks_mismatched']} mismatched, {len(results['missing_blocks'])} missing"
            )

            return results

        except Exception as e:
            logger.error(f"Verification failed: {e}")
            results["success"] = False
            results["error"] = str(e)
            return results


# Convenience functions
def migrate_to_unified(
    source_maif: str, source_manifest: str, target_path: str, **kwargs
) -> Dict:
    """Convenience function to migrate a legacy MAIF file to unified format."""
    migrator = MAIFMigrator()
    return migrator.migrate_file_to_unified(
        source_maif, source_manifest, target_path, **kwargs
    )


def migrate_to_aws(
    source_path: str, aws_bucket: str, aws_prefix: str = "maif/", **kwargs
) -> Dict:
    """Convenience function to migrate MAIF data to AWS S3."""
    migrator = MAIFMigrator()
    return migrator.migrate_between_backends(
        source_path=source_path,
        target_path=aws_prefix,
        source_is_aws=False,
        target_is_aws=True,
        target_bucket=aws_bucket,
        target_prefix=aws_prefix,
        **kwargs,
    )


def migrate_from_aws(
    aws_bucket: str, aws_prefix: str, target_path: str, **kwargs
) -> Dict:
    """Convenience function to migrate MAIF data from AWS S3 to local storage."""
    migrator = MAIFMigrator()
    return migrator.migrate_between_backends(
        source_path=aws_prefix,
        target_path=target_path,
        source_is_aws=True,
        target_is_aws=False,
        source_bucket=aws_bucket,
        source_prefix=aws_prefix,
        **kwargs,
    )


# Add missing import
import time
