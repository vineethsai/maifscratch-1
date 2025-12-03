"""
Cross-Backend Compatibility Demo
=================================

This example demonstrates how MAIF data can be seamlessly moved between
local file storage and AWS S3 storage using the unified format.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif.core import MAIFEncoder, MAIFParser
from maif.unified_storage import UnifiedStorage
from maif.migration_utils import MAIFMigrator, migrate_to_aws, migrate_from_aws
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_maif_local(file_path: str):
    """Create a sample MAIF file using local storage."""
    logger.info("Creating sample MAIF file with local storage")
    
    # Create encoder with unified format
    encoder = MAIFEncoder(
        agent_id="demo-agent",
        existing_maif_path=None,
        existing_manifest_path=None,
        enable_privacy=True,
        use_aws=False  # Local storage
    )
    
    # Add text blocks
    encoder.add_text_block(
        "This is a demonstration of cross-backend compatibility in MAIF.",
        metadata={"source": "local", "demo": True}
    )
    
    encoder.add_text_block(
        "MAIF can seamlessly move data between local files and AWS S3.",
        metadata={"feature": "portability"}
    )
    
    # Add embeddings
    embeddings = np.random.rand(10, 384).tolist()  # 10 embeddings of dimension 384
    encoder.add_embeddings_block(
        embeddings,
        metadata={"model": "demo-embedder", "dimension": 384}
    )
    
    # Add binary data
    binary_data = b"Binary content that works across backends"
    encoder.add_binary_block(
        binary_data,
        block_type="BDAT",
        metadata={"content_type": "application/octet-stream"}
    )
    
    # Save the file
    manifest_path = file_path.replace('.maif', '_manifest.json')
    encoder.save(file_path, manifest_path)
    
    logger.info(f"Created local MAIF file: {file_path}")
    return file_path, manifest_path


def demonstrate_aws_storage(aws_bucket: str = None):
    """Demonstrate storing MAIF data directly to AWS S3."""
    if not aws_bucket:
        logger.warning("AWS bucket not specified. Skipping AWS demo.")
        return
    
    logger.info("Creating MAIF data directly in AWS S3")
    
    # Create encoder with AWS backend
    encoder = MAIFEncoder(
        agent_id="aws-demo-agent",
        use_aws=True,
        aws_bucket=aws_bucket,
        aws_prefix="maif-demo/",
        enable_privacy=True
    )
    
    # Add content
    encoder.add_text_block(
        "This MAIF data is stored directly in AWS S3.",
        metadata={"storage": "s3", "direct": True}
    )
    
    encoder.add_embeddings_block(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadata={"test": True}
    )
    
    # Save to S3
    encoder.save("maif-demo/demo.maif", "")
    
    logger.info("Created MAIF data in AWS S3")


def demonstrate_migration(local_path: str, manifest_path: str, aws_bucket: str = None):
    """Demonstrate migrating MAIF data between backends."""
    if not aws_bucket:
        logger.warning("AWS bucket not specified. Skipping migration demo.")
        return
    
    logger.info("=== Migration Demo ===")
    
    # Create migrator
    migrator = MAIFMigrator()
    
    # 1. Migrate from local to AWS
    logger.info("Migrating from local file to AWS S3...")
    stats = migrate_to_aws(
        source_path=local_path,
        aws_bucket=aws_bucket,
        aws_prefix="migrated/",
        source_manifest_path=manifest_path
    )
    logger.info(f"Migration stats: {stats}")
    
    # 2. Verify the migration
    logger.info("Verifying migration integrity...")
    verification = migrator.verify_migration(
        source_path=local_path,
        target_path="migrated/",
        source_manifest_path=manifest_path,
        target_is_aws=True,
        target_bucket=aws_bucket
    )
    logger.info(f"Verification results: Success={verification['success']}, "
                f"Matched={verification['blocks_matched']}, "
                f"Mismatched={verification['blocks_mismatched']}")
    
    # 3. Migrate back from AWS to local
    with tempfile.NamedTemporaryFile(suffix='.maif', delete=False) as tmp:
        restored_path = tmp.name
    
    logger.info("Migrating back from AWS S3 to local file...")
    stats = migrate_from_aws(
        aws_bucket=aws_bucket,
        aws_prefix="migrated/",
        target_path=restored_path
    )
    logger.info(f"Reverse migration stats: {stats}")
    
    # 4. Parse and compare
    logger.info("Comparing original and restored data...")
    
    # Parse original
    original_parser = MAIFParser(local_path, manifest_path)
    original_content = original_parser.extract_content()
    
    # Parse restored (no manifest needed for unified format)
    restored_parser = MAIFParser(restored_path, "", use_aws=False)
    restored_content = restored_parser.extract_content()
    
    # Compare block counts
    original_blocks = len(original_content.get('text_blocks', []))
    restored_blocks = len(restored_content.get('text_blocks', []))
    
    logger.info(f"Original blocks: {original_blocks}, Restored blocks: {restored_blocks}")
    
    # Clean up
    os.unlink(restored_path)
    
    return verification['success']


def demonstrate_unified_storage():
    """Demonstrate unified storage abstraction."""
    logger.info("=== Unified Storage Demo ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = os.path.join(tmpdir, "unified.maif")
        
        # Create unified storage (local)
        storage = UnifiedStorage(
            storage_path=storage_path,
            use_aws=False,
            verify_signatures=True
        )
        
        # Store some blocks
        from maif.unified_block_format import UnifiedBlock, UnifiedBlockHeader
        import time
        import uuid
        
        # Create and store text block
        text_data = b"Unified storage works seamlessly"
        text_block = UnifiedBlock(
            header=UnifiedBlockHeader(
                magic=b'MAIF',
                version=1,
                size=len(text_data),
                block_type="TEXT",
                uuid=str(uuid.uuid4()),
                timestamp=time.time(),
                previous_hash=None,
                block_hash=None,
                flags=0,
                metadata_size=0,
                reserved=b'\x00' * 28
            ),
            data=text_data,
            metadata={"demo": True}
        )
        
        # Store block using backend directly
        block_id = storage.backend.store_block(text_block)
        logger.info(f"Stored block with ID: {block_id}")
        
        # Retrieve the block
        retrieved = storage.backend.retrieve_block(block_id)
        if retrieved:
            logger.info(f"Retrieved block type: {retrieved.header.block_type}")
            logger.info(f"Retrieved data: {retrieved.data.decode('utf-8')}")
        
        # List blocks
        blocks = storage.backend.list_blocks()
        logger.info(f"Total blocks in storage: {len(blocks)}")


def main():
    """Run the cross-backend compatibility demo."""
    print("=" * 60)
    print("MAIF Cross-Backend Compatibility Demo")
    print("=" * 60)
    
    # Get AWS configuration from environment
    aws_bucket = os.environ.get('MAIF_TEST_BUCKET')
    if not aws_bucket:
        print("\nNote: Set MAIF_TEST_BUCKET environment variable to test AWS features")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create local MAIF file
        local_path = os.path.join(tmpdir, "demo.maif")
        maif_path, manifest_path = create_sample_maif_local(local_path)
        
        # 2. Demonstrate unified storage
        demonstrate_unified_storage()
        
        # 3. Demonstrate AWS storage (if configured)
        if aws_bucket:
            demonstrate_aws_storage(aws_bucket)
            
            # 4. Demonstrate migration
            success = demonstrate_migration(maif_path, manifest_path, aws_bucket)
            
            if success:
                print("\n✅ Cross-backend compatibility test PASSED!")
            else:
                print("\n❌ Cross-backend compatibility test FAILED!")
        else:
            print("\n⚠️  AWS tests skipped (no bucket configured)")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()