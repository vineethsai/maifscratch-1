"""
MAIF SDK Client - High-performance native interface for MAIF operations.

This client provides the "hot path" for latency-sensitive operations with direct
memory-mapped I/O and optimized block handling as recommended in the decision memo.

Supports AWS backend integration with use_aws=True for seamless cloud storage,
encryption, compliance, and privacy features.
"""

import os
import mmap
import json
import time
import threading
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, BinaryIO, Any, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, asdict

# Import from maif package if available, otherwise use stubs
try:
    from maif.core import MAIFEncoder, MAIFDecoder, MAIFBlock
    from maif.security import SecurityManager
    from maif.compression_manager import CompressionManager
    from maif.privacy import PrivacyEngine, PrivacyPolicy
    MAIF_AVAILABLE = True
except ImportError:
    # Create minimal stubs for testing SDK without full maif package
    MAIF_AVAILABLE = False
    MAIFEncoder = None
    MAIFDecoder = None
    MAIFBlock = None
    SecurityManager = None
    CompressionManager = None
    PrivacyEngine = None
    PrivacyPolicy = None

from .types import ContentType, SecurityLevel, CompressionLevel, ContentMetadata, SecurityOptions, ProcessingOptions
from .artifact import Artifact

# Import AWS backend conditionally
try:
    from .aws_backend import AWSConfig, create_aws_backends
    AWS_BACKEND_AVAILABLE = True
except ImportError:
    AWS_BACKEND_AVAILABLE = False
    AWSConfig = None
    create_aws_backends = None

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for MAIF client."""
    agent_id: str = "default_agent"
    enable_mmap: bool = True
    buffer_size: int = 64 * 1024  # 64KB buffer for write combining
    max_concurrent_writers: int = 4
    enable_compression: bool = True
    default_security_level: SecurityLevel = SecurityLevel.PUBLIC
    cache_embeddings: bool = True
    validate_blocks: bool = True
    use_aws: bool = False
    aws_config: Optional[AWSConfig] = None


class WriteBuffer:
    """Write-combining buffer to coalesce multiple writes into single MAIF blocks."""
    
    def __init__(self, max_size: int = 64 * 1024):
        self.max_size = max_size
        self.buffer = []
        self.current_size = 0
        self.lock = threading.Lock()
    
    def add(self, data: bytes, metadata: Optional[Dict] = None) -> bool:
        """Add data to buffer. Returns True if buffer should be flushed."""
        with self.lock:
            entry_size = len(data)
            if self.current_size + entry_size > self.max_size and self.buffer:
                return True  # Signal flush needed
            
            self.buffer.append({
                'data': data,
                'metadata': metadata or {},
                'timestamp': time.time()
            })
            self.current_size += entry_size
            return False
    
    def flush(self) -> List[Dict]:
        """Flush buffer and return all entries."""
        with self.lock:
            entries = self.buffer.copy()
            self.buffer.clear()
            self.current_size = 0
            return entries


class MAIFClient:
    """
    High-performance MAIF client with direct memory-mapped I/O.
    
    This is the recommended "hot path" for latency-sensitive operations,
    providing native SDK performance without FUSE overhead.
    """
    
    def __init__(self, agent_id: str = "default_agent", use_aws: bool = False,
                 aws_config: Optional[AWSConfig] = None, **kwargs):
        """
        Initialize MAIF client with optional AWS backend support.
        
        Args:
            agent_id: Agent identifier
            use_aws: Enable AWS backend integrations
            aws_config: AWS configuration (uses environment if not provided)
            **kwargs: Additional configuration options
        """
        # Check if MAIF core is available
        if not MAIF_AVAILABLE:
            raise ImportError(
                "MAIF core package is not installed. Please install it to use the SDK.\n"
                "Run: pip install maif"
            )
        
        # Check if AWS backend is requested but not available
        if use_aws and not AWS_BACKEND_AVAILABLE:
            raise ImportError(
                "AWS backend dependencies are not installed. Please install boto3.\n"
                "Run: pip install boto3"
            )
        
        self.config = ClientConfig(
            agent_id=agent_id,
            use_aws=use_aws,
            aws_config=aws_config or (AWSConfig.from_environment() if use_aws else None),
            **kwargs
        )
        
        # Initialize AWS backends if enabled
        if self.config.use_aws:
            self._aws_backends = create_aws_backends(self.config.aws_config)
            
            # Use AWS backends for security, privacy, and storage
            self.security_engine = self._aws_backends.get('security', SecurityManager())
            self.compression_engine = CompressionManager()  # Still use local compression
            self.privacy_engine = self._aws_backends.get('privacy', PrivacyEngine())
            self.compliance_logger = self._aws_backends.get('compliance')
            self.storage_backend = self._aws_backends.get('storage')
            self.block_storage = self._aws_backends.get('block_storage')
            self.streaming_backend = self._aws_backends.get('streaming')
            self.encryption_backend = self._aws_backends.get('encryption')
        else:
            # Use local backends
            self.security_engine = SecurityManager(use_kms=False, require_encryption=False)
            self.compression_engine = CompressionManager()
            self.privacy_engine = PrivacyEngine()
            self.compliance_logger = None
            self.storage_backend = None
            self.block_storage = None
            self.streaming_backend = None
            self.encryption_backend = None
        
        self.write_buffer = WriteBuffer(self.config.buffer_size)
        self._encoders: Dict[str, MAIFEncoder] = {}
        self._decoders: Dict[str, MAIFDecoder] = {}
        self._mmaps: Dict[str, mmap.mmap] = {}
        self._lock = threading.RLock()
    
    def create_artifact(self, name: str, **kwargs) -> Artifact:
        """Create a new artifact with this client's configuration."""
        return Artifact(
            name=name,
            client=self,
            security_level=kwargs.get('security_level', self.config.default_security_level),
            **kwargs
        )
    
    @contextmanager
    def open_file(self, filepath: Union[str, Path], mode: str = 'r'):
        """
        Open a MAIF file with memory mapping for high performance.
        
        Args:
            filepath: Path to MAIF file
            mode: File mode ('r', 'w', 'a')
            
        Yields:
            Encoder/Decoder instance based on mode
        """
        filepath = str(filepath)
        
        try:
            with self._lock:
                if mode in ('w', 'a'):
                    if filepath in self._encoders:
                        yield self._encoders[filepath]
                        return
                    
                    # Create manifest path
                    manifest_path = filepath + '.manifest.json'
                    
                    # For append mode, load existing file if it exists
                    if mode == 'a' and os.path.exists(filepath):
                        encoder = MAIFEncoder(
                            agent_id=self.config.agent_id,
                            existing_maif_path=filepath,
                            existing_manifest_path=manifest_path if os.path.exists(manifest_path) else None,
                            enable_privacy=self.privacy_engine is not None
                        )
                    else:
                        encoder = MAIFEncoder(
                            agent_id=self.config.agent_id,
                            enable_privacy=self.privacy_engine is not None
                        )
                    
                    self._encoders[filepath] = encoder
                    yield encoder
                    
                else:  # mode == 'r'
                    if filepath in self._decoders:
                        yield self._decoders[filepath]
                        return
                    
                    manifest_path = filepath + '.manifest.json'
                    decoder = MAIFDecoder(
                        maif_path=filepath,
                        manifest_path=manifest_path if os.path.exists(manifest_path) else None,
                        privacy_engine=self.privacy_engine if self.privacy_engine else None,
                        requesting_agent=self.config.agent_id
                    )
                    
                    # Enable memory mapping for read operations if configured
                    if self.config.enable_mmap:
                        try:
                            with open(filepath, 'rb') as f:
                                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                                self._mmaps[filepath] = mm
                        except (OSError, ValueError):
                            # Fallback to regular file I/O
                            pass
                    
                    self._decoders[filepath] = decoder
                    yield decoder
            
        finally:
            # Keep file open for potential reuse, cleanup happens in close()
            pass
    
    def write_content(self, filepath: Union[str, Path], content: bytes, 
                     content_type: ContentType = ContentType.DATA,
                     metadata: Optional[ContentMetadata] = None,
                     security_options: Optional[SecurityOptions] = None,
                     processing_options: Optional[ProcessingOptions] = None,
                     flush_immediately: bool = False) -> str:
        """
        Write content to MAIF file with write buffering for performance.
        
        Args:
            filepath: Target MAIF file path
            content: Raw content bytes
            content_type: Type of content
            metadata: Content metadata
            security_options: Security configuration
            processing_options: Processing options
            flush_immediately: Skip buffering and write immediately
            
        Returns:
            Block ID of written content
        """
        filepath = str(filepath)
        
        # Prepare metadata
        meta_dict = {}
        if metadata:
            meta_dict.update(asdict(metadata))
        meta_dict['content_type'] = content_type.value
        meta_dict['agent_id'] = self.config.agent_id
        meta_dict['timestamp'] = time.time()
        
        # Apply security options
        if security_options:
            if security_options.encrypt:
                content = self.security_engine.encrypt_data(content)
                meta_dict['encrypted'] = True
            
            if security_options.sign:
                signature = self.security_engine.sign_data(content)
                meta_dict['signature'] = signature
        
        # Apply processing options
        if processing_options:
            if processing_options.compression != CompressionLevel.NONE:
                content = self.compression_engine.compress(
                    content, 
                    level=processing_options.compression.value
                )
                meta_dict['compressed'] = True
                meta_dict['compression_level'] = processing_options.compression.value
        
        # Use write buffering unless immediate flush requested
        if not flush_immediately:
            should_flush = self.write_buffer.add(content, meta_dict)
            if should_flush:
                self._flush_buffer_to_file(filepath)
        
        # Log to compliance logger if AWS enabled
        if self.config.use_aws and self.compliance_logger:
            try:
                from maif.compliance_logging import LogLevel, LogCategory
                self.compliance_logger.log(
                level=LogLevel.INFO,
                category=LogCategory.DATA,
                user_id=self.config.agent_id,
                action="write_content",
                resource_id=str(filepath),
                details={
                    "content_type": content_type.value,
                    "size": len(content),
                    "encrypted": meta_dict.get('encrypted', False),
                    "compressed": meta_dict.get('compressed', False)
                    }
                )
            except ImportError:
                # Compliance logging module not available
                pass
        
        # Use AWS S3 storage if enabled
        if self.config.use_aws and self.storage_backend:
            # Store in S3
            artifact_id = f"{content_type.value}-{uuid.uuid4()}"
            
            # Apply AWS KMS encryption if available
            if self.encryption_backend and security_options and security_options.encrypt:
                encrypted_data, encryption_metadata = self.encryption_backend.encrypt_with_context(
                    content,
                    context={"artifact_id": artifact_id, "content_type": content_type.value}
                )
                content = encrypted_data
                meta_dict.update(encryption_metadata)
            
            # Upload to S3
            self.storage_backend.upload_artifact(
                artifact_id=artifact_id,
                data=content,
                metadata=meta_dict
            )
            
            # Also save locally for hybrid approach
            with self.open_file(filepath, 'a') as encoder:
                block_id = encoder.add_binary_block(
                    data=b"",  # Don't store data locally, just metadata
                    block_type=content_type.value,
                    metadata={**meta_dict, "s3_artifact_id": artifact_id}
                )
                manifest_path = filepath + '.manifest.json'
                encoder.save(filepath, manifest_path)
            
            return artifact_id
        else:
            # Local storage only
            with self.open_file(filepath, 'a') as encoder:
                block_id = encoder.add_binary_block(
                    data=content,
                    block_type=content_type.value,
                    metadata=meta_dict
                )
                manifest_path = filepath + '.manifest.json'
                encoder.save(filepath, manifest_path)
                return block_id
    
    def _flush_buffer_to_file(self, filepath: str):
        """Flush write buffer to file as a single operation."""
        entries = self.write_buffer.flush()
        if not entries:
            return
        
        with self.open_file(filepath, 'a') as encoder:
            for entry in entries:
                encoder.add_binary_block(
                    data=entry['data'],
                    block_type=entry['metadata'].get('content_type', 'data'),
                    metadata=entry['metadata']
                )
            manifest_path = filepath + '.manifest.json'
            encoder.save(filepath, manifest_path)
    
    def read_content(self, filepath: Union[str, Path],
                    block_id: Optional[str] = None,
                    content_type: Optional[ContentType] = None) -> Iterator[Dict]:
        """
        Read content from MAIF file with memory-mapped access for performance.
        Supports reading from AWS S3 when use_aws=True.
        
        Args:
            filepath: MAIF file path
            block_id: Specific block ID to read (optional)
            content_type: Filter by content type (optional)
            
        Yields:
            Dictionary with block data and metadata
        """
        # Log read access if AWS enabled
        if self.config.use_aws and self.compliance_logger:
            try:
                from maif.compliance_logging import LogLevel, LogCategory
                self.compliance_logger.log(
                level=LogLevel.INFO,
                category=LogCategory.ACCESS,
                user_id=self.config.agent_id,
                action="read_content",
                resource_id=str(filepath),
                details={
                    "block_id": block_id,
                    "content_type": content_type.value if content_type else None
                    }
                )
            except ImportError:
                # Compliance logging module not available
                pass
        
        with self.open_file(filepath, 'r') as decoder:
            for block in decoder.blocks:
                # Apply filters
                if block_id and block.block_id != block_id:
                    continue
                if content_type and block.block_type != content_type.value:
                    continue
                
                # Check if data is stored in S3
                if self.config.use_aws and block.metadata and 's3_artifact_id' in block.metadata:
                    # Retrieve from S3
                    s3_artifact_id = block.metadata['s3_artifact_id']
                    
                    if self.storage_backend:
                        try:
                            data = self.storage_backend.download_artifact(s3_artifact_id)
                            
                            # Decrypt with AWS KMS if needed
                            if self.encryption_backend and block.metadata.get('kms_encrypted'):
                                data = self.encryption_backend.decrypt_with_context(
                                    data,
                                    block.metadata
                                )
                        except Exception as e:
                            # Fallback to local data if S3 fails
                            logger.warning(f"Failed to retrieve from S3: {e}, using local data")
                            data = block.data
                    else:
                        data = block.data
                else:
                    # Use local data
                    data = block.data
                
                # Decrypt if needed (local encryption)
                if block.metadata and block.metadata.get('encrypted') and not block.metadata.get('kms_encrypted'):
                    data = self.security_engine.decrypt_data(data)
                
                # Decompress if needed
                if block.metadata and block.metadata.get('compressed'):
                    data = self.compression_engine.decompress(data)
                
                # Apply privacy policy if AWS Macie is enabled
                if self.config.use_aws and self.privacy_engine and hasattr(self.privacy_engine, 'classify_data'):
                    privacy_level = self.privacy_engine.classify_data(data, block.block_id)
                    block.metadata['privacy_level'] = privacy_level.value
                
                yield {
                    'block_id': block.block_id,
                    'content_type': block.block_type,
                    'data': data,
                    'metadata': block.metadata or {},
                    'size': len(data),
                    'hash': block.hash_value
                }
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict:
        """Get information about a MAIF file."""
        with self.open_file(filepath, 'r') as decoder:
            return {
                'filepath': str(filepath),
                'total_blocks': len(decoder.blocks),
                'file_size': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
                'content_types': list(set(block.block_type for block in decoder.blocks)),
                'agents': list(set(
                    block.metadata.get('agent_id', 'unknown')
                    for block in decoder.blocks
                    if block.metadata
                )),
                'created': min(
                    (block.metadata.get('timestamp', 0) for block in decoder.blocks if block.metadata),
                    default=0
                ),
                'modified': max(
                    (block.metadata.get('timestamp', 0) for block in decoder.blocks if block.metadata),
                    default=0
                )
            }
    
    def flush_all_buffers(self):
        """Flush all pending write buffers."""
        flush_results = {
            'buffers_flushed': 0,
            'bytes_written': 0,
            'errors': []
        }
        
        with self._lock:
            # 1. Flush the main write buffer
            if self.write_buffer.buffer:
                try:
                    entries = self.write_buffer.flush()
                    if entries:
                        # Process buffered entries
                        for entry in entries:
                            # Find or create an encoder for temporary storage
                            temp_file = f"/tmp/maif_buffer_{uuid.uuid4().hex}.tmp"
                            encoder = MAIFEncoder(agent_id=self.config.agent_id)
                            
                            # Add buffered data as blocks
                            encoder.add_binary_block(
                                entry['data'],
                                content_type="application/octet-stream",
                                metadata=entry['metadata']
                            )
                            
                            # Write to temporary file
                            encoder.save(temp_file)
                            flush_results['bytes_written'] += len(entry['data'])
                            
                            # Clean up
                            try:
                                os.unlink(temp_file)
                            except (OSError, IOError):
                                pass  # File already deleted or inaccessible
                                
                        flush_results['buffers_flushed'] += 1
                except Exception as e:
                    flush_results['errors'].append(f"Write buffer flush error: {str(e)}")
            
            # 2. Flush all active encoders
            for filepath, encoder in list(self._encoders.items()):
                try:
                    if hasattr(encoder, 'flush') and callable(encoder.flush):
                        encoder.flush()
                    elif hasattr(encoder, 'save'):
                        # Force save to ensure data is written
                        encoder.save(filepath)
                    flush_results['buffers_flushed'] += 1
                except Exception as e:
                    flush_results['errors'].append(f"Encoder flush error for {filepath}: {str(e)}")
            
            # 3. Sync memory-mapped files
            for filepath, mm in list(self._mmaps.items()):
                try:
                    if mm and not mm.closed:
                        mm.flush()  # Flush to OS buffers
                        if hasattr(mm, 'fileno'):
                            os.fsync(mm.fileno())  # Force write to disk
                    flush_results['buffers_flushed'] += 1
                except Exception as e:
                    flush_results['errors'].append(f"Memory map flush error for {filepath}: {str(e)}")
            
            # 4. Flush AWS multipart uploads if using AWS backend
            if self.config.use_aws and self.storage_backend:
                try:
                    if hasattr(self.storage_backend, 'flush_all_uploads'):
                        aws_results = self.storage_backend.flush_all_uploads()
                        flush_results['buffers_flushed'] += aws_results.get('uploads_completed', 0)
                        flush_results['bytes_written'] += aws_results.get('bytes_uploaded', 0)
                    elif hasattr(self.storage_backend, '_multipart_uploads'):
                        # Handle S3 multipart uploads
                        for upload_id, upload_info in list(self.storage_backend._multipart_uploads.items()):
                            try:
                                self.storage_backend._complete_multipart_upload(
                                    upload_info['Bucket'],
                                    upload_info['Key'],
                                    upload_id,
                                    upload_info.get('Parts', [])
                                )
                                flush_results['buffers_flushed'] += 1
                            except Exception as e:
                                flush_results['errors'].append(f"S3 upload completion error: {str(e)}")
                except Exception as e:
                    flush_results['errors'].append(f"AWS backend flush error: {str(e)}")
            
            # 5. Flush streaming backend if available
            if self.streaming_backend and hasattr(self.streaming_backend, 'flush'):
                try:
                    self.streaming_backend.flush()
                    flush_results['buffers_flushed'] += 1
                except Exception as e:
                    flush_results['errors'].append(f"Streaming backend flush error: {str(e)}")
            
            # 6. Track buffer management
            if not hasattr(self, '_buffer_tracking'):
                self._buffer_tracking = {
                    'last_flush': time.time(),
                    'total_flushes': 0,
                    'total_bytes_flushed': 0
                }
            
            self._buffer_tracking['last_flush'] = time.time()
            self._buffer_tracking['total_flushes'] += 1
            self._buffer_tracking['total_bytes_flushed'] += flush_results['bytes_written']
        
        # Log results
        if flush_results['errors']:
            logger.warning(
                f"Buffer flush completed with errors: {len(flush_results['errors'])} errors, "
                f"{flush_results['buffers_flushed']} buffers flushed, "
                f"{flush_results['bytes_written']} bytes written"
            )
        else:
            logger.debug(
                f"Buffer flush completed: {flush_results['buffers_flushed']} buffers flushed, "
                f"{flush_results['bytes_written']} bytes written"
            )
        
        return flush_results
    
    def close(self):
        """Close all open files and clean up resources."""
        with self._lock:
            # Close memory maps
            for mm in self._mmaps.values():
                try:
                    mm.close()
                except Exception as e:
                    logger.debug(f"Failed to close memory map: {e}")
            self._mmaps.clear()
            
            # Close encoders
            for encoder in self._encoders.values():
                try:
                    if hasattr(encoder, 'close'):
                        encoder.close()
                except Exception as e:
                    logger.debug(f"Failed to close encoder: {e}")
            self._encoders.clear()
            
            # Decoders don't need explicit closing
            self._decoders.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience functions for quick operations
def quick_write(filepath: Union[str, Path], content: bytes, 
               content_type: ContentType = ContentType.DATA, **kwargs) -> str:
    """Quick write operation using default client."""
    with MAIFClient() as client:
        return client.write_content(filepath, content, content_type, **kwargs)


def quick_read(filepath: Union[str, Path], **kwargs) -> List[Dict]:
    """Quick read operation using default client."""
    with MAIFClient() as client:
        return list(client.read_content(filepath, **kwargs))


# Stub implementations for when MAIF core is not available
if not MAIF_AVAILABLE:
    class MAIFEncoderStub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MAIF core package is required. Install with: pip install maif")
    
    class MAIFDecoderStub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MAIF core package is required. Install with: pip install maif")
    
    class SecurityManagerStub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MAIF core package is required. Install with: pip install maif")
    
    class CompressionManagerStub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MAIF core package is required. Install with: pip install maif")
    
    class PrivacyEngineStub:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("MAIF core package is required. Install with: pip install maif")
    
    # Replace None values with stub classes
    MAIFEncoder = MAIFEncoder or MAIFEncoderStub
    MAIFDecoder = MAIFDecoder or MAIFDecoderStub
    SecurityManager = SecurityManager or SecurityManagerStub
    CompressionManager = CompressionManager or CompressionManagerStub
    PrivacyEngine = PrivacyEngine or PrivacyEngineStub