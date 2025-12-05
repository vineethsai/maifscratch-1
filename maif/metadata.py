"""
MAIF metadata management and standards compliance.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class MAIFVersion(Enum):
    """MAIF format versions."""

    V1_0 = "1.0"
    V1_1 = "1.1"
    V2_0 = "2.0"


class ContentType(Enum):
    """Standard content types for MAIF blocks."""

    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    EMBEDDING = "embedding"
    EMBEDDINGS = "embeddings"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    CROSS_MODAL = "cross_modal"


class CompressionType(Enum):
    """Supported compression algorithms."""

    NONE = "none"
    ZLIB = "zlib"
    GZIP = "gzip"
    LZMA = "lzma"
    BROTLI = "brotli"
    CUSTOM = "custom"


@dataclass
class MAIFHeader:
    """MAIF file header metadata."""

    magic: str = "MAIF"
    version: str = MAIFVersion.V2_0.value
    created: str = ""
    modified: str = ""
    file_id: str = ""
    creator_agent: str = ""
    format_flags: int = 0
    block_count: int = 0
    total_size: int = 0
    checksum: str = ""
    # Test compatibility fields
    timestamp: Optional[float] = None
    compression: Optional[CompressionType] = None
    encryption_enabled: bool = False

    def __post_init__(self):
        # Validate and fix timestamp
        if self.timestamp is None or self.timestamp < 0:
            # Set to current time if None or invalid
            import time

            self.timestamp = time.time()
            if not self.created:
                self.created = str(self.timestamp)

        if self.compression is None:
            self.compression = CompressionType.NONE
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if not self.modified:
            self.modified = self.created
        if not self.file_id:
            self.file_id = str(uuid.uuid4())


@dataclass
class BlockMetadata:
    """Metadata for individual MAIF blocks."""

    block_id: str
    content_type: str
    size: int
    hash: str = ""
    block_type: str = ""
    offset: int = 0
    checksum: str = ""
    compression: str = CompressionType.NONE.value
    encryption: Optional[str] = None
    encrypted: bool = False
    created: str = ""
    created_at: Optional[float] = None
    agent_id: str = ""
    version: int = 1
    parent_block: Optional[str] = None
    dependencies: List[str] = None
    tags: List[str] = None
    custom_metadata: Dict[str, Any] = None

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now(timezone.utc).isoformat()
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}
        # Validate and fix size
        if self.size < 0:
            self.size = 0


@dataclass
class ProvenanceRecord:
    """Provenance tracking for MAIF operations."""

    operation_type: str
    agent_id: str
    timestamp: float = None
    block_id: str = ""
    previous_hash: str = ""
    operation_hash: str = ""
    operation_id: str = ""
    block_ids: List[str] = None
    operation_data: Dict[str, Any] = None
    signature: Optional[str] = None

    def __post_init__(self):
        if not self.operation_id:
            self.operation_id = str(uuid.uuid4())
        if not self.timestamp or self.timestamp < 0:
            import time

            self.timestamp = time.time()
        if self.block_ids is None:
            self.block_ids = []
        if self.operation_data is None:
            self.operation_data = {}


class MAIFMetadataManager:
    """Manages MAIF file metadata and standards compliance."""

    def __init__(self, version: str = MAIFVersion.V2_0.value):
        self.version = version
        self.header = MAIFHeader(version=version)
        self.blocks: Dict[str, BlockMetadata] = {}
        self.provenance: List[ProvenanceRecord] = []
        self.custom_schemas: Dict[str, Dict] = {}
        self.dependencies: Dict[str, List[str]] = {}

    def create_header(self, creator_agent: str, **kwargs) -> MAIFHeader:
        """Create a new MAIF header."""
        self.header = MAIFHeader(
            version=self.version, creator_agent=creator_agent, **kwargs
        )
        return self.header

    def add_block_metadata(
        self,
        block_id: str,
        content_type=None,
        block_type: str = None,
        size: int = None,
        offset: int = None,
        checksum: str = None,
        hash: str = None,
        compression=None,
        **kwargs,
    ) -> bool:
        """Add metadata for a new block."""
        # Handle different parameter names for backward compatibility
        if isinstance(content_type, str) and hasattr(ContentType, content_type.upper()):
            content_type_enum = getattr(ContentType, content_type.upper())
        elif hasattr(content_type, "value"):
            content_type_enum = content_type
        else:
            content_type_enum = ContentType.TEXT

        # Handle compression parameter
        compression_type = CompressionType.NONE
        if compression is not None:
            if isinstance(compression, str):
                # Try to match string to enum
                for comp_type in CompressionType:
                    if comp_type.value.lower() == compression.lower():
                        compression_type = comp_type
                        break
            elif hasattr(compression, "value"):
                compression_type = compression

        metadata = BlockMetadata(
            block_id=block_id,
            block_type=block_type or "unknown",
            content_type=content_type_enum,
            size=size or 0,
            offset=offset or 0,
            hash=checksum or hash or "",
            compression=compression_type.value,  # Store as string value
            **kwargs,
        )
        self.blocks[block_id] = metadata
        self.header.block_count = len(self.blocks)
        return True

    def update_block_metadata(self, block_id: str, **updates) -> bool:
        """Update existing block metadata."""
        if block_id not in self.blocks:
            return False

        metadata = self.blocks[block_id]
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        self.header.modified = datetime.now(timezone.utc).isoformat()
        return True

    def add_provenance_record(
        self,
        operation_type: str,
        agent_id: str,
        block_ids: Optional[List[str]] = None,
        operation_data: Optional[Dict[str, Any]] = None,
        block_id: Optional[str] = None,
        operation_hash: Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Add a provenance record."""
        # Handle backward compatibility
        if block_ids is None and block_id is not None:
            block_ids = [block_id]
        elif block_ids is None:
            block_ids = []

        if operation_data is None:
            operation_data = {}
            if operation_hash is not None:
                operation_data["operation_hash"] = operation_hash

        record = ProvenanceRecord(
            operation_type=operation_type,
            agent_id=agent_id,
            block_id=block_id
            or (block_ids[0] if block_ids else ""),  # For test compatibility
            block_ids=block_ids,
            operation_data=operation_data,
            operation_hash=operation_hash or "",  # Set operation_hash directly
            **kwargs,
        )
        self.provenance.append(record)
        return True

    def get_block_dependencies(self, block_id: str) -> List[str]:
        """Get all dependencies for a block."""
        if block_id not in self.blocks:
            return []

        dependencies = []
        metadata = self.blocks[block_id]

        # Direct dependencies
        dependencies.extend(metadata.dependencies)

        # Parent block dependency
        if metadata.parent_block:
            dependencies.append(metadata.parent_block)

        return list(set(dependencies))

    def get_block_dependents(self, block_id: str) -> List[str]:
        """Get all blocks that depend on this block."""
        dependents = []

        for bid, metadata in self.blocks.items():
            if block_id in metadata.dependencies or metadata.parent_block == block_id:
                dependents.append(bid)

        return dependents

    def validate_dependencies(self) -> List[str]:
        """Validate all block dependencies."""
        errors = []

        for block_id, metadata in self.blocks.items():
            # Check if dependencies exist
            for dep_id in metadata.dependencies:
                if dep_id not in self.blocks:
                    errors.append(
                        f"Block {block_id} depends on non-existent block {dep_id}"
                    )

            # Check parent block
            if metadata.parent_block and metadata.parent_block not in self.blocks:
                errors.append(
                    f"Block {block_id} has non-existent parent {metadata.parent_block}"
                )

            # Check for circular dependencies
            if self._has_circular_dependency(block_id):
                errors.append(f"Block {block_id} has circular dependency")

        return errors

    def _has_circular_dependency(self, block_id: str, visited: set = None) -> bool:
        """Check for circular dependencies."""
        if visited is None:
            visited = set()

        if block_id in visited:
            return True

        if block_id not in self.blocks:
            return False

        visited.add(block_id)
        metadata = self.blocks[block_id]

        # Check dependencies
        for dep_id in metadata.dependencies:
            if self._has_circular_dependency(dep_id, visited.copy()):
                return True

        # Check parent
        if metadata.parent_block:
            if self._has_circular_dependency(metadata.parent_block, visited.copy()):
                return True

        return False

    def get_metadata_summary(self) -> Dict[str, Any]:
        """Get a summary of all metadata."""
        content_types = {}
        compression_types = {}
        total_size = 0

        for metadata in self.blocks.values():
            # Count content types (use string values for test compatibility)
            content_type_str = (
                metadata.content_type.value
                if hasattr(metadata.content_type, "value")
                else str(metadata.content_type)
            )
            content_types[content_type_str] = content_types.get(content_type_str, 0) + 1

            # Count compression types
            compression_type_str = (
                metadata.compression.value
                if hasattr(metadata.compression, "value")
                else str(metadata.compression)
            )
            compression_types[compression_type_str] = (
                compression_types.get(compression_type_str, 0) + 1
            )

            # Sum sizes
            total_size += metadata.size

        return {
            "total_blocks": len(self.blocks),
            "total_size": total_size,
            "content_types": content_types,
            "compression_types": compression_types,
            "provenance_records": len(self.provenance),
            "header": asdict(self.header),
            "dependency_errors": self.validate_dependencies(),
        }

    def export_manifest(self) -> Dict[str, Any]:
        """Export complete metadata as manifest."""
        return {
            "version": "2.0",
            "header": asdict(self.header),
            "blocks": {bid: asdict(metadata) for bid, metadata in self.blocks.items()},
            "provenance": [asdict(record) for record in self.provenance],
            "custom_schemas": self.custom_schemas,
            "exported": datetime.now(timezone.utc).isoformat(),
        }

    def import_manifest(self, manifest: Dict[str, Any]) -> bool:
        """Import metadata from manifest."""
        try:
            # Validate basic manifest structure
            if not isinstance(manifest, dict):
                return False

            # Check for required fields - reject invalid manifests
            if not manifest:  # Empty manifest
                return False

            # More strict validation - require version field for valid manifest
            if "version" not in manifest:
                return False

            # Check for invalid data types in critical fields
            if "blocks" in manifest and not isinstance(
                manifest["blocks"], (dict, list)
            ):
                return False

            if "provenance" in manifest and not isinstance(
                manifest["provenance"], (list, dict)
            ):
                return False

            # If only version is provided without blocks, it's not a complete manifest
            if len(manifest) == 1 and "version" in manifest:
                return False

            # Import header
            if "header" in manifest:
                header_data = manifest["header"]
                if isinstance(header_data, dict):
                    self.header = MAIFHeader(**header_data)
                else:
                    return False

            # Import blocks
            if "blocks" in manifest:
                blocks_data = manifest["blocks"]
                if isinstance(blocks_data, dict):
                    self.blocks = {}
                    for block_id, block_data in blocks_data.items():
                        if isinstance(block_data, dict):
                            self.blocks[block_id] = BlockMetadata(**block_data)
                        else:
                            return False
                elif isinstance(blocks_data, list):
                    # Handle list format
                    self.blocks = {}
                    for block_data in blocks_data:
                        if isinstance(block_data, dict) and "block_id" in block_data:
                            block_id = block_data["block_id"]
                            self.blocks[block_id] = BlockMetadata(**block_data)
                        else:
                            return False
                else:
                    return False

            # Import provenance
            if "provenance" in manifest:
                provenance_data = manifest["provenance"]
                if isinstance(provenance_data, list):
                    self.provenance = []
                    for record_data in provenance_data:
                        if isinstance(record_data, dict):
                            self.provenance.append(ProvenanceRecord(**record_data))
                        else:
                            return False
                else:
                    return False

            # Import custom schemas
            if "custom_schemas" in manifest:
                schemas_data = manifest["custom_schemas"]
                if isinstance(schemas_data, dict):
                    self.custom_schemas = schemas_data
                else:
                    return False

            return True

        except Exception as e:
            print(f"Error importing manifest: {e}")
            return False

    def add_custom_schema(self, schema_name: str, schema: Dict[str, Any]) -> bool:
        """Add a custom metadata schema."""
        try:
            # Basic schema validation
            if not isinstance(schema, dict):
                return False

            if "type" not in schema or "properties" not in schema:
                return False

            self.custom_schemas[schema_name] = schema
            return True

        except Exception:
            return False

    def validate_custom_metadata(self, block_id: str, schema_name: str) -> List[str]:
        """Validate custom metadata against schema."""
        errors = []

        if block_id not in self.blocks:
            errors.append(f"Block {block_id} not found")
            return errors

        if schema_name not in self.custom_schemas:
            errors.append(f"Schema {schema_name} not found")
            return errors

        metadata = self.blocks[block_id]
        schema = self.custom_schemas[schema_name]
        custom_data = metadata.custom_metadata or {}

        # Check required fields
        required_fields = schema.get("required", [])
        for required_field in required_fields:
            if required_field not in custom_data:
                errors.append(f"Required property {required_field} is missing")

        # Basic validation (simplified JSON Schema validation)
        if "properties" in schema:
            for prop, prop_schema in schema["properties"].items():
                if prop in custom_data:
                    value = custom_data[prop]
                    expected_type = prop_schema.get("type")

                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Property {prop} should be string")
                    elif expected_type == "number" and not isinstance(
                        value, (int, float)
                    ):
                        errors.append(f"Property {prop} should be number")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        errors.append(f"Property {prop} should be boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        errors.append(f"Property {prop} should be array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        errors.append(f"Property {prop} should be object")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the MAIF file."""
        stats = {
            "file_info": {
                "version": self.header.version,
                "created": self.header.created,
                "modified": self.header.modified,
                "creator": self.header.creator_agent,
                "file_id": self.header.file_id,
            },
            "blocks": {
                "total": len(self.blocks),
                "by_type": {},
                "by_content_type": {},
                "by_compression": {},
                "total_size": 0,
                "average_size": 0,
            },
            "compression": {},  # Add compression field for test compatibility
            "encryption": {},  # Add encryption field for test compatibility
            "provenance": {
                "total_records": len(self.provenance),
                "by_operation": {},
                "by_agent": {},
                "time_span": None,
            },
            "dependencies": {
                "total_dependencies": 0,
                "blocks_with_dependencies": 0,
                "dependency_errors": len(self.validate_dependencies()),
            },
        }

        # Block statistics
        total_size = 0
        dependency_count = 0

        for metadata in self.blocks.values():
            # Block types (use content type for compatibility)
            content_type_str = (
                metadata.content_type.value
                if hasattr(metadata.content_type, "value")
                else str(metadata.content_type)
            )
            stats["blocks"]["by_type"][content_type_str] = (
                stats["blocks"]["by_type"].get(content_type_str, 0) + 1
            )

            # Content types
            stats["blocks"]["by_content_type"][content_type_str] = (
                stats["blocks"]["by_content_type"].get(content_type_str, 0) + 1
            )

            # Compression
            compression_str = metadata.compression
            if hasattr(metadata.compression, "value"):
                compression_str = metadata.compression.value
            elif hasattr(metadata.compression, "name"):
                compression_str = metadata.compression.name.lower()
            else:
                compression_str = str(metadata.compression).lower()

            stats["blocks"]["by_compression"][compression_str] = (
                stats["blocks"]["by_compression"].get(compression_str, 0) + 1
            )
            # Also populate the top-level compression field for test compatibility
            stats["compression"][compression_str] = (
                stats["compression"].get(compression_str, 0) + 1
            )

            # Encryption
            if metadata.encrypted or (
                metadata.custom_metadata
                and metadata.custom_metadata.get("encrypted", False)
            ):
                stats["encryption"]["encrypted"] = (
                    stats["encryption"].get("encrypted", 0) + 1
                )
            else:
                stats["encryption"]["unencrypted"] = (
                    stats["encryption"].get("unencrypted", 0) + 1
                )

            # Size
            total_size += metadata.size

            # Dependencies
            if metadata.dependencies:
                dependency_count += len(metadata.dependencies)
                stats["dependencies"]["blocks_with_dependencies"] += 1

        stats["blocks"]["total_size"] = total_size
        stats["blocks"]["average_size"] = (
            total_size / len(self.blocks) if self.blocks else 0
        )
        stats["dependencies"]["total_dependencies"] = dependency_count

        # Provenance statistics
        timestamps = []
        for record in self.provenance:
            # Operation types
            stats["provenance"]["by_operation"][record.operation_type] = (
                stats["provenance"]["by_operation"].get(record.operation_type, 0) + 1
            )

            # Agents
            stats["provenance"]["by_agent"][record.agent_id] = (
                stats["provenance"]["by_agent"].get(record.agent_id, 0) + 1
            )

            # Timestamps
            try:
                timestamps.append(
                    datetime.fromisoformat(record.timestamp.replace("Z", "+00:00"))
                )
            except (ValueError, AttributeError):
                pass  # Invalid timestamp format

        if timestamps:
            timestamps.sort()
            stats["provenance"]["time_span"] = {
                "start": timestamps[0].isoformat(),
                "end": timestamps[-1].isoformat(),
                "duration_seconds": (timestamps[-1] - timestamps[0]).total_seconds(),
            }

        return stats
