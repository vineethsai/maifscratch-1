"""
Dynamic Version Management for MAIF
===================================

Implements automatic file format updates and schema evolution.
This module provides capabilities for handling version transitions,
schema migrations, and backward compatibility.
"""

import json
import time
import hashlib
import logging
import os
import struct
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable, TypeVar
from dataclasses import dataclass, field, asdict
import threading
from pathlib import Path
import copy
import re

from .binary_format import MAIFFileHeader, BlockHeader as MAIFBlockHeader
from .core import MAIFEncoder, MAIFDecoder

logger = logging.getLogger(__name__)


class VersionCompatibility(Enum):
    """Compatibility level between versions."""

    BACKWARD = "backward"  # New version can read old version
    FORWARD = "forward"  # Old version can read new version
    FULL = "full"  # Both backward and forward compatible
    NONE = "none"  # Not compatible


@dataclass
class SchemaField:
    """Field in a schema."""

    name: str
    field_type: str
    required: bool = True
    default_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Schema:
    """Schema definition for a MAIF version."""

    version: str
    fields: List[SchemaField]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "version": self.version,
            "fields": [
                {
                    "name": field.name,
                    "type": field.field_type,
                    "required": field.required,
                    "default_value": field.default_value,
                    "metadata": field.metadata,
                }
                for field in self.fields
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Schema":
        """Create schema from dictionary."""
        fields = [
            SchemaField(
                name=field["name"],
                field_type=field["type"],
                required=field.get("required", True),
                default_value=field.get("default_value"),
                metadata=field.get("metadata", {}),
            )
            for field in data["fields"]
        ]

        return cls(
            version=data["version"], fields=fields, metadata=data.get("metadata", {})
        )


class VersionTransition:
    """Transition between two versions."""

    def __init__(
        self,
        from_version: str,
        to_version: str,
        compatibility: VersionCompatibility,
        metadata: Dict[str, Any] = None,
    ):
        self.from_version = from_version
        self.to_version = to_version
        self.compatibility = compatibility
        self.metadata = metadata or {}
        self.upgrade_transformations: List[Dict[str, Any]] = []
        self.downgrade_transformations: List[Dict[str, Any]] = []

    def add_field_rename(self, old_name: str, new_name: str):
        """Add field rename transformation."""
        self.upgrade_transformations.append(
            {"type": "rename_field", "old_name": old_name, "new_name": new_name}
        )

        self.downgrade_transformations.append(
            {"type": "rename_field", "old_name": new_name, "new_name": old_name}
        )

    def add_field_type_change(
        self,
        field_name: str,
        old_type: str,
        new_type: str,
        converter_up: Dict[str, Any] = None,
        converter_down: Dict[str, Any] = None,
    ):
        """Add field type change transformation."""
        self.upgrade_transformations.append(
            {
                "type": "change_type",
                "field_name": field_name,
                "old_type": old_type,
                "new_type": new_type,
                "converter": converter_up,
            }
        )

        self.downgrade_transformations.append(
            {
                "type": "change_type",
                "field_name": field_name,
                "old_type": new_type,
                "new_type": old_type,
                "converter": converter_down,
            }
        )

    def add_field_add(
        self, field_name: str, field_type: str, default_value: Any = None
    ):
        """Add field addition transformation."""
        self.upgrade_transformations.append(
            {
                "type": "add_field",
                "field_name": field_name,
                "field_type": field_type,
                "default_value": default_value,
            }
        )

        self.downgrade_transformations.append(
            {"type": "remove_field", "field_name": field_name}
        )

    def add_field_remove(self, field_name: str):
        """Add field removal transformation."""
        self.upgrade_transformations.append(
            {"type": "remove_field", "field_name": field_name}
        )

        # No downgrade transformation for field removal
        # as we can't recover the data

    def add_custom_transformation(
        self,
        transform_up: Dict[str, Any],
        transform_down: Optional[Dict[str, Any]] = None,
    ):
        """Add custom transformation."""
        self.upgrade_transformations.append(transform_up)

        if transform_down:
            self.downgrade_transformations.append(transform_down)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary."""
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "compatibility": self.compatibility.value,
            "metadata": self.metadata,
            "upgrade_transformations": self.upgrade_transformations,
            "downgrade_transformations": self.downgrade_transformations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VersionTransition":
        """Create transition from dictionary."""
        transition = cls(
            from_version=data["from_version"],
            to_version=data["to_version"],
            compatibility=VersionCompatibility(data["compatibility"]),
            metadata=data.get("metadata", {}),
        )

        transition.upgrade_transformations = data.get("upgrade_transformations", [])
        transition.downgrade_transformations = data.get("downgrade_transformations", [])

        return transition


class SchemaRegistry:
    """Registry of schemas and version transitions."""

    def __init__(self):
        self.schemas: Dict[str, Schema] = {}
        self.transitions: Dict[Tuple[str, str], VersionTransition] = {}
        self._lock = threading.RLock()

    def register_schema(self, schema: Schema):
        """Register a schema."""
        with self._lock:
            self.schemas[schema.version] = schema

    def register_transition(self, transition: VersionTransition):
        """Register a version transition."""
        with self._lock:
            key = (transition.from_version, transition.to_version)
            self.transitions[key] = transition

    def get_schema(self, version: str) -> Optional[Schema]:
        """Get schema by version."""
        with self._lock:
            return self.schemas.get(version)

    def get_latest_version(self) -> str:
        """Get latest schema version."""
        with self._lock:
            if not self.schemas:
                return ""

            # Sort versions using semantic versioning
            versions = list(self.schemas.keys())
            versions.sort(key=lambda v: [int(x) for x in v.split(".")])

            return versions[-1]

    def get_transition(
        self, from_version: str, to_version: str
    ) -> Optional[VersionTransition]:
        """Get transition between versions."""
        with self._lock:
            return self.transitions.get((from_version, to_version))

    def find_upgrade_path(
        self, from_version: str, to_version: str
    ) -> List[VersionTransition]:
        """Find path to upgrade from one version to another."""
        with self._lock:
            if from_version == to_version:
                return []

            # Build graph of version transitions
            graph = {}
            for (src, dst), transition in self.transitions.items():
                if src not in graph:
                    graph[src] = []
                graph[src].append((dst, transition))

            # BFS to find shortest path
            queue = [(from_version, [])]
            visited = set()

            while queue:
                current, path = queue.pop(0)

                if current == to_version:
                    return path

                if current in visited:
                    continue

                visited.add(current)

                if current in graph:
                    for neighbor, transition in graph[current]:
                        if neighbor not in visited:
                            queue.append((neighbor, path + [transition]))

            return []

    def find_downgrade_path(
        self, from_version: str, to_version: str
    ) -> List[VersionTransition]:
        """Find path to downgrade from one version to another."""
        with self._lock:
            if from_version == to_version:
                return []

            # Build graph of version transitions
            graph = {}
            for (src, dst), transition in self.transitions.items():
                if dst not in graph:
                    graph[dst] = []
                graph[dst].append((src, transition))

            # BFS to find shortest path
            queue = [(from_version, [])]
            visited = set()

            while queue:
                current, path = queue.pop(0)

                if current == to_version:
                    return path

                if current in visited:
                    continue

                visited.add(current)

                if current in graph:
                    for neighbor, transition in graph[current]:
                        if neighbor not in visited:
                            queue.append((neighbor, path + [transition]))

            return []

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        with self._lock:
            return {
                "schemas": {
                    version: schema.to_dict()
                    for version, schema in self.schemas.items()
                },
                "transitions": {
                    f"{src}->{dst}": transition.to_dict()
                    for (src, dst), transition in self.transitions.items()
                },
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaRegistry":
        """Create registry from dictionary."""
        registry = cls()

        for version, schema_dict in data.get("schemas", {}).items():
            registry.register_schema(Schema.from_dict(schema_dict))

        for key, transition_dict in data.get("transitions", {}).items():
            registry.register_transition(VersionTransition.from_dict(transition_dict))

        return registry

    def save(self, file_path: str):
        """Save registry to file."""
        with self._lock:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "SchemaRegistry":
        """Load registry from file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


class DataTransformer:
    """Transforms data between schema versions."""

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry

    def transform(
        self, data: Dict[str, Any], from_version: str, to_version: str
    ) -> Dict[str, Any]:
        """
        Transform data from one version to another using copy-on-write semantics.

        Args:
            data: Data to transform
            from_version: Source version
            to_version: Target version

        Returns:
            Transformed data
        """
        # Copy-on-write: If versions are the same, return the original data without copying
        if from_version == to_version:
            return data

        # Find upgrade path
        upgrade_path = self.registry.find_upgrade_path(from_version, to_version)

        if upgrade_path:
            # Copy-on-write: Only create a copy if we need to transform
            result = data
            needs_transform = False

            # Check if any transformations will actually modify the data
            for transition in upgrade_path:
                if transition.upgrade_transformations:
                    needs_transform = True
                    break

            # Only make a deep copy if transformations are needed
            if needs_transform:
                result = copy.deepcopy(data)

                # Apply upgrades
                for transition in upgrade_path:
                    if transition.upgrade_transformations:
                        result = self._apply_transformations(
                            result, transition.upgrade_transformations
                        )

            return result

        # Find downgrade path
        downgrade_path = self.registry.find_downgrade_path(from_version, to_version)

        if downgrade_path:
            # Copy-on-write: Only create a copy if we need to transform
            result = data
            needs_transform = False

            # Check if any transformations will actually modify the data
            for transition in downgrade_path:
                if transition.downgrade_transformations:
                    needs_transform = True
                    break

            # Only make a deep copy if transformations are needed
            if needs_transform:
                result = copy.deepcopy(data)

                # Apply downgrades
                for transition in downgrade_path:
                    if transition.downgrade_transformations:
                        result = self._apply_transformations(
                            result, transition.downgrade_transformations
                        )

            return result

        raise ValueError(f"No transformation path from {from_version} to {to_version}")

    def _apply_transformations(
        self, data: Dict[str, Any], transformations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply transformations to data."""
        result = copy.deepcopy(data)

        for transform in transformations:
            transform_type = transform["type"]

            if transform_type == "rename_field":
                self._rename_field(result, transform["old_name"], transform["new_name"])

            elif transform_type == "change_type":
                self._change_type(
                    result,
                    transform["field_name"],
                    transform["old_type"],
                    transform["new_type"],
                    transform.get("converter"),
                )

            elif transform_type == "add_field":
                self._add_field(
                    result,
                    transform["field_name"],
                    transform["field_type"],
                    transform.get("default_value"),
                )

            elif transform_type == "remove_field":
                self._remove_field(result, transform["field_name"])

            elif transform_type == "custom":
                self._apply_custom_transform(result, transform)

        return result

    def _rename_field(self, data: Dict[str, Any], old_name: str, new_name: str):
        """Rename field in data."""
        if old_name in data:
            data[new_name] = data[old_name]
            del data[old_name]

    def _change_type(
        self,
        data: Dict[str, Any],
        field_name: str,
        old_type: str,
        new_type: str,
        converter: Optional[Dict[str, Any]],
    ):
        """Change field type in data."""
        if field_name in data:
            value = data[field_name]

            if converter:
                # Apply custom conversion
                if converter["type"] == "string_to_int":
                    data[field_name] = int(value)
                elif converter["type"] == "int_to_string":
                    data[field_name] = str(value)
                elif converter["type"] == "string_to_float":
                    data[field_name] = float(value)
                elif converter["type"] == "float_to_string":
                    data[field_name] = str(value)
                elif converter["type"] == "string_to_bool":
                    data[field_name] = value.lower() in ("true", "yes", "1")
                elif converter["type"] == "bool_to_string":
                    data[field_name] = "true" if value else "false"
                elif converter["type"] == "timestamp_to_string":
                    data[field_name] = time.strftime(
                        converter.get("format", "%Y-%m-%dT%H:%M:%SZ"),
                        time.gmtime(value),
                    )
                elif converter["type"] == "string_to_timestamp":
                    data[field_name] = time.mktime(
                        time.strptime(
                            value, converter.get("format", "%Y-%m-%dT%H:%M:%SZ")
                        )
                    )
                elif converter["type"] == "json_to_string":
                    data[field_name] = json.dumps(value)
                elif converter["type"] == "string_to_json":
                    data[field_name] = json.loads(value)
            else:
                # Basic type conversion
                if new_type == "string":
                    data[field_name] = str(value)
                elif new_type == "integer":
                    data[field_name] = int(value)
                elif new_type == "float":
                    data[field_name] = float(value)
                elif new_type == "boolean":
                    data[field_name] = bool(value)

    def _add_field(
        self, data: Dict[str, Any], field_name: str, field_type: str, default_value: Any
    ):
        """Add field to data."""
        if field_name not in data:
            data[field_name] = default_value

    def _remove_field(self, data: Dict[str, Any], field_name: str):
        """Remove field from data."""
        if field_name in data:
            del data[field_name]

    def _apply_custom_transform(self, data: Dict[str, Any], transform: Dict[str, Any]):
        """Apply custom transformation to data."""
        if transform.get("operation") == "merge_fields":
            # Merge multiple fields into one
            source_fields = transform["source_fields"]
            target_field = transform["target_field"]
            separator = transform.get("separator", " ")

            values = []
            for field in source_fields:
                if field in data:
                    values.append(str(data[field]))

            data[target_field] = separator.join(values)

            # Remove source fields if specified
            if transform.get("remove_sources", False):
                for field in source_fields:
                    if field in data:
                        del data[field]

        elif transform.get("operation") == "split_field":
            # Split field into multiple fields
            source_field = transform["source_field"]
            target_fields = transform["target_fields"]
            separator = transform.get("separator", " ")

            if source_field in data:
                parts = data[source_field].split(separator)

                for i, field in enumerate(target_fields):
                    if i < len(parts):
                        data[field] = parts[i]
                    else:
                        data[field] = ""

                # Remove source field if specified
                if transform.get("remove_source", False):
                    del data[source_field]

        elif transform.get("operation") == "regex_extract":
            # Extract data using regex
            source_field = transform["source_field"]
            pattern = transform["pattern"]
            target_fields = transform["target_fields"]

            if source_field in data:
                match = re.match(pattern, data[source_field])

                if match:
                    for i, field in enumerate(target_fields):
                        if i + 1 <= len(match.groups()):
                            data[field] = match.group(i + 1)

                # Remove source field if specified
                if transform.get("remove_source", False):
                    del data[source_field]


class VersionManager:
    """
    Manages MAIF file versions and schema evolution.

    Provides capabilities for automatic version detection, upgrades,
    and backward compatibility.
    """

    def __init__(self, registry: SchemaRegistry):
        self.registry = registry
        self.transformer = DataTransformer(registry)
        self._lock = threading.RLock()

    def detect_version(self, file_path: str) -> str:
        """
        Detect version of MAIF file.

        Args:
            file_path: Path to MAIF file

        Returns:
            Detected version
        """
        try:
            # Open file and read header
            with open(file_path, "rb") as f:
                # Read magic number
                magic = f.read(4)

                if magic != b"MAIF":
                    raise ValueError("Not a MAIF file")

                # Read version
                version_major = struct.unpack("<B", f.read(1))[0]
                version_minor = struct.unpack("<B", f.read(1))[0]

                # Read metadata offset
                f.seek(64)  # Skip to metadata offset
                metadata_offset = struct.unpack("<Q", f.read(8))[0]

                # Read metadata
                if metadata_offset > 0:
                    f.seek(metadata_offset)
                    metadata_size = struct.unpack("<I", f.read(4))[0]
                    metadata_bytes = f.read(metadata_size)

                    try:
                        metadata = json.loads(metadata_bytes.decode("utf-8"))
                        if "schema_version" in metadata:
                            return metadata["schema_version"]
                    except (KeyError, AttributeError):
                        pass  # Metadata not available

            # Fall back to binary version
            return f"{version_major}.{version_minor}"

        except Exception as e:
            logger.error(f"Error detecting version: {e}")
            return "unknown"

    def upgrade_file(
        self, source_path: str, target_path: str, target_version: Optional[str] = None
    ) -> str:
        """
        Upgrade MAIF file to target version.

        Args:
            source_path: Source file path
            target_path: Target file path
            target_version: Target version (None for latest)

        Returns:
            Resulting version
        """
        with self._lock:
            # Detect source version
            source_version = self.detect_version(source_path)

            if source_version == "unknown":
                raise ValueError("Unknown source version")

            # Determine target version
            if target_version is None:
                target_version = self.registry.get_latest_version()

            if source_version == target_version:
                # No upgrade needed
                if source_path != target_path:
                    import shutil

                    shutil.copy2(source_path, target_path)
                return source_version

            # Load source file (v3 format - self-contained)
            decoder = MAIFDecoder(source_path)
            decoder.load()

            # Create encoder for target file (v3 format)
            encoder = MAIFEncoder(target_path, agent_id="version_manager")

            # Copy blocks with transformation
            for block in decoder.blocks:
                # Get block data
                data = block.data

                # Transform metadata if needed
                if block.metadata:
                    # Convert metadata to target version
                    metadata = self.transformer.transform(
                        block.metadata, source_version, target_version
                    )
                else:
                    metadata = {}

                # Add schema version to metadata
                metadata["schema_version"] = target_version

                # Add block to encoder
                encoder.add_binary_block(
                    data, block.header.block_type, metadata=metadata
                )

            # Finalize (v3 format)
            encoder.finalize()

            return target_version

    def validate_schema(
        self, data: Dict[str, Any], version: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate data against schema.

        Args:
            data: Data to validate
            version: Schema version

        Returns:
            Tuple of (is_valid, errors)
        """
        with self._lock:
            schema = self.registry.get_schema(version)

            if not schema:
                return False, [f"Unknown schema version: {version}"]

            errors = []

            # Check required fields
            for field in schema.fields:
                if field.required and field.name not in data:
                    errors.append(f"Missing required field: {field.name}")

            # Check field types
            for field in schema.fields:
                if field.name in data and data[field.name] is not None:
                    value = data[field.name]

                    if field.field_type == "string" and not isinstance(value, str):
                        errors.append(
                            f"Field {field.name} should be string, got {type(value).__name__}"
                        )

                    elif field.field_type == "integer" and not isinstance(value, int):
                        errors.append(
                            f"Field {field.name} should be integer, got {type(value).__name__}"
                        )

                    elif field.field_type == "float" and not isinstance(
                        value, (int, float)
                    ):
                        errors.append(
                            f"Field {field.name} should be float, got {type(value).__name__}"
                        )

                    elif field.field_type == "boolean" and not isinstance(value, bool):
                        errors.append(
                            f"Field {field.name} should be boolean, got {type(value).__name__}"
                        )

                    elif field.field_type == "array" and not isinstance(value, list):
                        errors.append(
                            f"Field {field.name} should be array, got {type(value).__name__}"
                        )

                    elif field.field_type == "object" and not isinstance(value, dict):
                        errors.append(
                            f"Field {field.name} should be object, got {type(value).__name__}"
                        )

            return len(errors) == 0, errors
