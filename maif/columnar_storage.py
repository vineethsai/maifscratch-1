"""
Columnar Storage for MAIF
=========================

Implements Apache Parquet-inspired columnar storage for analytics.
This module provides efficient storage and retrieval of columnar data
with optimized compression and encoding schemes.
"""

import io
import json
import struct
import zlib
import pickle
import numpy as np
import hashlib
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union, BinaryIO
from dataclasses import dataclass, field
import threading
from pathlib import Path
import logging
import mmap
from collections import defaultdict

from .compression_manager import CompressionManager

logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Data types for columns."""

    INT32 = "int32"
    INT64 = "int64"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    BOOLEAN = "boolean"
    STRING = "string"
    BINARY = "binary"
    JSON = "json"
    TIMESTAMP = "timestamp"


class EncodingType(Enum):
    """Encoding types for columns."""

    PLAIN = "plain"  # No encoding
    RLE = "rle"  # Run-length encoding
    DICTIONARY = "dictionary"  # Dictionary encoding
    DELTA = "delta"  # Delta encoding
    DELTA_LENGTH_BYTE_ARRAY = "delta_length_byte_array"  # Delta encoding for strings


class CompressionType(Enum):
    """Compression types for column chunks."""

    NONE = "none"
    ZSTD = "zstd"
    GZIP = "gzip"
    LZ4 = "lz4"
    SNAPPY = "snappy"


@dataclass
class ColumnMetadata:
    """Metadata for a column."""

    name: str
    column_type: ColumnType
    encoding: EncodingType
    compression: CompressionType
    null_count: int = 0
    distinct_count: Optional[int] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    total_size: int = 0
    total_values: int = 0
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnChunk:
    """A chunk of column data."""

    metadata: ColumnMetadata
    data: bytes
    offset: int
    length: int
    row_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metadata": {
                "name": self.metadata.name,
                "column_type": self.metadata.column_type.value,
                "encoding": self.metadata.encoding.value,
                "compression": self.metadata.compression.value,
                "null_count": self.metadata.null_count,
                "distinct_count": self.metadata.distinct_count,
                "min_value": self._serialize_value(self.metadata.min_value),
                "max_value": self._serialize_value(self.metadata.max_value),
                "total_size": self.metadata.total_size,
                "total_values": self.metadata.total_values,
                "statistics": self.metadata.statistics,
            },
            "offset": self.offset,
            "length": self.length,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunk_data: bytes) -> "ColumnChunk":
        """Create from dictionary and data."""
        metadata = ColumnMetadata(
            name=data["metadata"]["name"],
            column_type=ColumnType(data["metadata"]["column_type"]),
            encoding=EncodingType(data["metadata"]["encoding"]),
            compression=CompressionType(data["metadata"]["compression"]),
            null_count=data["metadata"]["null_count"],
            distinct_count=data["metadata"]["distinct_count"],
            min_value=cls._deserialize_value(data["metadata"]["min_value"]),
            max_value=cls._deserialize_value(data["metadata"]["max_value"]),
            total_size=data["metadata"]["total_size"],
            total_values=data["metadata"]["total_values"],
            statistics=data["metadata"]["statistics"],
        )

        return cls(
            metadata=metadata,
            data=chunk_data,
            offset=data["offset"],
            length=data["length"],
            row_count=data["row_count"],
        )

    @staticmethod
    def _serialize_value(value: Any) -> Optional[str]:
        """Serialize value for JSON storage."""
        if value is None:
            return None

        if isinstance(value, (int, float, bool, str)):
            return str(value)

        return json.dumps(value)

    @staticmethod
    def _deserialize_value(value: Optional[str]) -> Any:
        """Deserialize value from JSON storage."""
        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError, TypeError):
            return value


@dataclass
class RowGroup:
    """A group of rows stored as column chunks."""

    row_count: int
    column_chunks: Dict[str, ColumnChunk] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row_count": self.row_count,
            "column_chunks": {
                name: chunk.to_dict() for name, chunk in self.column_chunks.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], chunk_data_provider) -> "RowGroup":
        """Create from dictionary and data provider function."""
        column_chunks = {}

        for name, chunk_dict in data["column_chunks"].items():
            chunk_data = chunk_data_provider(chunk_dict["offset"], chunk_dict["length"])
            column_chunks[name] = ColumnChunk.from_dict(chunk_dict, chunk_data)

        return cls(
            row_count=data["row_count"],
            column_chunks=column_chunks,
            metadata=data["metadata"],
        )


class ColumnEncoder:
    """Encodes column data with various encoding schemes."""

    def __init__(self, compression_manager: Optional[CompressionManager] = None):
        self.compression_manager = compression_manager or CompressionManager()

    def encode(
        self,
        values: List[Any],
        column_type: ColumnType,
        encoding: EncodingType,
        compression: CompressionType,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Encode column values.

        Args:
            values: Column values
            column_type: Column data type
            encoding: Encoding scheme
            compression: Compression algorithm

        Returns:
            Tuple of (encoded bytes, statistics)
        """
        # Convert values to appropriate type
        typed_values = self._convert_values(values, column_type)

        # Calculate statistics
        statistics = self._calculate_statistics(typed_values, column_type)

        # Encode values
        if encoding == EncodingType.PLAIN:
            encoded_data = self._encode_plain(typed_values, column_type)
        elif encoding == EncodingType.RLE:
            encoded_data = self._encode_rle(typed_values, column_type)
        elif encoding == EncodingType.DICTIONARY:
            encoded_data = self._encode_dictionary(typed_values, column_type)
        elif encoding == EncodingType.DELTA:
            encoded_data = self._encode_delta(typed_values, column_type)
        elif encoding == EncodingType.DELTA_LENGTH_BYTE_ARRAY:
            encoded_data = self._encode_delta_length_byte_array(
                typed_values, column_type
            )
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        # Compress data
        if compression != CompressionType.NONE:
            compressed_data = self._compress(encoded_data, compression)
        else:
            compressed_data = encoded_data

        return compressed_data, statistics

    def decode(
        self,
        data: bytes,
        column_type: ColumnType,
        encoding: EncodingType,
        compression: CompressionType,
        row_count: int,
    ) -> List[Any]:
        """
        Decode column data.

        Args:
            data: Encoded column data
            column_type: Column data type
            encoding: Encoding scheme
            compression: Compression algorithm
            row_count: Number of rows

        Returns:
            List of decoded values
        """
        # Decompress data
        if compression != CompressionType.NONE:
            decompressed_data = self._decompress(data, compression)
        else:
            decompressed_data = data

        # Decode values
        if encoding == EncodingType.PLAIN:
            values = self._decode_plain(decompressed_data, column_type, row_count)
        elif encoding == EncodingType.RLE:
            values = self._decode_rle(decompressed_data, column_type, row_count)
        elif encoding == EncodingType.DICTIONARY:
            values = self._decode_dictionary(decompressed_data, column_type, row_count)
        elif encoding == EncodingType.DELTA:
            values = self._decode_delta(decompressed_data, column_type, row_count)
        elif encoding == EncodingType.DELTA_LENGTH_BYTE_ARRAY:
            values = self._decode_delta_length_byte_array(
                decompressed_data, column_type, row_count
            )
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        return values

    def _convert_values(self, values: List[Any], column_type: ColumnType) -> List[Any]:
        """Convert values to appropriate type."""
        if column_type == ColumnType.INT32:
            return [int(v) if v is not None else None for v in values]
        elif column_type == ColumnType.INT64:
            return [int(v) if v is not None else None for v in values]
        elif column_type == ColumnType.FLOAT32:
            return [float(v) if v is not None else None for v in values]
        elif column_type == ColumnType.FLOAT64:
            return [float(v) if v is not None else None for v in values]
        elif column_type == ColumnType.BOOLEAN:
            return [bool(v) if v is not None else None for v in values]
        elif column_type == ColumnType.STRING:
            return [str(v) if v is not None else None for v in values]
        elif column_type == ColumnType.BINARY:
            return [bytes(v) if v is not None else None for v in values]
        elif column_type == ColumnType.JSON:
            return [json.dumps(v) if v is not None else None for v in values]
        elif column_type == ColumnType.TIMESTAMP:
            return [float(v) if v is not None else None for v in values]
        else:
            raise ValueError(f"Unsupported column type: {column_type}")

    def _calculate_statistics(
        self, values: List[Any], column_type: ColumnType
    ) -> Dict[str, Any]:
        """Calculate statistics for column values."""
        non_null_values = [v for v in values if v is not None]

        statistics = {
            "null_count": len(values) - len(non_null_values),
            "total_values": len(values),
        }

        if non_null_values:
            if column_type in (
                ColumnType.INT32,
                ColumnType.INT64,
                ColumnType.FLOAT32,
                ColumnType.FLOAT64,
                ColumnType.TIMESTAMP,
            ):
                statistics["min_value"] = min(non_null_values)
                statistics["max_value"] = max(non_null_values)
                statistics["sum"] = sum(non_null_values)
                statistics["mean"] = statistics["sum"] / len(non_null_values)

                if len(non_null_values) > 1:
                    variance = sum(
                        (x - statistics["mean"]) ** 2 for x in non_null_values
                    ) / len(non_null_values)
                    statistics["variance"] = variance
                    statistics["std_dev"] = variance**0.5

            if column_type in (ColumnType.STRING, ColumnType.BINARY):
                statistics["min_length"] = min(len(v) for v in non_null_values)
                statistics["max_length"] = max(len(v) for v in non_null_values)
                statistics["total_length"] = sum(len(v) for v in non_null_values)
                statistics["mean_length"] = statistics["total_length"] / len(
                    non_null_values
                )

            # Calculate distinct count (up to 1000 values)
            if len(values) <= 1000:
                statistics["distinct_count"] = len(set(values))
            else:
                # Estimate using HyperLogLog or similar in a real implementation
                statistics["distinct_count_estimated"] = True

        return statistics

    def _encode_plain(self, values: List[Any], column_type: ColumnType) -> bytes:
        """Encode values using plain encoding."""
        buffer = io.BytesIO()

        # Write null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        for i, value in enumerate(values):
            if value is not None:
                byte_index = i // 8
                bit_index = i % 8
                null_bitmap[byte_index] |= 1 << bit_index

        buffer.write(bytes(null_bitmap))

        # Write values
        if column_type == ColumnType.INT32:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<i", value))

        elif column_type == ColumnType.INT64:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<q", value))

        elif column_type == ColumnType.FLOAT32:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<f", value))

        elif column_type == ColumnType.FLOAT64:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<d", value))

        elif column_type == ColumnType.BOOLEAN:
            boolean_bytes = bytearray((len(values) + 7) // 8)
            for i, value in enumerate(values):
                if value is not None and value:
                    byte_index = i // 8
                    bit_index = i % 8
                    boolean_bytes[byte_index] |= 1 << bit_index

            buffer.write(bytes(boolean_bytes))

        elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
            for value in values:
                if value is not None:
                    encoded = value.encode("utf-8")
                    buffer.write(struct.pack("<I", len(encoded)))
                    buffer.write(encoded)

        elif column_type == ColumnType.BINARY:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<I", len(value)))
                    buffer.write(value)

        elif column_type == ColumnType.TIMESTAMP:
            for value in values:
                if value is not None:
                    buffer.write(struct.pack("<d", value))

        return buffer.getvalue()

    def _decode_plain(
        self, data: bytes, column_type: ColumnType, row_count: int
    ) -> List[Any]:
        """Decode values using plain encoding."""
        buffer = io.BytesIO(data)

        # Read null bitmap
        null_bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(null_bitmap_size)

        values = []

        if column_type == ColumnType.INT32:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(struct.unpack("<i", buffer.read(4))[0])

        elif column_type == ColumnType.INT64:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(struct.unpack("<q", buffer.read(8))[0])

        elif column_type == ColumnType.FLOAT32:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(struct.unpack("<f", buffer.read(4))[0])

        elif column_type == ColumnType.FLOAT64:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(struct.unpack("<d", buffer.read(8))[0])

        elif column_type == ColumnType.BOOLEAN:
            boolean_bytes = buffer.read((row_count + 7) // 8)
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(bool(boolean_bytes[i // 8] & (1 << (i % 8))))

        elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    length = struct.unpack("<I", buffer.read(4))[0]
                    values.append(buffer.read(length).decode("utf-8"))

        elif column_type == ColumnType.BINARY:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    length = struct.unpack("<I", buffer.read(4))[0]
                    values.append(buffer.read(length))

        elif column_type == ColumnType.TIMESTAMP:
            for i in range(row_count):
                is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
                if is_null:
                    values.append(None)
                else:
                    values.append(struct.unpack("<d", buffer.read(8))[0])

        return values

    def _encode_rle(self, values: List[Any], column_type: ColumnType) -> bytes:
        """Encode values using run-length encoding."""
        buffer = io.BytesIO()

        # Write null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        for i, value in enumerate(values):
            if value is not None:
                byte_index = i // 8
                bit_index = i % 8
                null_bitmap[byte_index] |= 1 << bit_index

        buffer.write(bytes(null_bitmap))

        # Encode runs
        if len(values) == 0:
            return buffer.getvalue()

        current_value = values[0]
        run_length = 1

        for i in range(1, len(values)):
            if values[i] == current_value:
                run_length += 1
            else:
                # Write run
                buffer.write(struct.pack("<I", run_length))

                # Write value
                if current_value is not None:
                    if column_type == ColumnType.INT32:
                        buffer.write(struct.pack("<i", current_value))
                    elif column_type == ColumnType.INT64:
                        buffer.write(struct.pack("<q", current_value))
                    elif column_type == ColumnType.FLOAT32:
                        buffer.write(struct.pack("<f", current_value))
                    elif column_type == ColumnType.FLOAT64:
                        buffer.write(struct.pack("<d", current_value))
                    elif column_type == ColumnType.BOOLEAN:
                        buffer.write(struct.pack("<?", current_value))
                    elif (
                        column_type == ColumnType.STRING
                        or column_type == ColumnType.JSON
                    ):
                        encoded = current_value.encode("utf-8")
                        buffer.write(struct.pack("<I", len(encoded)))
                        buffer.write(encoded)
                    elif column_type == ColumnType.BINARY:
                        buffer.write(struct.pack("<I", len(current_value)))
                        buffer.write(current_value)
                    elif column_type == ColumnType.TIMESTAMP:
                        buffer.write(struct.pack("<d", current_value))

                current_value = values[i]
                run_length = 1

        # Write final run
        buffer.write(struct.pack("<I", run_length))

        # Write final value
        if current_value is not None:
            if column_type == ColumnType.INT32:
                buffer.write(struct.pack("<i", current_value))
            elif column_type == ColumnType.INT64:
                buffer.write(struct.pack("<q", current_value))
            elif column_type == ColumnType.FLOAT32:
                buffer.write(struct.pack("<f", current_value))
            elif column_type == ColumnType.FLOAT64:
                buffer.write(struct.pack("<d", current_value))
            elif column_type == ColumnType.BOOLEAN:
                buffer.write(struct.pack("<?", current_value))
            elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                encoded = current_value.encode("utf-8")
                buffer.write(struct.pack("<I", len(encoded)))
                buffer.write(encoded)
            elif column_type == ColumnType.BINARY:
                buffer.write(struct.pack("<I", len(current_value)))
                buffer.write(current_value)
            elif column_type == ColumnType.TIMESTAMP:
                buffer.write(struct.pack("<d", current_value))

        return buffer.getvalue()

    def _decode_rle(
        self, data: bytes, column_type: ColumnType, row_count: int
    ) -> List[Any]:
        """Decode values using run-length encoding."""
        buffer = io.BytesIO(data)

        # Read null bitmap
        null_bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(null_bitmap_size)

        values = []
        remaining_rows = row_count

        while remaining_rows > 0:
            # Read run length
            run_length = min(remaining_rows, struct.unpack("<I", buffer.read(4))[0])

            # Read value
            value = None

            # Check if the first value in the run is null
            is_null = not (null_bitmap[len(values) // 8] & (1 << (len(values) % 8)))

            if not is_null:
                if column_type == ColumnType.INT32:
                    value = struct.unpack("<i", buffer.read(4))[0]
                elif column_type == ColumnType.INT64:
                    value = struct.unpack("<q", buffer.read(8))[0]
                elif column_type == ColumnType.FLOAT32:
                    value = struct.unpack("<f", buffer.read(4))[0]
                elif column_type == ColumnType.FLOAT64:
                    value = struct.unpack("<d", buffer.read(8))[0]
                elif column_type == ColumnType.BOOLEAN:
                    value = struct.unpack("<?", buffer.read(1))[0]
                elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                    length = struct.unpack("<I", buffer.read(4))[0]
                    value = buffer.read(length).decode("utf-8")
                elif column_type == ColumnType.BINARY:
                    length = struct.unpack("<I", buffer.read(4))[0]
                    value = buffer.read(length)
                elif column_type == ColumnType.TIMESTAMP:
                    value = struct.unpack("<d", buffer.read(8))[0]

            # Add values
            for _ in range(run_length):
                values.append(value)

            remaining_rows -= run_length

        return values

    def _encode_dictionary(self, values: List[Any], column_type: ColumnType) -> bytes:
        """Encode values using dictionary encoding."""
        buffer = io.BytesIO()

        # Write null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        for i, value in enumerate(values):
            if value is not None:
                byte_index = i // 8
                bit_index = i % 8
                null_bitmap[byte_index] |= 1 << bit_index

        buffer.write(bytes(null_bitmap))

        # Build dictionary
        dictionary = {}
        next_id = 0

        for value in values:
            if value is not None and value not in dictionary:
                dictionary[value] = next_id
                next_id += 1

        # Write dictionary size
        buffer.write(struct.pack("<I", len(dictionary)))

        # Write dictionary
        for value, value_id in sorted(dictionary.items(), key=lambda x: x[1]):
            if column_type == ColumnType.INT32:
                buffer.write(struct.pack("<i", value))
            elif column_type == ColumnType.INT64:
                buffer.write(struct.pack("<q", value))
            elif column_type == ColumnType.FLOAT32:
                buffer.write(struct.pack("<f", value))
            elif column_type == ColumnType.FLOAT64:
                buffer.write(struct.pack("<d", value))
            elif column_type == ColumnType.BOOLEAN:
                buffer.write(struct.pack("<?", value))
            elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                encoded = value.encode("utf-8")
                buffer.write(struct.pack("<I", len(encoded)))
                buffer.write(encoded)
            elif column_type == ColumnType.BINARY:
                buffer.write(struct.pack("<I", len(value)))
                buffer.write(value)
            elif column_type == ColumnType.TIMESTAMP:
                buffer.write(struct.pack("<d", value))

        # Write values as dictionary IDs
        for value in values:
            if value is not None:
                buffer.write(struct.pack("<I", dictionary[value]))

        return buffer.getvalue()

    def _decode_dictionary(
        self, data: bytes, column_type: ColumnType, row_count: int
    ) -> List[Any]:
        """Decode values using dictionary encoding."""
        buffer = io.BytesIO(data)

        # Read null bitmap
        null_bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(null_bitmap_size)

        # Read dictionary size
        dict_size = struct.unpack("<I", buffer.read(4))[0]

        # Read dictionary
        dictionary = {}

        for i in range(dict_size):
            if column_type == ColumnType.INT32:
                value = struct.unpack("<i", buffer.read(4))[0]
            elif column_type == ColumnType.INT64:
                value = struct.unpack("<q", buffer.read(8))[0]
            elif column_type == ColumnType.FLOAT32:
                value = struct.unpack("<f", buffer.read(4))[0]
            elif column_type == ColumnType.FLOAT64:
                value = struct.unpack("<d", buffer.read(8))[0]
            elif column_type == ColumnType.BOOLEAN:
                value = struct.unpack("<?", buffer.read(1))[0]
            elif column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                length = struct.unpack("<I", buffer.read(4))[0]
                value = buffer.read(length).decode("utf-8")
            elif column_type == ColumnType.BINARY:
                length = struct.unpack("<I", buffer.read(4))[0]
                value = buffer.read(length)
            elif column_type == ColumnType.TIMESTAMP:
                value = struct.unpack("<d", buffer.read(8))[0]

            dictionary[i] = value

        # Read values
        values = []

        for i in range(row_count):
            is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
            if is_null:
                values.append(None)
            else:
                dict_id = struct.unpack("<I", buffer.read(4))[0]
                values.append(dictionary[dict_id])

        return values

    def _encode_delta(self, values: List[Any], column_type: ColumnType) -> bytes:
        """Encode values using delta encoding."""
        buffer = io.BytesIO()

        # Write null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        for i, value in enumerate(values):
            if value is not None:
                byte_index = i // 8
                bit_index = i % 8
                null_bitmap[byte_index] |= 1 << bit_index

        buffer.write(bytes(null_bitmap))

        # Only applicable to numeric types
        if column_type not in (
            ColumnType.INT32,
            ColumnType.INT64,
            ColumnType.FLOAT32,
            ColumnType.FLOAT64,
            ColumnType.TIMESTAMP,
        ):
            # Fall back to plain encoding for non-numeric types
            return self._encode_plain(values, column_type)

        # Find first non-null value
        first_value = None
        for value in values:
            if value is not None:
                first_value = value
                break

        if first_value is None:
            # All values are null
            return buffer.getvalue()

        # Write first value
        if column_type == ColumnType.INT32:
            buffer.write(struct.pack("<i", first_value))
        elif column_type == ColumnType.INT64:
            buffer.write(struct.pack("<q", first_value))
        elif column_type == ColumnType.FLOAT32:
            buffer.write(struct.pack("<f", first_value))
        elif column_type == ColumnType.FLOAT64:
            buffer.write(struct.pack("<d", first_value))
        elif column_type == ColumnType.TIMESTAMP:
            buffer.write(struct.pack("<d", first_value))

        # Write deltas
        prev_value = first_value
        for value in values:
            if value is not None and value != first_value:
                delta = value - prev_value

                if column_type == ColumnType.INT32:
                    buffer.write(struct.pack("<i", delta))
                elif column_type == ColumnType.INT64:
                    buffer.write(struct.pack("<q", delta))
                elif column_type == ColumnType.FLOAT32:
                    buffer.write(struct.pack("<f", delta))
                elif column_type == ColumnType.FLOAT64:
                    buffer.write(struct.pack("<d", delta))
                elif column_type == ColumnType.TIMESTAMP:
                    buffer.write(struct.pack("<d", delta))

                prev_value = value

        return buffer.getvalue()

    def _decode_delta(
        self, data: bytes, column_type: ColumnType, row_count: int
    ) -> List[Any]:
        """Decode values using delta encoding."""
        buffer = io.BytesIO(data)

        # Read null bitmap
        null_bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(null_bitmap_size)

        # Only applicable to numeric types
        if column_type not in (
            ColumnType.INT32,
            ColumnType.INT64,
            ColumnType.FLOAT32,
            ColumnType.FLOAT64,
            ColumnType.TIMESTAMP,
        ):
            # Fall back to plain decoding for non-numeric types
            return self._decode_plain(data, column_type, row_count)

        # Check if all values are null
        if buffer.tell() >= len(data):
            return [None] * row_count

        # Read first value
        if column_type == ColumnType.INT32:
            first_value = struct.unpack("<i", buffer.read(4))[0]
        elif column_type == ColumnType.INT64:
            first_value = struct.unpack("<q", buffer.read(8))[0]
        elif column_type == ColumnType.FLOAT32:
            first_value = struct.unpack("<f", buffer.read(4))[0]
        elif column_type == ColumnType.FLOAT64:
            first_value = struct.unpack("<d", buffer.read(8))[0]
        elif column_type == ColumnType.TIMESTAMP:
            first_value = struct.unpack("<d", buffer.read(8))[0]

        # Initialize values
        values = [None] * row_count

        # Set first non-null value
        for i in range(row_count):
            is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
            if not is_null:
                values[i] = first_value
                break

        # Read deltas
        prev_value = first_value
        for i in range(row_count):
            is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
            if not is_null:
                if values[i] is None:  # Not the first value
                    if column_type == ColumnType.INT32:
                        delta = struct.unpack("<i", buffer.read(4))[0]
                    elif column_type == ColumnType.INT64:
                        delta = struct.unpack("<q", buffer.read(8))[0]
                    elif column_type == ColumnType.FLOAT32:
                        delta = struct.unpack("<f", buffer.read(4))[0]
                    elif column_type == ColumnType.FLOAT64:
                        delta = struct.unpack("<d", buffer.read(8))[0]
                    elif column_type == ColumnType.TIMESTAMP:
                        delta = struct.unpack("<d", buffer.read(8))[0]

                    values[i] = prev_value + delta
                    prev_value = values[i]

        return values

    def _encode_delta_length_byte_array(
        self, values: List[Any], column_type: ColumnType
    ) -> bytes:
        """Encode string/binary values using delta length encoding."""
        buffer = io.BytesIO()

        # Write null bitmap
        null_bitmap = bytearray((len(values) + 7) // 8)
        for i, value in enumerate(values):
            if value is not None:
                byte_index = i // 8
                bit_index = i % 8
                null_bitmap[byte_index] |= 1 << bit_index

        buffer.write(bytes(null_bitmap))

        # Only applicable to string/binary types
        if column_type not in (ColumnType.STRING, ColumnType.BINARY, ColumnType.JSON):
            # Fall back to plain encoding for non-string types
            return self._encode_plain(values, column_type)

        # Find first non-null value
        first_value = None
        for value in values:
            if value is not None:
                first_value = value
                break

        if first_value is None:
            # All values are null
            return buffer.getvalue()

        # Write first value length
        first_length = len(first_value)
        buffer.write(struct.pack("<I", first_length))

        # Write first value
        if column_type == ColumnType.STRING or column_type == ColumnType.JSON:
            buffer.write(first_value.encode("utf-8"))
        else:
            buffer.write(first_value)

        # Write length deltas and values
        prev_length = first_length
        for value in values:
            if value is not None and value != first_value:
                length = len(value)
                length_delta = length - prev_length

                # Write length delta
                buffer.write(struct.pack("<i", length_delta))

                # Write value
                if column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                    buffer.write(value.encode("utf-8"))
                else:
                    buffer.write(value)

                prev_length = length

        return buffer.getvalue()

    def _decode_delta_length_byte_array(
        self, data: bytes, column_type: ColumnType, row_count: int
    ) -> List[Any]:
        """Decode string/binary values using delta length encoding."""
        buffer = io.BytesIO(data)

        # Read null bitmap
        null_bitmap_size = (row_count + 7) // 8
        null_bitmap = buffer.read(null_bitmap_size)

        # Only applicable to string/binary types
        if column_type not in (ColumnType.STRING, ColumnType.BINARY, ColumnType.JSON):
            # Fall back to plain decoding for non-string types
            return self._decode_plain(data, column_type, row_count)

        # Check if all values are null
        if buffer.tell() >= len(data):
            return [None] * row_count

        # Read first value length
        first_length = struct.unpack("<I", buffer.read(4))[0]

        # Read first value
        first_value_bytes = buffer.read(first_length)
        if column_type == ColumnType.STRING or column_type == ColumnType.JSON:
            first_value = first_value_bytes.decode("utf-8")
        else:
            first_value = first_value_bytes

        # Initialize values
        values = [None] * row_count

        # Set first non-null value
        for i in range(row_count):
            is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
            if not is_null:
                values[i] = first_value
                break

        # Read length deltas and values
        prev_length = first_length
        for i in range(row_count):
            is_null = not (null_bitmap[i // 8] & (1 << (i % 8)))
            if not is_null and values[i] is None:  # Not the first value
                # Read length delta
                length_delta = struct.unpack("<i", buffer.read(4))[0]
                length = prev_length + length_delta

                # Read value
                value_bytes = buffer.read(length)
                if column_type == ColumnType.STRING or column_type == ColumnType.JSON:
                    values[i] = value_bytes.decode("utf-8")
                else:
                    values[i] = value_bytes

                prev_length = length

        return values

    def _compress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Compress data using specified algorithm."""
        if compression_type == CompressionType.ZSTD:
            return self.compression_manager.compress_zstd(data)
        elif compression_type == CompressionType.GZIP:
            return self.compression_manager.compress_gzip(data)
        elif compression_type == CompressionType.LZ4:
            return self.compression_manager.compress_lz4(data)
        elif compression_type == CompressionType.SNAPPY:
            return self.compression_manager.compress_snappy(data)
        else:
            return data

    def _decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """Decompress data using specified algorithm."""
        if compression_type == CompressionType.ZSTD:
            return self.compression_manager.decompress_zstd(data)
        elif compression_type == CompressionType.GZIP:
            return self.compression_manager.decompress_gzip(data)
        elif compression_type == CompressionType.LZ4:
            return self.compression_manager.decompress_lz4(data)
        elif compression_type == CompressionType.SNAPPY:
            return self.compression_manager.decompress_snappy(data)
        else:
            return data


class ColumnarFile:
    """
    Columnar storage file format for MAIF.

    Provides efficient storage and retrieval of columnar data with
    optimized compression and encoding schemes.
    """

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path) if file_path else None
        self.row_groups: List[RowGroup] = []
        self.schema: Dict[str, ColumnType] = {}
        self.metadata: Dict[str, Any] = {}
        self.encoder = ColumnEncoder()
        self._file_handle: Optional[BinaryIO] = None
        self._lock = threading.RLock()

        # Load file if path provided
        if self.file_path and self.file_path.exists():
            self._load_file()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close file handle."""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def add_column(self, name: str, column_type: ColumnType):
        """Add column to schema."""
        with self._lock:
            self.schema[name] = column_type

    def write_batch(
        self,
        data: Dict[str, List[Any]],
        row_group_size: int = 1000,
        encodings: Optional[Dict[str, EncodingType]] = None,
        compressions: Optional[Dict[str, CompressionType]] = None,
    ):
        """
        Write batch of data to file.

        Args:
            data: Dictionary of column name -> values
            row_group_size: Number of rows per row group
            encodings: Dictionary of column name -> encoding type
            compressions: Dictionary of column name -> compression type
        """
        with self._lock:
            # Validate schema
            for column_name in data:
                if column_name not in self.schema:
                    raise ValueError(f"Column {column_name} not in schema")

            # Copy-on-write: Check if data already exists and is identical
            if self.row_groups and all(
                column_name in data for column_name in self.schema
            ):
                # Check if this is an exact duplicate of existing data
                is_duplicate = False

                # Only check for duplicates if we have a reasonable amount of data
                if len(self.row_groups) > 0 and len(data) > 0:
                    # Get a hash of the new data for quick comparison
                    data_hash = self._hash_data(data)

                    # Check if we've seen this exact data before
                    for row_group in self.row_groups:
                        if row_group.metadata.get("data_hash") == data_hash:
                            # Found a duplicate, no need to write again
                            is_duplicate = True
                            break

                if is_duplicate:
                    # Copy-on-write: Data already exists, no need to write again
                    return

            # Use default encodings/compressions if not provided
            if encodings is None:
                encodings = {}
            if compressions is None:
                compressions = {}

            # Set default encodings/compressions
            for column_name in self.schema:
                if column_name not in encodings:
                    # Choose appropriate encoding based on column type
                    column_type = self.schema[column_name]
                    if column_type in (
                        ColumnType.INT32,
                        ColumnType.INT64,
                        ColumnType.FLOAT32,
                        ColumnType.FLOAT64,
                        ColumnType.TIMESTAMP,
                    ):
                        encodings[column_name] = EncodingType.DELTA
                    elif column_type in (
                        ColumnType.STRING,
                        ColumnType.BINARY,
                        ColumnType.JSON,
                    ):
                        encodings[column_name] = EncodingType.DICTIONARY
                    else:
                        encodings[column_name] = EncodingType.PLAIN

                if column_name not in compressions:
                    compressions[column_name] = CompressionType.ZSTD

            # Determine number of rows
            row_count = max(len(values) for values in data.values())

            # Pad columns to same length
            for column_name, values in data.items():
                if len(values) < row_count:
                    data[column_name] = values + [None] * (row_count - len(values))

            # Split into row groups
            for i in range(0, row_count, row_group_size):
                end_idx = min(i + row_group_size, row_count)
                row_group_data = {
                    column_name: values[i:end_idx]
                    for column_name, values in data.items()
                }

                # Add a hash of the data to the row group for future copy-on-write checks
                row_group_data_hash = self._hash_data(row_group_data)

                self._write_row_group(
                    row_group_data,
                    encodings,
                    compressions,
                    data_hash=row_group_data_hash,
                )

            # Write file if path provided
            if self.file_path:
                self._write_file()

    def _hash_data(self, data: Dict[str, List[Any]]) -> str:
        """Generate a hash of the data for copy-on-write comparisons."""
        import hashlib
        import json

        # Create a deterministic representation of the data
        # For simple types, we can use JSON
        try:
            # Try to create a JSON string for simple data types
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # For complex types, use a simpler approach
            hash_parts = []

            for column_name, values in sorted(data.items()):
                # Add column name to hash
                hash_parts.append(column_name)

                # Add a sample of values (first, middle, last)
                if values:
                    if len(values) <= 10:
                        # For small lists, use all values
                        for v in values:
                            hash_parts.append(str(v))
                    else:
                        # For larger lists, sample values
                        hash_parts.append(str(values[0]))
                        hash_parts.append(str(values[len(values) // 2]))
                        hash_parts.append(str(values[-1]))
                        # Add length as well
                        hash_parts.append(str(len(values)))

            # Create a hash of the combined parts
            combined = "|".join(hash_parts)
            return hashlib.sha256(combined.encode()).hexdigest()

    def _write_row_group(
        self,
        data: Dict[str, List[Any]],
        encodings: Dict[str, EncodingType],
        compressions: Dict[str, CompressionType],
        data_hash: Optional[str] = None,
    ):
        """Write row group to memory."""
        row_count = max(len(values) for values in data.values())

        # Create row group with metadata including data hash for copy-on-write
        metadata = {"timestamp": time.time()}

        # Add data hash if provided
        if data_hash:
            metadata["data_hash"] = data_hash

        # Create row group
        row_group = RowGroup(row_count=row_count, metadata=metadata)

        # Encode and compress columns
        for column_name, values in data.items():
            column_type = self.schema[column_name]
            encoding = encodings[column_name]
            compression = compressions[column_name]

            # Encode and compress data
            encoded_data, statistics = self.encoder.encode(
                values, column_type, encoding, compression
            )

            # Create column metadata
            metadata = ColumnMetadata(
                name=column_name,
                column_type=column_type,
                encoding=encoding,
                compression=compression,
                null_count=statistics.get("null_count", 0),
                distinct_count=statistics.get("distinct_count"),
                min_value=statistics.get("min_value"),
                max_value=statistics.get("max_value"),
                total_size=len(encoded_data),
                total_values=len(values),
                statistics=statistics,
            )

            # Create column chunk
            chunk = ColumnChunk(
                metadata=metadata,
                data=encoded_data,
                offset=0,  # Will be set when writing to file
                length=len(encoded_data),
                row_count=row_count,
            )

            # Add to row group
            row_group.column_chunks[column_name] = chunk

        # Add row group
        self.row_groups.append(row_group)

    def _write_file(self):
        """Write file to disk."""
        # Create directory if it doesn't exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.file_path, "wb") as f:
            # Write magic number
            f.write(b"MAIF_COL")

            # Write version
            f.write(struct.pack("<I", 1))

            # Write schema
            schema_json = json.dumps(
                {name: column_type.value for name, column_type in self.schema.items()}
            )
            f.write(struct.pack("<I", len(schema_json)))
            f.write(schema_json.encode("utf-8"))

            # Write metadata
            metadata_json = json.dumps(self.metadata)
            f.write(struct.pack("<I", len(metadata_json)))
            f.write(metadata_json.encode("utf-8"))

            # Write number of row groups
            f.write(struct.pack("<I", len(self.row_groups)))

            # Write row groups
            for row_group in self.row_groups:
                # Write row count
                f.write(struct.pack("<I", row_group.row_count))

                # Write number of columns
                f.write(struct.pack("<I", len(row_group.column_chunks)))

                # Write column chunks
                for column_name, chunk in row_group.column_chunks.items():
                    # Update offset
                    chunk.offset = f.tell() + 8  # 8 bytes for length

                    # Write column name
                    f.write(struct.pack("<I", len(column_name)))
                    f.write(column_name.encode("utf-8"))

                    # Write data length
                    f.write(struct.pack("<I", len(chunk.data)))

                    # Write data
                    f.write(chunk.data)

                # Write row group metadata
                metadata_json = json.dumps(row_group.metadata)
                f.write(struct.pack("<I", len(metadata_json)))
                f.write(metadata_json.encode("utf-8"))

    def _load_file(self):
        """Load file from disk."""
        with open(self.file_path, "rb") as f:
            # Read magic number
            magic = f.read(8)
            if magic != b"MAIF_COL":
                raise ValueError("Invalid file format")

            # Read version
            version = struct.unpack("<I", f.read(4))[0]
            if version != 1:
                raise ValueError(f"Unsupported version: {version}")

            # Read schema
            schema_len = struct.unpack("<I", f.read(4))[0]
            schema_json = f.read(schema_len).decode("utf-8")
            schema_dict = json.loads(schema_json)

            self.schema = {
                name: ColumnType(column_type)
                for name, column_type in schema_dict.items()
            }

            # Read metadata
            metadata_len = struct.unpack("<I", f.read(4))[0]
            metadata_json = f.read(metadata_len).decode("utf-8")
            self.metadata = json.loads(metadata_json)

            # Read number of row groups
            row_group_count = struct.unpack("<I", f.read(4))[0]

            # Read row groups
            for _ in range(row_group_count):
                # Read row count
                row_count = struct.unpack("<I", f.read(4))[0]

                # Read number of columns
                column_count = struct.unpack("<I", f.read(4))[0]

                # Create row group
                row_group = RowGroup(row_count=row_count)

                # Read column chunks
                for _ in range(column_count):
                    # Read column name
                    name_len = struct.unpack("<I", f.read(4))[0]
                    column_name = f.read(name_len).decode("utf-8")

                    # Read data length
                    data_len = struct.unpack("<I", f.read(4))[0]

                    # Record offset
                    offset = f.tell()

                    # Read data
                    data = f.read(data_len)

                    # Create placeholder chunk (will be populated later)
                    chunk = ColumnChunk(
                        metadata=ColumnMetadata(
                            name=column_name,
                            column_type=self.schema[column_name],
                            encoding=EncodingType.PLAIN,
                            compression=CompressionType.NONE,
                        ),
                        data=data,
                        offset=offset,
                        length=data_len,
                        row_count=row_count,
                    )

                    row_group.column_chunks[column_name] = chunk

                # Read row group metadata
                metadata_len = struct.unpack("<I", f.read(4))[0]
                metadata_json = f.read(metadata_len).decode("utf-8")
                row_group.metadata = json.loads(metadata_json)

                self.row_groups.append(row_group)

    def read_column(
        self, column_name: str, row_group_idx: Optional[int] = None
    ) -> List[Any]:
        """
        Read column from file.

        Args:
            column_name: Column name
            row_group_idx: Row group index (None for all row groups)

        Returns:
            List of values
        """
        with self._lock:
            if column_name not in self.schema:
                raise ValueError(f"Column {column_name} not in schema")

            if row_group_idx is not None:
                if row_group_idx < 0 or row_group_idx >= len(self.row_groups):
                    raise ValueError(f"Invalid row group index: {row_group_idx}")

                row_groups = [self.row_groups[row_group_idx]]
            else:
                row_groups = self.row_groups

            # Read column from row groups
            values = []

            for row_group in row_groups:
                if column_name in row_group.column_chunks:
                    chunk = row_group.column_chunks[column_name]

                    # Decode values
                    chunk_values = self.encoder.decode(
                        chunk.data,
                        self.schema[column_name],
                        chunk.metadata.encoding,
                        chunk.metadata.compression,
                        chunk.row_count,
                    )

                    values.extend(chunk_values)

            return values

    def read_row_group(self, row_group_idx: int) -> Dict[str, List[Any]]:
        """
        Read row group from file.

        Args:
            row_group_idx: Row group index

        Returns:
            Dictionary of column name -> values
        """
        with self._lock:
            if row_group_idx < 0 or row_group_idx >= len(self.row_groups):
                raise ValueError(f"Invalid row group index: {row_group_idx}")

            row_group = self.row_groups[row_group_idx]

            # Read columns
            data = {}

            for column_name, chunk in row_group.column_chunks.items():
                # Decode values
                values = self.encoder.decode(
                    chunk.data,
                    self.schema[column_name],
                    chunk.metadata.encoding,
                    chunk.metadata.compression,
                    chunk.row_count,
                )

                data[column_name] = values

            return data

    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all columns.

        Returns:
            Dictionary of column name -> statistics
        """
        with self._lock:
            statistics = {}

            for column_name in self.schema:
                column_stats = {
                    "row_count": 0,
                    "null_count": 0,
                    "min_value": None,
                    "max_value": None,
                }

                for row_group in self.row_groups:
                    if column_name in row_group.column_chunks:
                        chunk = row_group.column_chunks[column_name]

                        # Update statistics
                        column_stats["row_count"] += chunk.row_count
                        column_stats["null_count"] += chunk.metadata.null_count

                        if chunk.metadata.min_value is not None:
                            if column_stats["min_value"] is None:
                                column_stats["min_value"] = chunk.metadata.min_value
                            else:
                                column_stats["min_value"] = min(
                                    column_stats["min_value"], chunk.metadata.min_value
                                )

                        if chunk.metadata.max_value is not None:
                            if column_stats["max_value"] is None:
                                column_stats["max_value"] = chunk.metadata.max_value
                            else:
                                column_stats["max_value"] = max(
                                    column_stats["max_value"], chunk.metadata.max_value
                                )

                statistics[column_name] = column_stats

            return statistics
