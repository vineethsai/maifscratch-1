"""
MAIF Block Type Definitions and FourCC Implementation
Implements the hierarchical block structure specified in the paper.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import struct
import uuid


class BlockType(Enum):
    """Standard MAIF block types with FourCC identifiers."""

    HEADER = "HDER"
    TEXT_DATA = "TEXT"
    EMBEDDING = "EMBD"
    KNOWLEDGE_GRAPH = "KGRF"
    SECURITY = "SECU"
    BINARY_DATA = "BDAT"
    VIDEO_DATA = "VDAT"
    AUDIO_DATA = "AUDI"
    IMAGE_DATA = "IDAT"
    CROSS_MODAL = "XMOD"
    SEMANTIC_BINDING = "SBND"
    COMPRESSED_EMBEDDINGS = "CEMB"
    PROVENANCE = "PROV"
    ACCESS_CONTROL = "ACLS"
    LIFECYCLE = "LIFE"


@dataclass
class BlockHeader:
    """ISO BMFF-style block header."""

    size: int
    type: str  # FourCC
    version: int = 1
    flags: int = 0
    uuid: Optional[str] = None

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid4())

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        # Standard 32-byte header: size(4) + type(4) + version(4) + flags(4) + uuid(16)
        header = struct.pack(
            ">I4sII",
            self.size,
            self.type.encode("ascii")[:4].ljust(4, b"\0"),
            self.version,
            self.flags,
        )
        header += uuid.UUID(self.uuid).bytes
        return header

    @classmethod
    def from_bytes(cls, data: bytes) -> "BlockHeader":
        """Deserialize header from bytes."""
        size, type_bytes, version, flags = struct.unpack(">I4sII", data[:16])
        block_uuid = str(uuid.UUID(bytes=data[16:32]))
        return cls(
            size=size,
            type=type_bytes.decode("ascii").rstrip("\0"),
            version=version,
            flags=flags,
            uuid=block_uuid,
        )


class BlockFactory:
    """Factory for creating typed blocks."""

    @staticmethod
    def create_header_block(
        maif_version: str, creator_id: str, root_hash: str
    ) -> Dict[str, Any]:
        """Create HDER block."""
        return {
            "type": BlockType.HEADER.value,
            "data": {
                "maif_version": maif_version,
                "creator_id": creator_id,
                "root_hash": root_hash,
                "created_timestamp": __import__("time").time(),
                "format_features": [
                    "multimodal",
                    "semantic",
                    "cryptographic",
                    "streaming",
                ],
            },
        }

    @staticmethod
    def create_text_block(
        text: str, language: str = "en", encoding: str = "utf-8"
    ) -> Dict[str, Any]:
        """Create TEXT block."""
        return {
            "type": BlockType.TEXT_DATA.value,
            "data": text.encode(encoding),
            "metadata": {
                "language": language,
                "encoding": encoding,
                "length": len(text),
                "word_count": len(text.split()),
            },
        }

    @staticmethod
    def create_embedding_block(
        embeddings: List[List[float]], model_name: str, dimensions: int
    ) -> Dict[str, Any]:
        """Create EMBD block."""
        # Pack embeddings as binary data
        embedding_data = b""
        for embedding in embeddings:
            for value in embedding:
                embedding_data += struct.pack("f", value)

        return {
            "type": BlockType.EMBEDDING.value,
            "data": embedding_data,
            "metadata": {
                "model_name": model_name,
                "dimensions": dimensions,
                "count": len(embeddings),
                "data_type": "float32",
                "indexing": "dense",
            },
        }

    @staticmethod
    def create_knowledge_graph_block(
        triples: List[Dict], format_type: str = "json-ld"
    ) -> Dict[str, Any]:
        """Create KGRF block."""
        import json

        if format_type == "json-ld":
            kg_data = {
                "@context": "https://schema.org/",
                "@type": "KnowledgeGraph",
                "triples": triples,
            }
            data = json.dumps(kg_data).encode("utf-8")
        else:
            # Fallback to simple JSON
            data = json.dumps({"triples": triples}).encode("utf-8")

        return {
            "type": BlockType.KNOWLEDGE_GRAPH.value,
            "data": data,
            "metadata": {
                "format": format_type,
                "triple_count": len(triples),
                "compression": "none",
                "namespace_uris": [],
            },
        }

    @staticmethod
    def create_security_block(
        signatures: Dict, certificates: List, access_control: Dict
    ) -> Dict[str, Any]:
        """Create SECU block."""
        import json

        security_data = {
            "digital_signatures": signatures,
            "certificates": certificates,
            "access_control": access_control,
            "timestamp": __import__("time").time(),
        }

        return {
            "type": BlockType.SECURITY.value,
            "data": json.dumps(security_data).encode("utf-8"),
            "metadata": {
                "signature_algorithm": "ECDSA-P256",
                "hash_algorithm": "SHA-256",
                "certificate_format": "X.509",
                "access_control_version": "1.0",
            },
        }


class BlockValidator:
    """Validates block structure and content."""

    @staticmethod
    def validate_block_header(header: BlockHeader) -> List[str]:
        """Validate block header structure."""
        errors = []

        if header.size <= 32:  # Must be larger than header
            errors.append("Block size must be larger than header size")

        if len(header.type) != 4:
            errors.append("Block type must be 4 characters (FourCC)")

        if header.version < 1:
            errors.append("Block version must be >= 1")

        try:
            uuid.UUID(header.uuid)
        except ValueError:
            errors.append("Invalid UUID format")

        return errors

    @staticmethod
    def validate_block_type(block_type: str) -> bool:
        """Validate if block type is supported."""
        return any(bt.value == block_type for bt in BlockType)

    @staticmethod
    def validate_block_data(block_type: str, data: bytes, metadata: Dict) -> List[str]:
        """Validate block data based on type."""
        errors = []

        if block_type == BlockType.TEXT_DATA.value:
            encoding = metadata.get("encoding", "utf-8")
            try:
                data.decode(encoding)
            except UnicodeDecodeError:
                errors.append(f"Invalid {encoding} encoding in text block")

        elif block_type == BlockType.EMBEDDING.value:
            dimensions = metadata.get("dimensions", 0)
            count = metadata.get("count", 0)
            expected_size = dimensions * count * 4  # float32
            if len(data) != expected_size:
                errors.append(
                    f"Embedding data size mismatch: expected {expected_size}, got {len(data)}"
                )

        elif block_type == BlockType.KNOWLEDGE_GRAPH.value:
            try:
                import json

                json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                errors.append("Invalid JSON in knowledge graph block")

        return errors


# Export main classes
__all__ = ["BlockType", "BlockHeader", "BlockFactory", "BlockValidator"]
