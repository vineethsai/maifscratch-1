"""
MAIF Simple API - Easy-to-use interface for MAIF files

This API provides a simplified interface for creating and working with MAIF files.
MAIF v3 uses a self-contained secure format with Ed25519 signatures.

Usage:
    # Create a new MAIF
    maif = MAIF("my_agent")
    maif.add_text("Hello world!")
    maif.add_text("More content")
    maif.save("my_artifact.maif")

    # Load existing
    maif = MAIF.load("my_artifact.maif")
    print(maif.texts)
"""

import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path


class MAIF:
    """
    Simple MAIF API - One class to rule them all.

    This provides a high-level interface for creating and reading MAIF files
    without worrying about the underlying format details.

    Examples:
        # Create a new MAIF file
        maif = MAIF("my-agent")
        maif.add_text("Hello, world!")
        maif.add_text("This is AI-generated content")
        maif.save("output.maif")

        # Load and inspect
        maif = MAIF.load("output.maif")
        for text in maif.texts:
            print(text)

        # Check integrity
        if maif.verify():
            print("File is untampered")
    """

    def __init__(self, agent_id: str = "default-agent"):
        """
        Initialize MAIF with an agent identifier.

        Args:
            agent_id: Identifier for the agent creating this file
        """
        self.agent_id = agent_id
        self._name = f"{agent_id}_artifact"

        # Content storage (before saving)
        self._texts: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._binaries: List[Dict[str, Any]] = []

        # Loaded state (after loading)
        self._decoder = None
        self._file_path: Optional[str] = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        """Get the name of this MAIF artifact."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the name of this MAIF artifact."""
        self._name = value

    # =========================================================================
    # Content Addition Methods
    # =========================================================================

    def add_text(self, content: str, metadata: Optional[Dict] = None) -> "MAIF":
        """
        Add text content.

        Args:
            content: Text string to add
            metadata: Optional metadata dict

        Returns:
            self (for chaining)
        """
        self._texts.append({"content": content, "metadata": metadata or {}})
        return self

    def add_embeddings(
        self, vectors: List[List[float]], metadata: Optional[Dict] = None
    ) -> "MAIF":
        """
        Add embedding vectors.

        Args:
            vectors: List of embedding vectors
            metadata: Optional metadata

        Returns:
            self (for chaining)
        """
        self._embeddings.extend(vectors)
        return self

    def add_binary(
        self, data: bytes, name: str = "binary", metadata: Optional[Dict] = None
    ) -> "MAIF":
        """
        Add binary data.

        Args:
            data: Binary data
            name: Name for this binary block
            metadata: Optional metadata

        Returns:
            self (for chaining)
        """
        self._binaries.append({"data": data, "name": name, "metadata": metadata or {}})
        return self

    def add_image(self, path: str, metadata: Optional[Dict] = None) -> "MAIF":
        """
        Add image from file path.

        Args:
            path: Path to image file
            metadata: Optional metadata

        Returns:
            self (for chaining)
        """
        with open(path, "rb") as f:
            data = f.read()

        meta = metadata or {}
        meta["source_path"] = path
        meta["filename"] = os.path.basename(path)

        return self.add_binary(data, f"image:{os.path.basename(path)}", meta)

    # =========================================================================
    # Save and Load
    # =========================================================================

    def save(self, filepath: str) -> str:
        """
        Save to a MAIF file.

        Args:
            filepath: Output file path

        Returns:
            Path to created file
        """
        from maif import MAIFEncoder, BlockType

        encoder = MAIFEncoder(filepath, agent_id=self.agent_id)

        # Add text blocks
        for item in self._texts:
            encoder.add_text_block(item["content"], item["metadata"])

        # Add embeddings
        if self._embeddings:
            encoder.add_embeddings_block(self._embeddings)

        # Add binary blocks
        for item in self._binaries:
            encoder.add_binary_block(item["data"], BlockType.BINARY, item["metadata"])

        encoder.finalize()
        self._file_path = filepath
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "MAIF":
        """
        Load an existing MAIF file.

        Args:
            filepath: Path to MAIF file

        Returns:
            MAIF instance with loaded content
        """
        from maif import MAIFDecoder

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MAIF file not found: {filepath}")

        instance = cls()
        instance._file_path = filepath
        instance._decoder = MAIFDecoder(filepath)
        instance._decoder.load()
        instance._is_loaded = True

        # Extract agent ID from file info
        file_info = instance._decoder.get_file_info()
        instance.agent_id = file_info.get("agent_did", "unknown").replace(
            "did:maif:", ""
        )

        return instance

    # =========================================================================
    # Content Access (after loading)
    # =========================================================================

    @property
    def texts(self) -> List[str]:
        """Get all text content."""
        if self._is_loaded and self._decoder:
            from maif import BlockType

            result = []
            for i, block in enumerate(self._decoder.blocks):
                if block.header.block_type == BlockType.TEXT:
                    text = self._decoder.get_text_content(i)
                    if text:
                        result.append(text)
            return result
        return [item["content"] for item in self._texts]

    @property
    def blocks(self) -> List[Dict]:
        """Get all blocks as dictionaries."""
        if self._is_loaded and self._decoder:
            return [
                {
                    "index": i,
                    "type": block.header.block_type,
                    "size": block.header.size,
                    "hash": block.header.content_hash.hex(),
                    "metadata": block.metadata,
                }
                for i, block in enumerate(self._decoder.blocks)
            ]
        return []

    @property
    def provenance(self) -> List[Dict]:
        """Get provenance chain."""
        if self._is_loaded and self._decoder:
            return [e.to_dict() for e in self._decoder.provenance]
        return []

    @property
    def file_info(self) -> Dict[str, Any]:
        """Get file information."""
        if self._is_loaded and self._decoder:
            return self._decoder.get_file_info()
        return {
            "agent_id": self.agent_id,
            "text_count": len(self._texts),
            "embedding_count": len(self._embeddings),
            "binary_count": len(self._binaries),
        }

    # =========================================================================
    # Verification
    # =========================================================================

    def verify(self) -> bool:
        """
        Verify file integrity.

        Returns:
            True if file is valid and untampered
        """
        if not self._is_loaded or not self._decoder:
            return True  # Nothing to verify yet

        is_valid, _ = self._decoder.verify_integrity()
        return is_valid

    def is_tampered(self) -> bool:
        """Check if file has been tampered with."""
        if not self._is_loaded or not self._decoder:
            return False
        return self._decoder.is_tampered()

    def get_content_list(self) -> List[Dict[str, Any]]:
        """
        Get a list of all content items with metadata.

        Returns:
            List of dicts with 'type', 'content', and 'metadata' keys
        """
        content_list = []

        # Add texts
        for i, item in enumerate(self._texts):
            content_list.append(
                {
                    "index": i,
                    "type": "text",
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                }
            )

        # Add embeddings info
        if self._embeddings:
            content_list.append(
                {
                    "index": len(self._texts),
                    "type": "embeddings",
                    "content": f"{len(self._embeddings)} vectors",
                    "metadata": {},
                }
            )

        # If loaded from file, get from decoder
        if self._is_loaded and self._decoder:
            content_list = []
            # Block type constants (match SecureBlockType values)
            BLOCK_TYPE_TEXT = 0x54455854  # 'TEXT'
            BLOCK_TYPE_EMBEDDINGS = 0x454D4244  # 'EMBD'

            for i, block in enumerate(self._decoder.blocks):
                block_type = block.header.block_type
                if block_type == BLOCK_TYPE_TEXT:
                    type_name = "text"
                elif block_type == BLOCK_TYPE_EMBEDDINGS:
                    type_name = "embeddings"
                else:
                    type_name = "binary"

                content_list.append(
                    {
                        "index": i,
                        "type": type_name,
                        "content": block.data.decode("utf-8", errors="replace")
                        if block.data
                        else "",
                        "metadata": block.metadata or {},
                    }
                )

        return content_list

    # =========================================================================
    # Utilities
    # =========================================================================

    def __repr__(self) -> str:
        if self._is_loaded:
            return f"<MAIF loaded from '{self._file_path}', {len(self._decoder.blocks)} blocks>"
        return f"<MAIF agent='{self.agent_id}', {len(self._texts)} texts, {len(self._embeddings)} embeddings>"

    def summary(self) -> str:
        """Get a summary of this MAIF."""
        lines = [f"MAIF Summary: {self.name}"]
        lines.append(f"  Agent: {self.agent_id}")

        if self._is_loaded:
            info = self.file_info
            lines.append(f"  File: {self._file_path}")
            lines.append(f"  Version: {info.get('version', 'unknown')}")
            lines.append(f"  Blocks: {info.get('block_count', 0)}")
            lines.append(f"  Signed: {'Yes' if info.get('is_signed') else 'No'}")
            lines.append(f"  Finalized: {'Yes' if info.get('is_finalized') else 'No'}")
        else:
            lines.append(f"  Texts: {len(self._texts)}")
            lines.append(f"  Embeddings: {len(self._embeddings)}")
            lines.append(f"  Binaries: {len(self._binaries)}")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_maif(
    output_path: str,
    texts: List[str] = None,
    embeddings: List[List[float]] = None,
    agent_id: str = "default-agent",
) -> str:
    """
    Quick function to create a MAIF file.

    Args:
        output_path: Output file path
        texts: List of text strings
        embeddings: List of embedding vectors
        agent_id: Agent identifier

    Returns:
        Path to created file
    """
    maif = MAIF(agent_id)

    if texts:
        for text in texts:
            maif.add_text(text)

    if embeddings:
        maif.add_embeddings(embeddings)

    return maif.save(output_path)


def load_maif(filepath: str) -> MAIF:
    """
    Load a MAIF file.

    Args:
        filepath: Path to MAIF file

    Returns:
        MAIF instance
    """
    return MAIF.load(filepath)


def quick_text_maif(text: str, output_path: str, agent_id: str = "quick") -> str:
    """
    Create a simple MAIF file with just text.

    Args:
        text: Text content
        output_path: Output path
        agent_id: Agent ID

    Returns:
        Path to created file
    """
    return create_maif(output_path, texts=[text], agent_id=agent_id)


def quick_multimodal_maif(
    texts: List[str],
    embeddings: List[List[float]],
    output_path: str,
    agent_id: str = "multimodal",
) -> str:
    """
    Create a MAIF file with text and embeddings.

    Args:
        texts: List of texts
        embeddings: List of embedding vectors
        output_path: Output path
        agent_id: Agent ID

    Returns:
        Path to created file
    """
    return create_maif(
        output_path, texts=texts, embeddings=embeddings, agent_id=agent_id
    )


__all__ = [
    "MAIF",
    "create_maif",
    "load_maif",
    "quick_text_maif",
    "quick_multimodal_maif",
]
