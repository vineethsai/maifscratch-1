"""
MAIF utilities for CrewAI enhanced demo (sessions + KB artifacts).
"""

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Any

# Ensure repository root is on path (two levels up from examples/crewai)
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from maif import MAIFEncoder, MAIFDecoder
from maif.core.block_storage import BlockStorage
from maif.core.block_types import BlockType
from maif.core.secure_format import SecureBlockType


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = BASE_DIR / "data" / "sessions"
DEFAULT_KB_DIR = BASE_DIR / "data" / "kb"


class SessionManager:
    """Handles MAIF session artifacts for the CrewAI RAG flow."""

    def __init__(self, sessions_dir: Optional[str] = None):
        self.sessions_dir = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _add_block(self, session_path: str, block_type: str, data: bytes, metadata: Dict) -> str:
        """
        Append a block to an existing secure-format file without clobbering.
        BlockStorage does not load existing blocks, so we explicitly seek to EOF
        before writing to preserve prior blocks and ensure MAIFDecoder can read all.
        """
        with BlockStorage(session_path) as storage:
            if storage.file_handle:
                storage.file_handle.seek(0, os.SEEK_END)
            return storage.add_block(block_type=block_type, data=data, metadata=metadata)

    def create_session(self, session_id: Optional[str] = None) -> str:
        session_id = session_id or str(uuid.uuid4())
        session_path = self.sessions_dir / f"{session_id}.maif"

        # Create new session using BlockStorage (compatible with _add_block)
        session_metadata = {
            "type": "session_init",
            "session_id": session_id,
            "created_at": time.time(),
            "version": "1.0",
        }

        with BlockStorage(str(session_path)) as storage:
            storage.add_block(
                block_type="BDAT",
                data=json.dumps(session_metadata).encode("utf-8"),
                metadata={"type": "session_init"},
            )

        return str(session_path)

    def log_user_message(
        self, session_path: str, question: str, metadata: Optional[Dict] = None
    ) -> str:
        msg_metadata = metadata or {}
        msg_metadata.update({"type": "user_message", "timestamp": time.time()})
        return self._add_block(
            session_path=session_path,
            block_type=BlockType.TEXT_DATA.value,
            data=question.encode("utf-8"),
            metadata=msg_metadata,
        )

    def log_retrieval_event(
        self,
        session_path: str,
        query: str,
        results: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> str:
        retrieval_data = {
            "type": "retrieval_event",
            "query": query,
            "num_results": len(results),
            "results": [
                {
                    "doc_id": r.get("doc_id"),
                    "chunk_index": r.get("chunk_index"),
                    "score": r.get("score"),
                    "text_preview": r.get("text", "")[:100],
                }
                for r in results
            ],
            "timestamp": time.time(),
        }
        if metadata:
            retrieval_data.update(metadata)
        return self._add_block(
            session_path=session_path,
            block_type="BDAT",
            data=json.dumps(retrieval_data).encode("utf-8"),
            metadata={"type": "retrieval_event"},
        )

    def log_model_response(
        self,
        session_path: str,
        response: str,
        model: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        response_metadata = metadata or {}
        response_metadata.update(
            {"type": "model_response", "model": model, "timestamp": time.time()}
        )
        return self._add_block(
            session_path=session_path,
            block_type=BlockType.TEXT_DATA.value,
            data=response.encode("utf-8"),
            metadata=response_metadata,
        )

    def log_verification(
        self,
        session_path: str,
        verification_results: Dict,
        metadata: Optional[Dict] = None,
    ) -> str:
        verif_metadata = metadata or {}
        verif_metadata.update({"type": "verification", "timestamp": time.time()})
        return self._add_block(
            session_path=session_path,
            block_type="BDAT",
            data=json.dumps(verification_results).encode("utf-8"),
            metadata=verif_metadata,
        )

    def log_citations(
        self, session_path: str, citations: List[Dict], metadata: Optional[Dict] = None
    ) -> str:
        citation_metadata = metadata or {}
        citation_metadata.update(
            {
                "type": "citations",
                "num_citations": len(citations),
                "timestamp": time.time(),
            }
        )
        return self._add_block(
            session_path=session_path,
            block_type="BDAT",
            data=json.dumps({"citations": citations}).encode("utf-8"),
            metadata=citation_metadata,
        )

    def get_session_history(self, session_path: str) -> List[Dict]:
        decoder = MAIFDecoder(session_path)
        decoder.load()
        history: List[Dict] = []
        for block in decoder.blocks:
            content = None
            try:
                block_data = getattr(block, "data", None)
                if block_data:
                    try:
                        content = block_data.decode("utf-8")
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    except UnicodeDecodeError:
                        content = "<binary data>"
            except Exception as e:
                content = f"<error reading block: {e}>"

            history.append(
                {
                    "block_id": getattr(block, "header", block).block_id
                    if hasattr(block, "header") and hasattr(block.header, "block_id")
                    else getattr(block, "block_id", None),
                    "block_type": getattr(block, "header", block).block_type
                    if hasattr(block, "header") and hasattr(block.header, "block_type")
                    else getattr(block, "block_type", None),
                    "metadata": getattr(block, "metadata", {}),
                    "content": content,
                }
            )
        return history


class KBManager:
    """Handles MAIF KB artifacts for documents + embeddings."""

    def __init__(self, kb_dir: Optional[str] = None):
        self.kb_dir = Path(kb_dir) if kb_dir else DEFAULT_KB_DIR
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def create_kb_artifact(
        self, doc_id: str, chunks: List[Dict], document_metadata: Optional[Dict] = None
    ) -> str:
        kb_path = self.kb_dir / f"{doc_id}.maif"
        encoder = MAIFEncoder(str(kb_path), agent_id=f"kb_{doc_id}")

        doc_meta = document_metadata or {}
        doc_meta.update(
            {
                "type": "document_metadata",
                "doc_id": doc_id,
                "num_chunks": len(chunks),
                "created_at": time.time(),
            }
        )

        encoder.add_binary_block(
            data=json.dumps(doc_meta).encode("utf-8"),
            block_type=SecureBlockType.METADATA,
            metadata={"type": "document_metadata"},
        )

        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            chunk_meta = chunk.get("metadata", {})
            chunk_meta.update({"chunk_index": i, "doc_id": doc_id})

            encoder.add_text_block(text=chunk_text, metadata=chunk_meta)

            if "embedding" in chunk:
                import struct

                embedding = chunk["embedding"]
                embedding_bytes = b"".join(struct.pack("f", x) for x in embedding)
                encoder.add_binary_block(
                    data=embedding_bytes,
                    block_type=SecureBlockType.EMBEDDINGS,
                    metadata={
                        "chunk_index": i,
                        "doc_id": doc_id,
                        "dimensions": len(embedding),
                    },
                )

        encoder.finalize()
        return str(kb_path)

    def get_kb_chunks(self, kb_path: str) -> List[Dict]:
        try:
            decoder = MAIFDecoder(kb_path)
            decoder.load()
            chunks = []
            for block in decoder.blocks:
                if block.block_type == SecureBlockType.TEXT:
                    block_data = decoder.get_block_data(block.block_id)
                    if block_data:
                        text = block_data.decode("utf-8")
                        chunks.append(
                            {
                                "block_id": block.block_id,
                                "text": text,
                                "metadata": block.metadata,
                                "chunk_index": block.metadata.get("chunk_index", -1),
                            }
                        )
            chunks.sort(key=lambda x: x["chunk_index"])
            return chunks
        except Exception:
            return []


