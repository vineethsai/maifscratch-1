"""
MAIF utility functions for session and KB management.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Any
import sys
import os

# Add parent directory to path for MAIF imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from maif.block_storage import BlockStorage
from maif.block_types import BlockType
from maif.core import MAIFEncoder, MAIFDecoder
from maif.security import MAIFSigner


class SessionManager:
    """Manages MAIF session artifacts for conversation tracking."""
    
    def __init__(self, sessions_dir: str = "data/sessions"):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session artifact.
        
        Returns:
            session_artifact_path: Path to the created MAIF artifact
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        session_path = self.sessions_dir / f"{session_id}.maif"
        
        # Create new session using MAIFEncoder
        encoder = MAIFEncoder(agent_id=f"session_{session_id}")
        
        # Add session initialization metadata
        session_metadata = {
            "type": "session_init",
            "session_id": session_id,
            "created_at": time.time(),
            "version": "1.0"
        }
        
        encoder.add_binary_block(
            data=json.dumps(session_metadata).encode('utf-8'),
            block_type="BDAT",  # Binary data - 4 chars
            metadata={"type": "session_init"}
        )
        
        # Save the session artifact
        manifest_path = str(session_path).replace('.maif', '_manifest.json')
        encoder.build_maif(str(session_path), manifest_path)
        
        return str(session_path)
    
    def log_user_message(self, session_path: str, question: str, metadata: Optional[Dict] = None) -> str:
        """
        Log a user message to the session artifact.
        
        Returns:
            block_id: ID of the created block
        """
        with BlockStorage(session_path) as storage:
            msg_metadata = metadata or {}
            msg_metadata.update({
                "type": "user_message",
                "timestamp": time.time(),
            })
            
            block_id = storage.add_block(
                block_type=BlockType.TEXT_DATA.value,
                data=question.encode('utf-8'),
                metadata=msg_metadata
            )
            
        return block_id
    
    def log_retrieval_event(self, session_path: str, query: str, 
                           results: List[Dict], metadata: Optional[Dict] = None) -> str:
        """
        Log a retrieval event to the session artifact.
        
        Args:
            session_path: Path to session artifact
            query: The query that was used
            results: List of retrieved chunks
            metadata: Optional additional metadata
            
        Returns:
            block_id: ID of the created block
        """
        with BlockStorage(session_path) as storage:
            retrieval_data = {
                "type": "retrieval_event",
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "doc_id": r.get("doc_id"),
                        "chunk_index": r.get("chunk_index"),
                        "score": r.get("score"),
                        "text_preview": r.get("text", "")[:100]  # First 100 chars
                    }
                    for r in results
                ],
                "timestamp": time.time(),
            }
            
            if metadata:
                retrieval_data.update(metadata)
            
            block_id = storage.add_block(
                block_type="BDAT",  # Binary data - 4 chars
                data=json.dumps(retrieval_data).encode('utf-8'),
                metadata={"type": "retrieval_event"}
            )
            
        return block_id
    
    def log_model_response(self, session_path: str, response: str, 
                          model: str, metadata: Optional[Dict] = None) -> str:
        """
        Log a model response to the session artifact.
        
        Returns:
            block_id: ID of the created block
        """
        with BlockStorage(session_path) as storage:
            response_metadata = metadata or {}
            response_metadata.update({
                "type": "model_response",
                "model": model,
                "timestamp": time.time(),
            })
            
            block_id = storage.add_block(
                block_type=BlockType.TEXT_DATA.value,
                data=response.encode('utf-8'),
                metadata=response_metadata
            )
            
        return block_id
    
    def log_verification(self, session_path: str, verification_results: Dict,
                        metadata: Optional[Dict] = None) -> str:
        """
        Log fact-checking verification results.
        
        Returns:
            block_id: ID of the created block
        """
        with BlockStorage(session_path) as storage:
            verif_metadata = metadata or {}
            verif_metadata.update({
                "type": "verification",
                "timestamp": time.time(),
            })
            
            block_id = storage.add_block(
                block_type="BDAT",  # Binary data - 4 chars
                data=json.dumps(verification_results).encode('utf-8'),
                metadata=verif_metadata
            )
            
        return block_id
    
    def log_citations(self, session_path: str, citations: List[Dict],
                     metadata: Optional[Dict] = None) -> str:
        """
        Log citations to the session artifact.
        
        Returns:
            block_id: ID of the created block
        """
        with BlockStorage(session_path) as storage:
            citation_metadata = metadata or {}
            citation_metadata.update({
                "type": "citations",
                "num_citations": len(citations),
                "timestamp": time.time(),
            })
            
            block_id = storage.add_block(
                block_type="BDAT",  # Binary data - 4 chars
                data=json.dumps({"citations": citations}).encode('utf-8'),
                metadata=citation_metadata
            )
            
        return block_id
    
    def get_session_history(self, session_path: str) -> List[Dict]:
        """
        Get the full history from a session artifact.
        
        Returns:
            List of blocks with their data and metadata
        """
        manifest_path = session_path.replace('.maif', '_manifest.json')
        
        try:
            decoder = MAIFDecoder(session_path, manifest_path)
            
            history = []
            for block in decoder.blocks:
                block_data = decoder.get_block_data(block.block_id)
                
                if block_data:
                    try:
                        # Try to decode as text
                        content = block_data.decode('utf-8')
                        # Try to parse as JSON
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass  # Keep as string
                    except UnicodeDecodeError:
                        content = "<binary data>"
                else:
                    content = None
                
                history.append({
                    "block_id": block.block_id,
                    "block_type": block.block_type,
                    "metadata": block.metadata,
                    "content": content
                })
            
            return history
        except Exception as e:
            print(f"Error reading session history: {e}")
            return []


class KBManager:
    """Manages MAIF knowledge base artifacts."""
    
    def __init__(self, kb_dir: str = "data/kb"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
    
    def create_kb_artifact(self, doc_id: str, chunks: List[Dict],
                          document_metadata: Optional[Dict] = None) -> str:
        """
        Create a KB artifact from document chunks.
        
        Args:
            doc_id: Unique document identifier
            chunks: List of chunks, each with 'text', 'embedding' (optional), 'metadata' (optional)
            document_metadata: Optional metadata for the entire document
            
        Returns:
            kb_artifact_path: Path to the created MAIF artifact
        """
        kb_path = self.kb_dir / f"{doc_id}.maif"
        
        encoder = MAIFEncoder(agent_id=f"kb_{doc_id}")
        
        # Add document metadata block
        doc_meta = document_metadata or {}
        doc_meta.update({
            "type": "document_metadata",
            "doc_id": doc_id,
            "num_chunks": len(chunks),
            "created_at": time.time()
        })
        
        encoder.add_binary_block(
            data=json.dumps(doc_meta).encode('utf-8'),
            block_type="METADATA",
            metadata={"type": "document_metadata"}
        )
        
        # Add each chunk
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get('text', '')
            chunk_meta = chunk.get('metadata', {})
            chunk_meta.update({
                "chunk_index": i,
                "doc_id": doc_id
            })
            
            # Add text block
            encoder.add_text_block(
                text=chunk_text,
                metadata=chunk_meta
            )
            
            # Add embedding if present
            if 'embedding' in chunk:
                embedding = chunk['embedding']
                # Convert to bytes
                import struct
                embedding_bytes = b''.join(struct.pack('f', x) for x in embedding)
                
                encoder.add_binary_block(
                    data=embedding_bytes,
                    block_type="EMBEDDING",
                    metadata={
                        "chunk_index": i,
                        "doc_id": doc_id,
                        "dimensions": len(embedding)
                    }
                )
        
        # Save KB artifact
        manifest_path = str(kb_path).replace('.maif', '_manifest.json')
        encoder.build_maif(str(kb_path), manifest_path)
        
        return str(kb_path)
    
    def get_kb_chunks(self, kb_path: str) -> List[Dict]:
        """
        Get all chunks from a KB artifact.
        
        Returns:
            List of chunks with text and metadata
        """
        manifest_path = kb_path.replace('.maif', '_manifest.json')
        
        try:
            decoder = MAIFDecoder(kb_path, manifest_path)
            
            chunks = []
            for block in decoder.blocks:
                if block.block_type == "text":
                    block_data = decoder.get_block_data(block.block_id)
                    if block_data:
                        text = block_data.decode('utf-8')
                        chunks.append({
                            "block_id": block.block_id,
                            "text": text,
                            "metadata": block.metadata,
                            "chunk_index": block.metadata.get("chunk_index", -1)
                        })
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x['chunk_index'])
            return chunks
        except Exception as e:
            print(f"Error reading KB chunks: {e}")
            return []

