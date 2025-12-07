"""
MAIF VectorStore for LangChain.

A vector store that persists embeddings to MAIF artifacts
with full provenance tracking.
"""

import time
import hashlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from pathlib import Path

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    VectorStore = object
    Document = dict
    Embeddings = object

import numpy as np

from maif.integrations._base import EventType, MAIFProvenanceTracker
from maif.integrations._utils import safe_serialize


class MAIFVectorStore(VectorStore if LANGCHAIN_AVAILABLE else object):
    """MAIF-backed vector store for LangChain.
    
    Stores document embeddings in a MAIF artifact with cryptographic
    provenance tracking. Every add, update, and search operation is
    logged for audit purposes.
    
    Usage:
        from langchain_openai import OpenAIEmbeddings
        from maif.integrations.langchain import MAIFVectorStore
        
        embeddings = OpenAIEmbeddings()
        vectorstore = MAIFVectorStore(
            embedding=embeddings,
            artifact_path="vectors.maif"
        )
        
        # Add documents
        vectorstore.add_texts(["Hello world", "Goodbye world"])
        
        # Search
        results = vectorstore.similarity_search("Hello")
        
        # Finalize
        vectorstore.finalize()
    """
    
    def __init__(
        self,
        embedding: Embeddings,
        artifact_path: str,
        agent_id: str = "maif_vectorstore",
    ):
        """Initialize the vector store.
        
        Args:
            embedding: Embeddings model to use
            artifact_path: Path to the MAIF artifact
            agent_id: Identifier for this store
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is required for MAIFVectorStore. "
                "Install with: pip install langchain-core"
            )
        
        self._embedding = embedding
        self.artifact_path = Path(artifact_path)
        self._agent_id = agent_id
        
        # Initialize tracker
        self._tracker = MAIFProvenanceTracker(
            artifact_path=artifact_path,
            agent_id=agent_id,
            auto_finalize=False,
        )
        
        # In-memory storage
        self._documents: List[Document] = []
        self._embeddings: List[List[float]] = []
        self._ids: List[str] = []
        
        # Load existing if artifact exists
        self._load_existing()
    
    def _load_existing(self) -> None:
        """Load existing vectors from MAIF artifact."""
        if not self.artifact_path.exists():
            return
        
        try:
            from maif import MAIFDecoder
            import json
            
            decoder = MAIFDecoder(str(self.artifact_path))
            decoder.load()
            
            for block in decoder.blocks:
                meta = block.metadata or {}
                if meta.get("type") == "vector_add":
                    try:
                        data = block.data
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        event_data = json.loads(data).get("data", {})
                        
                        doc_id = event_data.get("id")
                        content = event_data.get("content", "")
                        doc_metadata = event_data.get("metadata", {})
                        embedding = event_data.get("embedding", [])
                        
                        if doc_id and doc_id not in self._ids:
                            self._ids.append(doc_id)
                            self._documents.append(Document(
                                page_content=content,
                                metadata=doc_metadata,
                            ))
                            self._embeddings.append(embedding)
                    except:
                        pass
        except Exception:
            pass
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    @property
    def embeddings(self) -> Embeddings:
        """Return the embeddings model."""
        return self._embedding
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.
        
        Args:
            texts: Texts to add
            metadatas: Optional metadata for each text
            
        Returns:
            List of IDs for the added texts
        """
        texts_list = list(texts)
        metadatas = metadatas or [{}] * len(texts_list)
        
        # Generate embeddings
        embeddings = self._embedding.embed_documents(texts_list)
        
        ids = []
        for i, (text, embedding, metadata) in enumerate(zip(texts_list, embeddings, metadatas)):
            doc_id = self._generate_id(text)
            ids.append(doc_id)
            
            # Store in memory
            self._ids.append(doc_id)
            self._documents.append(Document(page_content=text, metadata=metadata))
            self._embeddings.append(embedding)
            
            # Log to MAIF
            self._tracker.log_event(
                event_type=EventType.STATE_CHECKPOINT,
                data={
                    "id": doc_id,
                    "content": text[:1000],  # Limit size
                    "metadata": metadata,
                    "embedding": embedding[:10],  # Store sample
                    "embedding_dim": len(embedding),
                },
                metadata={
                    "type": "vector_add",
                    "doc_id": doc_id,
                    "timestamp": time.time(),
                },
            )
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of similar documents
        """
        # Get query embedding
        query_embedding = self._embedding.embed_query(query)
        
        # Calculate similarities
        results = self.similarity_search_by_vector(query_embedding, k=k)
        
        # Log search
        self._tracker.log_event(
            event_type=EventType.NODE_END,
            data={
                "query": query[:500],
                "k": k,
                "num_results": len(results),
                "result_ids": [self._generate_id(d.page_content) for d in results],
            },
            metadata={
                "type": "vector_search",
                "timestamp": time.time(),
            },
        )
        
        return results
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search by embedding vector.
        
        Args:
            embedding: Query embedding
            k: Number of results
            
        Returns:
            List of similar documents
        """
        if not self._embeddings:
            return []
        
        # Calculate cosine similarity
        query_vec = np.array(embedding)
        similarities = []
        
        for i, doc_embedding in enumerate(self._embeddings):
            doc_vec = np.array(doc_embedding)
            
            # Cosine similarity
            dot = np.dot(query_vec, doc_vec)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            similarity = dot / norm if norm > 0 else 0
            
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        results = []
        for idx, _ in similarities[:k]:
            results.append(self._documents[idx])
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search with similarity scores.
        
        Args:
            query: Query text
            k: Number of results
            
        Returns:
            List of (document, score) tuples
        """
        query_embedding = self._embedding.embed_query(query)
        
        if not self._embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        results = []
        
        for i, doc_embedding in enumerate(self._embeddings):
            doc_vec = np.array(doc_embedding)
            dot = np.dot(query_vec, doc_vec)
            norm = np.linalg.norm(query_vec) * np.linalg.norm(doc_vec)
            similarity = dot / norm if norm > 0 else 0
            results.append((self._documents[i], similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    @classmethod
    def from_texts(
        cls: Type["MAIFVectorStore"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        artifact_path: str = "vectorstore.maif",
        **kwargs: Any,
    ) -> "MAIFVectorStore":
        """Create vector store from texts.
        
        Args:
            texts: Texts to add
            embedding: Embeddings model
            metadatas: Optional metadata
            artifact_path: Path for MAIF artifact
            
        Returns:
            Initialized vector store
        """
        store = cls(embedding=embedding, artifact_path=artifact_path)
        store.add_texts(texts, metadatas=metadatas)
        return store
    
    def finalize(self) -> None:
        """Finalize the MAIF artifact."""
        self._tracker.finalize()
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self._documents)
    
    def __enter__(self) -> "MAIFVectorStore":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()

