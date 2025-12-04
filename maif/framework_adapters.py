"""
MAIF Framework Integration Adapters
Provides native integration with popular AI frameworks like LangChain, LlamaIndex, and MemGPT.
"""

import os
import json
import time
import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Iterator, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Block type constants (match SecureBlockType values)
BLOCK_TYPE_TEXT = 0x54455854       # 'TEXT'
BLOCK_TYPE_EMBEDDINGS = 0x454D4244 # 'EMBD'
BLOCK_TYPE_BINARY = 0x42494E41     # 'BINA'

def _get_block_type(block) -> int:
    """Get block type as int from a SecureBlock."""
    if hasattr(block, 'header'):
        return block.header.block_type
    return getattr(block, 'block_type', 0)

def _is_text_block(block) -> bool:
    """Check if block is a text block."""
    block_type = _get_block_type(block)
    return block_type == BLOCK_TYPE_TEXT or block_type == 1

def _is_embeddings_block(block) -> bool:
    """Check if block is an embeddings block."""
    block_type = _get_block_type(block)
    return block_type == BLOCK_TYPE_EMBEDDINGS or block_type == 2

# Base classes for framework compatibility
class BaseVectorStore(ABC):
    """Base class for vector store implementations."""
    
    @abstractmethod
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, 
                  ids: Optional[List[str]] = None, **kwargs) -> List[str]:
        """Add texts to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Any]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> List[Tuple[Any, float]]:
        """Search with similarity scores."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> Optional[bool]:
        """Delete documents by ID."""
        pass

@dataclass
class Document:
    """Document class compatible with LangChain/LlamaIndex."""
    page_content: str
    metadata: Dict[str, Any]
    
    def __str__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"

# LangChain VectorStore Adapter
class MAIFLangChainVectorStore(BaseVectorStore):
    """
    LangChain-compatible VectorStore implementation using MAIF.
    Drop-in replacement for Chroma, Pinecone, etc.
    """
    
    def __init__(self, maif_path: str, collection_name: str = "default",
                 embedding_function: Optional[Any] = None):
        self.maif_path = Path(maif_path)
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Initialize MAIF components
        from .core import MAIFEncoder, MAIFDecoder
        from .semantic_optimized import OptimizedSemanticEmbedder
        
        # Create or load MAIF (v3 format - self-contained)
        if self.maif_path.exists():
            try:
                self.decoder = MAIFDecoder(str(self.maif_path))
                self.decoder.load()
                self.encoder = None  # Will create when needed
            except Exception:
                # If loading fails, start fresh
                self.encoder = MAIFEncoder(str(self.maif_path), agent_id="langchain_adapter")
                self.decoder = None
        else:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="langchain_adapter")
            self.decoder = None
        
        # Use provided embedding function or default
        if embedding_function:
            self.embedder = embedding_function
        else:
            self.embedder = OptimizedSemanticEmbedder()
        
        # Cache for fast retrieval
        self._document_cache: Dict[str, Document] = {}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load documents and embeddings into cache."""
        if not self.decoder:
            return
        
        # Load text blocks as documents
        for block in self.decoder.blocks:
            if _is_text_block(block):
                meta = block.metadata or {}
                doc_id = meta.get("doc_id", block.header.block_id.hex())
                content = block.data.decode('utf-8') if block.data else ""
                
                self._document_cache[doc_id] = Document(
                    page_content=content,
                    metadata=block.metadata or {}
                )
        
        # Load embeddings
        for block in self.decoder.blocks:
            if _is_embeddings_block(block):
                meta = block.metadata or {}
                doc_id = meta.get("doc_id")
                if doc_id:
                    embedding_data = block.data
                    # Deserialize embedding
                    import struct
                    num_floats = len(embedding_data) // 4
                    embedding = struct.unpack(f'{num_floats}f', embedding_data)
                    self._embedding_cache[doc_id] = np.array(embedding)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None,
                  ids: Optional[List[str]] = None, **kwargs) -> List[str]:
        """Add texts to the MAIF vector store."""
        if not metadatas:
            metadatas = [{} for _ in texts]
        
        if not ids:
            ids = [hashlib.md5(text.encode()).hexdigest() for text in texts]
        
        added_ids = []
        
        for text, metadata, doc_id in zip(texts, metadatas, ids):
            # Add text block
            metadata["doc_id"] = doc_id
            metadata["collection"] = self.collection_name
            metadata["timestamp"] = time.time()
            
            self.encoder.add_text_block(text, metadata)
            
            # Generate and store embedding
            if hasattr(self.embedder, 'embed_texts'):
                embeddings = self.embedder.embed_texts([text])
                embedding = embeddings[0].vector if hasattr(embeddings[0], 'vector') else embeddings[0]
            else:
                # Fallback for simple embedding functions
                embedding = self.embedder([text])[0]
            
            # Store embedding
            embedding_metadata = {
                "doc_id": doc_id,
                "dimensions": len(embedding),
                "model": getattr(self.embedder, 'model_name', 'unknown')
            }
            
            self.encoder.add_embeddings_block([embedding], embedding_metadata)
            
            # Update cache
            self._document_cache[doc_id] = Document(page_content=text, metadata=metadata)
            self._embedding_cache[doc_id] = np.array(embedding)
            
            added_ids.append(doc_id)
        
        # Save MAIF
        self._save()
        
        return added_ids
    
    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Search for similar documents."""
        results = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(self, query: str, k: int = 4, 
                                   **kwargs) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores."""
        # Generate query embedding
        if hasattr(self.embedder, 'embed_texts'):
            query_embeddings = self.embedder.embed_texts([query])
            query_embedding = query_embeddings[0].vector if hasattr(query_embeddings[0], 'vector') else query_embeddings[0]
        else:
            query_embedding = self.embedder([query])[0]
        
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_embedding in self._embedding_cache.items():
            # Cosine similarity
            similarity = np.dot(query_vec, doc_embedding) / (
                np.linalg.norm(query_vec) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for doc_id, score in similarities[:k]:
            if doc_id in self._document_cache:
                results.append((self._document_cache[doc_id], float(score)))
        
        return results
    
    def delete(self, ids: List[str], **kwargs) -> Optional[bool]:
        """Delete documents by ID."""
        # Note: MAIF is append-only, so we mark as deleted
        for doc_id in ids:
            if doc_id in self._document_cache:
                # Add deletion marker
                self.encoder.add_text_block(
                    "",
                    {"doc_id": doc_id, "deleted": True, "timestamp": time.time()}
                )
                
                # Remove from cache
                self._document_cache.pop(doc_id, None)
                self._embedding_cache.pop(doc_id, None)
        
        self._save()
        return True
    
    def _save(self):
        """Save MAIF file (v3 format)."""
        from .core import MAIFEncoder, MAIFDecoder
        if self.encoder is None:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="langchain_adapter")
        self.encoder.finalize()
        # Reload decoder
        self.decoder = MAIFDecoder(str(self.maif_path))
        self.decoder.load()
    
    def as_retriever(self, **kwargs):
        """Return a retriever interface."""
        return MAIFRetriever(vectorstore=self, **kwargs)

class MAIFRetriever:
    """Retriever interface for LangChain compatibility."""
    
    def __init__(self, vectorstore: MAIFLangChainVectorStore, search_kwargs: Optional[Dict] = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 4}
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for a query."""
        return self.vectorstore.similarity_search(query, **self.search_kwargs)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version (currently synchronous)."""
        return self.get_relevant_documents(query)

# LlamaIndex Integration
class MAIFLlamaIndexVectorStore:
    """
    LlamaIndex-compatible VectorStore implementation using MAIF.
    Native DocumentStore and VectorIndex implementation.
    """
    
    def __init__(self, maif_path: str, dim: int = 384):
        self.maif_path = Path(maif_path)
        self.dim = dim
        
        # Initialize MAIF (v3 format - self-contained)
        from .core import MAIFEncoder, MAIFDecoder
        
        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
            self.encoder = None
        else:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="llamaindex_adapter")
            self.decoder = None
        
        # Node storage
        self._nodes: Dict[str, Any] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._load_nodes()
    
    def _load_nodes(self):
        """Load nodes from MAIF."""
        if not self.decoder:
            return
        
        from .core import BlockType
        for block in self.decoder.blocks:
            block_type = block.header.block_type
            if block_type == BlockType.TEXT and block.metadata and block.metadata.get("node_id"):
                node_id = block.metadata["node_id"]
                content = block.data.decode('utf-8')
                
                self._nodes[node_id] = {
                    "id": node_id,
                    "text": content,
                    "metadata": block.metadata
                }
        
        # Load embeddings
        for block in self.decoder.blocks:
            if _is_embeddings_block(block) and (block.metadata or {}).get("node_id"):
                node_id = block.metadata["node_id"]
                embedding_data = block.data
                
                import struct
                num_floats = len(embedding_data) // 4
                embedding = struct.unpack(f'{num_floats}f', embedding_data)
                self._embeddings[node_id] = np.array(embedding)
    
    def add(self, nodes: List[Any]) -> List[str]:
        """Add nodes to the store."""
        ids = []
        
        for node in nodes:
            node_id = getattr(node, 'id_', None) or hashlib.md5(
                str(node).encode()
            ).hexdigest()
            
            # Extract text and embedding
            text = getattr(node, 'text', str(node))
            embedding = getattr(node, 'embedding', None)
            metadata = getattr(node, 'metadata', {})
            
            # Store in MAIF
            metadata["node_id"] = node_id
            metadata["timestamp"] = time.time()
            
            self.encoder.add_text_block(text, metadata)
            
            if embedding is not None:
                self.encoder.add_embeddings_block(
                    [embedding],
                    {"node_id": node_id, "dimensions": len(embedding)}
                )
                self._embeddings[node_id] = np.array(embedding)
            
            self._nodes[node_id] = {
                "id": node_id,
                "text": text,
                "metadata": metadata
            }
            
            ids.append(node_id)
        
        self._save()
        return ids
    
    def delete(self, node_id: str, **kwargs) -> None:
        """Delete a node."""
        if node_id in self._nodes:
            # Mark as deleted in MAIF
            self.encoder.add_text_block(
                "",
                {"node_id": node_id, "deleted": True, "timestamp": time.time()}
            )
            
            self._nodes.pop(node_id, None)
            self._embeddings.pop(node_id, None)
            
            self._save()
    
    def query(self, query_embedding: np.ndarray, similarity_top_k: int = 10,
              **kwargs) -> List[Tuple[str, float]]:
        """Query for similar nodes."""
        results = []
        
        for node_id, embedding in self._embeddings.items():
            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            results.append((node_id, float(similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:similarity_top_k]
    
    def get(self, node_id: str) -> Optional[Any]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def _save(self):
        """Save MAIF file (v3 format)."""
        from .core import MAIFEncoder, MAIFDecoder
        if self.encoder is None:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="llamaindex_adapter")
        self.encoder.finalize()
        # Reload decoder
        self.decoder = MAIFDecoder(str(self.maif_path))
        self.decoder.load()

# MemGPT Paging Backend
class MAIFMemGPTBackend:
    """
    MemGPT-compatible paging backend using MAIF blocks as pages.
    Implements memory hierarchy with MAIF as persistent storage.
    """
    
    def __init__(self, maif_path: str, page_size: int = 4096, max_pages_in_memory: int = 10):
        self.maif_path = Path(maif_path)
        self.page_size = page_size
        self.max_pages_in_memory = max_pages_in_memory
        
        # Initialize MAIF (v3 format - self-contained)
        from .core import MAIFEncoder, MAIFDecoder
        
        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
            self.encoder = None
        else:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="memgpt_adapter")
            self.decoder = None
        
        # Page management
        self._pages: Dict[int, Dict[str, Any]] = {}  # page_id -> page_data
        self._page_access_times: Dict[int, float] = {}  # LRU tracking
        self._dirty_pages: set = set()  # Pages that need to be written
        self._next_page_id = 0
        
        self._load_pages()
    
    def _load_pages(self):
        """Load pages from MAIF."""
        if not self.decoder:
            return
        
        for block in self.decoder.blocks:
            block_type_name = block.header.block_type.name if hasattr(block.header.block_type, 'name') else str(block.header.block_type)
            if block_type_name == "memory_page" or (block.metadata and block.metadata.get("type") == "memory_page"):
                page_id = block.metadata.get("page_id", 0) if block.metadata else 0
                page_data = json.loads(block.data.decode('utf-8'))
                
                self._pages[page_id] = page_data
                self._next_page_id = max(self._next_page_id, page_id + 1)
    
    def read_page(self, page_id: int) -> Optional[Dict[str, Any]]:
        """Read a page from memory or disk."""
        # Update access time
        self._page_access_times[page_id] = time.time()
        
        # Check if in memory
        if page_id in self._pages:
            return self._pages[page_id]
        
        # Load from MAIF if exists
        if self.decoder:
            for block in self.decoder.blocks:
                meta = block.metadata or {}
                if (meta.get("type") == "memory_page" and 
                    meta.get("page_id") == page_id):
                    page_data = json.loads(
                        block.data.decode('utf-8') if block.data else '{}'
                    )
                    
                    # Add to memory cache
                    self._pages[page_id] = page_data
                    self._evict_if_needed()
                    
                    return page_data
        
        return None
    
    def write_page(self, page_id: int, page_data: Dict[str, Any]):
        """Write a page to memory."""
        self._pages[page_id] = page_data
        self._page_access_times[page_id] = time.time()
        self._dirty_pages.add(page_id)
        
        self._evict_if_needed()
    
    def allocate_page(self) -> int:
        """Allocate a new page."""
        page_id = self._next_page_id
        self._next_page_id += 1
        
        # Initialize empty page
        self.write_page(page_id, {
            "data": "",
            "metadata": {
                "created": time.time(),
                "modified": time.time()
            }
        })
        
        return page_id
    
    def _evict_if_needed(self):
        """Evict pages if memory limit exceeded."""
        while len(self._pages) > self.max_pages_in_memory:
            # Find LRU page
            lru_page_id = min(self._page_access_times.keys(), 
                            key=lambda k: self._page_access_times[k])
            
            # Write to MAIF if dirty
            if lru_page_id in self._dirty_pages:
                self._write_page_to_maif(lru_page_id)
            
            # Remove from memory
            self._pages.pop(lru_page_id, None)
            self._page_access_times.pop(lru_page_id, None)
            self._dirty_pages.discard(lru_page_id)
    
    def _write_page_to_maif(self, page_id: int):
        """Write a page to MAIF storage."""
        if page_id not in self._pages:
            return
        
        page_data = self._pages[page_id]
        
        # Serialize page
        page_json = json.dumps(page_data)
        
        # Ensure encoder exists
        from .core import MAIFEncoder, BlockType
        if self.encoder is None:
            self.encoder = MAIFEncoder(str(self.maif_path), agent_id="memgpt_adapter")
        
        # Add to MAIF
        metadata = {
            "page_id": page_id,
            "size": len(page_json),
            "timestamp": time.time(),
            "type": "memory_page"
        }
        
        self.encoder.add_binary_block(
            page_json.encode('utf-8'),
            BlockType.BINARY,
            metadata=metadata
        )
    
    def flush(self):
        """Flush all dirty pages to MAIF (v3 format)."""
        from .core import MAIFEncoder, MAIFDecoder
        
        for page_id in list(self._dirty_pages):
            self._write_page_to_maif(page_id)
        
        self._dirty_pages.clear()
        
        # Finalize MAIF (v3 format)
        if self.encoder:
            self.encoder.finalize()
            # Reload decoder
            self.decoder = MAIFDecoder(str(self.maif_path))
            self.decoder.load()
            self.encoder = None
    
    def get_memory_context(self, start_page: int, num_pages: int) -> str:
        """Get memory context for LLM processing."""
        context_parts = []
        
        for page_id in range(start_page, start_page + num_pages):
            page_data = self.read_page(page_id)
            if page_data and "data" in page_data:
                context_parts.append(page_data["data"])
        
        return "\n".join(context_parts)
    
    def update_memory_context(self, start_page: int, new_context: str):
        """Update memory context from LLM output."""
        # Split context into pages
        pages = []
        for i in range(0, len(new_context), self.page_size):
            pages.append(new_context[i:i + self.page_size])
        
        # Update pages
        for i, page_content in enumerate(pages):
            page_id = start_page + i
            
            # Read existing page or allocate new
            page_data = self.read_page(page_id)
            if page_data is None:
                page_id = self.allocate_page()
                page_data = self.read_page(page_id)
            
            # Update content
            page_data["data"] = page_content
            page_data["metadata"]["modified"] = time.time()
            
            self.write_page(page_id, page_data)

# Semantic Kernel Connector
class MAIFSemanticKernelConnector:
    """
    Microsoft Semantic Kernel connector for MAIF.
    Provides memory and skill integration.
    """
    
    def __init__(self, maif_path: str):
        self.maif_path = Path(maif_path)
        
        # Initialize vector store for semantic memory
        self.vector_store = MAIFLangChainVectorStore(str(maif_path))
        
        # Initialize paging backend for working memory
        self.paging_backend = MAIFMemGPTBackend(str(maif_path))
    
    async def save_information_async(self, collection: str, text: str, 
                                   id: str, description: str = "", 
                                   additional_metadata: Optional[Dict] = None) -> str:
        """Save information to semantic memory."""
        metadata = {
            "collection": collection,
            "description": description,
            "timestamp": time.time()
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        ids = self.vector_store.add_texts([text], [metadata], [id])
        return ids[0]
    
    async def search_async(self, collection: str, query: str, 
                          limit: int = 5, min_relevance_score: float = 0.7) -> List[Any]:
        """Search semantic memory."""
        results = self.vector_store.similarity_search_with_score(query, k=limit)
        
        # Filter by collection and relevance
        filtered_results = []
        for doc, score in results:
            if (doc.metadata.get("collection") == collection and 
                score >= min_relevance_score):
                filtered_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance": score
                })
        
        return filtered_results
    
    def create_memory_skill(self) -> Dict[str, Any]:
        """Create a memory skill for Semantic Kernel."""
        return {
            "name": "MAIFMemory",
            "description": "MAIF-based memory system",
            "functions": {
                "save": self.save_information_async,
                "search": self.search_async
            }
        }

# Export all adapters
__all__ = [
    'MAIFLangChainVectorStore',
    'MAIFLlamaIndexVectorStore',
    'MAIFMemGPTBackend',
    'MAIFSemanticKernelConnector',
    'Document'
]