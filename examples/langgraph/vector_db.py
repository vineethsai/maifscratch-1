"""
Real Vector Database Integration using ChromaDB.

This module replaces the mock vector search with a real ChromaDB implementation.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import json


class VectorDB:
    """Real vector database using ChromaDB with sentence-transformers."""

    def __init__(
        self,
        persist_directory: str = "examples/langgraph/data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize ChromaDB client with persistent storage.

        Args:
            persist_directory: Where to store the ChromaDB database
            embedding_model: Sentence-transformers model to use
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Initialize embedding model
        print(f"📊 Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"   ✅ Model loaded (dimension: {self.embedding_dim})")

        # Get or create collection
        self.collection_name = "knowledge_base"
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None,  # We handle embeddings ourselves
            )
            print(f"📚 Loaded existing collection: {self.collection_name}")
            print(f"   Documents: {collection.count()}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"description": "MAIF knowledge base for RAG"},
            )
            print(f"📚 Created new collection: {self.collection_name}")

        return collection

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        if show_progress:
            from tqdm import tqdm

            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        else:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        return embeddings

    def add_documents(
        self, doc_id: str, chunks: List[Dict], document_metadata: Optional[Dict] = None
    ):
        """
        Add document chunks to the vector database.

        Args:
            doc_id: Unique document identifier
            chunks: List of chunks with 'text' and optional 'metadata'
            document_metadata: Optional metadata for the entire document
        """
        print(f"\n📥 Adding document to vector DB: {doc_id}")
        print(f"   Chunks: {len(chunks)}")

        # Extract texts
        texts = [chunk["text"] for chunk in chunks]

        # Generate embeddings
        print(f"   Generating embeddings...")
        embeddings = self.generate_embeddings(texts, show_progress=True)
        print(f"   ✅ Generated {len(embeddings)} embeddings")

        # Prepare IDs
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        # Prepare metadata
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {"doc_id": doc_id, "chunk_index": i, "chunk_id": f"{doc_id}_{i}"}

            # Add chunk-specific metadata
            if "metadata" in chunk:
                meta.update(chunk["metadata"])

            # Add document-level metadata
            if document_metadata:
                for key, value in document_metadata.items():
                    if key not in meta:  # Don't override chunk metadata
                        meta[f"doc_{key}"] = str(value)

            metadatas.append(meta)

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

        print(f"   ✅ Added to ChromaDB")
        print(f"   Total documents in DB: {self.collection.count()}")

    def search(
        self, query: str, top_k: int = 5, filter_doc_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_doc_ids: Optional list of doc_ids to filter by

        Returns:
            List of retrieved chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)

        # Build filter if needed
        where = None
        if filter_doc_ids:
            where = {"doc_id": {"$in": filter_doc_ids}}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        chunks = []
        for i in range(len(results["ids"][0])):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # Lower distance = higher similarity
            # Convert to 0-1 scale (approximate)
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - (distance / 2.0))  # Rough conversion

            chunk = {
                "doc_id": results["metadatas"][0][i].get("doc_id", "unknown"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "text": results["documents"][0][i],
                "score": float(score),
                "block_id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
            }
            chunks.append(chunk)

        return chunks

    def delete_document(self, doc_id: str):
        """Delete all chunks for a document."""
        # Get all IDs for this document
        results = self.collection.get(where={"doc_id": doc_id})

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"🗑️  Deleted {len(results['ids'])} chunks for document: {doc_id}")

    def clear(self):
        """Clear the entire collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        print("🗑️  Cleared vector database")

    def get_stats(self) -> Dict:
        """Get database statistics."""
        count = self.collection.count()

        # Get all metadata to count documents
        if count > 0:
            results = self.collection.get(limit=count, include=["metadatas"])
            doc_ids = set(
                meta.get("doc_id", "unknown") for meta in results["metadatas"]
            )
            num_documents = len(doc_ids)
        else:
            num_documents = 0

        return {
            "total_chunks": count,
            "num_documents": num_documents,
            "embedding_dimension": self.embedding_dim,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }


# Singleton instance
_vector_db_instance = None


def get_vector_db() -> VectorDB:
    """Get or create the global vector DB instance."""
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDB()
    return _vector_db_instance
