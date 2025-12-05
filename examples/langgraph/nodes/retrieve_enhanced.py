"""
Enhanced Retrieve Node - Uses REAL ChromaDB vector search.
"""

import sys
import os

# Add parent directories to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from examples.langgraph.state import RAGState
from examples.langgraph.maif_utils import SessionManager
from examples.langgraph.vector_db import get_vector_db


def retrieve_node_enhanced(state: RAGState) -> RAGState:
    """
    Retrieve relevant chunks using REAL vector database (ChromaDB).

    This node:
    1. Takes the user question
    2. Generates embedding using sentence-transformers
    3. Queries ChromaDB for semantic similarity
    4. Logs the retrieval event to MAIF
    5. Returns retrieved chunks in state

    Args:
        state: Current RAG state

    Returns:
        Updated state with retrieved_chunks populated
    """
    print(f"🔍 [retrieve_enhanced] Searching with REAL vector DB...")
    print(f"   Question: {state['question']}")

    question = state["question"]
    session_manager = SessionManager()

    # Get vector DB instance
    try:
        vector_db = get_vector_db()

        # Check if DB has content
        stats = vector_db.get_stats()
        print(
            f"   📊 Vector DB stats: {stats['total_chunks']} chunks, {stats['num_documents']} docs"
        )

        if stats["total_chunks"] == 0:
            print(f"   ⚠️  Vector DB is empty! Run create_kb_enhanced.py first")
            # Return empty results
            state["retrieved_chunks"] = []
            state["error"] = "Vector database is empty"
            return state

        # Perform real semantic search
        print(f"   🧠 Generating query embedding...")
        retrieved_chunks = vector_db.search(question, top_k=5)

        print(f"   ✅ Retrieved {len(retrieved_chunks)} chunks (REAL semantic search!)")
        for i, chunk in enumerate(retrieved_chunks[:3], 1):  # Show first 3
            print(f"      {i}. [{chunk['doc_id']}] Score: {chunk['score']:.3f}")
            print(f"         {chunk['text'][:80]}...")

    except Exception as e:
        print(f"   ❌ Vector DB error: {e}")
        print(f"   ℹ️  Falling back to mock retrieval...")

        # Fallback to mock (shouldn't happen in enhanced mode)
        from examples.langgraph.nodes.retrieve import mock_vector_search

        retrieved_chunks = mock_vector_search(
            question, state.get("kb_artifact_paths", {})
        )

    # Log retrieval event to MAIF
    block_id = session_manager.log_retrieval_event(
        session_path=state["session_artifact_path"],
        query=question,
        results=retrieved_chunks,
        metadata={"node": "retrieve_enhanced", "method": "chromadb_semantic_search"},
    )

    print(f"   📝 Logged to MAIF (block: {block_id[:8]}...)")

    # Update state
    state["retrieved_chunks"] = retrieved_chunks
    state["current_turn_block_ids"].append(block_id)

    return state
