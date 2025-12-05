"""
Hybrid Retrieve Node - Uses local KB first, then web search if needed.
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
from examples.langgraph.web_search_agent import (
    should_use_web_search,
    search_web,
    format_web_results_as_chunks,
)


def retrieve_node_hybrid(state: RAGState) -> RAGState:
    """
    Hybrid retrieval: Try local KB first, fallback to web search if needed.

    This node:
    1. Searches local ChromaDB
    2. If results are poor quality → searches web
    3. Combines results
    4. Logs everything to MAIF

    Args:
        state: Current RAG state

    Returns:
        Updated state with retrieved_chunks
    """
    print(f"🔍 [retrieve_hybrid] Intelligent search (local KB + web fallback)...")
    print(f"   Question: {state['question']}")

    question = state["question"]
    session_manager = SessionManager()

    # First, try local KB
    try:
        vector_db = get_vector_db()
        stats = vector_db.get_stats()

        print(f"   📚 Searching local KB ({stats['total_chunks']} chunks)...")
        local_chunks = vector_db.search(question, top_k=5)

        print(f"   ✅ Found {len(local_chunks)} local chunks")
        for i, chunk in enumerate(local_chunks[:3], 1):
            print(f"      {i}. [{chunk['doc_id']}] Score: {chunk['score']:.3f}")
            print(f"         {chunk['text'][:80]}...")

    except Exception as e:
        print(f"   ⚠️  Local KB error: {e}")
        local_chunks = []

    # Check if web search is needed
    use_web = should_use_web_search(local_chunks, threshold=0.35)

    if use_web:
        best_score = local_chunks[0]["score"] if local_chunks else 0
        print(
            f"\n   ℹ️  Local KB doesn't have good answer (best score: {best_score:.3f})"
        )
        print(f"   🌐 Falling back to web search...")

        # Search web
        web_results = search_web(question, num_results=3)

        if web_results:
            # Format as chunks
            web_chunks = format_web_results_as_chunks(web_results)

            # Combine with local results
            retrieved_chunks = web_chunks + local_chunks[:2]  # Web results first

            print(
                f"   ✅ Combined: {len(web_chunks)} web + {min(2, len(local_chunks))} local"
            )
        else:
            retrieved_chunks = local_chunks
            print(f"   ⚠️  Web search failed, using local results only")
    else:
        retrieved_chunks = local_chunks
        print(f"   ✅ Local KB has good results, using them")

    # Log retrieval event to MAIF
    block_id = session_manager.log_retrieval_event(
        session_path=state["session_artifact_path"],
        query=question,
        results=retrieved_chunks,
        metadata={
            "node": "retrieve_hybrid",
            "used_web_search": use_web,
            "num_local": len(local_chunks),
            "num_web": len(
                [c for c in retrieved_chunks if c["doc_id"].startswith("web_")]
            ),
            "method": "hybrid_chromadb_web",
        },
    )

    print(f"   📝 Logged to MAIF (block: {block_id[:8]}...)")

    # Update state
    state["retrieved_chunks"] = retrieved_chunks
    state["current_turn_block_ids"].append(block_id)

    return state
