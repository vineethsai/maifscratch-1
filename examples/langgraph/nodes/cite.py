"""
Citation Node - Adds citations to the final answer.
"""

import sys
import os
import re
from typing import List

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from examples.langgraph.state import RAGState, Citation
from examples.langgraph.maif_utils import SessionManager


def citation_node(state: RAGState) -> RAGState:
    """
    Add citations to the verified answer.
    
    This node:
    1. Takes the draft answer and verification results
    2. Maps claims to their supporting sources
    3. Formats the final answer with citations
    4. Logs citations to MAIF
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with final answer and citations
    """
    print(f"📚 [cite] Adding citations to answer...")
    
    draft_answer = state.get('draft_answer', '')
    chunks = state.get('retrieved_chunks', [])
    verification_results = state.get('verification_results', {})
    
    # Build citations list
    citations = build_citations(
        draft_answer,
        chunks,
        verification_results.get('verified_claims', [])
    )
    
    print(f"   ✅ Generated {len(citations)} citations")
    
    # Format final answer with inline citations
    final_answer = format_answer_with_citations(draft_answer, chunks, citations)
    
    # Add references section
    final_answer_with_refs = add_references_section(final_answer, chunks, citations)
    
    # Log citations to MAIF
    session_manager = SessionManager()
    block_id = session_manager.log_citations(
        session_path=state['session_artifact_path'],
        citations=[
            {
                "claim": c['claim'],
                "source_doc": c['source_doc'],
                "source_chunk": c['source_chunk'],
                "confidence": c['confidence']
            }
            for c in citations
        ],
        metadata={"node": "cite", "num_citations": len(citations)}
    )
    
    print(f"   📝 Logged to MAIF (block: {block_id[:8]}...)")
    print(f"\n📄 Final answer ({len(final_answer_with_refs)} chars):\n")
    print("   " + "\n   ".join(final_answer_with_refs.split('\n')[:5]))
    if len(final_answer_with_refs.split('\n')) > 5:
        print("   ...")
    
    # Update state
    state['answer'] = final_answer_with_refs
    state['citations'] = citations
    state['current_turn_block_ids'].append(block_id)
    
    return state


def build_citations(answer: str, chunks: List[dict], verified_claims: List[str]) -> List[Citation]:
    """
    Build citations mapping claims to sources.
    """
    citations = []
    
    # Simple approach: For each verified claim, find best supporting chunk
    for claim in verified_claims:
        best_chunk = find_best_supporting_chunk(claim, chunks)
        
        if best_chunk:
            citation = Citation(
                claim=claim[:100],  # Truncate for brevity
                source_doc=best_chunk['doc_id'],
                source_chunk=best_chunk['chunk_index'],
                confidence=best_chunk['score'],
                text_snippet=best_chunk['text'][:150]
            )
            citations.append(citation)
    
    return citations


def find_best_supporting_chunk(claim: str, chunks: List[dict]) -> dict:
    """Find the chunk that best supports a claim."""
    claim_lower = claim.lower()
    claim_words = set(re.findall(r'\w+', claim_lower))
    
    best_chunk = None
    best_score = 0.0
    
    for chunk in chunks:
        chunk_text = chunk.get('text', '').lower()
        chunk_words = set(re.findall(r'\w+', chunk_text))
        
        # Calculate overlap
        overlap = len(claim_words & chunk_words)
        score = overlap / len(claim_words) if claim_words else 0.0
        
        if score > best_score:
            best_score = score
            best_chunk = chunk
    
    return best_chunk


def format_answer_with_citations(answer: str, chunks: List[dict], citations: List[Citation]) -> str:
    """
    Format answer with inline citations.
    
    Simple approach: Add [1], [2] markers.
    In production, intelligently place citations near relevant sentences.
    """
    # For now, just return the answer as-is
    # In production, parse answer and insert citation markers
    return answer


def add_references_section(answer: str, chunks: List[dict], citations: List[Citation]) -> str:
    """Add a references section to the answer."""
    if not citations:
        return answer
    
    # Create unique list of sources
    sources = {}
    for citation in citations:
        key = (citation['source_doc'], citation['source_chunk'])
        if key not in sources:
            sources[key] = {
                'doc_id': citation['source_doc'],
                'chunk_index': citation['source_chunk'],
                'snippet': citation['text_snippet']
            }
    
    # Build references section
    references = ["\n\n---\n**References:**\n"]
    
    for i, (key, source) in enumerate(sources.items(), 1):
        doc_id = source['doc_id']
        chunk_idx = source['chunk_index']
        snippet = source['snippet']
        
        ref = f"[{i}] {doc_id}, chunk {chunk_idx}: \"{snippet}...\""
        references.append(ref)
    
    return answer + "\n".join(references)

