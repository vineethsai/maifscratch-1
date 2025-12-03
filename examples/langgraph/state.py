"""
State definition for the LangGraph RAG system.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph.message import add_messages


class RAGState(TypedDict):
    """
    State for the Research Assistant with Fact-Checking Pipeline.
    
    This state is managed by LangGraph and persisted to checkpointer.
    MAIF artifacts store the durable, auditable history.
    """
    
    # User interaction
    question: str
    """The user's question"""
    
    answer: Optional[str]
    """The final answer (after fact-checking and citations)"""
    
    messages: Annotated[list, add_messages]
    """LangGraph message history for conversational context"""
    
    # MAIF persistence layer
    session_id: str
    """Unique session identifier"""
    
    session_artifact_path: str
    """Path to the session MAIF artifact (e.g., 'data/sessions/abc123.maif')"""
    
    kb_artifact_paths: Dict[str, str]
    """Mapping of doc_id -> KB artifact path"""
    
    # Working memory (ephemeral)
    retrieved_chunks: List[Dict]
    """
    Retrieved chunks from vector DB.
    Format: [{"doc_id": str, "chunk_index": int, "text": str, "score": float, "block_id": str}, ...]
    """
    
    current_turn_block_ids: List[str]
    """Block IDs created in this turn (for provenance linking)"""
    
    draft_answer: Optional[str]
    """Draft answer before fact-checking"""
    
    # Fact-checking state
    verification_results: Optional[Dict]
    """
    Results of fact-checking.
    Format: {
        "verified_claims": [...],
        "unverified_claims": [...],
        "contradictions": [...],
        "confidence": float
    }
    """
    
    verification_status: Optional[str]
    """One of: 'pending', 'verified', 'needs_revision', 'failed'"""
    
    needs_revision: bool
    """Whether the answer needs to be revised based on fact-checking"""
    
    iteration_count: int
    """Number of synthesis-fact_check iterations (prevent infinite loops)"""
    
    max_iterations: int
    """Maximum allowed iterations (default: 3)"""
    
    # Citations
    citations: Optional[List[Dict]]
    """
    Final citations.
    Format: [{"claim": str, "source_doc": str, "source_chunk": int, "confidence": float}, ...]
    """
    
    # Error handling
    error: Optional[str]
    """Error message if something fails"""


# Type for retrieved chunks
class RetrievedChunk(TypedDict):
    """Structure for a retrieved chunk."""
    doc_id: str
    chunk_index: int
    text: str
    score: float
    block_id: str  # Block ID in the KB MAIF artifact


# Type for verification results
class VerificationResult(TypedDict):
    """Structure for fact-checking verification."""
    claim: str
    verified: bool
    supporting_chunks: List[str]  # List of chunk texts
    confidence: float
    reason: Optional[str]


# Type for citation
class Citation(TypedDict):
    """Structure for a citation."""
    claim: str
    source_doc: str
    source_chunk: int
    confidence: float
    text_snippet: str

