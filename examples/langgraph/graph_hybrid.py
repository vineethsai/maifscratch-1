"""
HYBRID LangGraph with local KB + web search fallback.
"""

import sys
import os
import sqlite3
from pathlib import Path

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from examples.langgraph.state import RAGState
from examples.langgraph.nodes.init_session import init_session_node
from examples.langgraph.nodes.retrieve_hybrid import retrieve_node_hybrid
from examples.langgraph.nodes.synthesize import synthesize_node
from examples.langgraph.nodes.fact_check_enhanced import fact_check_node_enhanced
from examples.langgraph.nodes.cite import citation_node


def build_hybrid_graph() -> StateGraph:
    """
    Build HYBRID LangGraph with local KB + web search.
    
    Flow:
        START
          ↓
        init_session
          ↓
        retrieve_hybrid → Tries local KB first, falls back to web
          ↓
        synthesize
          ↓
        fact_check_enhanced
          ↓
          ├─ verified? → cite → END
          └─ needs_revision? → synthesize
    
    Returns:
        StateGraph
    """
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("init_session", init_session_node)
    graph.add_node("retrieve", retrieve_node_hybrid)  # Hybrid retrieval!
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("fact_check", fact_check_node_enhanced)
    graph.add_node("cite", citation_node)
    
    # Add edges
    graph.add_edge(START, "init_session")
    graph.add_edge("init_session", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "fact_check")
    
    # Conditional routing
    def route_after_fact_check(state: RAGState) -> str:
        verification_status = state.get('verification_status', 'unknown')
        needs_revision = state.get('needs_revision', False)
        
        if needs_revision and verification_status == 'needs_revision':
            return "synthesize"
        else:
            return "cite"
    
    graph.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {
            "synthesize": "synthesize",
            "cite": "cite"
        }
    )
    
    graph.add_edge("cite", END)
    
    return graph


def create_hybrid_app(checkpoints_db: str = "examples/langgraph/data/checkpoints_hybrid.db"):
    """Create hybrid app with local KB + web search."""
    Path(checkpoints_db).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(checkpoints_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    graph = build_hybrid_graph()
    app = graph.compile(checkpointer=checkpointer)
    
    return app

