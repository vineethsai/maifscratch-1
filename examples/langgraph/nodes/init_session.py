"""
Init Session Node - Ensures session artifact exists.
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from examples.langgraph.state import RAGState
from examples.langgraph.maif_utils import SessionManager


def init_session_node(state: RAGState) -> RAGState:
    """
    Initialize or load a session artifact.

    This node:
    1. Checks if a session artifact already exists
    2. If not, creates a new one
    3. Sets the session_artifact_path in state

    Args:
        state: Current RAG state

    Returns:
        Updated state with session_artifact_path set
    """
    print(f"🔧 [init_session] Initializing session: {state['session_id']}")

    session_manager = SessionManager()

    # Check if session artifact already exists
    session_path = f"examples/langgraph/data/sessions/{state['session_id']}.maif"

    if not Path(session_path).exists():
        print(f"   Creating new session artifact...")
        session_path = session_manager.create_session(state["session_id"])
        print(f"   ✅ Session artifact created: {session_path}")
    else:
        print(f"   ✅ Session artifact exists: {session_path}")

    # Update state
    state["session_artifact_path"] = session_path

    # Initialize other state fields if not present
    if "current_turn_block_ids" not in state or state["current_turn_block_ids"] is None:
        state["current_turn_block_ids"] = []

    if "iteration_count" not in state or state["iteration_count"] is None:
        state["iteration_count"] = 0

    if "max_iterations" not in state or state["max_iterations"] is None:
        state["max_iterations"] = 3

    if "needs_revision" not in state or state["needs_revision"] is None:
        state["needs_revision"] = False

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    return state
