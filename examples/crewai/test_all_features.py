#!/usr/bin/env python3
"""
Smoke test for CrewAI enhanced demo.

Runs KB creation (if needed) and a single QA pipeline in offline mode.
Also includes test for block append fix.
"""

import os
import sys
import tempfile
import shutil
import uuid
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from maif import MAIFDecoder

from create_kb_enhanced import build_kb
from maif_utils import SessionManager
from nodes.retrieve import run_retrieve
from nodes.synthesize import run_synthesize
from nodes.fact_check import run_fact_check
from nodes.cite import run_citations
from state import RAGState


def ensure_kb():
    kb_dir = Path("examples/crewai_enhanced/data/kb")
    if not any(kb_dir.glob("*.maif")):
        build_kb()


def run_smoke():
    # Force offline path for LLM calls if key not present
    os.environ.setdefault("GEMINI_API_KEY", "")

    ensure_kb()
    session_mgr = SessionManager("examples/crewai_enhanced/data/sessions")
    session_path = session_mgr.create_session("smoke-test")

    state: RAGState = {
        "question": "What are key climate change mitigation strategies?",
        "session_artifact_path": session_path,
        "retrieved_chunks": [],
        "messages": [],
    }

    session_mgr.log_user_message(session_path, state["question"])
    state = run_retrieve(state, session_mgr)
    state = run_synthesize(state, session_mgr)
    state = run_fact_check(state, session_mgr)
    state = run_citations(state, session_mgr)

    print("Answer:", state.get("answer"))
    print("Confidence:", state.get("confidence"))
    print("Citations:", state.get("citations"))
    print("Session artifact:", session_path)


def test_block_append_fix():
    """
    Test: Block append fix - ensures all blocks are preserved when appending.
    
    This test demonstrates the critical bug fix in _add_block that seeks to EOF
    before writing. Without this fix, BlockStorage would overwrite existing blocks
    when opening an existing file, causing data loss.
    """
    print("\n" + "=" * 80)
    print("TEST: Block Append Fix (EOF Seek)")
    print("=" * 80)

    try:
        # Create temporary directory for test
        temp_dir = tempfile.mkdtemp()
        test_sessions_dir = Path(temp_dir) / "sessions"
        test_sessions_dir.mkdir(parents=True, exist_ok=True)

        session_manager = SessionManager(str(test_sessions_dir))
        test_session_id = f"append_test_{uuid.uuid4().hex[:8]}"

        # Create initial session
        session_path = session_manager.create_session(test_session_id)
        print(f"✓ Session created: {Path(session_path).name}")

        # Add multiple blocks sequentially (simulating a real RAG flow)
        print("\nAdding blocks sequentially...")
        session_manager.log_user_message(session_path, "What is climate change?")
        print("  ✓ User message logged")

        session_manager.log_retrieval_event(
            session_path,
            query="climate change",
            results=[
                {"doc_id": "doc1", "chunk_index": 0, "score": 0.95, "text": "Climate change refers to..."}
            ],
        )
        print("  ✓ Retrieval event logged")

        session_manager.log_model_response(
            session_path,
            response="Climate change is a long-term shift in global weather patterns.",
            model="gemini-2.0-flash",
        )
        print("  ✓ Model response logged")

        session_manager.log_verification(
            session_path,
            verification_results={"verified": True, "confidence": 0.85},
        )
        print("  ✓ Verification logged")

        session_manager.log_citations(
            session_path,
            citations=[{"doc_id": "doc1", "chunk_index": 0, "text": "Climate change refers to..."}],
        )
        print("  ✓ Citations logged")

        # Verify all blocks are preserved
        decoder = MAIFDecoder(session_path)
        decoder.load()
        total_blocks = len(decoder.blocks)
        print(f"\n✓ Total blocks in artifact: {total_blocks}")

        # Expected: 1 (session_init) + 5 (user_message, retrieval, model_response, verification, citations) = 6
        expected_min_blocks = 6
        if total_blocks < expected_min_blocks:
            print(f"  ✗ FAILED: Expected at least {expected_min_blocks} blocks, got {total_blocks}")
            print("  This indicates blocks were overwritten (bug exists)")
            return False

        # Verify we can read all blocks back
        history = session_manager.get_session_history(session_path)
        user_messages = [h for h in history if h.get("metadata", {}).get("type") == "user_message"]
        retrieval_events = [h for h in history if h.get("metadata", {}).get("type") == "retrieval_event"]
        model_responses = [h for h in history if h.get("metadata", {}).get("type") == "model_response"]
        verifications = [h for h in history if h.get("metadata", {}).get("type") == "verification"]
        citations = [h for h in history if h.get("metadata", {}).get("type") == "citations"]

        print(f"\n✓ Block type breakdown:")
        print(f"  - User messages: {len(user_messages)}")
        print(f"  - Retrieval events: {len(retrieval_events)}")
        print(f"  - Model responses: {len(model_responses)}")
        print(f"  - Verifications: {len(verifications)}")
        print(f"  - Citations: {len(citations)}")

        if len(user_messages) >= 1 and len(retrieval_events) >= 1 and len(model_responses) >= 1:
            print("\n✓ SUCCESS: All blocks preserved correctly!")
            print("  The _add_block fix (EOF seek) is working correctly.")
            print("  Without this fix, BlockStorage would overwrite blocks when")
            print("  opening existing files, causing data loss.")
        else:
            print("\n✗ FAILED: Some block types are missing")
            return False

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

        return True
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test_append_fix":
        # Run just the block append fix test
        success = test_block_append_fix()
        sys.exit(0 if success else 1)
    else:
        # Run smoke test
        run_smoke()
        print("\n" + "=" * 80)
        print("To test block append fix, run: python test_all_features.py test_append_fix")
        print("=" * 80)


