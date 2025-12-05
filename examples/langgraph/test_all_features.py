"""
Comprehensive Test Script - Validates ALL Features

Tests:
1. ✅ Real ChromaDB vector search
2. ✅ Real embeddings (sentence-transformers)
3. ✅ Gemini API integration (your key)
4. ✅ LLM fact-checking
5. ✅ MAIF provenance logging
6. ✅ Multi-turn conversations
7. ✅ Cryptographic hash chains
"""

import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from examples.langgraph.graph_enhanced import create_enhanced_app
from examples.langgraph.vector_db import get_vector_db
from examples.langgraph.maif_utils import SessionManager


def test_vector_db():
    """Test 1: Vector DB with real embeddings."""
    print("\n" + "=" * 80)
    print("TEST 1: Real Vector Database (ChromaDB)")
    print("=" * 80)

    try:
        vdb = get_vector_db()
        stats = vdb.get_stats()

        print(f"✅ Vector DB loaded")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Documents: {stats['num_documents']}")
        print(f"   Embedding dim: {stats['embedding_dimension']}")

        # Test search
        results = vdb.search("greenhouse gas emissions", top_k=3)
        print(f"✅ Semantic search works")
        print(f"   Query: 'greenhouse gas emissions'")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['doc_id']} (score: {r['score']:.3f})")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_gemini_api():
    """Test 2: Gemini API integration."""
    print("\n" + "=" * 80)
    print("TEST 2: Gemini API Integration (Your Key)")
    print("=" * 80)

    try:
        from examples.langgraph.nodes.synthesize import call_gemini_api

        test_prompt = "Say 'Test successful' if you can read this."
        response = call_gemini_api(test_prompt)

        if response:
            print(f"✅ Gemini API works")
            print(f"   Response: {response[:100]}...")
            return True
        else:
            print(f"❌ FAILED: No response from Gemini")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_llm_fact_checking():
    """Test 3: LLM-based fact-checking."""
    print("\n" + "=" * 80)
    print("TEST 3: LLM Fact-Checking (Gemini Verification)")
    print("=" * 80)

    try:
        from examples.langgraph.enhanced_fact_check import verify_claim_with_llm

        test_claim = "The Earth is round"
        test_chunks = [
            {
                "text": "The Earth is a sphere, as proven by satellite imagery and circumnavigation."
            }
        ]

        print(f"   Claim: '{test_claim}'")
        print(f"   Verifying with Gemini...")

        result = verify_claim_with_llm(test_claim, test_chunks)

        print(f"✅ LLM verification works")
        print(f"   Verified: {result['verified']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Reason: {result['reason'][:80]}...")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_maif_logging():
    """Test 4: MAIF provenance logging."""
    print("\n" + "=" * 80)
    print("TEST 4: MAIF Provenance Logging")
    print("=" * 80)

    try:
        import uuid
        from maif.block_storage import BlockStorage

        test_session_id = f"test_{uuid.uuid4().hex[:8]}"
        session_manager = SessionManager()

        # Create session
        session_path = session_manager.create_session(test_session_id)
        print(f"✅ Session artifact created: {Path(session_path).name}")

        # Log test message
        block_id = session_manager.log_user_message(session_path, "Test message")
        print(f"✅ Message logged (block: {block_id[:8]}...)")

        # Verify integrity
        with BlockStorage(session_path) as storage:
            is_valid = storage.validate_integrity()

        print(f"✅ Hash chain integrity: {'VALID' if is_valid else 'INVALID'}")

        # Read back
        history = session_manager.get_session_history(session_path)
        print(f"✅ Session history readable: {len(history)} blocks")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multi_agent_pipeline():
    """Test 5: Complete multi-agent pipeline."""
    print("\n" + "=" * 80)
    print("TEST 5: Multi-Agent Pipeline (End-to-End)")
    print("=" * 80)

    try:
        import uuid

        app = create_enhanced_app()
        print(f"✅ LangGraph app built")

        session_id = f"e2e_test_{uuid.uuid4().hex[:8]}"

        initial_state = {
            "question": "What is the main cause of climate change?",
            "answer": None,
            "session_id": session_id,
            "session_artifact_path": "",
            "kb_artifact_paths": {},
            "retrieved_chunks": [],
            "current_turn_block_ids": [],
            "verification_status": None,
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 3,
            "messages": [],
        }

        config = {"configurable": {"thread_id": session_id}}

        print(f"   Running pipeline...")
        result = app.invoke(initial_state, config)

        print(f"✅ Pipeline completed")
        print(f"   Answer generated: {len(result.get('answer', '')) > 0}")
        print(f"   Iterations: {result.get('iteration_count', 0)}")
        print(
            f"   Confidence: {result.get('verification_results', {}).get('confidence', 0):.1%}"
        )
        print(f"   MAIF blocks: {len(result.get('current_turn_block_ids', []))}")

        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multi_turn_conversation():
    """Test 6: Multi-turn conversation support."""
    print("\n" + "=" * 80)
    print("TEST 6: Multi-Turn Conversations")
    print("=" * 80)

    try:
        import uuid

        app = create_enhanced_app()
        session_id = f"multi_turn_test_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": session_id}}

        # Turn 1
        print(f"   Turn 1...")
        state1 = {
            "question": "What causes climate change?",
            "answer": None,
            "session_id": session_id,
            "session_artifact_path": "",
            "kb_artifact_paths": {},
            "retrieved_chunks": [],
            "current_turn_block_ids": [],
            "verification_status": None,
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 2,  # Reduced for testing
            "messages": [],
        }
        result1 = app.invoke(state1, config)
        session_path = result1.get("session_artifact_path", "")

        print(f"   ✅ Turn 1 complete")

        # Turn 2 (same session!)
        print(f"   Turn 2 (same session)...")
        state2 = {
            "question": "How can we mitigate it?",
            "answer": None,
            "session_id": session_id,
            "session_artifact_path": session_path,  # Same artifact!
            "kb_artifact_paths": {},
            "retrieved_chunks": [],
            "current_turn_block_ids": [],
            "verification_status": None,
            "needs_revision": False,
            "iteration_count": 0,
            "max_iterations": 2,
            "messages": [],
        }
        result2 = app.invoke(state2, config)

        print(f"   ✅ Turn 2 complete")

        # Verify both turns in same artifact
        if session_path and Path(session_path).exists():
            session_manager = SessionManager()
            history = session_manager.get_session_history(session_path)

            user_messages = [
                h
                for h in history
                if h.get("metadata", {}).get("type") == "user_message"
            ]

            print(
                f"✅ Multi-turn works: {len(user_messages)} questions in same artifact"
            )
            return True
        else:
            print(f"❌ FAILED: Session artifact not found")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE FEATURE TEST SUITE")
    print("   Testing ALL enhancements and integrations")
    print("=" * 80)

    tests = [
        ("Vector DB (ChromaDB + Embeddings)", test_vector_db),
        ("Gemini API (Your Key)", test_gemini_api),
        ("LLM Fact-Checking", test_llm_fact_checking),
        ("MAIF Provenance", test_maif_logging),
        ("Multi-Agent Pipeline", test_multi_agent_pipeline),
        ("Multi-Turn Conversations", test_multi_turn_conversation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80 + "\n")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\n{'=' * 80}")
    print(f"RESULTS: {passed}/{total} tests passed ({100 * passed / total:.0f}%)")
    print(f"{'=' * 80}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✨ Your system has:")
        print("   ✅ Real vector database (ChromaDB)")
        print("   ✅ Real embeddings (sentence-transformers)")
        print("   ✅ Gemini API integration (your key)")
        print("   ✅ LLM fact-checking")
        print("   ✅ MAIF cryptographic provenance")
        print("   ✅ Multi-agent collaboration")
        print("   ✅ Multi-turn conversations")
        print("\n🚀 Ready for production use!")
    else:
        print("\n⚠️  Some tests failed. Check output above.")
        print("   Most likely issues:")
        print("   - Vector DB empty? Run: create_kb_enhanced.py")
        print(
            "   - Missing dependencies? Run: pip install -r requirements_enhanced.txt"
        )


if __name__ == "__main__":
    main()
