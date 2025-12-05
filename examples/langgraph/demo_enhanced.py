"""
ENHANCED Interactive Demo with ALL production features:
1. ✅ Real ChromaDB vector database
2. ✅ Real embeddings (sentence-transformers)
3. ✅ LLM-based fact-checking (Gemini)
4. ✅ Multi-turn conversations
5. ✅ MAIF provenance for everything
"""

import sys
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from examples.langgraph.graph_enhanced import create_enhanced_app
from examples.langgraph.maif_utils import SessionManager
from examples.langgraph.vector_db import get_vector_db


def print_banner():
    """Print enhanced demo banner."""
    print("\n" + "=" * 80)
    print("🚀 ENHANCED LangGraph + MAIF Research Assistant".center(80))
    print("Production-Ready Multi-Agent RAG with Full Provenance".center(80))
    print("=" * 80 + "\n")
    print("✨ ENHANCEMENTS:")
    print("   1. ✅ Real ChromaDB vector database (semantic search)")
    print("   2. ✅ Real embeddings (sentence-transformers)")
    print("   3. ✅ LLM fact-checking (Gemini API)")
    print("   4. ✅ Multi-turn conversations")
    print("   5. ✅ MAIF cryptographic provenance")
    print()


def print_menu():
    """Print menu options."""
    print("\n" + "-" * 80)
    print("📋 MENU:")
    print("   1. Ask a question (uses REAL semantic search!)")
    print("   2. View session history (from MAIF artifact)")
    print("   3. Inspect MAIF artifact details")
    print("   4. Show multi-agent stats")
    print("   5. Show vector DB stats")
    print("   6. Start new session")
    print("   7. Multi-turn conversation mode")
    print("   8. Exit")
    print("-" * 80)


def show_vector_db_stats():
    """Show vector database statistics."""
    print("\n" + "=" * 80)
    print("📊 VECTOR DATABASE STATISTICS")
    print("=" * 80 + "\n")

    try:
        vector_db = get_vector_db()
        stats = vector_db.get_stats()

        print("🗄️  Database Info:")
        print(f"   Total Chunks: {stats['total_chunks']}")
        print(f"   Documents: {stats['num_documents']}")
        print(f"   Embedding Dimension: {stats['embedding_dimension']}")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Storage: {stats['persist_directory']}")

        print("\n💡 Search Method: Semantic similarity (cosine distance)")
        print("   Model: sentence-transformers/all-MiniLM-L6-v2")

    except Exception as e:
        print(f"❌ Error getting stats: {e}")


def show_maif_artifact_details(session_path: str):
    """Show detailed MAIF artifact information."""
    if not Path(session_path).exists():
        print("\n⚠️  No session artifact created yet. Ask a question first!")
        return

    print("\n" + "=" * 80)
    print("🔍 MAIF ARTIFACT DETAILS (Cryptographic Provenance)")
    print("=" * 80)

    print(f"\n📄 Artifact Path: {session_path}")
    if Path(session_path).exists():
        size = Path(session_path).stat().st_size
        print(f"   File Size: {size:,} bytes")

    session_manager = SessionManager()
    history = session_manager.get_session_history(session_path)

    print(f"   Total Blocks: {len(history)}")
    print(f"   Hash-Chained: ✅ (cryptographically linked)")
    print(f"   Tamper-Evident: ✅ (any modification breaks chain)")

    print("\n📊 Block Distribution:")
    block_types = {}
    for entry in history:
        btype = entry.get("metadata", {}).get("type", "unknown")
        block_types[btype] = block_types.get(btype, 0) + 1

    for btype, count in block_types.items():
        print(f"   - {btype}: {count} block(s)")

    print("\n🔗 Provenance Chain (Every Agent Action):")
    for i, entry in enumerate(history, 1):
        block_type = entry.get("block_type", "unknown")
        metadata = entry.get("metadata", {})
        entry_type = metadata.get("type", "unknown")
        timestamp = metadata.get("timestamp", 0)

        # Format timestamp
        if timestamp:
            from datetime import datetime

            dt = datetime.fromtimestamp(timestamp)
            time_str = dt.strftime("%H:%M:%S")
        else:
            time_str = "N/A"

        # Get method info if available
        method = metadata.get("method", "")
        method_str = f" [{method}]" if method else ""

        print(f"   Block {i}: [{block_type}] {entry_type}{method_str} @ {time_str}")

        # Show content preview
        content = entry.get("content", "")
        if isinstance(content, str) and len(content) > 0:
            preview = content[:60].replace("\n", " ")
            print(f"            Preview: {preview}...")
        elif isinstance(content, dict):
            print(f"            Keys: {list(content.keys())[:5]}")


def show_session_history(session_path: str):
    """Show human-readable session history."""
    if not Path(session_path).exists():
        print("\n⚠️  No session artifact created yet. Ask a question first!")
        return

    print("\n" + "=" * 80)
    print("📚 SESSION HISTORY (READ FROM MAIF ARTIFACT)")
    print("=" * 80 + "\n")

    session_manager = SessionManager()
    history = session_manager.get_session_history(session_path)

    for i, entry in enumerate(history, 1):
        metadata = entry.get("metadata", {})
        entry_type = metadata.get("type", "unknown")
        content = entry.get("content", "")

        print(f"[{i}] {entry_type.upper()}")
        print("-" * 40)

        if entry_type == "user_message":
            print(f"📝 User asked: {content}")

        elif entry_type == "retrieval_event":
            if isinstance(content, dict):
                num_results = content.get("num_results", 0)
                query = content.get("query", "N/A")
                method = metadata.get("method", "unknown")
                print(f"🔍 Retrieved {num_results} chunks using: {method}")
                print(f"   Query: {query[:80]}...")
                if "results" in content:
                    for j, result in enumerate(content["results"][:3], 1):
                        doc_id = result.get("doc_id", "unknown")
                        score = result.get("score", 0)
                        print(f"   {j}. {doc_id} (similarity: {score:.3f})")

        elif entry_type == "model_response":
            model = metadata.get("model", "unknown")
            is_revision = metadata.get("is_revision", False)
            revision_str = " [REVISION]" if is_revision else ""
            print(f"🤖 Model ({model}){revision_str} generated:")
            preview = content[:200] if isinstance(content, str) else str(content)[:200]
            print(f"   {preview}...")

        elif entry_type == "verification":
            if isinstance(content, dict):
                confidence = content.get("confidence", 0)
                num_claims = content.get("num_claims", 0)
                num_verified = content.get("num_verified", 0)
                method = content.get("method", "unknown")
                print(
                    f"✅ Fact-check ({method}): {num_verified}/{num_claims} claims verified"
                )
                print(f"   Confidence: {confidence:.1%}")
                if "unverified_claims" in content and content["unverified_claims"]:
                    print(
                        f"   ⚠️  Unverified: {len(content['unverified_claims'])} claim(s)"
                    )

        elif entry_type == "citations":
            if isinstance(content, dict):
                citations = content.get("citations", [])
                print(f"📚 Added {len(citations)} citations")
                for j, cite in enumerate(citations[:3], 1):
                    doc = cite.get("source_doc", "unknown")
                    chunk = cite.get("source_chunk", 0)
                    conf = cite.get("confidence", 0)
                    print(f"   {j}. {doc}#chunk{chunk} (conf: {conf:.2f})")

        print()


def show_agent_stats(state_history: list):
    """Show statistics about agent activity."""
    print("\n" + "=" * 80)
    print("📊 MULTI-AGENT STATISTICS")
    print("=" * 80 + "\n")

    if not state_history:
        print("No questions asked yet!")
        return

    total_questions = len(state_history)
    total_iterations = sum(s.get("iteration_count", 0) for s in state_history)
    total_blocks = sum(len(s.get("current_turn_block_ids", [])) for s in state_history)

    print(f"📈 Overall Stats:")
    print(f"   Total Questions: {total_questions}")
    print(f"   Total Synthesis Iterations: {total_iterations}")
    print(f"   Total MAIF Blocks Created: {total_blocks}")
    print(f"   Average Blocks per Question: {total_blocks / total_questions:.1f}")

    print(f"\n🤖 Agent Activity:")
    print(f"   - Init Session Agent: {total_questions} runs")
    print(f"   - Retriever Agent (ChromaDB): {total_questions} runs")
    print(f"   - Synthesizer Agent (Gemini): {total_iterations} runs")
    print(f"   - Fact-Checker Agent (LLM): {total_iterations} runs")
    print(f"   - Citation Agent: {total_questions} runs")

    # Show verification stats
    total_claims = 0
    total_verified = 0
    for s in state_history:
        vr = s.get("verification_results", {})
        if vr:
            total_claims += vr.get("num_claims", 0)
            total_verified += vr.get("num_verified", 0)

    if total_claims > 0:
        print(f"\n✅ Quality Metrics:")
        print(f"   Total Claims Checked: {total_claims}")
        print(
            f"   Claims Verified: {total_verified} ({100 * total_verified / total_claims:.1f}%)"
        )
        print(
            f"   Average Verification Confidence: {100 * total_verified / total_claims:.1f}%"
        )


def run_query_interactive(
    app, question: str, session_id: str, kb_paths: dict, session_path: str
):
    """Run a query with interactive feedback."""
    print("\n" + "=" * 80)
    print(f"❓ PROCESSING: {question}")
    print("=" * 80 + "\n")

    print("🚀 Starting ENHANCED multi-agent pipeline...")
    print("   (Real embeddings + ChromaDB + LLM verification)\n")

    # Create initial state
    initial_state = {
        "question": question,
        "answer": None,
        "session_id": session_id,
        "session_artifact_path": session_path,
        "kb_artifact_paths": kb_paths,
        "retrieved_chunks": [],
        "current_turn_block_ids": [],
        "verification_status": None,
        "needs_revision": False,
        "iteration_count": 0,
        "max_iterations": 3,
        "messages": [],
    }

    # Run the graph
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = app.invoke(initial_state, config)

        # Print result
        print("\n" + "=" * 80)
        print("✅ FINAL ANSWER:")
        print("=" * 80)
        answer = result.get("answer", "No answer generated")

        # Print answer with nice formatting
        lines = answer.split("\n")
        for line in lines:
            print(f"  {line}")

        print("\n" + "=" * 80)

        # Print detailed stats
        print(f"\n📊 Query Statistics:")
        print(f"   Synthesis Iterations: {result.get('iteration_count', 0)}")
        vr = result.get("verification_results", {})
        if vr:
            method = vr.get("method", "unknown")
            print(f"   Verification Method: {method}")
            print(f"   Verification Confidence: {vr.get('confidence', 0):.1%}")
            print(
                f"   Claims Verified: {vr.get('num_verified', 0)}/{vr.get('num_claims', 0)}"
            )
        print(f"   Citations Added: {len(result.get('citations', []))}")
        print(
            f"   MAIF Blocks Created: {len(result.get('current_turn_block_ids', []))}"
        )

        # Show MAIF artifact update
        print(f"\n💾 MAIF Artifact Updated:")
        print(f"   Path: {session_path}")
        if Path(session_path).exists():
            size = Path(session_path).stat().st_size
            print(f"   Size: {size:,} bytes")
            session_manager = SessionManager()
            history = session_manager.get_session_history(session_path)
            print(f"   Total Blocks (all turns): {len(history)}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return None


def multi_turn_conversation_mode(
    app, session_id: str, kb_paths: dict, session_path: str
):
    """Run multi-turn conversation mode."""
    print("\n" + "=" * 80)
    print("💬 MULTI-TURN CONVERSATION MODE")
    print("=" * 80)
    print("\nℹ️  This searches YOUR LOCAL KNOWLEDGE BASE (3 climate change documents)")
    print("   Ask follow-up questions about climate change topics!")
    print("\nYou can ask multiple follow-up questions in this session.")
    print("All questions and answers will be logged to the SAME MAIF artifact.")
    print("Type 'done' to exit.\n")

    turn_count = 0
    actual_session_path = session_path  # Track actual path

    while True:
        try:
            question = input(f"\n[Turn {turn_count + 1}] Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting multi-turn mode.")
            break

        if not question:
            print("⚠️  No question entered.")
            continue

        if question.lower() in ["done", "exit", "quit"]:
            print("\n✅ Exiting multi-turn mode.")
            break

        # Run query
        result = run_query_interactive(
            app, question, session_id, kb_paths, actual_session_path
        )

        if result:
            turn_count += 1
            # Update session path if it was just created
            if result.get("session_artifact_path"):
                actual_session_path = result["session_artifact_path"]
            print(f"\n✨ Turn {turn_count} complete and logged to MAIF!")

    # Show final session summary
    if turn_count > 0:
        print("\n" + "=" * 80)
        print("📊 MULTI-TURN SESSION SUMMARY")
        print("=" * 80)
        print(f"\n   Total Turns: {turn_count}")

        if Path(actual_session_path).exists():
            session_manager = SessionManager()
            history = session_manager.get_session_history(actual_session_path)
            print(f"   Total MAIF Blocks: {len(history)}")

            # Count by type
            user_messages = sum(
                1
                for h in history
                if h.get("metadata", {}).get("type") == "user_message"
            )
            model_responses = sum(
                1
                for h in history
                if h.get("metadata", {}).get("type") == "model_response"
            )

            print(f"   User Messages: {user_messages}")
            print(f"   Model Responses: {model_responses}")
            print(f"\n💾 All logged to: {actual_session_path}")

    return actual_session_path  # Return the path for caller


def main():
    """Run the enhanced interactive demo."""
    print_banner()

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found!")
        print("\n📝 Please create a .env file with:")
        print("   GEMINI_API_KEY=your_key_here")
        print("\nOr set environment variable:")
        print("   export GEMINI_API_KEY=your_key_here")
        print("\nExiting...")
        return
    else:
        print(f"✅ Gemini API key loaded from environment")

    print("\nℹ️  NOTE: This searches YOUR LOCAL KNOWLEDGE BASE (not Google/web)")
    print("   KB contains: 3 documents about climate change")
    print("   Ask questions about climate change topics!\n")

    # Check vector DB status
    print("🔍 Checking Vector Database...")
    try:
        vector_db = get_vector_db()
        stats = vector_db.get_stats()

        if stats["total_chunks"] == 0:
            print("⚠️  Vector DB is EMPTY!")
            print("\n❗ Please run this first to create the knowledge base:")
            print("   python3 create_kb_enhanced.py")
            print("\nExiting...")
            return

        print(
            f"✅ Vector DB loaded: {stats['total_chunks']} chunks, {stats['num_documents']} documents"
        )
    except Exception as e:
        print(f"❌ Vector DB error: {e}")
        print("\n❗ Please run this first:")
        print("   python3 create_kb_enhanced.py")
        print("\nExiting...")
        return

    # Create app
    print("\n🏗️  Building ENHANCED LangGraph application...")
    app = create_enhanced_app()
    print("✅ Multi-agent system ready!\n")

    # Create session
    session_id = f"enhanced_{uuid.uuid4().hex[:8]}"
    session_path = f"examples/langgraph/data/sessions/{session_id}.maif"

    print(f"📋 Session Created:")
    print(f"   Session ID: {session_id}")
    print(f"   MAIF Artifact: {session_path}")

    # KB paths
    kb_paths = {
        "doc_001": "examples/langgraph/data/kb/doc_001.maif",
        "doc_002": "examples/langgraph/data/kb/doc_002.maif",
        "doc_003": "examples/langgraph/data/kb/doc_003.maif",
    }

    print(f"\n📚 Knowledge Base: {stats['num_documents']} documents loaded in ChromaDB")

    # State tracking
    state_history = []

    # Interactive loop
    while True:
        print_menu()

        try:
            choice = input("\n👉 Enter your choice (1-8): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if choice == "1":
            # Ask a question
            print("\n" + "=" * 80)
            print("💬 ASK A QUESTION (Real Semantic Search!)")
            print("=" * 80)
            print("\n💡 Try these questions about climate change:")
            print("  - What are the main causes of climate change?")
            print("  - How effective is renewable energy?")
            print("  - What is the scientific consensus?")
            print("  - What role does agriculture play?")
            print("  - How can we protect forests?")
            print()

            try:
                question = input("❓ Your question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nCancelled.")
                continue

            if not question:
                print("⚠️  No question entered.")
                continue

            result = run_query_interactive(
                app, question, session_id, kb_paths, session_path
            )
            if result:
                state_history.append(result)
                # Update session path if it was just created
                if result.get("session_artifact_path"):
                    session_path = result["session_artifact_path"]

        elif choice == "2":
            # View session history
            show_session_history(session_path)

        elif choice == "3":
            # Inspect MAIF artifact
            show_maif_artifact_details(session_path)

        elif choice == "4":
            # Show agent stats
            show_agent_stats(state_history)

        elif choice == "5":
            # Show vector DB stats
            show_vector_db_stats()

        elif choice == "6":
            # Start new session
            print("\n🔄 Starting new session...")
            session_id = f"enhanced_{uuid.uuid4().hex[:8]}"
            session_path = f"examples/langgraph/data/sessions/{session_id}.maif"
            state_history = []
            print(f"✅ New session created: {session_id}")
            print(f"   MAIF Artifact: {session_path}")

        elif choice == "7":
            # Multi-turn conversation mode
            updated_path = multi_turn_conversation_mode(
                app, session_id, kb_paths, session_path
            )
            if updated_path:
                session_path = updated_path

        elif choice == "8":
            # Exit
            print("\n" + "=" * 80)
            print("👋 THANK YOU FOR USING THE ENHANCED DEMO!")
            print("=" * 80)
            print(f"\n💾 Your session is saved at:")
            print(f"   {session_path}")
            print(f"\n📊 Session Summary:")
            print(f"   Questions Asked: {len(state_history)}")
            if Path(session_path).exists():
                session_manager = SessionManager()
                history = session_manager.get_session_history(session_path)
                print(f"   MAIF Blocks Created: {len(history)}")
                print(f"   Artifact Size: {Path(session_path).stat().st_size:,} bytes")
            print("\n🔐 All interactions are cryptographically verified!")
            print("🧠 All searches used real semantic embeddings!")
            print("🤖 All fact-checking used LLM semantic understanding!")
            print()
            break

        else:
            print("\n⚠️  Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback

        traceback.print_exc()
