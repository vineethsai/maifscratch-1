"""
HYBRID Demo - Searches local KB first, falls back to web if needed.

This version adds web search capability so you can ask ANY question,
not just questions about the local knowledge base.
"""

import sys
import os
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from examples.langgraph.graph_hybrid import create_hybrid_app
from examples.langgraph.demo_enhanced import (
    print_menu,
    show_session_history,
    show_maif_artifact_details,
    show_agent_stats,
    show_vector_db_stats,
    run_query_interactive,
)


def print_banner():
    """Print hybrid demo banner."""
    print("\n" + "=" * 80)
    print("🌐 HYBRID LangGraph + MAIF Research Assistant".center(80))
    print("Local KB + Web Search with Cryptographic Provenance".center(80))
    print("=" * 80 + "\n")
    print("✨ FEATURES:")
    print("   1. ✅ Searches local KB first (ChromaDB semantic search)")
    print("   2. ✅ Falls back to web if KB doesn't have answer")
    print("   3. ✅ Real embeddings (sentence-transformers)")
    print("   4. ✅ LLM fact-checking (Gemini API)")
    print("   5. ✅ Multi-turn conversations")
    print("   6. ✅ MAIF cryptographic provenance")
    print()
    print("💡 TIP: You can now ask questions about:")
    print("   - Climate change (from local KB)")
    print("   - OR anything else (from web search)")
    print()


def main():
    """Run hybrid demo."""
    print_banner()

    # Create app
    print("🏗️  Building HYBRID LangGraph application...")
    app = create_hybrid_app()
    print("✅ Multi-agent system ready!\n")

    # Create session
    session_id = f"hybrid_{uuid.uuid4().hex[:8]}"
    session_path = f"examples/langgraph/data/sessions/{session_id}.maif"

    print(f"📋 Session Created:")
    print(f"   Session ID: {session_id}")
    print(f"   MAIF Artifact: {session_path}")

    # Check vector DB
    from examples.langgraph.vector_db import get_vector_db

    try:
        vdb = get_vector_db()
        stats = vdb.get_stats()
        print(
            f"\n📚 Local Knowledge Base: {stats['num_documents']} documents, {stats['total_chunks']} chunks"
        )
        print("🌐 Web Search: Enabled (DuckDuckGo fallback)")
    except:
        print("\n⚠️  Local KB not loaded, will use web search only")

    kb_paths = {
        "doc_001": "examples/langgraph/data/kb/doc_001.maif",
        "doc_002": "examples/langgraph/data/kb/doc_002.maif",
        "doc_003": "examples/langgraph/data/kb/doc_003.maif",
    }

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
            print("\n" + "=" * 80)
            print("💬 ASK ANY QUESTION (Hybrid Search!)")
            print("=" * 80)
            print("\n💡 Try these questions:")
            print("  📚 Local KB (climate change):")
            print("     - What causes climate change?")
            print("     - How effective is renewable energy?")
            print()
            print("  🌐 Web Search (anything else):")
            print("     - What is the capital of France?")
            print("     - Who won the 2024 Olympics?")
            print("     - What is quantum computing?")
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
                if result.get("session_artifact_path"):
                    session_path = result["session_artifact_path"]

        elif choice == "2":
            show_session_history(session_path)

        elif choice == "3":
            show_maif_artifact_details(session_path)

        elif choice == "4":
            show_agent_stats(state_history)

        elif choice == "5":
            show_vector_db_stats()

        elif choice == "6":
            print("\n🔄 Starting new session...")
            session_id = f"hybrid_{uuid.uuid4().hex[:8]}"
            session_path = f"examples/langgraph/data/sessions/{session_id}.maif"
            state_history = []
            print(f"✅ New session: {session_id}")

        elif choice == "7":
            print("\n💬 Multi-turn mode coming soon in hybrid version!")
            print("   Use single questions for now")

        elif choice == "8":
            print("\n" + "=" * 80)
            print("👋 THANK YOU!")
            print("=" * 80)
            print(f"\n💾 Session: {session_path}")
            if Path(session_path).exists():
                print(f"   Questions: {len(state_history)}")
            print("\n🌐 You asked questions using hybrid search!")
            print()
            break

        else:
            print("\n⚠️  Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Bye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
