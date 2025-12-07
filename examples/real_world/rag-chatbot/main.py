#!/usr/bin/env python3
"""
RAG Chatbot with MAIF Provenance

A production-ready RAG chatbot that demonstrates:
- Document ingestion with provenance tracking
- Semantic search with citation tracking
- Multi-turn conversation with state persistence
- Full audit trail for compliance

Usage:
    # Set your API key
    export OPENAI_API_KEY=your_key
    
    # Run the chatbot
    python main.py

Requirements:
    pip install maif langgraph langchain-openai chromadb
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional
from operator import add
from datetime import datetime

# Add parent path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer


# =============================================================================
# State Definition
# =============================================================================

class Message(TypedDict):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: Optional[str]
    citations: Optional[List[str]]


class RAGState(TypedDict):
    """State for the RAG chatbot."""
    messages: Annotated[List[Message], add]
    query: str
    context: List[str]
    sources: List[str]


# =============================================================================
# Mock Components (Replace with real implementations)
# =============================================================================

# Sample knowledge base
KNOWLEDGE_BASE = {
    "product_features": {
        "content": "Our product includes real-time analytics, custom dashboards, and API access. The analytics engine processes data in under 100ms.",
        "source": "docs/product-guide.md"
    },
    "pricing": {
        "content": "Pricing starts at $29/month for the starter plan, $99/month for professional, and custom pricing for enterprise.",
        "source": "docs/pricing.md"
    },
    "support": {
        "content": "Support is available 24/7 via chat, email, and phone for enterprise customers. Response time SLA is 4 hours.",
        "source": "docs/support-policy.md"
    },
    "api": {
        "content": "The REST API supports JSON and XML formats. Authentication uses OAuth 2.0 with JWT tokens. Rate limit is 1000 requests/minute.",
        "source": "docs/api-reference.md"
    },
    "security": {
        "content": "All data is encrypted at rest (AES-256) and in transit (TLS 1.3). SOC2 Type II certified. GDPR compliant.",
        "source": "docs/security.md"
    },
}


def retrieve_documents(query: str, top_k: int = 3) -> List[dict]:
    """Simple keyword-based retrieval (replace with vector search)."""
    results = []
    query_lower = query.lower()
    
    for key, doc in KNOWLEDGE_BASE.items():
        # Simple relevance scoring
        score = sum(1 for word in query_lower.split() if word in doc["content"].lower())
        if score > 0:
            results.append({
                "content": doc["content"],
                "source": doc["source"],
                "score": score,
            })
    
    # Sort by score and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def generate_response(query: str, context: List[str], history: List[Message]) -> str:
    """Generate response (replace with LLM call)."""
    # Check for OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key and api_key != "your_key":
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
            
            messages = [
                SystemMessage(content=f"""You are a helpful assistant. Use the following context to answer questions.
                
Context:
{chr(10).join(context)}

Be concise and cite your sources when possible.""")
            ]
            
            # Add history
            for msg in history[-6:]:  # Last 3 exchanges
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            
            messages.append(HumanMessage(content=query))
            
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"[LLM Error: {e}, using mock response]")
    
    # Mock response
    if context:
        return f"Based on the documentation: {context[0][:200]}..."
    return "I don't have specific information about that. Could you rephrase your question?"


# =============================================================================
# Graph Nodes
# =============================================================================

def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents."""
    query = state["query"]
    docs = retrieve_documents(query)
    
    context = [d["content"] for d in docs]
    sources = [d["source"] for d in docs]
    
    print(f"[RETRIEVE] Found {len(docs)} relevant documents")
    
    return {
        "context": context,
        "sources": sources,
        "messages": [],
    }


def generate_node(state: RAGState) -> RAGState:
    """Generate response with citations."""
    query = state["query"]
    context = state["context"]
    sources = state["sources"]
    history = state.get("messages", [])
    
    response = generate_response(query, context, history)
    
    # Create message with citations
    message: Message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat(),
        "citations": sources if sources else None,
    }
    
    print(f"[GENERATE] Response generated with {len(sources)} citations")
    
    return {
        "messages": [message],
        "context": [],
        "sources": [],
    }


# =============================================================================
# Build Graph
# =============================================================================

def create_rag_chatbot(artifact_path: str = "rag_chatbot.maif"):
    """Create the RAG chatbot graph with MAIF provenance."""
    
    graph = StateGraph(RAGState)
    
    # Add nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    
    # Add edges
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    
    # Compile with MAIF checkpointer
    checkpointer = MAIFCheckpointer(artifact_path, agent_id="rag_chatbot")
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


# =============================================================================
# Main Application
# =============================================================================

def main():
    print("=" * 60)
    print("RAG Chatbot with MAIF Provenance")
    print("=" * 60)
    print()
    print("This chatbot demonstrates:")
    print("- Document retrieval with citations")
    print("- Multi-turn conversation")
    print("- Full cryptographic audit trail")
    print()
    print("Type 'quit' to exit, 'history' to see conversation")
    print("Type 'audit' to see provenance summary")
    print("-" * 60)
    print()
    
    # Create chatbot
    app, checkpointer = create_rag_chatbot()
    
    # Session config
    session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}
    
    conversation: List[Message] = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            break
        
        if user_input.lower() == "history":
            print("\n--- Conversation History ---")
            for msg in conversation:
                role = "You" if msg["role"] == "user" else "Assistant"
                print(f"{role}: {msg['content'][:100]}...")
                if msg.get("citations"):
                    print(f"  Sources: {', '.join(msg['citations'])}")
            print("---")
            continue
        
        if user_input.lower() == "audit":
            print("\n--- Audit Summary ---")
            from maif import MAIFDecoder
            try:
                decoder = MAIFDecoder(checkpointer.get_artifact_path())
                decoder.load()
                print(f"Total events: {len(decoder.blocks)}")
                is_valid, _ = decoder.verify_integrity()
                print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")
            except:
                print("No audit trail yet (start a conversation first)")
            print("---")
            continue
        
        # Add user message
        user_msg: Message = {
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
            "citations": None,
        }
        conversation.append(user_msg)
        
        # Process through graph
        result = app.invoke(
            {
                "messages": conversation.copy(),
                "query": user_input,
                "context": [],
                "sources": [],
            },
            config=config
        )
        
        # Get assistant response
        assistant_msgs = [m for m in result["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            assistant_msg = assistant_msgs[-1]
            conversation.append(assistant_msg)
            
            print(f"\nAssistant: {assistant_msg['content']}")
            if assistant_msg.get("citations"):
                print(f"\n  Sources: {', '.join(assistant_msg['citations'])}")
    
    # Finalize
    print("\n" + "=" * 60)
    print("Finalizing session...")
    checkpointer.finalize()
    
    # Show summary
    from maif import MAIFDecoder
    decoder = MAIFDecoder(checkpointer.get_artifact_path())
    decoder.load()
    
    print(f"\nSession saved: {checkpointer.get_artifact_path()}")
    print(f"Total events: {len(decoder.blocks)}")
    print(f"Conversation turns: {len(conversation) // 2}")
    
    is_valid, _ = decoder.verify_integrity()
    print(f"Integrity: {'VERIFIED' if is_valid else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

