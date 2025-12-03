"""
Synthesize Node - Generates answer using Gemini API.
"""

import sys
import os
import json
import requests
from typing import Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from examples.langgraph.state import RAGState
from examples.langgraph.maif_utils import SessionManager


def synthesize_node(state: RAGState) -> RAGState:
    """
    Generate answer using Gemini API based on retrieved chunks.
    
    This node:
    1. Takes the question and retrieved chunks
    2. Constructs a prompt with context
    3. Calls Gemini API to generate answer
    4. Logs the response to MAIF
    5. Returns the draft answer
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with draft_answer populated
    """
    print(f"✍️  [synthesize] Generating answer (iteration {state['iteration_count'] + 1})...")
    
    question = state['question']
    chunks = state.get('retrieved_chunks', [])
    needs_revision = state.get('needs_revision', False)
    verification_results = state.get('verification_results')
    
    # Build context from retrieved chunks
    context = build_context(chunks)
    
    # Build prompt
    if needs_revision and verification_results:
        prompt = build_revision_prompt(question, context, verification_results)
        print(f"   🔄 Revising previous answer based on fact-check feedback...")
    else:
        prompt = build_initial_prompt(question, context)
        print(f"   📝 Generating initial answer...")
    
    # Call Gemini API
    answer = call_gemini_api(prompt)
    
    if answer:
        print(f"   ✅ Answer generated ({len(answer)} chars)")
        print(f"      Preview: {answer[:100]}...")
    else:
        print(f"   ❌ Failed to generate answer")
        state['error'] = "Failed to generate answer from Gemini API"
        return state
    
    # Log to MAIF
    session_manager = SessionManager()
    block_id = session_manager.log_model_response(
        session_path=state['session_artifact_path'],
        response=answer,
        model="gemini-2.0-flash",
        metadata={
            "node": "synthesize",
            "iteration": state['iteration_count'] + 1,
            "num_chunks_used": len(chunks),
            "is_revision": needs_revision
        }
    )
    
    print(f"   📝 Logged to MAIF (block: {block_id[:8]}...)")
    
    # Update state
    state['draft_answer'] = answer
    state['current_turn_block_ids'].append(block_id)
    state['iteration_count'] = state['iteration_count'] + 1
    state['needs_revision'] = False  # Reset revision flag
    
    return state


def build_context(chunks: list) -> str:
    """Build context string from retrieved chunks."""
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        doc_id = chunk.get('doc_id', 'unknown')
        chunk_idx = chunk.get('chunk_index', 0)
        text = chunk.get('text', '')
        score = chunk.get('score', 0.0)
        
        context_parts.append(
            f"[Source {i}: {doc_id}#chunk{chunk_idx}, relevance: {score:.2f}]\n{text}\n"
        )
    
    return "\n".join(context_parts)


def build_initial_prompt(question: str, context: str) -> str:
    """Build initial prompt for answer generation."""
    return f"""You are a research assistant that provides accurate, well-cited answers.

Question: {question}

Context from sources:
{context}

Instructions:
1. Answer the question based ONLY on the provided context
2. Be specific and cite which source supports each claim
3. If the context doesn't fully answer the question, say so
4. Use clear, concise language
5. Structure your answer with clear paragraphs

Answer:"""


def build_revision_prompt(question: str, context: str, verification_results: dict) -> str:
    """Build prompt for revising answer based on fact-check feedback."""
    unverified = verification_results.get('unverified_claims', [])
    contradictions = verification_results.get('contradictions', [])
    
    feedback = []
    if unverified:
        feedback.append(f"The following claims need better support: {', '.join(unverified)}")
    if contradictions:
        feedback.append(f"The following claims contradict the sources: {', '.join(contradictions)}")
    
    feedback_text = "\n".join(feedback)
    
    return f"""You are revising an answer based on fact-checking feedback.

Question: {question}

Context from sources:
{context}

Fact-checking feedback:
{feedback_text}

Instructions:
1. Revise the answer to address the feedback
2. Only include claims that are well-supported by the sources
3. Remove or qualify any unsupported claims
4. Cite sources clearly
5. Be more cautious and precise

Revised answer:"""


def call_gemini_api(prompt: str, api_key: Optional[str] = None) -> Optional[str]:
    """
    Call Gemini API to generate text.
    
    Args:
        prompt: The prompt to send to Gemini
        api_key: Optional API key (if not provided, uses env var)
        
    Returns:
        Generated text or None if failed
    """
    # Get API key from environment
    if api_key is None:
        api_key = os.getenv('GEMINI_API_KEY')
    
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env file or environment variable.")
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': api_key
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract text from response
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text']
        
        print(f"   ⚠️  Unexpected response format: {json.dumps(data)[:200]}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API request failed: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Error calling Gemini API: {e}")
        return None

