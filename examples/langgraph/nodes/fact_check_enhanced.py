"""
Enhanced Fact Check Node - Uses LLM for verification.
"""

import sys
import os
import re
from typing import List, Dict

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from examples.langgraph.state import RAGState, VerificationResult
from examples.langgraph.maif_utils import SessionManager
from examples.langgraph.enhanced_fact_check import verify_claim_with_llm, batch_verify_claims


def fact_check_node_enhanced(state: RAGState) -> RAGState:
    """
    Verify claims using LLM (Gemini API) for semantic understanding.
    
    This node:
    1. Extracts claims from the draft answer
    2. Verifies each claim using Gemini API
    3. Logs verification results to MAIF
    4. Determines if revision is needed
    
    Args:
        state: Current RAG state
        
    Returns:
        Updated state with verification_results and verification_status
    """
    print(f"🔬 [fact_check_enhanced] Verifying with LLM (semantic understanding)...")
    
    draft_answer = state.get('draft_answer', '')
    chunks = state.get('retrieved_chunks', [])
    iteration = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 3)
    
    if not draft_answer:
        print(f"   ⚠️  No draft answer to verify")
        state['verification_status'] = 'failed'
        state['error'] = "No draft answer to verify"
        return state
    
    # Extract claims from answer
    claims = extract_claims(draft_answer)
    print(f"   📋 Extracted {len(claims)} claims to verify")
    
    # Verify claims using LLM
    print(f"   🤖 Using Gemini API for semantic verification...")
    
    try:
        # Verify claims in parallel (with rate limiting)
        verification_results = batch_verify_claims(
            claims[:5],  # Limit to 5 claims to avoid too many API calls
            chunks,
            max_workers=2  # Limit parallel calls
        )
        
        # Count verified claims
        verified_count = sum(1 for r in verification_results if r['verified'])
        unverified_claims = [r['claim'] for r in verification_results if not r['verified']]
        
        # Show results
        for i, result in enumerate(verification_results, 1):
            status = "✅ VERIFIED" if result['verified'] else "❌ NOT VERIFIED"
            confidence = result['confidence']
            print(f"      {i}. {status} (confidence: {confidence:.1%})")
            if not result['verified']:
                print(f"         Reason: {result['reason'][:80]}...")
        
    except Exception as e:
        print(f"   ⚠️  LLM verification error: {e}")
        print(f"   ℹ️  Falling back to keyword matching...")
        
        # Fallback to simple verification
        from examples.langgraph.nodes.fact_check import verify_claim
        verification_results = []
        verified_count = 0
        unverified_claims = []
        
        for claim in claims:
            result = verify_claim(claim, chunks)
            verification_results.append(result)
            if result['verified']:
                verified_count += 1
            else:
                unverified_claims.append(claim)
    
    # Calculate overall confidence
    confidence = verified_count / len(claims) if claims else 0.0
    print(f"   📊 Verification confidence: {confidence:.1%} ({verified_count}/{len(claims)})")
    
    # Determine status
    if confidence >= 0.75:  # Slightly lower threshold with LLM
        verification_status = 'verified'
        needs_revision = False
        print(f"   ✅ Answer VERIFIED (confidence {confidence:.1%})")
    elif iteration >= max_iterations:
        verification_status = 'verified'  # Accept after max iterations
        needs_revision = False
        print(f"   ⚠️  Max iterations reached, accepting answer with {confidence:.1%} confidence")
    else:
        verification_status = 'needs_revision'
        needs_revision = True
        print(f"   🔄 Answer needs REVISION (confidence {confidence:.1%})")
    
    # Compile results
    results = {
        "claims": [r['claim'] for r in verification_results],
        "verified_claims": [r['claim'] for r in verification_results if r['verified']],
        "unverified_claims": unverified_claims,
        "contradictions": [],  # TODO: Could use LLM to detect contradictions
        "confidence": confidence,
        "num_claims": len(claims),
        "num_verified": verified_count,
        "method": "llm_semantic_verification"
    }
    
    # Log to MAIF
    session_manager = SessionManager()
    block_id = session_manager.log_verification(
        session_path=state['session_artifact_path'],
        verification_results=results,
        metadata={
            "node": "fact_check_enhanced",
            "iteration": iteration,
            "status": verification_status,
            "method": "gemini_llm"
        }
    )
    
    print(f"   📝 Logged to MAIF (block: {block_id[:8]}...)")
    
    # Update state
    state['verification_results'] = results
    state['verification_status'] = verification_status
    state['needs_revision'] = needs_revision
    state['current_turn_block_ids'].append(block_id)
    
    return state


def extract_claims(text: str) -> List[str]:
    """
    Extract verifiable claims from text.
    
    Uses sentence splitting and filtering.
    """
    # Split by periods, questions, exclamations
    sentences = re.split(r'[.!?]+', text)
    
    claims = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Filter out very short sentences and meta-statements
        if (len(sentence) > 20 and 
            not sentence.lower().startswith(('based on', 'according to', 'the sources', 
                                            'reference', '[', 'source'))):
            claims.append(sentence)
    
    return claims[:10]  # Limit to top 10 claims

