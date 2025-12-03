"""
Enhanced Fact-Checking using LLM (Gemini API).

This module replaces simple keyword matching with real semantic verification.
"""

import os
import json
import requests
from typing import List, Dict, Optional
from examples.langgraph.state import VerificationResult


def verify_claim_with_llm(claim: str, source_chunks: List[Dict],
                          api_key: Optional[str] = None) -> VerificationResult:
    """
    Verify a claim against source chunks using Gemini API.
    
    Args:
        claim: The claim to verify
        source_chunks: List of source chunks
        api_key: Optional Gemini API key
        
    Returns:
        VerificationResult with verification details
    """
    # Build context from source chunks
    context = "\n\n".join([
        f"[Source {i+1}] {chunk.get('text', '')}"
        for i, chunk in enumerate(source_chunks[:5])  # Use top 5 chunks
    ])
    
    # Build verification prompt
    prompt = f"""You are a fact-checker. Your job is to verify if a claim is supported by the provided sources.

Claim to verify:
"{claim}"

Sources:
{context}

Task:
1. Determine if the claim is SUPPORTED, CONTRADICTED, or UNVERIFIED by the sources
2. Provide a confidence score (0.0 to 1.0)
3. Explain your reasoning

Respond in JSON format:
{{
    "verdict": "SUPPORTED" | "CONTRADICTED" | "UNVERIFIED",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}

JSON Response:"""
    
    # Call Gemini API
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
                    response_text = parts[0]['text']
                    
                    # Try to parse JSON
                    try:
                        # Extract JSON from response (might have markdown code blocks)
                        if '```json' in response_text:
                            json_start = response_text.find('```json') + 7
                            json_end = response_text.find('```', json_start)
                            response_text = response_text[json_start:json_end]
                        elif '```' in response_text:
                            json_start = response_text.find('```') + 3
                            json_end = response_text.find('```', json_start)
                            response_text = response_text[json_start:json_end]
                        
                        result = json.loads(response_text.strip())
                        
                        verdict = result.get('verdict', 'UNVERIFIED')
                        confidence = float(result.get('confidence', 0.5))
                        reasoning = result.get('reasoning', 'No reasoning provided')
                        
                        # Determine if verified
                        verified = verdict == "SUPPORTED" and confidence > 0.6
                        
                        return VerificationResult(
                            claim=claim,
                            verified=verified,
                            supporting_chunks=[chunk['text'][:100] for chunk in source_chunks[:3]],
                            confidence=confidence,
                            reason=f"{verdict}: {reasoning}"
                        )
                        
                    except json.JSONDecodeError:
                        # Fallback: Simple keyword matching
                        print(f"      ⚠️  Could not parse JSON response, using fallback")
                        return _fallback_verification(claim, source_chunks)
        
        # Fallback if response format unexpected
        return _fallback_verification(claim, source_chunks)
        
    except Exception as e:
        print(f"      ⚠️  API error: {e}, using fallback")
        return _fallback_verification(claim, source_chunks)


def _fallback_verification(claim: str, chunks: List[Dict]) -> VerificationResult:
    """Fallback verification using keyword matching."""
    import re
    
    claim_lower = claim.lower()
    claim_words = set(re.findall(r'\w+', claim_lower))
    
    best_match_score = 0.0
    supporting_chunks = []
    
    for chunk in chunks:
        chunk_text = chunk.get('text', '').lower()
        chunk_words = set(re.findall(r'\w+', chunk_text))
        
        # Calculate simple overlap score
        overlap = len(claim_words & chunk_words)
        score = overlap / len(claim_words) if claim_words else 0.0
        
        if score > 0.3:
            supporting_chunks.append(chunk['text'][:100])
            best_match_score = max(best_match_score, score)
    
    verified = best_match_score > 0.4
    
    return VerificationResult(
        claim=claim,
        verified=verified,
        supporting_chunks=supporting_chunks,
        confidence=best_match_score,
        reason="Keyword-based fallback verification"
    )


def batch_verify_claims(claims: List[str], chunks: List[Dict],
                       max_workers: int = 3) -> List[VerificationResult]:
    """
    Verify multiple claims in parallel (with rate limiting).
    
    Args:
        claims: List of claims to verify
        chunks: Source chunks
        max_workers: Maximum parallel API calls
        
    Returns:
        List of verification results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = []
    
    # Use ThreadPoolExecutor for parallel API calls (with limit)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all verification tasks
        future_to_claim = {
            executor.submit(verify_claim_with_llm, claim, chunks): claim
            for claim in claims
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_claim):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                claim = future_to_claim[future]
                print(f"      ⚠️  Error verifying claim: {e}")
                # Add failed result
                results.append(VerificationResult(
                    claim=claim,
                    verified=False,
                    supporting_chunks=[],
                    confidence=0.0,
                    reason=f"Verification failed: {str(e)}"
                ))
    
    return results

