"""
Web Search Agent - Fallback when local KB doesn't have the answer.

Adds ability to search the web if the local knowledge base doesn't contain
relevant information for the user's question.
"""

import sys
import os
import requests
from typing import List, Dict, Optional
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables
load_dotenv()

# Try to import SerpAPI
try:
    from serpapi import GoogleSearch

    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False


def should_use_web_search(retrieved_chunks: List[Dict], threshold: float = 0.3) -> bool:
    """
    Determine if web search is needed based on retrieval quality.

    Args:
        retrieved_chunks: Chunks from local KB
        threshold: Minimum score threshold for relevance

    Returns:
        True if web search should be used
    """
    if not retrieved_chunks:
        return True

    # Check if best match is below threshold
    best_score = max(chunk.get("score", 0) for chunk in retrieved_chunks)

    return best_score < threshold


def search_web(query: str, num_results: int = 3) -> List[Dict]:
    """
    Search the web using available search APIs.

    Supports (in order of priority):
    1. SerpAPI (if SERPAPI_KEY is set)
    2. ValueSERP (if VALUESERPAPI_KEY is set)
    3. DuckDuckGo HTML scraping (free fallback)

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results
    """
    print(f"   🌐 Searching the web for: '{query}'")

    # Try SerpAPI first (most reliable)
    serpapi_key = os.getenv("SERPAPI_KEY")
    if serpapi_key:
        try:
            return _search_with_serpapi(query, num_results, serpapi_key)
        except Exception as e:
            print(f"   ⚠️  SerpAPI failed: {e}")

    # Try ValueSERP
    valueserpapi_key = os.getenv("VALUESERPAPI_KEY")
    if valueserpapi_key:
        try:
            return _search_with_valueserp(query, num_results, valueserpapi_key)
        except Exception as e:
            print(f"   ⚠️  ValueSERP failed: {e}")

    # Fallback to DuckDuckGo HTML scraping
    print(f"   ℹ️  Using DuckDuckGo HTML search (no API key needed)")

    try:
        # Use DuckDuckGo HTML search
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        data = {"q": query, "b": "", "kl": "us-en"}

        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML (basic extraction)
        html = response.text
        results = []

        # Simple regex-based extraction (not robust, but works for demo)
        import re

        # Find result snippets
        snippet_pattern = r'class="result__snippet"[^>]*>([^<]+)'
        snippets = re.findall(snippet_pattern, html)

        # Find result titles
        title_pattern = r'class="result__a"[^>]*>([^<]+)'
        titles = re.findall(title_pattern, html)

        # Find URLs
        url_pattern = r'class="result__url"[^>]*>([^<]+)'
        urls = re.findall(url_pattern, html)

        # Combine results
        for i in range(min(num_results, len(snippets), len(titles))):
            if snippets[i].strip() and titles[i].strip():
                results.append(
                    {
                        "title": titles[i].strip(),
                        "snippet": snippets[i].strip(),
                        "url": urls[i].strip() if i < len(urls) else "",
                        "source": "DuckDuckGo",
                    }
                )

        if results:
            print(f"   ✅ Found {len(results)} web results")
            return results
        else:
            print(
                f"   ⚠️  No web results found (try installing duckduckgo-search: pip install duckduckgo-search)"
            )
            return []

    except Exception as e:
        print(f"   ⚠️  Web search failed: {e}")
        print(
            f"   💡 For better web search, set SERPAPI_KEY or VALUESERPAPI_KEY in .env"
        )
        return []


def _search_with_serpapi(query: str, num_results: int, api_key: str) -> List[Dict]:
    """Search using SerpAPI (Google Search)."""
    if not SERPAPI_AVAILABLE:
        raise ImportError(
            "serpapi library not installed. Run: pip install google-search-results"
        )

    print(f"   ℹ️  Using SerpAPI (Google Search)")

    params = {
        "api_key": api_key,
        "engine": "google",
        "q": query,
        "num": num_results,
        "gl": "us",
        "hl": "en",
    }

    search = GoogleSearch(params)
    data = search.get_dict()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "Google (SerpAPI)",
            }
        )

    if results:
        print(f"   ✅ Found {len(results)} web results from SerpAPI")
    return results


def _search_with_valueserp(query: str, num_results: int, api_key: str) -> List[Dict]:
    """Search using ValueSERP API."""
    print(f"   ℹ️  Using ValueSERP API")

    url = "https://api.valueserp.com/search"
    params = {"q": query, "api_key": api_key, "num": num_results, "search_type": "web"}

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url": item.get("link", ""),
                "source": "ValueSERP",
            }
        )

    if results:
        print(f"   ✅ Found {len(results)} web results from ValueSERP")
    return results


def format_web_results_as_chunks(web_results: List[Dict]) -> List[Dict]:
    """
    Format web search results as chunks for RAG pipeline.

    Args:
        web_results: Results from web search

    Returns:
        Formatted chunks compatible with RAG system
    """
    chunks = []

    for i, result in enumerate(web_results):
        chunk = {
            "doc_id": f"web_{i}",
            "chunk_index": 0,
            "text": f"{result.get('title', '')}\n\n{result.get('snippet', '')}",
            "score": 0.8,  # Give web results decent score
            "block_id": f"web_result_{i}",
            "metadata": {
                "source": "web_search",
                "url": result.get("url", ""),
                "search_engine": result.get("source", "unknown"),
            },
        }
        chunks.append(chunk)

    return chunks
