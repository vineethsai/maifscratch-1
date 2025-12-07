#!/usr/bin/env python3
"""
Research Assistant with MAIF Provenance

A research assistant that demonstrates:
- Document analysis with citation tracking
- Multi-step research workflows
- Source verification and fact-checking
- Complete research audit trail

Usage:
    python main.py

Requirements:
    pip install maif langgraph
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, List, Annotated, Optional
from operator import add
from datetime import datetime
import hashlib

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from langgraph.graph import StateGraph, START, END
from maif.integrations.langgraph import MAIFCheckpointer


# =============================================================================
# State
# =============================================================================

class Source(TypedDict):
    title: str
    url: str
    snippet: str
    reliability_score: float
    accessed_at: str


class Finding(TypedDict):
    claim: str
    sources: List[str]
    confidence: float
    verified: bool


class ResearchState(TypedDict):
    """State for research workflow."""
    messages: Annotated[List[dict], add]
    query: str
    sources: List[Source]
    findings: List[Finding]
    summary: str
    research_log: Annotated[List[dict], add]


# =============================================================================
# Mock Data Sources
# =============================================================================

MOCK_SOURCES = {
    "climate": [
        {
            "title": "IPCC Climate Report 2023",
            "url": "https://ipcc.ch/report/2023",
            "snippet": "Global temperatures have risen 1.1°C above pre-industrial levels. Without significant emissions reductions, warming could reach 2.7°C by 2100.",
            "reliability_score": 0.95,
        },
        {
            "title": "NASA Climate Data",
            "url": "https://climate.nasa.gov/data",
            "snippet": "Arctic sea ice has declined by 13% per decade since 1979. The last decade was the warmest on record.",
            "reliability_score": 0.98,
        },
    ],
    "ai": [
        {
            "title": "State of AI Report 2024",
            "url": "https://stateofai.com/2024",
            "snippet": "Large language models have achieved human-level performance on many benchmarks. AI adoption in enterprise grew 250% in 2023.",
            "reliability_score": 0.85,
        },
        {
            "title": "AI Safety Research Survey",
            "url": "https://arxiv.org/abs/ai-safety-2024",
            "snippet": "Alignment research has accelerated with 3x more papers published in 2023. Key challenges remain in interpretability and robustness.",
            "reliability_score": 0.90,
        },
    ],
    "health": [
        {
            "title": "WHO Health Statistics 2024",
            "url": "https://who.int/statistics/2024",
            "snippet": "Global life expectancy increased to 73.4 years. Cardiovascular disease remains the leading cause of death worldwide.",
            "reliability_score": 0.97,
        },
    ],
}


# =============================================================================
# Research Agents
# =============================================================================

def search_agent(state: ResearchState) -> ResearchState:
    """Search for relevant sources."""
    query = state["query"].lower()
    
    found_sources = []
    for topic, sources in MOCK_SOURCES.items():
        if topic in query or any(word in query for word in topic.split()):
            for src in sources:
                found_sources.append({
                    **src,
                    "accessed_at": datetime.now().isoformat(),
                })
    
    # Default sources if none found
    if not found_sources:
        found_sources = [{
            "title": "General Knowledge Base",
            "url": "https://example.com/kb",
            "snippet": "This is a general knowledge source for your query.",
            "reliability_score": 0.70,
            "accessed_at": datetime.now().isoformat(),
        }]
    
    print(f"[SEARCH] Found {len(found_sources)} sources")
    
    return {
        "sources": found_sources,
        "messages": [],
        "findings": [],
        "summary": "",
        "research_log": [{
            "agent": "search",
            "action": "search_sources",
            "sources_found": len(found_sources),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def analyze_agent(state: ResearchState) -> ResearchState:
    """Analyze sources and extract findings."""
    sources = state["sources"]
    
    findings = []
    for src in sources:
        # Extract key claims from snippet
        snippet = src["snippet"]
        sentences = snippet.split(". ")
        
        for sentence in sentences:
            if len(sentence) > 20:  # Skip very short sentences
                finding: Finding = {
                    "claim": sentence.strip(),
                    "sources": [src["url"]],
                    "confidence": src["reliability_score"],
                    "verified": src["reliability_score"] > 0.85,
                }
                findings.append(finding)
    
    print(f"[ANALYZE] Extracted {len(findings)} findings")
    
    return {
        "findings": findings,
        "messages": [],
        "sources": [],
        "summary": "",
        "research_log": [{
            "agent": "analyze",
            "action": "extract_findings",
            "findings_count": len(findings),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def verify_agent(state: ResearchState) -> ResearchState:
    """Verify findings and cross-reference."""
    findings = state["findings"]
    
    verified_count = sum(1 for f in findings if f["verified"])
    
    print(f"[VERIFY] {verified_count}/{len(findings)} findings verified")
    
    return {
        "findings": [],
        "messages": [],
        "sources": [],
        "summary": "",
        "research_log": [{
            "agent": "verify",
            "action": "cross_reference",
            "verified_count": verified_count,
            "total_count": len(findings),
            "timestamp": datetime.now().isoformat(),
        }],
    }


def synthesize_agent(state: ResearchState) -> ResearchState:
    """Synthesize findings into a summary."""
    findings = state["findings"]
    sources = state["sources"]
    
    # Build summary
    summary_parts = [
        f"## Research Summary: {state['query']}",
        f"\n**Sources Analyzed:** {len(sources) if sources else 'Multiple'}",
        f"**Key Findings:** {len(findings)}",
        "\n### Findings:\n",
    ]
    
    for i, finding in enumerate(findings[:5], 1):  # Top 5 findings
        confidence = "High" if finding["confidence"] > 0.9 else "Medium" if finding["confidence"] > 0.8 else "Low"
        summary_parts.append(f"{i}. {finding['claim']} (Confidence: {confidence})")
    
    summary_parts.append("\n### Sources:")
    seen_urls = set()
    for finding in findings:
        for url in finding["sources"]:
            if url not in seen_urls:
                seen_urls.add(url)
                summary_parts.append(f"- {url}")
    
    summary = "\n".join(summary_parts)
    
    print(f"[SYNTHESIZE] Generated summary ({len(summary)} chars)")
    
    return {
        "summary": summary,
        "messages": [{
            "role": "assistant",
            "content": summary,
        }],
        "findings": [],
        "sources": [],
        "research_log": [{
            "agent": "synthesize",
            "action": "generate_summary",
            "summary_length": len(summary),
            "timestamp": datetime.now().isoformat(),
        }],
    }


# =============================================================================
# Build Graph
# =============================================================================

def create_research_assistant(artifact_path: str = "research.maif"):
    """Create the research assistant graph."""
    
    graph = StateGraph(ResearchState)
    
    graph.add_node("search", search_agent)
    graph.add_node("analyze", analyze_agent)
    graph.add_node("verify", verify_agent)
    graph.add_node("synthesize", synthesize_agent)
    
    graph.add_edge(START, "search")
    graph.add_edge("search", "analyze")
    graph.add_edge("analyze", "verify")
    graph.add_edge("verify", "synthesize")
    graph.add_edge("synthesize", END)
    
    checkpointer = MAIFCheckpointer(artifact_path, agent_id="research_assistant")
    app = graph.compile(checkpointer=checkpointer)
    
    return app, checkpointer


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Research Assistant with MAIF Provenance")
    print("=" * 60)
    print()
    print("Pipeline: Search → Analyze → Verify → Synthesize")
    print("Try: 'climate change', 'AI trends', 'health statistics'")
    print("Type 'quit' to exit")
    print("-" * 60)
    
    app, checkpointer = create_research_assistant()
    config = {"configurable": {"thread_id": f"research-{datetime.now().strftime('%H%M%S')}"}}
    
    while True:
        try:
            query = input("\nResearch Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not query:
            continue
        if query.lower() == "quit":
            break
        
        print("\nResearching...")
        
        result = app.invoke(
            {
                "messages": [],
                "query": query,
                "sources": [],
                "findings": [],
                "summary": "",
                "research_log": [],
            },
            config=config
        )
        
        print("\n" + "=" * 60)
        print(result["summary"])
        print("=" * 60)
        
        # Show research log
        print("\n[Research Log]")
        for entry in result.get("research_log", []):
            print(f"  - {entry['agent']}: {entry['action']}")
    
    checkpointer.finalize()
    print(f"\nResearch saved to: {checkpointer.get_artifact_path()}")


if __name__ == "__main__":
    main()

