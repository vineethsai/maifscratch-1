"""
MAIF Convenience API - Simple, intuitive methods for common operations.
Makes MAIF as easy to use as a Python dictionary with AI superpowers.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging

from .core import MAIFEncoder, MAIFDecoder
from .semantic_optimized import OptimizedSemanticEmbedder
from .privacy import PrivacyEngine, PrivacyLevel

logger = logging.getLogger(__name__)


class SimpleMAIFAgent:
    """
    High-level convenience API for MAIF operations.
    Provides intuitive methods like remember(), forget(), and explain().
    """
    
    def __init__(self, agent_id: str, maif_path: Optional[str] = None):
        """
        Initialize a simple MAIF agent.
        
        Args:
            agent_id: Unique identifier for this agent
            maif_path: Path to MAIF file (defaults to {agent_id}.maif)
        """
        self.agent_id = agent_id
        self.maif_path = Path(maif_path or f"{agent_id}.maif")
        
        # Initialize components
        self.embedder = OptimizedSemanticEmbedder()
        self.privacy_engine = PrivacyEngine()
        
        # Load or create MAIF (v3 format - self-contained)
        self.encoder = MAIFEncoder(str(self.maif_path), agent_id=agent_id)
        if self.maif_path.exists():
            self.decoder = MAIFDecoder(str(self.maif_path))
        else:
            self.decoder = None
        
        # Memory cache for fast access
        self._memory_cache = {}
        self._load_memories()
    
    def remember(self, content: Union[str, Dict[str, Any]], 
                 context: Optional[str] = None,
                 importance: float = 1.0) -> str:
        """
        Remember something - as simple as it gets!
        
        Args:
            content: What to remember (text or structured data)
            context: Optional context (e.g., "user preference", "learned fact")
            importance: How important this memory is (0-1)
            
        Returns:
            Memory ID for future reference
            
        Example:
            agent.remember("User likes pizza")
            agent.remember({"name": "John", "age": 30}, context="user profile")
        """
        # Convert to string if needed
        if isinstance(content, dict):
            content_str = json.dumps(content)
            content_type = "json"
        else:
            content_str = str(content)
            content_type = "text"
        
        # Create metadata
        metadata = {
            "timestamp": time.time(),
            "context": context or "general",
            "importance": importance,
            "content_type": content_type,
            "agent_id": self.agent_id
        }
        
        # Generate embedding for semantic search
        embedding = self.embedder.embed_text(content_str)
        
        # Add to MAIF
        block_id = self.encoder.add_text_block(
            content_str,
            metadata=metadata
        )
        
        # Add embedding
        self.encoder.add_embedding_block(
            embedding.vector,
            block_id=f"{block_id}_embedding",
            metadata={
                "source_block": block_id,
                "model": embedding.model
            }
        )
        
        # Update cache
        self._memory_cache[block_id] = {
            "content": content,
            "metadata": metadata,
            "embedding": embedding
        }
        
        # Save periodically (every 10 memories)
        if len(self._memory_cache) % 10 == 0:
            self.save()
        
        logger.info(f"Remembered: {content_str[:50]}... with ID {block_id}")
        return block_id
    
    def recall(self, query: str, limit: int = 5, 
               min_relevance: float = 0.7) -> List[Dict[str, Any]]:
        """
        Recall memories related to a query.
        
        Args:
            query: What to search for
            limit: Maximum number of memories to return
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            List of relevant memories with scores
            
        Example:
            memories = agent.recall("pizza preferences")
            memories = agent.recall("what does the user like?")
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search in memory cache
        results = []
        for memory_id, memory_data in self._memory_cache.items():
            # Calculate similarity
            similarity = self._cosine_similarity(
                query_embedding.vector,
                memory_data["embedding"].vector
            )
            
            if similarity >= min_relevance:
                results.append({
                    "id": memory_id,
                    "content": memory_data["content"],
                    "context": memory_data["metadata"].get("context", "general"),
                    "timestamp": memory_data["metadata"].get("timestamp", 0),
                    "relevance": similarity
                })
        
        # Sort by relevance and limit
        results.sort(key=lambda x: x["relevance"], reverse=True)
        results = results[:limit]
        
        logger.info(f"Recalled {len(results)} memories for query: {query}")
        return results
    
    def forget(self, memory_id: Optional[str] = None,
               older_than: Optional[Union[int, timedelta]] = None,
               context: Optional[str] = None,
               importance_below: Optional[float] = None) -> int:
        """
        Forget memories based on various criteria.
        
        Args:
            memory_id: Specific memory to forget
            older_than: Forget memories older than X days (int) or timedelta
            context: Forget all memories with this context
            importance_below: Forget memories with importance below this threshold
            
        Returns:
            Number of memories forgotten
            
        Examples:
            agent.forget(memory_id="abc123")  # Forget specific memory
            agent.forget(older_than=30)  # Forget memories older than 30 days
            agent.forget(context="temporary")  # Forget all temporary memories
            agent.forget(importance_below=0.3)  # Forget unimportant memories
        """
        forgotten_count = 0
        current_time = time.time()
        
        # Convert older_than to seconds
        if isinstance(older_than, int):
            older_than_seconds = older_than * 86400  # days to seconds
        elif isinstance(older_than, timedelta):
            older_than_seconds = older_than.total_seconds()
        else:
            older_than_seconds = None
        
        # Filter memories to forget
        memories_to_forget = []
        
        for mem_id, memory_data in self._memory_cache.items():
            should_forget = False
            
            # Check specific ID
            if memory_id and mem_id == memory_id:
                should_forget = True
            
            # Check age
            if older_than_seconds:
                age = current_time - memory_data["metadata"].get("timestamp", 0)
                if age > older_than_seconds:
                    should_forget = True
            
            # Check context
            if context and memory_data["metadata"].get("context") == context:
                should_forget = True
            
            # Check importance
            if importance_below is not None:
                importance = memory_data["metadata"].get("importance", 1.0)
                if importance < importance_below:
                    should_forget = True
            
            if should_forget:
                memories_to_forget.append(mem_id)
        
        # Remove from cache
        for mem_id in memories_to_forget:
            del self._memory_cache[mem_id]
            forgotten_count += 1
        
        # TODO: Mark as deleted in MAIF file (needs core support)
        
        logger.info(f"Forgot {forgotten_count} memories")
        return forgotten_count
    
    def explain_reasoning(self, decision: str, 
                         context_limit: int = 5) -> Dict[str, Any]:
        """
        Explain the reasoning behind a decision by showing relevant memories.
        
        Args:
            decision: The decision or conclusion to explain
            context_limit: Maximum number of supporting memories to show
            
        Returns:
            Explanation with supporting memories and confidence
            
        Example:
            explanation = agent.explain_reasoning("User would enjoy Italian food")
        """
        # Find relevant memories
        relevant_memories = self.recall(decision, limit=context_limit)
        
        # Calculate confidence based on relevance scores
        if relevant_memories:
            avg_relevance = sum(m["relevance"] for m in relevant_memories) / len(relevant_memories)
            confidence = min(avg_relevance * 1.2, 1.0)  # Boost slightly, cap at 1.0
        else:
            confidence = 0.0
        
        # Build explanation
        explanation = {
            "decision": decision,
            "confidence": confidence,
            "reasoning": f"Based on {len(relevant_memories)} relevant memories",
            "supporting_evidence": relevant_memories,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add natural language explanation
        if relevant_memories:
            evidence_summary = []
            for mem in relevant_memories[:3]:  # Top 3
                evidence_summary.append(
                    f"- {mem['content']} (relevance: {mem['relevance']:.2f})"
                )
            explanation["explanation"] = (
                f"I believe '{decision}' because:\n" + 
                "\n".join(evidence_summary)
            )
        else:
            explanation["explanation"] = (
                f"I have no strong evidence to support '{decision}'"
            )
        
        return explanation
    
    def save(self):
        """Save the current state to MAIF file (v3 format)."""
        self.encoder.finalize()
        logger.info(f"Saved agent memory to {self.maif_path}")
    
    def _load_memories(self):
        """Load memories from MAIF file into cache."""
        if not self.decoder:
            return
        
        # Load text blocks as memories
        for block in self.decoder.blocks:
            if block.block_type == "text":
                content = self.decoder.get_block_data(block.block_id).decode('utf-8')
                
                # Try to parse JSON
                try:
                    if block.metadata.get("content_type") == "json":
                        content = json.loads(content)
                except:
                    pass
                
                # Find corresponding embedding
                embedding = None
                embedding_id = f"{block.block_id}_embedding"
                for emb_block in self.decoder.blocks:
                    if emb_block.block_id == embedding_id:
                        embedding_data = self.decoder.get_block_data(embedding_id)
                        # TODO: Deserialize embedding properly
                        break
                
                self._memory_cache[block.block_id] = {
                    "content": content,
                    "metadata": block.metadata,
                    "embedding": embedding
                }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# Convenience functions for even simpler usage
def create_agent(agent_id: str) -> SimpleMAIFAgent:
    """Create a simple MAIF agent."""
    return SimpleMAIFAgent(agent_id)