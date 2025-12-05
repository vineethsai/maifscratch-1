"""
Multi-Agent MAIF Exchange Protocol and Semantic Alignment

This module implements:
1. MAIF Exchange Protocol - Standardized protocol for MAIF exchange between agents
2. Advanced Semantic Alignment - Beyond basic interoperability to semantic understanding
"""

import asyncio
import json
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Tuple, Set, Protocol, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path
import pickle
import struct
import zlib

from .core import MAIFEncoder, MAIFDecoder, MAIFBlock
from .block_types import BlockType
from .semantic import SemanticEmbedder
from .security import SecurityManager
from .compression_manager import CompressionManager


# Type aliases for forward references
class SemanticProcessor:
    """Placeholder semantic processor for multi-agent coordination."""
    
    def __init__(self):
        self.embedder = SemanticEmbedder()
    
    def process(self, data: Any) -> Any:
        return data


class Ontology:
    """Placeholder ontology class for semantic alignment."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.concepts: Dict[str, Any] = {}


class MAIF:
    """Placeholder MAIF class for multi-agent exchange."""
    
    def __init__(self, id: str):
        self.id = id
        self.header: Dict[str, Any] = {}
        self.blocks: List[Any] = []
    
    def add_block(self, block: Any):
        self.blocks.append(block)


class Block:
    """Placeholder Block class for multi-agent exchange."""
    
    def __init__(self, block_type: BlockType, data: bytes, metadata: Optional[Dict] = None):
        self.block_type = block_type
        self.data = data
        self.metadata = metadata or {}


class ExchangeProtocolVersion(Enum):
    """MAIF Exchange Protocol versions"""

    V1_0 = "1.0"  # Basic exchange
    V2_0 = "2.0"  # With semantic alignment
    V3_0 = "3.0"  # With negotiation


class MessageType(Enum):
    """Types of messages in the exchange protocol"""

    # Discovery
    HELLO = "HELLO"
    CAPABILITIES = "CAPABILITIES"

    # Negotiation
    PROPOSE_EXCHANGE = "PROPOSE_EXCHANGE"
    ACCEPT_EXCHANGE = "ACCEPT_EXCHANGE"
    REJECT_EXCHANGE = "REJECT_EXCHANGE"

    # Transfer
    REQUEST_MAIF = "REQUEST_MAIF"
    SEND_MAIF = "SEND_MAIF"
    SEND_BLOCK = "SEND_BLOCK"

    # Semantic Alignment
    REQUEST_ALIGNMENT = "REQUEST_ALIGNMENT"
    SEND_ALIGNMENT = "SEND_ALIGNMENT"
    CONFIRM_ALIGNMENT = "CONFIRM_ALIGNMENT"

    # Control
    ACK = "ACK"
    NACK = "NACK"
    ERROR = "ERROR"
    GOODBYE = "GOODBYE"


@dataclass
class AgentCapabilities:
    """Capabilities of an agent in the exchange protocol"""

    agent_id: str
    name: str
    version: str
    supported_protocols: List[ExchangeProtocolVersion]
    supported_block_types: List[BlockType]
    semantic_models: List[str]  # List of semantic model identifiers
    compression_algorithms: List[str]
    max_maif_size: int  # Maximum MAIF size agent can handle
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeMessage:
    """Message in the MAIF exchange protocol"""

    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    timestamp: datetime
    payload: Dict[str, Any]
    signature: Optional[bytes] = None

    def to_bytes(self) -> bytes:
        """Serialize message to bytes"""
        data = {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
        }
        serialized = json.dumps(data).encode("utf-8")

        # Add length prefix for framing
        length = struct.pack("!I", len(serialized))
        return length + serialized

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExchangeMessage":
        """Deserialize message from bytes"""
        # Skip length prefix
        serialized = data[4:]
        parsed = json.loads(serialized.decode("utf-8"))

        return cls(
            message_id=parsed["message_id"],
            sender_id=parsed["sender_id"],
            recipient_id=parsed["recipient_id"],
            message_type=MessageType(parsed["message_type"]),
            timestamp=datetime.fromisoformat(parsed["timestamp"]),
            payload=parsed["payload"],
        )


@dataclass
class SemanticAlignment:
    """Semantic alignment between two agents"""

    source_agent: str
    target_agent: str
    concept_mappings: Dict[str, str]  # source concept -> target concept
    confidence_scores: Dict[str, float]  # mapping -> confidence
    transformation_rules: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAIFExchangeProtocol:
    """
    MAIF Exchange Protocol implementation

    Provides standardized communication for MAIF exchange between agents
    """

    def __init__(self, agent_id: str, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.active_sessions: Dict[str, "ExchangeSession"] = {}
        self.semantic_processor = SemanticProcessor()
        self.security_manager = SecurityManager()

    async def initiate_exchange(
        self, target_agent: "MAIFExchangeProtocol", maif_id: str
    ) -> bool:
        """Initiate MAIF exchange with another agent"""
        session_id = str(uuid.uuid4())

        # Send HELLO message
        hello_msg = self._create_message(
            target_agent.agent_id,
            MessageType.HELLO,
            {
                "session_id": session_id,
                "protocol_version": ExchangeProtocolVersion.V3_0.value,
            },
        )

        response = await target_agent.handle_message(hello_msg)
        if response.message_type != MessageType.CAPABILITIES:
            return False

        # Negotiate capabilities
        target_caps = AgentCapabilities(**response.payload["capabilities"])
        if not self._check_compatibility(target_caps):
            return False

        # Propose exchange
        propose_msg = self._create_message(
            target_agent.agent_id,
            MessageType.PROPOSE_EXCHANGE,
            {
                "session_id": session_id,
                "maif_id": maif_id,
                "transfer_mode": "streaming",
                "compression": "zstd",
            },
        )

        response = await target_agent.handle_message(propose_msg)
        return response.message_type == MessageType.ACCEPT_EXCHANGE

    async def handle_message(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle incoming exchange protocol message"""
        handlers = {
            MessageType.HELLO: self._handle_hello,
            MessageType.PROPOSE_EXCHANGE: self._handle_propose,
            MessageType.REQUEST_MAIF: self._handle_request_maif,
            MessageType.REQUEST_ALIGNMENT: self._handle_request_alignment,
        }

        handler = handlers.get(message.message_type)
        if handler:
            return await handler(message)
        else:
            return self._create_error_response(message, "Unsupported message type")

    async def _handle_hello(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle HELLO message"""
        # Send capabilities in response
        return self._create_message(
            message.sender_id,
            MessageType.CAPABILITIES,
            {"capabilities": self.capabilities.__dict__},
        )

    async def _handle_propose(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle exchange proposal"""
        # Simple acceptance for now
        return self._create_message(
            message.sender_id,
            MessageType.ACCEPT_EXCHANGE,
            {"session_id": message.payload["session_id"]},
        )

    async def _handle_request_maif(self, message: ExchangeMessage) -> ExchangeMessage:
        """Handle MAIF request"""
        maif_id = message.payload["maif_id"]
        # In real implementation, would retrieve MAIF from storage
        # For now, return acknowledgment
        return self._create_message(
            message.sender_id, MessageType.ACK, {"maif_id": maif_id, "status": "ready"}
        )

    async def _handle_request_alignment(
        self, message: ExchangeMessage
    ) -> ExchangeMessage:
        """Handle semantic alignment request"""
        source_concepts = message.payload.get("concepts", [])
        alignment = self._compute_alignment(message.sender_id, source_concepts)

        return self._create_message(
            message.sender_id,
            MessageType.SEND_ALIGNMENT,
            {"alignment": alignment.__dict__},
        )

    def _create_message(
        self, recipient_id: str, msg_type: MessageType, payload: Dict[str, Any]
    ) -> ExchangeMessage:
        """Create a new exchange message"""
        return ExchangeMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=msg_type,
            timestamp=datetime.now(),
            payload=payload,
        )

    def _create_error_response(
        self, original: ExchangeMessage, error: str
    ) -> ExchangeMessage:
        """Create error response message"""
        return self._create_message(
            original.sender_id,
            MessageType.ERROR,
            {"error": error, "original_message_id": original.message_id},
        )

    def _check_compatibility(self, target_caps: AgentCapabilities) -> bool:
        """Check if two agents are compatible for exchange"""
        # Check protocol version compatibility
        common_protocols = set(self.capabilities.supported_protocols) & set(
            target_caps.supported_protocols
        )
        if not common_protocols:
            return False

        # Check block type compatibility
        common_blocks = set(self.capabilities.supported_block_types) & set(
            target_caps.supported_block_types
        )
        if not common_blocks:
            return False

        return True

    def _compute_alignment(
        self, target_agent_id: str, source_concepts: List[str]
    ) -> SemanticAlignment:
        """Compute semantic alignment with another agent"""
        # Simplified alignment - in practice would use ontology matching
        concept_mappings = {}
        confidence_scores = {}

        for concept in source_concepts:
            # Simple heuristic mapping
            if concept in self.capabilities.semantic_models:
                concept_mappings[concept] = concept
                confidence_scores[concept] = 1.0
            else:
                # Try fuzzy matching
                best_match, score = self._fuzzy_match(
                    concept, self.capabilities.semantic_models
                )
                if score > 0.7:
                    concept_mappings[concept] = best_match
                    confidence_scores[concept] = score

        return SemanticAlignment(
            source_agent=target_agent_id,
            target_agent=self.agent_id,
            concept_mappings=concept_mappings,
            confidence_scores=confidence_scores,
            transformation_rules=[],
        )

    def _fuzzy_match(self, concept: str, candidates: List[str]) -> Tuple[str, float]:
        """Simple fuzzy matching for concept alignment"""
        # Simplified - in practice would use embeddings or ontology
        best_match = ""
        best_score = 0.0

        concept_lower = concept.lower()
        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Check substring match
            if concept_lower in candidate_lower or candidate_lower in concept_lower:
                score = 0.8
            else:
                # Check word overlap
                concept_words = set(concept_lower.split())
                candidate_words = set(candidate_lower.split())
                overlap = len(concept_words & candidate_words)
                total = len(concept_words | candidate_words)
                score = overlap / total if total > 0 else 0.0

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, best_score


class SemanticAlignmentEngine:
    """
    Advanced Semantic Alignment Engine

    Provides deep semantic understanding and alignment between agents
    """

    def __init__(self):
        self.ontology_cache: Dict[str, "Ontology"] = {}
        self.alignment_cache: Dict[Tuple[str, str], SemanticAlignment] = {}
        self.semantic_processor = SemanticProcessor()

    async def align_agents(
        self, agent1: AgentCapabilities, agent2: AgentCapabilities
    ) -> SemanticAlignment:
        """Perform deep semantic alignment between two agents"""
        cache_key = (agent1.agent_id, agent2.agent_id)
        if cache_key in self.alignment_cache:
            return self.alignment_cache[cache_key]

        # Extract semantic models
        models1 = set(agent1.semantic_models)
        models2 = set(agent2.semantic_models)

        # Find common ground
        common_models = models1 & models2
        unique_to_1 = models1 - models2
        unique_to_2 = models2 - models1

        # Build concept mappings
        concept_mappings = {}
        confidence_scores = {}

        # Direct mappings for common models
        for model in common_models:
            concept_mappings[model] = model
            confidence_scores[model] = 1.0

        # Compute alignments for unique concepts
        for concept1 in unique_to_1:
            best_match, score = await self._find_best_alignment(concept1, unique_to_2)
            if score > 0.5:  # Threshold for acceptance
                concept_mappings[concept1] = best_match
                confidence_scores[concept1] = score

        # Generate transformation rules
        transformation_rules = self._generate_transformation_rules(
            concept_mappings, agent1, agent2
        )

        alignment = SemanticAlignment(
            source_agent=agent1.agent_id,
            target_agent=agent2.agent_id,
            concept_mappings=concept_mappings,
            confidence_scores=confidence_scores,
            transformation_rules=transformation_rules,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "common_models": list(common_models),
                "alignment_quality": np.mean(list(confidence_scores.values())),
            },
        )

        self.alignment_cache[cache_key] = alignment
        return alignment

    async def _find_best_alignment(
        self, source_concept: str, target_concepts: Set[str]
    ) -> Tuple[str, float]:
        """Find best semantic alignment for a concept"""
        if not target_concepts:
            return "", 0.0

        # Use semantic embeddings for alignment
        source_embedding = await self._get_concept_embedding(source_concept)

        best_match = ""
        best_score = 0.0

        for target in target_concepts:
            target_embedding = await self._get_concept_embedding(target)

            # Cosine similarity
            score = self._cosine_similarity(source_embedding, target_embedding)

            # Boost score if there's lexical similarity
            lexical_score = self._lexical_similarity(source_concept, target)
            score = 0.7 * score + 0.3 * lexical_score

            if score > best_score:
                best_score = score
                best_match = target

        return best_match, best_score

    async def _get_concept_embedding(self, concept: str) -> np.ndarray:
        """Get semantic embedding for a concept"""
        # In practice, would use a pre-trained model
        # For now, use a simple hash-based embedding
        hash_bytes = hashlib.sha256(concept.encode()).digest()
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        return embedding / np.linalg.norm(embedding)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return float(dot_product / norm_product) if norm_product > 0 else 0.0

    def _lexical_similarity(self, str1: str, str2: str) -> float:
        """Compute lexical similarity between strings"""

        # Simple Jaccard similarity on character n-grams
        def get_ngrams(s: str, n: int = 3) -> Set[str]:
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ngrams1 = get_ngrams(str1.lower())
        ngrams2 = get_ngrams(str2.lower())

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _generate_transformation_rules(
        self,
        mappings: Dict[str, str],
        source: AgentCapabilities,
        target: AgentCapabilities,
    ) -> List[Dict[str, Any]]:
        """Generate transformation rules based on alignment"""
        rules = []

        for source_concept, target_concept in mappings.items():
            if source_concept != target_concept:
                rule = {
                    "type": "concept_mapping",
                    "source": source_concept,
                    "target": target_concept,
                    "confidence": mappings.get(source_concept, 0.0),
                    "transformations": [],
                }

                # Add data type transformations if needed
                source_features = source.features.get(source_concept, {})
                target_features = target.features.get(target_concept, {})

                if source_features.get("data_type") != target_features.get("data_type"):
                    rule["transformations"].append(
                        {
                            "type": "type_conversion",
                            "from": source_features.get("data_type"),
                            "to": target_features.get("data_type"),
                        }
                    )

                rules.append(rule)

        return rules

    async def transform_maif(self, maif: MAIF, alignment: SemanticAlignment) -> MAIF:
        """Transform MAIF according to semantic alignment"""
        transformed = MAIF(f"{maif.id}_transformed")

        # Copy header with updated metadata
        transformed.header = maif.header.copy()
        transformed.header.metadata["semantic_alignment"] = {
            "source_agent": alignment.source_agent,
            "target_agent": alignment.target_agent,
            "alignment_quality": alignment.metadata.get("alignment_quality", 0.0),
        }

        # Transform blocks according to alignment rules
        for block in maif.blocks:
            transformed_block = await self._transform_block(block, alignment)
            transformed.add_block(transformed_block)

        return transformed

    async def _transform_block(
        self, block: Block, alignment: SemanticAlignment
    ) -> Block:
        """Transform a block according to alignment rules"""
        # Check if block type needs transformation
        block_type_str = block.block_type.name

        if block_type_str in alignment.concept_mappings:
            target_type_str = alignment.concept_mappings[block_type_str]

            # Create new block with transformed type
            try:
                target_type = BlockType[target_type_str]
            except KeyError:
                # If target type doesn't exist, keep original
                target_type = block.block_type

            transformed = Block(
                block_type=target_type, data=block.data, metadata=block.metadata.copy()
            )

            # Apply transformation rules
            for rule in alignment.transformation_rules:
                if rule["source"] == block_type_str:
                    transformed = await self._apply_transformation_rule(
                        transformed, rule
                    )

            return transformed
        else:
            # No transformation needed
            return block

    async def _apply_transformation_rule(
        self, block: Block, rule: Dict[str, Any]
    ) -> Block:
        """Apply a specific transformation rule to a block"""
        for transform in rule.get("transformations", []):
            if transform["type"] == "type_conversion":
                # Simple type conversion example
                if transform["from"] == "json" and transform["to"] == "msgpack":
                    import msgpack

                    data = json.loads(block.data)
                    block.data = msgpack.packb(data)
                elif transform["from"] == "msgpack" and transform["to"] == "json":
                    import msgpack

                    data = msgpack.unpackb(block.data)
                    block.data = json.dumps(data).encode()

        return block


class MultiAgentOrchestrator:
    """
    Orchestrates multi-agent MAIF exchanges and collaborations
    """

    def __init__(self):
        self.agents: Dict[str, MAIFExchangeProtocol] = {}
        self.alignment_engine = SemanticAlignmentEngine()
        self.exchange_history: List[Dict[str, Any]] = []

    def register_agent(self, agent: MAIFExchangeProtocol):
        """Register an agent in the orchestrator"""
        self.agents[agent.agent_id] = agent

    async def facilitate_exchange(
        self, source_agent_id: str, target_agent_id: str, maif_id: str
    ) -> bool:
        """Facilitate MAIF exchange between two agents"""
        source = self.agents.get(source_agent_id)
        target = self.agents.get(target_agent_id)

        if not source or not target:
            return False

        # Compute semantic alignment
        alignment = await self.alignment_engine.align_agents(
            source.capabilities, target.capabilities
        )

        # Check if alignment quality is sufficient
        quality = alignment.metadata.get("alignment_quality", 0.0)
        if quality < 0.6:  # Threshold for minimum alignment
            return False

        # Initiate exchange with alignment context
        success = await source.initiate_exchange(target, maif_id)

        # Record exchange
        self.exchange_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "source": source_agent_id,
                "target": target_agent_id,
                "maif_id": maif_id,
                "alignment_quality": quality,
                "success": success,
            }
        )

        return success

    async def broadcast_maif(
        self, source_agent_id: str, maif_id: str, min_alignment: float = 0.7
    ) -> List[str]:
        """Broadcast MAIF to all compatible agents"""
        source = self.agents.get(source_agent_id)
        if not source:
            return []

        successful_agents = []

        for agent_id, agent in self.agents.items():
            if agent_id == source_agent_id:
                continue

            # Check alignment
            alignment = await self.alignment_engine.align_agents(
                source.capabilities, agent.capabilities
            )

            quality = alignment.metadata.get("alignment_quality", 0.0)
            if quality >= min_alignment:
                success = await source.initiate_exchange(agent, maif_id)
                if success:
                    successful_agents.append(agent_id)

        return successful_agents

    def get_exchange_analytics(self) -> Dict[str, Any]:
        """Get analytics on multi-agent exchanges"""
        if not self.exchange_history:
            return {}

        total_exchanges = len(self.exchange_history)
        successful_exchanges = sum(1 for e in self.exchange_history if e["success"])

        alignment_scores = [e["alignment_quality"] for e in self.exchange_history]

        # Agent participation
        agent_exchanges = {}
        for exchange in self.exchange_history:
            for agent in [exchange["source"], exchange["target"]]:
                agent_exchanges[agent] = agent_exchanges.get(agent, 0) + 1

        return {
            "total_exchanges": total_exchanges,
            "successful_exchanges": successful_exchanges,
            "success_rate": successful_exchanges / total_exchanges
            if total_exchanges > 0
            else 0,
            "average_alignment": np.mean(alignment_scores) if alignment_scores else 0,
            "min_alignment": min(alignment_scores) if alignment_scores else 0,
            "max_alignment": max(alignment_scores) if alignment_scores else 0,
            "most_active_agents": sorted(
                agent_exchanges.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "exchange_timeline": self.exchange_history[-10:],  # Last 10 exchanges
        }


# Example usage and testing
if __name__ == "__main__":

    async def demo_multi_agent_exchange():
        """Demonstrate multi-agent MAIF exchange"""

        # Create agent capabilities
        agent1_caps = AgentCapabilities(
            agent_id="agent_1",
            name="Research Agent",
            version="1.0",
            supported_protocols=[ExchangeProtocolVersion.V3_0],
            supported_block_types=[
                BlockType.TEXT,
                BlockType.EMBEDDING,
                BlockType.METADATA,
            ],
            semantic_models=[
                "research_ontology",
                "scientific_concepts",
                "paper_structure",
            ],
            compression_algorithms=["zstd", "lz4"],
            max_maif_size=1024 * 1024 * 1024,  # 1GB
            features={
                "research_ontology": {"data_type": "json", "version": "2.0"},
                "scientific_concepts": {"data_type": "msgpack", "version": "1.5"},
            },
        )

        agent2_caps = AgentCapabilities(
            agent_id="agent_2",
            name="Analysis Agent",
            version="1.0",
            supported_protocols=[
                ExchangeProtocolVersion.V3_0,
                ExchangeProtocolVersion.V2_0,
            ],
            supported_block_types=[BlockType.TEXT, BlockType.EMBEDDING, BlockType.CODE],
            semantic_models=[
                "analysis_framework",
                "scientific_concepts",
                "data_models",
            ],
            compression_algorithms=["zstd", "gzip"],
            max_maif_size=512 * 1024 * 1024,  # 512MB
            features={
                "analysis_framework": {"data_type": "json", "version": "1.0"},
                "scientific_concepts": {"data_type": "json", "version": "1.5"},
            },
        )

        agent3_caps = AgentCapabilities(
            agent_id="agent_3",
            name="Visualization Agent",
            version="1.0",
            supported_protocols=[ExchangeProtocolVersion.V3_0],
            supported_block_types=[BlockType.IMAGE, BlockType.METADATA],
            semantic_models=["visualization_types", "chart_ontology"],
            compression_algorithms=["zstd"],
            max_maif_size=256 * 1024 * 1024,  # 256MB
            features={},
        )

        # Create agents
        agent1 = MAIFExchangeProtocol("agent_1", agent1_caps)
        agent2 = MAIFExchangeProtocol("agent_2", agent2_caps)
        agent3 = MAIFExchangeProtocol("agent_3", agent3_caps)

        # Create orchestrator
        orchestrator = MultiAgentOrchestrator()
        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)
        orchestrator.register_agent(agent3)

        print("=== Multi-Agent MAIF Exchange Demo ===\n")

        # Test semantic alignment
        print("1. Testing Semantic Alignment:")
        alignment_engine = SemanticAlignmentEngine()

        alignment_1_2 = await alignment_engine.align_agents(agent1_caps, agent2_caps)
        print(
            f"   Agent 1 <-> Agent 2 alignment quality: {alignment_1_2.metadata['alignment_quality']:.2f}"
        )
        print(f"   Common models: {alignment_1_2.metadata['common_models']}")
        print(f"   Concept mappings: {alignment_1_2.concept_mappings}")

        alignment_1_3 = await alignment_engine.align_agents(agent1_caps, agent3_caps)
        print(
            f"\n   Agent 1 <-> Agent 3 alignment quality: {alignment_1_3.metadata['alignment_quality']:.2f}"
        )
        print(f"   Common models: {alignment_1_3.metadata['common_models']}")

        # Test direct exchange
        print("\n2. Testing Direct Exchange:")
        success = await orchestrator.facilitate_exchange(
            "agent_1", "agent_2", "research_maif_001"
        )
        print(f"   Exchange agent_1 -> agent_2: {'Success' if success else 'Failed'}")

        # Test broadcast
        print("\n3. Testing Broadcast:")
        recipients = await orchestrator.broadcast_maif(
            "agent_1", "research_maif_002", min_alignment=0.3
        )
        print(f"   Broadcast from agent_1 reached: {recipients}")

        # Show analytics
        print("\n4. Exchange Analytics:")
        analytics = orchestrator.get_exchange_analytics()
        for key, value in analytics.items():
            if key != "exchange_timeline" and key != "most_active_agents":
                print(f"   {key}: {value}")

        # Test MAIF transformation
        print("\n5. Testing MAIF Transformation:")

        # Create a sample MAIF
        test_maif = MAIF("test_transformation")
        test_maif.add_block(
            Block(
                block_type=BlockType.TEXT,
                data=b"Research findings on quantum computing",
                metadata={"concept": "research_ontology"},
            )
        )
        test_maif.add_block(
            Block(
                block_type=BlockType.METADATA,
                data=json.dumps({"experiment": "quantum_simulation"}).encode(),
                metadata={"concept": "scientific_concepts", "format": "msgpack"},
            )
        )

        # Transform MAIF according to alignment
        transformed = await alignment_engine.transform_maif(test_maif, alignment_1_2)
        print(f"   Original blocks: {len(test_maif.blocks)}")
        print(f"   Transformed blocks: {len(transformed.blocks)}")
        print(
            f"   Transformation metadata: {transformed.header.metadata.get('semantic_alignment', {})}"
        )

    # Run the demo
    asyncio.run(demo_multi_agent_exchange())
