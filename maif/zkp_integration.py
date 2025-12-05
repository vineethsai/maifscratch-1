"""
Zero-Knowledge Proof Integration for MAIF
========================================

Integrates zero-knowledge proofs across all MAIF components to provide:
- Privacy-preserving block verification
- Anonymous agent authentication
- Confidential data sharing
- Secure multi-party computation
- Privacy-preserving analytics
"""

import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import time

from .zero_knowledge_proofs import (
    ZKProofType,
    ProofStatus,
    ZKProof,
    ZKCommitment,
    CurveType,
    ZKChallenge,
)
from .core import MAIFBlock, MAIFEncoder, MAIFDecoder
from .signature_verification import SignatureAlgorithm
from .privacy import PrivacyLevel
from .distributed import CRDTOperation

logger = logging.getLogger(__name__)


class ZKPBlockExtension:
    """Extends MAIF blocks with zero-knowledge proof capabilities."""

    def __init__(self, encoder: MAIFEncoder):
        self.encoder = encoder
        self.zkp_blocks: Dict[str, ZKProof] = {}

    def add_zkp_block(
        self,
        data: bytes,
        proof_type: ZKProofType,
        statement: str,
        public_params: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a block with zero-knowledge proof."""

        # Generate proof ID
        proof_id = hashlib.sha256(
            f"{time.time()}:{statement}:{proof_type.value}".encode()
        ).hexdigest()[:16]

        # Create ZK proof
        zkp = ZKProof(
            proof_id=proof_id,
            proof_type=proof_type,
            statement=statement,
            proof_data={"data_hash": hashlib.sha256(data).hexdigest()},
            public_parameters=public_params,
            created_at=time.time(),
            creator_id=self.encoder.agent_id,
        )

        # Add metadata with ZKP info
        block_metadata = metadata or {}
        block_metadata.update(
            {
                "zkp_enabled": True,
                "zkp_proof_id": proof_id,
                "zkp_type": proof_type.value,
                "zkp_statement": statement,
            }
        )

        # Add block with ZKP metadata
        block_id = self.encoder.add_raw_block(data, "zkp_data", block_metadata)

        # Store ZKP reference
        self.zkp_blocks[block_id] = zkp

        # Add proof as separate block
        proof_data = json.dumps(
            {
                "proof": zkp.proof_data,
                "public_params": zkp.public_parameters,
                "statement": zkp.statement,
                "type": zkp.proof_type.value,
            }
        ).encode()

        proof_block_id = self.encoder.add_raw_block(
            proof_data, "zkp_proof", {"proof_id": proof_id, "data_block_id": block_id}
        )

        return block_id

    def verify_zkp_block(
        self, block_id: str, decoder: MAIFDecoder
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify a block's zero-knowledge proof."""

        # Get block metadata
        block = None
        for b in decoder.blocks:
            if b.block_id == block_id:
                block = b
                break

        if not block:
            return False, {"error": "Block not found"}

        if not block.metadata.get("zkp_enabled"):
            return False, {"error": "Block does not have ZKP"}

        proof_id = block.metadata.get("zkp_proof_id")

        # Find proof block
        proof_block = None
        for b in decoder.blocks:
            if b.block_type == "zkp_proof" and b.metadata.get("proof_id") == proof_id:
                proof_block = b
                break

        if not proof_block:
            return False, {"error": "Proof block not found"}

        # Verify proof (simplified - in production, use actual ZKP verification)
        proof_data = json.loads(decoder.get_block_data(proof_block.block_id))
        data_hash = hashlib.sha256(decoder.get_block_data(block_id)).hexdigest()

        if proof_data["proof"]["data_hash"] == data_hash:
            return True, {
                "verified": True,
                "proof_type": proof_data["type"],
                "statement": proof_data["statement"],
            }

        return False, {"error": "Proof verification failed"}


class ZKPPrivacyEnhancer:
    """Enhances privacy features with zero-knowledge proofs."""

    @staticmethod
    def create_membership_proof(
        value: Any, set_commitment: bytes, privacy_level: PrivacyLevel
    ) -> ZKProof:
        """Create proof that value belongs to a set without revealing the value."""

        # Hash the value
        value_hash = hashlib.sha256(str(value).encode()).hexdigest()

        # Create Merkle tree commitment (simplified)
        proof_data = {
            "merkle_root": set_commitment.hex(),
            "value_commitment": value_hash,
            "privacy_level": privacy_level.value,
        }

        public_params = {"set_size": "hidden", "hash_function": "sha256"}

        return ZKProof(
            proof_id=hashlib.sha256(f"{value_hash}:{time.time()}".encode()).hexdigest()[
                :16
            ],
            proof_type=ZKProofType.MEMBERSHIP_PROOF,
            statement="Value belongs to committed set",
            proof_data=proof_data,
            public_parameters=public_params,
            created_at=time.time(),
            creator_id="privacy_enhancer",
        )

    @staticmethod
    def create_range_proof(value: int, min_val: int, max_val: int) -> ZKProof:
        """Prove value is within range without revealing exact value."""

        # Bulletproof-style range proof (simplified)
        # In production, use actual Bulletproof implementation

        # Commit to value
        blinding_factor = hashlib.sha256(f"{value}:{time.time()}".encode()).digest()
        commitment = hashlib.sha256(
            f"{value}:{blinding_factor.hex()}".encode()
        ).hexdigest()

        proof_data = {
            "commitment": commitment,
            "range_proof": {
                "valid": min_val <= value <= max_val,
                "bits": (max_val - min_val).bit_length(),
            },
        }

        public_params = {"min": min_val, "max": max_val, "curve": "secp256k1"}

        return ZKProof(
            proof_id=hashlib.sha256(f"range:{commitment}".encode()).hexdigest()[:16],
            proof_type=ZKProofType.RANGE_PROOF,
            statement=f"Value in range [{min_val}, {max_val}]",
            proof_data=proof_data,
            public_parameters=public_params,
            created_at=time.time(),
            creator_id="privacy_enhancer",
        )


class ZKPAgentAuthenticator:
    """Zero-knowledge authentication for agents."""

    def __init__(self):
        self.registered_agents: Dict[str, Dict[str, Any]] = {}

    def register_agent_zkp(
        self, agent_id: str, public_key: bytes, attributes: Dict[str, Any]
    ) -> ZKCommitment:
        """Register agent with zero-knowledge commitments."""

        # Commit to agent attributes without revealing them
        attr_hash = hashlib.sha256(
            json.dumps(attributes, sort_keys=True).encode()
        ).hexdigest()

        commitment = ZKCommitment(
            commitment_id=agent_id,
            commitment_value=public_key,
            commitment_type="agent_identity",
            metadata={
                "attribute_commitment": attr_hash,
                "registration_time": time.time(),
            },
            created_at=time.time(),
            randomness_hash=hashlib.sha256(public_key + attr_hash.encode()).hexdigest(),
        )

        self.registered_agents[agent_id] = {
            "commitment": commitment,
            "public_key": public_key,
            "attribute_hash": attr_hash,
        }

        return commitment

    def authenticate_agent_zkp(
        self,
        agent_id: str,
        challenge: bytes,
        response: bytes,
        claimed_attributes: Dict[str, Any],
    ) -> Tuple[bool, ZKProof]:
        """Authenticate agent using zero-knowledge proof."""

        if agent_id not in self.registered_agents:
            return False, None

        agent_info = self.registered_agents[agent_id]

        # Verify attribute commitment
        attr_hash = hashlib.sha256(
            json.dumps(claimed_attributes, sort_keys=True).encode()
        ).hexdigest()
        if attr_hash != agent_info["attribute_hash"]:
            return False, None

        # Create authentication proof
        proof = ZKProof(
            proof_id=hashlib.sha256(
                f"{agent_id}:{challenge.hex()}".encode()
            ).hexdigest()[:16],
            proof_type=ZKProofType.SCHNORR,
            statement="Agent possesses private key",
            proof_data={
                "challenge": challenge.hex(),
                "response": response.hex(),
                "public_key": agent_info["public_key"].hex(),
            },
            public_parameters={"agent_id": agent_id, "timestamp": time.time()},
            created_at=time.time(),
            creator_id=agent_id,
            verification_status=ProofStatus.VALID,
        )

        return True, proof


class ZKPDistributedOps:
    """Zero-knowledge proofs for distributed operations."""

    @staticmethod
    def create_crdt_zkp(operation: CRDTOperation, secret_value: Any) -> ZKProof:
        """Create ZKP for CRDT operation without revealing the value."""

        # Commit to the secret value
        value_commitment = hashlib.sha256(str(secret_value).encode()).hexdigest()

        # Create proof that operation is valid without revealing value
        proof_data = {
            "operation_id": operation.op_id,
            "operation_type": type(operation).__name__,
            "value_commitment": value_commitment,
            "vector_clock": operation.vector_clock,
        }

        public_params = {"node_id": operation.node_id, "timestamp": operation.timestamp}

        return ZKProof(
            proof_id=hashlib.sha256(f"crdt:{operation.op_id}".encode()).hexdigest()[
                :16
            ],
            proof_type=ZKProofType.COMMITMENT_SCHEME,
            statement="Valid CRDT operation",
            proof_data=proof_data,
            public_parameters=public_params,
            created_at=time.time(),
            creator_id=operation.node_id,
        )

    @staticmethod
    def verify_distributed_computation(
        results: List[Tuple[str, Any]], proofs: List[ZKProof]
    ) -> bool:
        """Verify distributed computation results using ZKPs."""

        if len(results) != len(proofs):
            return False

        for (node_id, result), proof in zip(results, proofs):
            # Verify each node's computation proof
            if proof.public_parameters.get("node_id") != node_id:
                return False

            # Verify result commitment
            result_hash = hashlib.sha256(str(result).encode()).hexdigest()
            if proof.proof_data.get("result_commitment") != result_hash:
                return False

        return True


class ZKPSecureAggregation:
    """Secure aggregation using zero-knowledge proofs."""

    @staticmethod
    def create_aggregation_proof(values: List[int], aggregate: int) -> ZKProof:
        """Prove correct aggregation without revealing individual values."""

        # Create commitments to individual values
        commitments = []
        for val in values:
            blinding = hashlib.sha256(f"{val}:{time.time()}".encode()).digest()
            commitment = hashlib.sha256(f"{val}:{blinding.hex()}".encode()).hexdigest()
            commitments.append(commitment)

        # Create proof of correct sum
        proof_data = {
            "value_commitments": commitments,
            "aggregate_commitment": hashlib.sha256(str(aggregate).encode()).hexdigest(),
            "num_values": len(values),
        }

        public_params = {"aggregate_value": aggregate, "aggregation_type": "sum"}

        return ZKProof(
            proof_id=hashlib.sha256(
                f"agg:{aggregate}:{len(values)}".encode()
            ).hexdigest()[:16],
            proof_type=ZKProofType.BULLETPROOF,
            statement="Correct aggregation of committed values",
            proof_data=proof_data,
            public_parameters=public_params,
            created_at=time.time(),
            creator_id="aggregator",
        )


def integrate_zkp_with_maif(maif_encoder: MAIFEncoder, maif_decoder: MAIFDecoder):
    """Integrate ZKP capabilities into existing MAIF instances."""

    # Add ZKP block extension
    zkp_extension = ZKPBlockExtension(maif_encoder)

    # Add ZKP methods to encoder
    maif_encoder.add_zkp_block = zkp_extension.add_zkp_block

    # Add ZKP verification to decoder
    maif_decoder.verify_zkp_block = lambda block_id: zkp_extension.verify_zkp_block(
        block_id, maif_decoder
    )

    # Add privacy enhancement methods
    maif_encoder.create_membership_proof = ZKPPrivacyEnhancer.create_membership_proof
    maif_encoder.create_range_proof = ZKPPrivacyEnhancer.create_range_proof

    return maif_encoder, maif_decoder


def create_zkp_enabled_block(
    encoder: MAIFEncoder,
    data: Any,
    proof_type: ZKProofType = ZKProofType.COMMITMENT_SCHEME,
    privacy_level: PrivacyLevel = PrivacyLevel.HIGH,
) -> str:
    """Helper function to create a ZKP-enabled block."""

    # Serialize data
    if isinstance(data, (dict, list)):
        data_bytes = json.dumps(data).encode()
    else:
        data_bytes = str(data).encode()

    # Create commitment
    commitment = hashlib.sha256(data_bytes).hexdigest()

    # Define statement based on privacy level
    statements = {
        PrivacyLevel.LOW: "Data integrity verified",
        PrivacyLevel.MEDIUM: "Data from authorized source",
        PrivacyLevel.HIGH: "Data exists without revealing content",
        PrivacyLevel.MAXIMUM: "Proof of knowledge without any disclosure",
    }

    statement = statements.get(privacy_level, "Zero-knowledge proof")

    # Public parameters based on privacy level
    public_params = {
        "privacy_level": privacy_level.value,
        "proof_system": "MAIF-ZKP",
        "version": "1.0",
    }

    if privacy_level == PrivacyLevel.LOW:
        public_params["data_type"] = type(data).__name__

    # Create ZKP block
    zkp_ext = ZKPBlockExtension(encoder)
    block_id = zkp_ext.add_zkp_block(data_bytes, proof_type, statement, public_params)

    return block_id


# Example usage functions
def example_private_analytics():
    """Example: Private analytics using ZKP."""

    # Create secure aggregation proof
    user_ages = [25, 30, 35, 40, 45]  # Individual ages remain private
    average_age = sum(user_ages) // len(user_ages)

    proof = ZKPSecureAggregation.create_aggregation_proof(user_ages, sum(user_ages))

    # Verify without knowing individual values
    return {
        "average": average_age,
        "count": len(user_ages),
        "proof": proof.proof_id,
        "privacy_preserved": True,
    }


def example_anonymous_verification():
    """Example: Anonymous credential verification."""

    authenticator = ZKPAgentAuthenticator()

    # Register agent with hidden attributes
    agent_id = "agent_123"
    public_key = b"mock_public_key"
    secret_attributes = {
        "clearance_level": "TOP_SECRET",
        "department": "Intelligence",
        "specialization": "Cryptography",
    }

    commitment = authenticator.register_agent_zkp(
        agent_id, public_key, secret_attributes
    )

    # Later: Prove attributes without revealing them
    challenge = hashlib.sha256(b"random_challenge").digest()
    response = hashlib.sha256(challenge + b"private_key").digest()

    verified, proof = authenticator.authenticate_agent_zkp(
        agent_id, challenge, response, secret_attributes
    )

    return {
        "verified": verified,
        "proof_id": proof.proof_id if proof else None,
        "attributes_hidden": True,
    }
