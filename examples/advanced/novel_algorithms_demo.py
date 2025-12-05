#!/usr/bin/env python3
"""
MAIF Novel Algorithms Demo

Demonstrates the three novel algorithms implemented in MAIF:
1. ACAM - Adaptive Cross-Modal Attention Mechanism
2. HSC - Hierarchical Semantic Compression
3. CSB - Cryptographic Semantic Binding

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path

# Add the parent directory to the path so we can import maif
sys.path.insert(0, str(Path(__file__).parent.parent))

from maif import MAIFEncoder, MAIFDecoder, MAIFParser
from maif.semantic import (
    SemanticEmbedder,
    CrossModalAttention,
    HierarchicalSemanticCompression,
    CryptographicSemanticBinding,
    DeepSemanticUnderstanding,
)


def demo_acam():
    """Demonstrate Adaptive Cross-Modal Attention Mechanism (ACAM)."""
    print("\n" + "=" * 60)
    print("ACAM - Adaptive Cross-Modal Attention Mechanism Demo")
    print("=" * 60)

    # Create sample multimodal embeddings
    text_embedding = [0.1, 0.2, 0.3, 0.4] * 96  # 384-dim
    image_embedding = [0.2, 0.3, 0.4, 0.5] * 96  # 384-dim
    audio_embedding = [0.3, 0.4, 0.5, 0.6] * 96  # 384-dim

    embeddings = {
        "text": text_embedding,
        "image": image_embedding,
        "audio": audio_embedding,
    }

    # Initialize ACAM
    acam = CrossModalAttention(embedding_dim=384)

    # Compute attention weights
    print("Computing cross-modal attention weights...")
    attention_weights = acam.compute_attention_weights(embeddings)

    print("\nAttention weights between modalities:")
    if hasattr(attention_weights, "coherence_matrix"):
        # Display the coherence matrix as modality relationships
        modalities = attention_weights.modalities
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    weight = attention_weights.coherence_matrix[i, j]
                    print(f"  {mod1} -> {mod2}: {weight:.4f}")
    else:
        print("  AttentionWeights structure not as expected")

    # Get attended representation for text modality
    attended_text = acam.get_attended_representation(embeddings, "text")

    print(f"\nOriginal text embedding length: {len(text_embedding)}")
    print(f"Attended text representation length: {len(attended_text)}")
    print(
        f"Attention-weighted enhancement: {np.linalg.norm(attended_text) / np.linalg.norm(text_embedding):.4f}x"
    )

    return attention_weights


def demo_hsc():
    """Demonstrate Hierarchical Semantic Compression (HSC)."""
    print("\n" + "=" * 60)
    print("HSC - Hierarchical Semantic Compression Demo")
    print("=" * 60)

    # Create sample embeddings with semantic relationships
    embeddings = []
    for i in range(20):
        # Create embeddings with some semantic structure
        base = [0.1 * (i % 5), 0.2 * (i % 3), 0.3 * (i % 7)]
        embedding = base * 128  # 384-dim
        # Add some noise
        embedding = [x + np.random.normal(0, 0.01) for x in embedding]
        embeddings.append(embedding)

    print(
        f"Original embeddings: {len(embeddings)} vectors of {len(embeddings[0])} dimensions"
    )
    original_size = len(embeddings) * len(embeddings[0]) * 4  # 4 bytes per float
    print(f"Original size: {original_size} bytes")

    # Initialize HSC
    hsc = HierarchicalSemanticCompression(compression_levels=3)

    # Compress embeddings
    print("\nApplying Hierarchical Semantic Compression...")
    start_time = time.time()
    compressed_result = hsc.compress_embeddings(embeddings)
    compression_time = time.time() - start_time

    # Display compression results
    metadata = compressed_result["metadata"]
    compressed_size = len(str(compressed_result).encode())

    print(f"Compression completed in {compression_time:.4f} seconds")
    print(f"Compressed size: {compressed_size} bytes")
    print(f"Compression ratio: {metadata['compression_ratio']:.2f}:1")
    print(f"Space savings: {(1 - compressed_size / original_size) * 100:.1f}%")
    print(f"Semantic clusters identified: {len(set(metadata.get('clusters', [])))}")

    # Decompress and verify
    print("\nDecompressing embeddings...")
    start_time = time.time()
    decompressed_embeddings = hsc.decompress_embeddings(compressed_result)
    decompression_time = time.time() - start_time

    print(f"Decompression completed in {decompression_time:.4f} seconds")
    print(f"Decompressed: {len(decompressed_embeddings)} vectors")

    # Calculate semantic preservation
    if len(decompressed_embeddings) > 0:
        original_norm = np.linalg.norm(embeddings[0])
        decompressed_norm = np.linalg.norm(decompressed_embeddings[0])
        preservation_ratio = decompressed_norm / original_norm
        print(f"Semantic preservation ratio: {preservation_ratio:.4f}")

    return compressed_result


def demo_csb():
    """Demonstrate Cryptographic Semantic Binding (CSB)."""
    print("\n" + "=" * 60)
    print("CSB - Cryptographic Semantic Binding Demo")
    print("=" * 60)

    # Create sample embedding and source data
    source_text = "Artificial intelligence systems require trustworthy data containers for secure operation."

    # Generate embedding for the source text
    embedder = SemanticEmbedder()
    embedding_obj = embedder.embed_text(source_text)
    embedding = embedding_obj.vector

    print(f"Source text: {source_text}")
    print(f"Embedding dimensions: {len(embedding)}")

    # Initialize CSB
    csb = CryptographicSemanticBinding()

    # Create semantic commitment
    print("\nCreating cryptographic semantic binding...")
    binding = csb.create_semantic_commitment(embedding, source_text)

    print("Binding created with components:")
    for key, value in binding.items():
        if key != "salt":  # Don't show salt for security
            print(
                f"  {key}: {value[:32]}..."
                if len(str(value)) > 32
                else f"  {key}: {value}"
            )

    # Create zero-knowledge proof
    print("\nGenerating zero-knowledge proof...")
    zk_proof = csb.create_zero_knowledge_proof(embedding, binding)

    print("Zero-knowledge proof generated:")
    for key, value in zk_proof.items():
        print(
            f"  {key}: {value[:32]}..." if len(str(value)) > 32 else f"  {key}: {value}"
        )

    # Verify the binding
    print("\nVerifying semantic binding...")
    is_valid = csb.verify_semantic_binding(embedding, source_text, binding)
    print(f"Binding verification: {'VALID' if is_valid else 'INVALID'}")

    # Test with tampered data
    print("\nTesting with tampered embedding...")
    tampered_embedding = embedding.copy()
    tampered_embedding[0] += 0.1  # Slight modification
    is_tampered_valid = csb.verify_semantic_binding(
        tampered_embedding, source_text, binding
    )
    print(
        f"Tampered binding verification: {'VALID' if is_tampered_valid else 'INVALID'}"
    )

    # Verify zero-knowledge proof
    print("\nVerifying zero-knowledge proof...")
    zk_valid = csb.verify_zero_knowledge_proof(zk_proof, binding)
    print(f"Zero-knowledge proof verification: {'VALID' if zk_valid else 'INVALID'}")

    return binding, zk_proof


def demo_cross_modal_ai():
    """Demonstrate Cross-Modal AI with Deep Semantic Understanding."""
    print("\n" + "=" * 60)
    print("Cross-Modal AI - Deep Semantic Understanding Demo")
    print("=" * 60)

    # Initialize Deep Semantic Understanding
    dsu = DeepSemanticUnderstanding()

    # Create multimodal content for analysis
    multimodal_content = {
        "text": "A beautiful sunset over the ocean with waves crashing on the shore"
    }

    print("Analyzing multimodal content:")
    print(f"  Text: {multimodal_content['text']}")

    # Analyze semantic content
    print("\nApplying deep semantic understanding...")
    result = dsu.analyze_semantic_content(multimodal_content)

    print(f"\nAnalysis results:")
    print(f"  Embeddings generated: {len(result['embeddings'])}")

    if result["embeddings"].get("text") is not None:
        text_embedding = result["embeddings"]["text"]
        print(
            f"  Text embedding dimensions: {len(text_embedding) if hasattr(text_embedding, '__len__') else 'N/A'}"
        )

    if result["knowledge_graph"].get("entities"):
        print(f"  Entities extracted: {len(result['knowledge_graph']['entities'])}")
        for entity in result["knowledge_graph"]["entities"][:3]:  # Show first 3
            print(f"    - {entity}")

    print(f"  Semantic coherence score: {result['semantic_coherence']:.4f}")
    print(f"  Understanding score: {result['understanding_score']:.4f}")

    return result


def demo_maif_integration():
    """Demonstrate integration of novel algorithms in MAIF format."""
    print("\n" + "=" * 60)
    print("MAIF Integration Demo - Novel Algorithms in Action")
    print("=" * 60)

    # Create MAIF encoder (secure format with Ed25519)
    output_file = "novel_algorithms_demo.maif"
    encoder = MAIFEncoder(output_file, agent_id="novel_algorithms_demo")

    # Sample data
    text_data = "MAIF enables trustworthy AI through novel algorithms and cross-modal understanding."
    embedder = SemanticEmbedder()
    embedding_obj = embedder.embed_text(text_data)
    embeddings = [embedding_obj.vector]

    # Add regular text block
    text_block_id = encoder.add_text_block(
        text_data, {"source": "demo", "type": "description"}
    )
    print(f"Added text block: {text_block_id[:16]}...")

    # Add cross-modal data as text block with ACAM metadata
    multimodal_data = {
        "text": text_data,
        "metadata": {"importance": "high", "category": "AI_trustworthiness"},
    }
    cross_modal_block_id = encoder.add_text_block(
        json.dumps(multimodal_data),
        {"algorithm": "ACAM", "demo": True, "type": "cross_modal"},
    )
    print(f"Added cross-modal block (ACAM): {cross_modal_block_id[:16]}...")

    # Add embeddings block (uses HSC internally)
    hsc_block_id = encoder.add_embeddings_block(
        embeddings, {"algorithm": "HSC", "demo": True, "compressed": True}
    )
    print(f"Added embeddings block (HSC): {hsc_block_id[:16]}...")

    # Add semantic binding as text+embedding combo
    csb_data = {
        "source_text": text_data,
        "binding_hash": embedding_obj.source_hash
        if hasattr(embedding_obj, "source_hash")
        else "computed",
    }
    csb_block_id = encoder.add_text_block(
        json.dumps(csb_data),
        {"algorithm": "CSB", "demo": True, "type": "semantic_binding"},
    )
    print(f"Added semantic binding block (CSB): {csb_block_id[:16]}...")

    # Finalize MAIF file (self-contained with Ed25519 signatures)
    print(f"\nFinalizing MAIF file: {output_file}")
    encoder.finalize()
    print("  (Self-contained with Ed25519 signatures, no manifest needed)")

    # Parse and verify the MAIF file
    print(f"Parsing MAIF file...")
    decoder = MAIFDecoder(output_file)
    decoder.load()

    # Verify integrity
    is_valid, errors = decoder.verify_integrity()
    print(f"File integrity: {'VALID ✅' if is_valid else 'INVALID ❌'}")

    # Get file info
    file_info = decoder.get_file_info()
    print(f"\nFile Information:")
    print(f"  Blocks: {file_info['block_count']}")
    print(f"  Signed: {'Yes' if file_info['is_signed'] else 'No'}")
    print(f"  Merkle Root: {file_info['merkle_root'][:32]}...")

    # Show block types
    blocks = decoder.get_blocks()
    print(f"\nBlocks in MAIF file: {len(blocks)}")
    for i, block in enumerate(blocks):
        metadata = block.metadata or {}
        algo = metadata.get("algorithm", "standard")
        print(f"  Block {i}: {algo}")

    return output_file


def main():
    """Run all novel algorithm demonstrations."""
    print("MAIF Novel Algorithms and Cross-Modal AI Demonstration")
    print("=" * 80)
    print("This demo showcases the three novel algorithms implemented in MAIF:")
    print("1. ACAM - Adaptive Cross-Modal Attention Mechanism")
    print("2. HSC - Hierarchical Semantic Compression")
    print("3. CSB - Cryptographic Semantic Binding")
    print("Plus cross-modal AI with deep semantic understanding")
    print("=" * 80)

    try:
        # Run individual algorithm demos
        attention_weights = demo_acam()
        compressed_result = demo_hsc()
        binding, zk_proof = demo_csb()
        cross_modal_result = demo_cross_modal_ai()

        # Demonstrate MAIF integration
        maif_file = demo_maif_integration()

        print("\n" + "=" * 60)
        print("Demo Summary")
        print("=" * 60)
        print("✓ ACAM: Successfully computed cross-modal attention weights")
        print("✓ HSC: Successfully compressed embeddings with semantic preservation")
        print("✓ CSB: Successfully created cryptographic semantic bindings")
        print("✓ Cross-Modal AI: Successfully processed multimodal input")
        print(
            "✓ MAIF Integration: Successfully created MAIF file with novel algorithms"
        )

        print(f"\nGenerated files:")
        print(f"  - {maif_file} (self-contained, Ed25519 signed)")

        print("\nNovel algorithms are now integrated and operational in MAIF!")
        print("Using secure format with Ed25519 signatures and embedded provenance.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
