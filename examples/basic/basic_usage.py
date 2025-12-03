"""
Basic usage example for MAIF library.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
from maif import MAIFEncoder, MAIFParser, MAIFSigner, MAIFVerifier
from maif.semantic import SemanticEmbedder, KnowledgeGraphBuilder, SemanticEmbedding
from maif.forensics import ForensicAnalyzer

# Set up logging
logger = logging.getLogger(__name__)

def create_sample_maif():
    """Create a sample MAIF file with multimodal content."""
    print("Creating sample MAIF file...")
    
    # Initialize components
    encoder = MAIFEncoder()
    signer = MAIFSigner(agent_id="demo-agent-001")
    embedder = SemanticEmbedder()
    kg_builder = KnowledgeGraphBuilder()
    
    # Add text content
    texts = [
        "Artificial intelligence is transforming how we work and live.",
        "Machine learning models require careful validation and testing.",
        "Trustworthy AI systems need transparency and accountability."
    ]
    
    for i, text in enumerate(texts):
        # Add text block
        text_hash = encoder.add_text_block(text, metadata={"source": f"document_{i}.txt"})
        
        # Add provenance entry
        signer.add_provenance_entry(f"add_text_block", text_hash)
        
        # Generate embeddings
        embedding = embedder.embed_text(text)
        
        # Extract entities for knowledge graph
        entities = kg_builder.extract_entities_from_text(text, source=f"document_{i}")
    
    # Add embeddings block
    embeddings = [emb.vector for emb in embedder.embeddings]
    embed_hash = encoder.add_embeddings_block(embeddings, metadata={
        "model": embedder.model_name,
        "dimensions": len(embeddings[0]) if embeddings else 0
    })
    signer.add_provenance_entry("add_embeddings_block", embed_hash)
    
    # Add knowledge graph block
    kg_data = kg_builder.export_to_json()
    kg_json = json.dumps(kg_data).encode('utf-8')
    kg_hash = encoder.add_binary_block(kg_json, "knowledge_graph", metadata={
        "format": "json",
        "triples_count": len(kg_builder.triples)
    })
    signer.add_provenance_entry("add_knowledge_graph", kg_hash)
    
    # Build MAIF file
    encoder.build_maif("sample.maif", "sample_manifest.json")
    
    # Sign the manifest
    with open("sample_manifest.json", "r") as f:
        manifest = json.load(f)
    
    signed_manifest = signer.sign_maif_manifest(manifest)
    
    with open("sample_manifest.json", "w") as f:
        json.dump(signed_manifest, f, indent=2)
    
    print("✓ Sample MAIF file created: sample.maif")
    print("✓ Signed manifest created: sample_manifest.json")
    
    return "sample.maif", "sample_manifest.json"

def verify_and_analyze_maif(maif_path, manifest_path):
    """Verify and analyze a MAIF file."""
    print(f"\nVerifying and analyzing {maif_path}...")
    
    # Parse MAIF file
    parser = MAIFParser(maif_path, manifest_path)
    
    # Verify integrity
    integrity_ok = parser.verify_integrity()
    print(f"✓ File integrity: {'VALID' if integrity_ok else 'INVALID'}")
    
    # Verify signatures and provenance
    verifier = MAIFVerifier()
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    manifest_valid, errors = verifier.verify_maif_manifest(manifest)
    print(f"✓ Manifest verification: {'VALID' if manifest_valid else 'INVALID'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Extract content
    content = parser.extract_content()
    print(f"✓ Extracted {len(content['texts'])} text blocks")
    print(f"✓ Extracted {len(content['embeddings'])} embeddings")
    
    # Perform forensic analysis
    print("\nPerforming forensic analysis...")
    forensic_analyzer = ForensicAnalyzer()
    report = forensic_analyzer.analyze_maif_file(maif_path, manifest_path)
    
    print(f"✓ Forensic status: {report.get('status', 'analyzed')}")
    print(f"✓ Risk assessment: {report.get('risk_assessment', {}).get('overall_risk', 'unknown')}")
    print(f"✓ Total evidence: {report.get('total_evidence', 0)}")
    
    evidence_list = report.get('evidence', [])
    if evidence_list:
        print(f"⚠ Found {len(evidence_list)} pieces of evidence:")
        for evidence in evidence_list[:3]:  # Show first 3
            print(f"  - {evidence.get('severity', 'unknown').upper()}: {evidence.get('description', 'No description')}")
    
    recommendations = report.get('recommendations', [])
    print(f"✓ Recommendations: {len(recommendations)}")
    for rec in recommendations[:3]:  # Show first 3 recommendations
        print(f"  - {rec}")
    
    return report

def demonstrate_semantic_search():
    """Demonstrate semantic search capabilities."""
    print("\nDemonstrating semantic search...")
    
    # Parse the MAIF file
    parser = MAIFParser("sample.maif", "sample_manifest.json")
    content = parser.extract_content()
    
    if not content['embeddings']:
        print("No embeddings found for semantic search")
        return
    
    # Initialize embedder for query
    embedder = SemanticEmbedder()
    
    # Create query embedding
    query = "How can we make AI more trustworthy?"
    query_embedding = embedder.embed_text(query)
    
    # Simple similarity search (in a real implementation, you'd use FAISS)
    similarities = []
    for i, embedding_vector in enumerate(content['embeddings']):
        # Create embedding object for comparison
        doc_embedding = SemanticEmbedding(
            vector=embedding_vector,
            source_hash="",
            model_name=embedder.model_name,
            timestamp=0
        )
        
        similarity = embedder.compute_similarity(query_embedding, doc_embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Query: '{query}'")
    print("Most similar content:")
    for i, (doc_idx, similarity) in enumerate(similarities[:2]):
        if doc_idx < len(content['texts']):
            print(f"  {i+1}. (similarity: {similarity:.3f}) {content['texts'][doc_idx][:100]}...")

def main():
    """Main demonstration function."""
    print("MAIF Library Demonstration")
    print("=" * 50)
    
    try:
        # Create sample MAIF
        maif_path, manifest_path = create_sample_maif()
        
        # Verify and analyze
        report = verify_and_analyze_maif(maif_path, manifest_path)
        
        # Demonstrate semantic search
        demonstrate_semantic_search()
        
        # Save forensic report
        with open("forensic_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n✓ Forensic report saved: forensic_report.json")
        
        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print("Files created:")
        print("  - sample.maif (MAIF container)")
        print("  - sample_manifest.json (signed manifest)")
        print("  - forensic_report.json (analysis report)")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()