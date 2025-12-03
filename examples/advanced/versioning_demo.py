"""
Demonstration of MAIF versioning and forensic capabilities.
"""

import json
import time
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maif import MAIFEncoder, MAIFParser, MAIFSigner, MAIFVerifier
from maif.forensics import ForensicAnalyzer

def demonstrate_versioning():
    """Demonstrate MAIF versioning with append-on-write."""
    print("MAIF Versioning and Forensics Demonstration")
    print("=" * 50)
    
    # Create initial MAIF with agent 1
    print("\n1. Creating initial MAIF file...")
    agent1_id = "agent-001-alice"
    encoder = MAIFEncoder(agent_id=agent1_id)
    signer = MAIFSigner(agent_id=agent1_id)
    
    # Add initial content
    original_text = "This is the original document content."
    text_hash = encoder.add_text_block(original_text, metadata={
        "author": "Alice",
        "document_type": "report",
        "change_description": "Initial document creation"
    })
    signer.add_provenance_entry("create_document", text_hash)
    
    # Get the block ID for future updates
    text_block = encoder.blocks[0]
    text_block_id = text_block.block_id
    print(f"   Created block ID: {text_block_id}")
    
    # Build initial MAIF
    encoder.build_maif("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Sign manifest
    with open("versioned_doc_manifest.json", "r") as f:
        manifest = json.load(f)
    signed_manifest = signer.sign_maif_manifest(manifest)
    with open("versioned_doc_manifest.json", "w") as f:
        json.dump(signed_manifest, f, indent=2)
    
    print("   ✓ Initial MAIF created and signed")
    
    # Simulate some time passing
    time.sleep(1)
    
    # Update the document with agent 2 (append-on-write)
    print("\n2. Updating document with second agent...")
    agent2_id = "agent-002-bob"
    
    # Load existing MAIF for append-on-write
    encoder2 = MAIFEncoder(
        agent_id=agent2_id,
        existing_maif_path="versioned_doc.maif",
        existing_manifest_path="versioned_doc_manifest.json"
    )
    signer2 = MAIFSigner(agent_id=agent2_id)
    
    # Update the text block (this creates version 2)
    updated_text = "This is the original document content. UPDATED: Added new section by Bob."
    updated_hash = encoder2.add_text_block(updated_text, metadata={
        "author": "Bob", 
        "change_description": "Added new section with additional information"
    }, update_block_id=text_block_id)
    signer2.add_provenance_entry("update_document", updated_hash)
    
    print(f"   Updated block ID: {text_block_id} (now version 2)")
    
    # Build updated MAIF
    encoder2.build_maif("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Sign updated manifest
    with open("versioned_doc_manifest.json", "r") as f:
        manifest = json.load(f)
    signed_manifest = signer2.sign_maif_manifest(manifest)
    with open("versioned_doc_manifest.json", "w") as f:
        json.dump(signed_manifest, f, indent=2)
    
    print("   ✓ Document updated and signed")
    
    # Simulate more time and another update
    time.sleep(1)
    
    # Third update by agent 1 again
    print("\n3. Another update by original agent...")
    encoder3 = MAIFEncoder(
        agent_id=agent1_id,
        existing_maif_path="versioned_doc.maif",
        existing_manifest_path="versioned_doc_manifest.json"
    )
    signer3 = MAIFSigner(agent_id=agent1_id)
    
    final_text = "This is the original document content. UPDATED: Added new section by Bob. FINAL: Alice's final review and approval."
    final_hash = encoder3.add_text_block(final_text, metadata={
        "author": "Alice",
        "change_description": "Final review and approval",
        "status": "approved"
    }, update_block_id=text_block_id)
    signer3.add_provenance_entry("approve_document", final_hash)
    
    print(f"   Updated block ID: {text_block_id} (now version 3)")
    
    # Build final MAIF
    encoder3.build_maif("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Sign final manifest
    with open("versioned_doc_manifest.json", "r") as f:
        manifest = json.load(f)
    signed_manifest = signer3.sign_maif_manifest(manifest)
    with open("versioned_doc_manifest.json", "w") as f:
        json.dump(signed_manifest, f, indent=2)
    
    print("   ✓ Final version created and signed")
    
    return text_block_id

def analyze_version_history(block_id):
    """Analyze the version history and perform forensic analysis."""
    print("\n4. Analyzing version history...")
    
    # Parse the versioned MAIF
    parser = MAIFParser("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Get version history for our block
    versions = parser.decoder.get_block_versions(block_id)
    print(f"   Found {len(versions)} versions of block {block_id}")
    
    for version in versions:
        print(f"   - Version {version.version}: {version.hash_value[:16]}... "
              f"(author: {version.metadata.get('author', 'unknown')})")
    
    # Get complete timeline
    timeline = parser.decoder.get_version_timeline()
    print(f"\n   Complete timeline ({len(timeline)} events):")
    for event in timeline:
        datetime_str = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        print(f"   - {datetime_str}: {event.operation} by {event.agent_id}")
        if event.change_description:
            print(f"     Description: {event.change_description}")
    
    # Check if any blocks are deleted
    is_deleted = parser.decoder.is_block_deleted(block_id)
    print(f"\n   Block deletion status: {'DELETED' if is_deleted else 'ACTIVE'}")
    
    # Get changes by each agent
    print("\n   Changes by agent:")
    for agent_id in ["agent-001-alice", "agent-002-bob"]:
        changes = parser.decoder.get_changes_by_agent(agent_id)
        print(f"   - {agent_id}: {len(changes)} changes")
    
    return parser

def perform_forensic_analysis(parser):
    """Perform comprehensive forensic analysis."""
    print("\n5. Performing forensic analysis...")
    
    verifier = MAIFVerifier()
    analyzer = ForensicAnalyzer()
    
    # Perform analysis
    report = analyzer.analyze_maif_file("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Extract key information from the report dictionary
    print(f"   Forensic Status: {report.get('status', 'Unknown')}")
    print(f"   Risk Assessment: {report.get('risk_assessment', {}).get('overall_risk', 'Unknown')}")
    print(f"   Total Evidence: {len(report.get('evidence', []))}")
    print(f"   Recommendations: {len(report.get('recommendations', []))}")
    
    # Show version analysis details if available
    version_history = report.get('version_history', {})
    if version_history.get('version_count', 0) > 0:
        print("\n   Version Analysis Details:")
        print(f"   - Total versions: {version_history.get('version_count', 0)}")
        print(f"   - Active agents: {version_history.get('agent_count', 0)}")
        if version_history.get('anomalies'):
            print(f"   - Anomalies detected: {len(version_history.get('anomalies', []))}")
    
    # Show evidence if any
    evidence = report.get('evidence', [])
    if evidence:
        print("\n   Evidence Found:")
        for i, ev in enumerate(evidence[:3], 1):  # Show first 3 pieces of evidence
            severity = ev.get('severity', 'Unknown')
            description = ev.get('description', 'No description')
            print(f"   {i}. [{severity}] {description}")
    
    # Show recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n   Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec.get('description', 'No description')}")
    
    # Save forensic report
    with open("versioning_forensic_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n   ✓ Forensic report saved: versioning_forensic_report.json")
    
    return report

def demonstrate_data_recovery():
    """Demonstrate data recovery from version history."""
    print("\n6. Demonstrating data recovery...")
    
    parser = MAIFParser("versioned_doc.maif", "versioned_doc_manifest.json")
    
    # Get all text blocks and their versions
    text_blocks = [block for block in parser.decoder.blocks if block.block_type == "text_data"]
    
    if text_blocks:
        block_id = text_blocks[0].block_id
        versions = parser.decoder.get_block_versions(block_id)
        
        print(f"   Recovering all versions of block {block_id[:8]}...")
        
        for version in versions:
            # In a real implementation, you'd extract the actual content
            # For demo purposes, we'll show the metadata
            author = version.metadata.get('author', 'unknown')
            description = version.metadata.get('change_description', 'no description')
            print(f"   - Version {version.version} by {author}: {description}")
        
        # Get latest version
        latest = parser.decoder.get_latest_block_version(block_id)
        if latest:
            print(f"\n   Latest version: {latest.version} (hash: {latest.hash_value[:16]}...)")
    
    print("   ✓ Data recovery demonstration complete")

def main():
    """Main demonstration function."""
    try:
        # Demonstrate versioning
        block_id = demonstrate_versioning()
        
        # Analyze version history
        parser = analyze_version_history(block_id)
        
        # Perform forensic analysis
        report = perform_forensic_analysis(parser)
        
        # Demonstrate data recovery
        demonstrate_data_recovery()
        
        print("\n" + "=" * 50)
        print("Versioning and Forensics Demonstration Complete!")
        print("\nFiles created:")
        print("  - versioned_doc.maif (versioned MAIF container)")
        print("  - versioned_doc_manifest.json (signed manifest with version history)")
        print("  - versioning_forensic_report.json (comprehensive forensic analysis)")
        
        print(f"\nKey achievements:")
        # Count total versions across all blocks
        total_versions = sum(len(versions) for versions in parser.decoder.version_history.values())
        # Get unique agents from all versions
        all_agents = set()
        for versions in parser.decoder.version_history.values():
            for v in versions:
                all_agents.add(v.agent_id)
        
        print(f"  - Created {total_versions} version entries")
        print(f"  - Tracked changes by {len(all_agents)} agents")
        print(f"  - Preserved complete audit trail with forensic analysis")
        print(f"  - Demonstrated append-on-write with data preservation")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()