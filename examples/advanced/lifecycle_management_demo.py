"""
MAIF Lifecycle Management Demo
Demonstrates merging, splitting, and self-governing data fabric capabilities.

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
"""

import asyncio
import json
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAIF components
from maif.lifecycle_management import (
    MAIFMerger,
    MAIFSplitter,
    SelfGoverningMAIF,
    MAIFLifecycleManager,
    GovernanceRule,
    MAIFLifecycleState
)
from maif import MAIFEncoder
from maif_api import create_maif

def create_sample_maifs(workspace: Path, num_files: int = 3) -> list:
    """Create sample MAIF files for testing (secure format with Ed25519)."""
    maif_paths = []
    
    for i in range(num_files):
        maif_path = workspace / f"sample_{i}.maif"
        encoder = MAIFEncoder(str(maif_path), agent_id=f"lifecycle-agent-{i}")
        
        # Add various content
        for j in range(20):
            encoder.add_text_block(
                f"Sample text {j} from file {i}: This is test content for lifecycle management demo.",
                {"file_id": i, "block_id": j, "timestamp": time.time()}
            )
        
        # Add some binary data
        encoder.add_binary_block(
            b"Binary data sample" * 100,
            metadata={"type": "test_data", "file_id": i}
        )
        
        # Finalize (signs with Ed25519, self-contained)
        encoder.finalize()
        
        maif_paths.append(str(maif_path))
        logger.info(f"Created sample MAIF: {maif_path}")
    
    return maif_paths

def demonstrate_merging():
    """Demonstrate MAIF merging capabilities."""
    print("\n=== MAIF Merging Demo ===")
    
    workspace = Path("./demo_workspace/merging")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create sample MAIFs
    maif_paths = create_sample_maifs(workspace)
    
    merger = MAIFMerger()
    
    # Test different merge strategies
    strategies = ["append", "semantic", "temporal"]
    
    for strategy in strategies:
        print(f"\nTesting {strategy} merge strategy...")
        
        output_path = workspace / f"merged_{strategy}.maif"
        
        stats = merger.merge(
            maif_paths,
            str(output_path),
            merge_strategy=strategy,
            deduplication=True
        )
        
        print(f"Merge statistics for {strategy}:")
        print(f"  - Total blocks: {stats['total_blocks']}")
        print(f"  - Merged blocks: {stats['merged_blocks']}")
        print(f"  - Duplicates removed: {stats['duplicate_blocks']}")
        print(f"  - Merge time: {stats['merge_time']:.2f}s")
        print(f"  - Valid: {stats['is_valid']}")

def demonstrate_splitting():
    """Demonstrate MAIF splitting capabilities."""
    print("\n=== MAIF Splitting Demo ===")
    
    workspace = Path("./demo_workspace/splitting")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create a large MAIF (secure format)
    maif_path = workspace / "large_file.maif"
    encoder = MAIFEncoder(str(maif_path), agent_id="splitting-demo")
    
    # Add content of different types
    for i in range(100):
        encoder.add_text_block(
            f"Text block {i}: " + "Sample content " * 50,
            {"block_id": i, "type": "text", "category": f"cat_{i % 5}"}
        )
    
    for i in range(50):
        encoder.add_binary_block(
            b"Binary data " * 1000,
            metadata={"block_id": i, "type": "binary", "subtype": "image" if i % 2 == 0 else "data"}
        )
    
    # Finalize (self-contained with Ed25519 signatures)
    encoder.finalize()
    
    splitter = MAIFSplitter()
    
    # Test different split strategies
    strategies = [
        ("size", {"max_size_mb": 1.0}),
        ("count", {"blocks_per_file": 30}),
        ("type", {}),
        ("semantic", {"num_clusters": 3})
    ]
    
    for strategy, kwargs in strategies:
        print(f"\nTesting {strategy} split strategy...")
        
        output_dir = workspace / f"split_{strategy}"
        
        parts = splitter.split(
            str(maif_path),
            str(output_dir),
            split_strategy=strategy,
            **kwargs
        )
        
        print(f"Split into {len(parts)} parts:")
        for part in parts:
            size_mb = Path(part).stat().st_size / (1024 * 1024)
            print(f"  - {Path(part).name}: {size_mb:.2f} MB")

def demonstrate_self_governance():
    """Demonstrate self-governing data fabric."""
    print("\n=== Self-Governing Data Fabric Demo ===")
    
    workspace = Path("./demo_workspace/governance")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create a MAIF file (secure format)
    maif_path = workspace / "governed.maif"
    encoder = MAIFEncoder(str(maif_path), agent_id="governance-demo")
    
    for i in range(200):
        encoder.add_text_block(
            f"Block {i}: " + "Content " * 100,
            {"block_id": i, "timestamp": time.time()}
        )
    
    # Finalize (self-contained with Ed25519 signatures)
    encoder.finalize()
    
    # Create custom governance rules
    rules = [
        {
            "rule_id": "small_file_merge",
            "condition": "metrics.size_bytes < 1048576 and metrics.block_count < 10",
            "action": "merge_similar",
            "priority": 6,
            "enabled": True
        },
        {
            "rule_id": "optimize_frequent",
            "condition": "metrics.access_frequency > 5.0",
            "action": "optimize_hot",
            "priority": 8,
            "enabled": True
        }
    ]
    
    rules_path = workspace / "governance_rules.json"
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2)
    
    # Create self-governing MAIF
    governed = SelfGoverningMAIF(str(maif_path), str(rules_path))
    
    print("Self-governing MAIF created with custom rules")
    print("\nInitial governance report:")
    report = governed.get_governance_report()
    print(json.dumps(report, indent=2))
    
    # Simulate some activity
    print("\nSimulating file activity...")
    
    # Simulate high access
    for _ in range(20):
        governed.optimizer.record_access("block_1", 0.001)
    
    # Force metrics update
    governed._update_metrics()
    
    # Manually trigger rule evaluation for demo
    print("\nEvaluating governance rules...")
    actions = governed._evaluate_rules()
    print(f"Actions to execute: {actions}")
    
    # Get final report
    print("\nFinal governance report:")
    report = governed.get_governance_report()
    print(json.dumps(report, indent=2))
    
    # Stop governance
    governed.stop_governance()

def demonstrate_lifecycle_manager():
    """Demonstrate lifecycle manager for multiple MAIFs."""
    print("\n=== Lifecycle Manager Demo ===")
    
    workspace = Path("./demo_workspace/lifecycle_manager")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create multiple MAIFs
    maif_paths = create_sample_maifs(workspace, num_files=5)
    
    # Create lifecycle manager
    manager = MAIFLifecycleManager(str(workspace))
    
    # Add MAIFs to management
    for path in maif_paths:
        manager.add_maif(path)
    
    print(f"\nManaging {len(maif_paths)} MAIF files")
    
    # Get status
    print("\nLifecycle status:")
    status = manager.get_status()
    for path, report in status.items():
        print(f"\n{Path(path).name}:")
        print(f"  State: {report['state']}")
        print(f"  Size: {report['metrics']['size_mb']:.2f} MB")
        print(f"  Blocks: {report['metrics']['block_count']}")
        print(f"  Access frequency: {report['metrics']['access_frequency']:.2f}")
    
    # Test merging through manager
    print("\nMerging first 3 MAIFs...")
    merge_output = workspace / "manager_merged.maif"
    
    merge_stats = manager.merge_maifs(
        maif_paths[:3],
        str(merge_output),
        strategy="semantic"
    )
    
    print(f"Merged {merge_stats['merged_blocks']} blocks")
    
    # Test splitting through manager
    print("\nSplitting merged file...")
    split_output = workspace / "manager_split"
    
    parts = manager.split_maif(
        str(merge_output),
        str(split_output),
        strategy="count",
        blocks_per_file=20
    )
    
    print(f"Split into {len(parts)} parts")
    
    # Clean up
    for path in maif_paths:
        manager.remove_maif(path)

def demonstrate_advanced_governance():
    """Demonstrate advanced self-governance scenarios."""
    print("\n=== Advanced Self-Governance Demo ===")
    
    workspace = Path("./demo_workspace/advanced_governance")
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Create a MAIF that will trigger various rules (secure format)
    maif_path = workspace / "fragmented.maif"
    encoder = MAIFEncoder(str(maif_path), agent_id="advanced-governance-demo")
    
    # Add fragmented content (non-sequential)
    import random
    indices = list(range(100))
    random.shuffle(indices)
    
    for i in indices:
        encoder.add_text_block(
            f"Fragmented block {i}",
            {"original_index": i, "timestamp": time.time() - random.randint(0, 86400)}
        )
    
    # Finalize (self-contained with Ed25519 signatures)
    encoder.finalize()
    
    # Create governance with all default rules
    governed = SelfGoverningMAIF(str(maif_path))
    
    # Add custom rule for demonstration
    custom_rule = GovernanceRule(
        rule_id="demo_rule",
        condition="metrics.block_count > 50",
        action="semantic_reorganize",
        priority=10,
        enabled=True
    )
    governed.add_rule(custom_rule)
    
    print("Created fragmented MAIF with advanced governance")
    
    # Force immediate evaluation
    governed._update_metrics()
    actions = governed._evaluate_rules()
    
    print(f"\nTriggered actions: {actions}")
    
    # Execute one action for demo
    if actions:
        print(f"\nExecuting action: {actions[0]}")
        governed._execute_action(actions[0])
    
    # Show history
    report = governed.get_governance_report()
    print("\nGovernance history:")
    for event in report['history']:
        print(f"  - {event['action']} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['timestamp']))}")
    
    governed.stop_governance()

def main():
    """Run all lifecycle management demonstrations."""
    print("MAIF Lifecycle Management Demo")
    print("==============================")
    
    # Run demonstrations
    demonstrate_merging()
    demonstrate_splitting()
    demonstrate_self_governance()
    demonstrate_lifecycle_manager()
    demonstrate_advanced_governance()
    
    print("\n=== All Demos Completed ===")
    print("\nKey Features Demonstrated:")
    print("- MAIF Merging with multiple strategies (append, semantic, temporal)")
    print("- MAIF Splitting by size, count, type, and semantic similarity")
    print("- Self-Governing Data Fabric with rule-based autonomous management")
    print("- Lifecycle Manager for coordinating multiple MAIFs")
    print("- Advanced governance scenarios with custom rules")
    print("- Automatic reorganization, archiving, and optimization")
    print("\nSecure Format Features:")
    print("- Ed25519 signatures (64 bytes per block)")
    print("- Self-contained files (no external manifest needed)")
    print("- Embedded provenance chain for audit trail")

if __name__ == "__main__":
    main()