"""
MAIF Integrated Features Demo
=============================

This example demonstrates how to use the integrated features of MAIF:
- Event Sourcing
- Columnar Storage
- Dynamic Version Management
- Adaptation Rules Engine

The demo shows how these features work together in a cohesive way.
"""

import os
import json
import time
import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# matplotlib is optional - for visualization only
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import MAIF components
from maif.integration_enhanced import EnhancedMAIF, EnhancedMAIFManager
from maif.adaptation_rules import (
    AdaptationRule, RulePriority, RuleStatus,
    ActionType, TriggerType, ComparisonOperator,
    MetricCondition, Action, ActionParameter
)

def setup_workspace():
    """Set up workspace for the demo."""
    workspace = Path("./demo_workspace/integrated_features")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace

def demonstrate_basic_usage(workspace: Path):
    """Demonstrate basic usage of integrated features."""
    print("\n=== Basic Usage of Integrated Features ===")
    
    # Create an enhanced MAIF
    maif_path = workspace / "basic_demo.maif"
    maif = EnhancedMAIF(
        str(maif_path),
        agent_id="demo-agent",
        enable_event_sourcing=True,
        enable_columnar_storage=True,
        enable_version_management=True,
        enable_adaptation_rules=True
    )
    
    # Add some content
    print("\nAdding content to MAIF...")
    for i in range(10):
        text = f"Sample text block {i}: This demonstrates integrated features."
        metadata = {
            "block_id": f"block_{i}",
            "timestamp": time.time(),
            "category": f"category_{i % 3}",
            "priority": i % 5
        }
        block_id = maif.add_text_block(text, metadata)
        print(f"  Added text block {i} with ID: {block_id}")
    
    # Add binary data
    binary_data = b"Binary data sample" * 100
    binary_id = maif.add_binary_block(
        binary_data,
        metadata={"type": "test_data", "timestamp": time.time()}
    )
    print(f"  Added binary block with ID: {binary_id}")
    
    # Save MAIF
    maif.save()
    print(f"\nMAIF saved to {maif_path}")
    
    # Show event history
    print("\nEvent history:")
    for event in maif.get_history():
        print(f"  - {event.event_type.value} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
    
    # Show schema version
    print(f"\nCurrent schema version: {maif.get_schema_version()}")
    
    # Show columnar statistics
    print("\nColumnar storage statistics:")
    stats = maif.get_columnar_statistics()
    for column, column_stats in stats.items():
        print(f"  - {column}: {column_stats}")
    
    # Evaluate adaptation rules
    print("\nEvaluating adaptation rules...")
    actions = maif.evaluate_rules()
    print(f"Actions triggered: {actions}")
    
    return maif

def demonstrate_event_sourcing_features(maif: EnhancedMAIF):
    """Demonstrate event sourcing features."""
    print("\n=== Event Sourcing Features ===")
    
    # Show event history (already captured during basic usage)
    print("\nReviewing events captured during file creation...")
    
    # Get history
    history = maif.get_history()
    
    print(f"Total events recorded: {len(history)}")
    
    # Note: In the secure format, files are immutable after finalization.
    # This demonstrates the event sourcing for the blocks that were added
    # before finalization. New blocks would require creating a new MAIF file.
    
    # Show latest events
    print("\nLatest events:")
    latest_events = maif.get_history()[-3:]
    for event in latest_events:
        print(f"  - {event.event_type.value} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}")
        print(f"    Agent: {event.agent_id}")
        print(f"    Payload: {event.payload}")
    
    # Save changes
    maif.save()
    print("\nChanges saved with event tracking")

def demonstrate_columnar_storage_features(workspace: Path):
    """Demonstrate columnar storage features."""
    print("\n=== Columnar Storage Features ===")
    
    # Create a new MAIF with columnar storage
    maif_path = workspace / "columnar_demo.maif"
    maif = EnhancedMAIF(
        str(maif_path),
        agent_id="columnar-agent",
        enable_event_sourcing=False,
        enable_columnar_storage=True,
        enable_version_management=False,
        enable_adaptation_rules=False
    )
    
    # Add structured data
    print("\nAdding structured data for columnar storage...")
    
    # Generate some structured data
    categories = ["news", "sports", "technology", "entertainment", "business"]
    ratings = [1, 2, 3, 4, 5]
    
    for i in range(100):
        category = categories[i % len(categories)]
        rating = ratings[i % len(ratings)]
        timestamp = time.time() - (i * 3600)  # Spread over time
        
        text = f"Article {i}: This is a sample article about {category}."
        metadata = {
            "category": category,
            "rating": rating,
            "timestamp": timestamp,
            "word_count": len(text.split()),
            "is_featured": (i % 10 == 0)
        }
        
        maif.add_text_block(text, metadata)
    
    # Save MAIF
    maif.save()
    print(f"\nColumnar MAIF saved to {maif_path}")
    
    # Show columnar statistics
    print("\nColumnar storage statistics:")
    stats = maif.get_columnar_statistics()
    for column, column_stats in stats.items():
        if isinstance(column_stats, dict) and "min_value" in column_stats and "max_value" in column_stats:
            print(f"  - {column}: min={column_stats['min_value']}, max={column_stats['max_value']}, null_count={column_stats.get('null_count', 0)}")
    
    return maif

def demonstrate_version_management_features(workspace: Path):
    """Demonstrate version management features."""
    print("\n=== Version Management Features ===")
    
    # Create a new MAIF with version management
    maif_path = workspace / "versioned_demo.maif"
    maif = EnhancedMAIF(
        str(maif_path),
        agent_id="version-agent",
        enable_event_sourcing=False,
        enable_columnar_storage=False,
        enable_version_management=True,
        enable_adaptation_rules=False
    )
    
    # Add some content with the initial schema
    print("\nAdding content with initial schema (v1.0.0)...")
    for i in range(5):
        text = f"Content for version 1.0.0 - item {i}"
        metadata = {
            "id": f"item_{i}",
            "type": "text",
            "content": text,
            "metadata": {"version": "1.0.0", "index": i}
        }
        maif.add_text_block(text, metadata)
    
    # Save MAIF
    maif.save()
    
    # Create a new schema version
    print("\nCreating new schema version (v1.1.0)...")
    registry = maif.schema_registry
    
    # Define new schema
    from maif.version_management import Schema, SchemaField
    new_schema = Schema(
        version="1.1.0",
        fields=[
            SchemaField(name="id", field_type="string", required=True),
            SchemaField(name="type", field_type="string", required=True),
            SchemaField(name="content", field_type="string", required=True),
            SchemaField(name="metadata", field_type="json", required=False),
            SchemaField(name="tags", field_type="json", required=False, default_value=[]),
            SchemaField(name="rating", field_type="float", required=False, default_value=0.0)
        ],
        metadata={"created_at": time.time()}
    )
    
    # Register new schema
    registry.register_schema(new_schema)
    
    # Define transition
    from maif.version_management import VersionTransition, VersionCompatibility
    transition = VersionTransition(
        from_version="1.0.0",
        to_version="1.1.0",
        compatibility=VersionCompatibility.BACKWARD
    )
    
    # Add field transformations
    transition.add_field_add("tags", "json", [])
    transition.add_field_add("rating", "float", 0.0)
    
    # Register transition
    registry.register_transition(transition)
    
    # Save registry
    registry_path = maif.maif_path.with_suffix('.schema')
    registry.save(str(registry_path))
    
    print(f"Schema updated to version {registry.get_latest_version()}")
    
    # Note: In the secure format, MAIF files are immutable after finalization.
    # To add content with a new schema, create a new MAIF file.
    print("\nCreating new MAIF with updated schema (v1.1.0)...")
    
    maif_path_v2 = workspace / "versioned_demo_v1.1.maif"
    maif_v2 = EnhancedMAIF(
        str(maif_path_v2),
        agent_id="version-agent-v1.1",
        enable_event_sourcing=False,
        enable_columnar_storage=False,
        enable_version_management=True,
        enable_adaptation_rules=False
    )
    
    for i in range(5, 10):
        text = f"Content for version 1.1.0 - item {i}"
        metadata = {
            "id": f"item_{i}",
            "type": "text",
            "content": text,
            "metadata": {"version": "1.1.0", "index": i},
            "tags": [f"tag_{j}" for j in range(i % 3 + 1)],
            "rating": i / 2.0
        }
        maif_v2.add_text_block(text, metadata)
    
    # Save the new MAIF
    maif_v2.save()
    print(f"  New versioned MAIF saved to {maif_path_v2}")
    
    # Original MAIF info
    print(f"\nOriginal versioned MAIF: {maif_path}")
    
    return maif

def demonstrate_adaptation_rules_features(workspace: Path):
    """Demonstrate adaptation rules features."""
    print("\n=== Adaptation Rules Features ===")
    
    # Create a new MAIF with adaptation rules
    maif_path = workspace / "adaptive_demo.maif"
    maif = EnhancedMAIF(
        str(maif_path),
        agent_id="adaptive-agent",
        enable_event_sourcing=True,
        enable_columnar_storage=True,
        enable_version_management=True,
        enable_adaptation_rules=True
    )
    
    # Add some content
    print("\nAdding content to trigger adaptation rules...")
    
    # Add enough content to trigger size-based rules
    for i in range(200):
        text = f"Sample text {i}: " + "Content " * 50
        metadata = {
            "block_id": f"block_{i}",
            "timestamp": time.time(),
            "category": f"category_{i % 5}"
        }
        maif.add_text_block(text, metadata)
    
    # Save MAIF
    maif.save()
    
    # Add custom rule
    print("\nAdding custom adaptation rule...")
    
    # Create a custom rule for semantic reorganization
    semantic_condition = MetricCondition(
        metric_name="semantic_coherence",
        operator=ComparisonOperator.LESS_THAN,
        threshold=0.8
    )
    
    semantic_action = Action(
        action_type=ActionType.TRANSFORM,
        parameters=[
            ActionParameter(name="operation", value="semantic_reorganize"),
            ActionParameter(name="threshold", value=0.7)
        ]
    )
    
    semantic_rule = AdaptationRule(
        rule_id="semantic_reorganize",
        name="Semantic Reorganization",
        description="Reorganize content based on semantic similarity",
        priority=RulePriority.HIGH,
        trigger=TriggerType.METRIC,
        condition=semantic_condition,
        actions=[semantic_action],
        status=RuleStatus.ACTIVE
    )
    
    # Register custom rule
    maif.rules_engine.register_rule(semantic_rule)
    
    # Simulate metrics that would trigger rules
    print("\nSimulating metrics to trigger rules...")
    maif.metrics.size_bytes = 150 * 1024 * 1024  # 150 MB
    maif.metrics.block_count = 200
    maif.metrics.semantic_coherence = 0.6
    
    # Evaluate rules
    print("\nEvaluating adaptation rules...")
    actions = maif.evaluate_rules()
    print(f"Actions triggered: {actions}")
    
    return maif

def demonstrate_integrated_manager(workspace: Path):
    """Demonstrate the integrated MAIF manager."""
    print("\n=== Integrated MAIF Manager ===")
    
    # Create manager
    manager = EnhancedMAIFManager(str(workspace / "managed"))
    
    # Create multiple MAIFs with different feature combinations
    print("\nCreating multiple MAIFs with different feature combinations...")
    
    # MAIF with all features
    maif1 = manager.create_maif(
        "complete",
        agent_id="agent-complete",
        enable_event_sourcing=True,
        enable_columnar_storage=True,
        enable_version_management=True,
        enable_adaptation_rules=True
    )
    
    # MAIF with event sourcing and columnar storage
    maif2 = manager.create_maif(
        "data_focused",
        agent_id="agent-data",
        enable_event_sourcing=True,
        enable_columnar_storage=True,
        enable_version_management=False,
        enable_adaptation_rules=False
    )
    
    # MAIF with version management and adaptation rules
    maif3 = manager.create_maif(
        "management_focused",
        agent_id="agent-management",
        enable_event_sourcing=False,
        enable_columnar_storage=False,
        enable_version_management=True,
        enable_adaptation_rules=True
    )
    
    # Add content to each MAIF
    for i in range(50):
        maif1.add_text_block(f"Complete MAIF content {i}", {"index": i})
        maif2.add_text_block(f"Data-focused MAIF content {i}", {"index": i})
        maif3.add_text_block(f"Management-focused MAIF content {i}", {"index": i})
    
    # Save all MAIFs
    manager.save_all()
    
    # Get status of all MAIFs
    print("\nStatus of all managed MAIFs:")
    status = manager.get_status()
    for name, maif_status in status.items():
        print(f"\n{name}:")
        print(f"  Path: {maif_status['path']}")
        print(f"  Agent: {maif_status['agent_id']}")
        print(f"  State: {maif_status['state']}")
        print(f"  Size: {maif_status['metrics']['size_bytes']} bytes")
        print(f"  Blocks: {maif_status['metrics']['block_count']}")
        print("  Features enabled:")
        for feature, enabled in maif_status['features'].items():
            print(f"    - {feature}: {enabled}")
    
    # Evaluate rules for all MAIFs
    print("\nEvaluating rules for all MAIFs...")
    rule_results = manager.evaluate_all_rules()
    for name, actions in rule_results.items():
        print(f"  {name}: {actions}")
    
    return manager

def main():
    """Run the integrated features demo."""
    print("MAIF Integrated Features Demo")
    print("=============================")
    
    # Set up workspace
    workspace = setup_workspace()
    
    # Run demonstrations
    basic_maif = demonstrate_basic_usage(workspace)
    demonstrate_event_sourcing_features(basic_maif)
    demonstrate_columnar_storage_features(workspace)
    demonstrate_version_management_features(workspace)
    demonstrate_adaptation_rules_features(workspace)
    demonstrate_integrated_manager(workspace)
    
    print("\n=== Demo Completed ===")
    print("\nKey Features Demonstrated:")
    print("- Event Sourcing: Complete history tracking and state reconstruction")
    print("- Columnar Storage: Efficient storage and retrieval with optimized compression")
    print("- Dynamic Version Management: Automatic file format updates and schema evolution")
    print("- Adaptation Rules Engine: Rules defining when/how MAIF can transition states")
    print("- Integrated Manager: Centralized management of multiple MAIF instances")
    print("\nAll features work together in a cohesive, integrated manner.")

if __name__ == "__main__":
    main()