"""
Enhanced MAIF Lifecycle Management Demo
=======================================

This example demonstrates the enhanced lifecycle management capabilities
using the Adaptation Rules Engine for more sophisticated governance.

Uses the secure MAIF format with:
- Ed25519 signatures (64 bytes per block)
- Self-contained files (no external manifest)
- Embedded provenance chain
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# matplotlib is optional - for visualization only
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import MAIF components
from maif.agents.lifecycle_management_enhanced import (
    EnhancedSelfGoverningMAIF,
    EnhancedMAIFLifecycleManager,
    MAIFLifecycleState,
)
from maif.performance.adaptation_rules import (
    AdaptationRule,
    RulePriority,
    RuleStatus,
    ActionType,
    TriggerType,
    ComparisonOperator,
    LogicalOperator,
    MetricCondition,
    ScheduleCondition,
    EventCondition,
    CompositeCondition,
    Action,
    ActionParameter,
)
from maif import MAIFEncoder


def setup_workspace():
    """Set up workspace for the demo."""
    workspace = Path("./demo_workspace/enhanced_lifecycle")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def create_sample_maif(workspace: Path, name: str, size_mb: float = 10.0) -> str:
    """
    Create a sample MAIF file of specified size (secure format with Ed25519).

    Args:
        workspace: Workspace directory
        name: MAIF name
        size_mb: Approximate size in MB

    Returns:
        Path to created MAIF file
    """
    maif_path = workspace / f"{name}.maif"
    encoder = MAIFEncoder(str(maif_path), agent_id=f"lifecycle-{name}")

    # Calculate number of blocks needed to reach target size
    # Each block is approximately 10KB
    blocks_needed = int((size_mb * 1024) / 10)

    logger.info(f"Creating {name} with approximately {blocks_needed} blocks")

    # Add text blocks
    for i in range(blocks_needed):
        # Create text content of approximately 10KB
        text = f"Block {i}: " + "Content " * 500

        # Add metadata
        metadata = {
            "block_id": f"block_{i}",
            "timestamp": time.time(),
            "category": f"category_{i % 5}",
            "priority": i % 3,
        }

        # Add block
        encoder.add_text_block(text, metadata)

    # Finalize MAIF (self-contained with Ed25519 signatures)
    encoder.finalize()

    logger.info(
        f"Created {name} at {maif_path} ({maif_path.stat().st_size / (1024 * 1024):.2f} MB)"
    )

    return str(maif_path)


def demonstrate_enhanced_governance():
    """Demonstrate enhanced self-governance with adaptation rules."""
    print("\n=== Enhanced Self-Governance Demo ===")

    workspace = setup_workspace()

    # Create a MAIF file
    maif_path = create_sample_maif(workspace, "governed", size_mb=50.0)

    # Create custom rules
    rules_path = workspace / "enhanced_rules.json"

    # Define custom rules in the new format
    custom_rules = [
        {
            "rule_id": "custom_size_rule",
            "name": "Custom Size Rule",
            "description": "Split files larger than 20MB",
            "priority": 80,
            "trigger": "metric",
            "condition": {
                "type": "metric",
                "metric_name": "size_bytes",
                "operator": "gt",
                "threshold": 20 * 1024 * 1024,  # 20MB
            },
            "actions": [
                {
                    "action_type": "split",
                    "parameters": [
                        {"name": "strategy", "value": "size"},
                        {"name": "max_size_mb", "value": 10.0},
                    ],
                }
            ],
            "status": "active",
        },
        {
            "rule_id": "custom_composite_rule",
            "name": "Custom Composite Rule",
            "description": "Optimize files with high access and low fragmentation",
            "priority": 90,
            "trigger": "composite",
            "condition": {
                "type": "composite",
                "operator": "and",
                "conditions": [
                    {
                        "type": "metric",
                        "metric_name": "access_frequency",
                        "operator": "gt",
                        "threshold": 5.0,
                    },
                    {
                        "type": "metric",
                        "metric_name": "fragmentation",
                        "operator": "lt",
                        "threshold": 0.3,
                    },
                ],
            },
            "actions": [{"action_type": "optimize", "parameters": []}],
            "status": "active",
        },
    ]

    # Save custom rules
    with open(rules_path, "w") as f:
        json.dump(custom_rules, f, indent=2)

    print(f"Created custom rules at {rules_path}")

    # Create enhanced self-governing MAIF
    governed = EnhancedSelfGoverningMAIF(maif_path, str(rules_path))

    print("Enhanced self-governing MAIF created with custom rules")

    # Get initial governance report
    print("\nInitial governance report:")
    report = governed.get_governance_report()
    print(json.dumps(report, indent=2))

    # Simulate metrics that would trigger rules
    print("\nSimulating metrics to trigger rules...")

    # Simulate high access frequency
    for _ in range(50):
        governed.optimizer.record_access("block_1", 0.001)

    # Force metrics update
    governed._update_metrics()

    # Manually set metrics for demonstration
    governed.metrics.size_bytes = 25 * 1024 * 1024  # 25MB
    governed.metrics.access_frequency = 8.0
    governed.metrics.fragmentation = 0.2

    # Evaluate rules
    print("\nEvaluating adaptation rules...")
    actions = governed._evaluate_rules()
    print(f"Actions triggered: {actions}")

    # Get final report
    print("\nFinal governance report:")
    report = governed.get_governance_report()
    print(json.dumps(report, indent=2))

    # Stop governance
    governed.stop_governance()

    return governed


def demonstrate_enhanced_lifecycle_manager():
    """Demonstrate enhanced lifecycle manager."""
    print("\n=== Enhanced Lifecycle Manager Demo ===")

    workspace = setup_workspace()

    # Create manager
    manager = EnhancedMAIFLifecycleManager(str(workspace / "managed"))

    # Create multiple MAIFs
    maif_paths = []
    for i, size in enumerate([5.0, 15.0, 30.0]):
        path = create_sample_maif(workspace, f"managed_{i}", size_mb=size)
        maif_paths.append(path)

    # Add MAIFs to management
    for path in maif_paths:
        manager.add_maif(path)

    print(f"\nManaging {len(maif_paths)} MAIF files")

    # Get status
    print("\nLifecycle status:")
    status = manager.get_status()
    for path, report in status.items():
        print(f"\n{Path(path).name}:")
        print(f"State: {report['state']}")
        print(f"Size: {report['metrics']['size_mb']:.2f} MB")
        print(f"Blocks: {report['metrics']['block_count']}")

    # Create a custom rule
    custom_rule = AdaptationRule(
        rule_id="manager_custom_rule",
        name="Manager Custom Rule",
        description="Archive files older than 1 day",
        priority=RulePriority.HIGH,
        trigger=TriggerType.METRIC,
        condition=MetricCondition(
            metric_name="age_days",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=1.0,
        ),
        actions=[Action(action_type=ActionType.ARCHIVE, parameters=[])],
        status=RuleStatus.ACTIVE,
    )

    # Add custom rule to first MAIF
    print("\nAdding custom rule to first MAIF...")
    first_maif = manager.governed_maifs[maif_paths[0]]
    first_maif.add_rule(custom_rule)

    # Simulate metrics for demonstration
    for path, governed in manager.governed_maifs.items():
        # Set different metrics for each MAIF
        if "managed_0" in path:
            governed.metrics.access_frequency = 12.0
            governed.metrics.fragmentation = 0.1
        elif "managed_1" in path:
            governed.metrics.access_frequency = 0.05
            governed.metrics.age_days = 45.0
        elif "managed_2" in path:
            governed.metrics.size_bytes = 35 * 1024 * 1024
            governed.metrics.fragmentation = 0.6

    # Evaluate rules for all MAIFs
    print("\nEvaluating rules for all MAIFs...")
    for path, governed in manager.governed_maifs.items():
        actions = governed._evaluate_rules()
        print(f"{Path(path).name}: {actions}")

    # Get updated status
    print("\nUpdated lifecycle status:")
    status = manager.get_status()
    for path, report in status.items():
        print(f"\n{Path(path).name}:")
        print(f"State: {report['state']}")
        print(f"Size: {report['metrics']['size_mb']:.2f} MB")
        print(f"History: {len(report['history'])} events")

    # Clean up
    for path in maif_paths:
        manager.remove_maif(path)

    return manager


def demonstrate_programmatic_rule_creation():
    """Demonstrate programmatic creation of adaptation rules."""
    print("\n=== Programmatic Rule Creation Demo ===")

    workspace = setup_workspace()

    # Create a MAIF file
    maif_path = create_sample_maif(workspace, "programmatic", size_mb=20.0)

    # Create enhanced self-governing MAIF
    governed = EnhancedSelfGoverningMAIF(maif_path)

    print("Creating custom rules programmatically...")

    # Rule 1: Schedule-based rule
    schedule_condition = ScheduleCondition(
        cron_expression="0 2 * * *",  # 2:00 AM daily
        timezone="UTC",
    )

    schedule_action = Action(
        action_type=ActionType.OPTIMIZE,
        parameters=[ActionParameter(name="mode", value="deep")],
    )

    schedule_rule = AdaptationRule(
        rule_id="nightly_optimization",
        name="Nightly Optimization",
        description="Optimize MAIF every night at 2:00 AM",
        priority=RulePriority.MEDIUM,
        trigger=TriggerType.SCHEDULE,
        condition=schedule_condition,
        actions=[schedule_action],
        status=RuleStatus.ACTIVE,
    )

    # Rule 2: Complex composite rule
    composite_condition = CompositeCondition(
        operator=LogicalOperator.OR,
        conditions=[
            MetricCondition(
                metric_name="size_bytes",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=50 * 1024 * 1024,  # 50MB
            ),
            CompositeCondition(
                operator=LogicalOperator.AND,
                conditions=[
                    MetricCondition(
                        metric_name="fragmentation",
                        operator=ComparisonOperator.GREATER_THAN,
                        threshold=0.4,
                    ),
                    MetricCondition(
                        metric_name="access_frequency",
                        operator=ComparisonOperator.GREATER_THAN,
                        threshold=3.0,
                    ),
                ],
            ),
        ],
    )

    composite_action = Action(
        action_type=ActionType.REORGANIZE,
        parameters=[ActionParameter(name="strategy", value="optimal")],
    )

    composite_rule = AdaptationRule(
        rule_id="complex_reorganization",
        name="Complex Reorganization",
        description="Reorganize based on complex conditions",
        priority=RulePriority.HIGH,
        trigger=TriggerType.COMPOSITE,
        condition=composite_condition,
        actions=[composite_action],
        status=RuleStatus.ACTIVE,
    )

    # Rule 3: Event-based rule
    event_condition = EventCondition(
        event_type="file_access",
        event_source="external_system",
        event_properties={"access_type": "write", "user_role": "admin"},
    )

    event_action = Action(
        action_type=ActionType.CUSTOM,
        parameters=[
            ActionParameter(name="operation", value="security_audit"),
            ActionParameter(name="level", value="detailed"),
        ],
    )

    event_rule = AdaptationRule(
        rule_id="security_audit",
        name="Security Audit",
        description="Trigger security audit on admin write",
        priority=RulePriority.CRITICAL,
        trigger=TriggerType.EVENT,
        condition=event_condition,
        actions=[event_action],
        status=RuleStatus.ACTIVE,
    )

    # Add rules to governance
    governed.add_rule(schedule_rule)
    governed.add_rule(composite_rule)
    governed.add_rule(event_rule)

    print("Added 3 custom rules programmatically")

    # Simulate event for demonstration
    print("\nSimulating event trigger...")
    event = {
        "type": "file_access",
        "source": "external_system",
        "access_type": "write",
        "user_role": "admin",
        "timestamp": time.time(),
    }

    # Create context with event
    context = {
        "metrics": governed.metrics.__dict__,
        "current_time": time.time(),
        "event": event,
    }

    # Evaluate rules with event context
    triggered_rules = governed.rules_engine.evaluate_rules(context)

    print(f"Rules triggered by event: {[rule.rule_id for rule in triggered_rules]}")

    # Get governance report
    print("\nGovernance report with custom rules:")
    report = governed.get_governance_report()
    print(json.dumps(report, indent=2))

    # Stop governance
    governed.stop_governance()

    return governed


def main():
    """Run the enhanced lifecycle management demo."""
    print("Enhanced MAIF Lifecycle Management Demo")
    print("=======================================")

    # Set up workspace
    workspace = setup_workspace()

    # Run demonstrations
    demonstrate_enhanced_governance()
    demonstrate_enhanced_lifecycle_manager()
    demonstrate_programmatic_rule_creation()

    print("\n=== Demo Completed ===")
    print("\nKey Features Demonstrated:")
    print("- Enhanced Self-Governance with Adaptation Rules Engine")
    print("- Complex rule conditions (metric, schedule, event, composite)")
    print("- Programmatic rule creation and management")
    print("- Integration with lifecycle management")
    print("- Centralized management of multiple MAIF instances")
    print("\nAll features work together in a cohesive, integrated manner.")


if __name__ == "__main__":
    main()
