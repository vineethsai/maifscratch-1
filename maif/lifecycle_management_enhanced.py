"""
Enhanced MAIF Lifecycle Management
==================================

Implements merging/splitting operations and self-governing data fabric
with integration of the Adaptation Rules Engine.
"""

import os
import json
import time
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

# Import MAIF components
from .core import MAIFEncoder, MAIFDecoder
from .validation import MAIFValidator
from .semantic_optimized import OptimizedSemanticEmbedder, CryptographicSemanticBinding
from .self_optimizing import SelfOptimizingMAIF
from .distributed import DistributedCoordinator

# Import adaptation rules
from .adaptation_rules import (
    AdaptationRulesEngine,
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

logger = logging.getLogger(__name__)


# Lifecycle States
class MAIFLifecycleState(Enum):
    """MAIF lifecycle states."""

    CREATED = "created"
    ACTIVE = "active"
    MERGING = "merging"
    SPLITTING = "splitting"
    OPTIMIZING = "optimizing"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


@dataclass
class MAIFMetrics:
    """Metrics for self-governance decisions."""

    size_bytes: int = 0
    block_count: int = 0
    access_frequency: float = 0.0
    last_accessed: float = 0.0
    compression_ratio: float = 1.0
    fragmentation: float = 0.0
    age_days: float = 0.0
    semantic_coherence: float = 1.0


# Reuse MAIFMerger and MAIFSplitter from original implementation
from .lifecycle_management import MAIFMerger, MAIFSplitter


class EnhancedSelfGoverningMAIF:
    """
    Enhanced self-governing data fabric for MAIF files.

    Implements autonomous lifecycle management using the Adaptation Rules Engine
    for more sophisticated rule evaluation and action execution.
    """

    def __init__(self, maif_path: str, rules_path: Optional[str] = None):
        """
        Initialize enhanced self-governing MAIF.

        Args:
            maif_path: Path to MAIF file
            rules_path: Optional path to rules JSON file
        """
        self.maif_path = Path(maif_path)
        self.rules_path = Path(rules_path) if rules_path else None

        # Components
        self.optimizer = SelfOptimizingMAIF(str(maif_path))
        self.merger = MAIFMerger()
        self.splitter = MAIFSplitter()

        # Governance state
        self.state = MAIFLifecycleState.CREATED
        self.metrics = MAIFMetrics()
        self.history: List[Dict[str, Any]] = []

        # Adaptation Rules Engine
        self.rules_engine = AdaptationRulesEngine()

        # Threading
        self._lock = threading.RLock()
        self._governance_thread = None
        self._running = False

        # Register action handlers
        self._register_action_handlers()

        # Load rules
        self._load_rules()

        # Start governance
        self.start_governance()

    def _register_action_handlers(self):
        """Register action handlers for the rules engine."""

        def handle_split(action: Action, context: Dict[str, Any]) -> bool:
            """Handle split action."""
            try:
                self._action_split()
                return True
            except Exception as e:
                logger.error(f"Error handling split action: {e}")
                return False

        def handle_reorganize(action: Action, context: Dict[str, Any]) -> bool:
            """Handle reorganize action."""
            try:
                self._action_reorganize()
                return True
            except Exception as e:
                logger.error(f"Error handling reorganize action: {e}")
                return False

        def handle_archive(action: Action, context: Dict[str, Any]) -> bool:
            """Handle archive action."""
            try:
                self._action_archive()
                return True
            except Exception as e:
                logger.error(f"Error handling archive action: {e}")
                return False

        def handle_optimize(action: Action, context: Dict[str, Any]) -> bool:
            """Handle optimize action."""
            try:
                self._action_optimize_hot()
                return True
            except Exception as e:
                logger.error(f"Error handling optimize action: {e}")
                return False

        def handle_semantic_reorganize(action: Action, context: Dict[str, Any]) -> bool:
            """Handle semantic reorganize action."""
            try:
                self._action_semantic_reorganize()
                return True
            except Exception as e:
                logger.error(f"Error handling semantic reorganize action: {e}")
                return False

        # Register handlers
        self.rules_engine.register_action_handler(ActionType.SPLIT, handle_split)
        self.rules_engine.register_action_handler(
            ActionType.REORGANIZE, handle_reorganize
        )
        self.rules_engine.register_action_handler(ActionType.ARCHIVE, handle_archive)
        self.rules_engine.register_action_handler(ActionType.OPTIMIZE, handle_optimize)
        self.rules_engine.register_action_handler(
            ActionType.TRANSFORM, handle_semantic_reorganize
        )

    def _load_rules(self):
        """Load adaptation rules."""
        # Default rules

        # Rule 1: Split large files
        size_condition = MetricCondition(
            metric_name="size_bytes",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=1073741824,  # 1GB
        )

        split_action = Action(action_type=ActionType.SPLIT, parameters=[])

        size_rule = AdaptationRule(
            rule_id="size_limit",
            name="Size Limit",
            description="Split files larger than 1GB",
            priority=RulePriority.HIGH,
            trigger=TriggerType.METRIC,
            condition=size_condition,
            actions=[split_action],
            status=RuleStatus.ACTIVE,
        )

        # Rule 2: Reorganize fragmented files
        fragmentation_condition = MetricCondition(
            metric_name="fragmentation",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=0.5,
        )

        reorganize_action = Action(action_type=ActionType.REORGANIZE, parameters=[])

        fragmentation_rule = AdaptationRule(
            rule_id="fragmentation",
            name="Fragmentation",
            description="Reorganize fragmented files",
            priority=RulePriority.MEDIUM,
            trigger=TriggerType.METRIC,
            condition=fragmentation_condition,
            actions=[reorganize_action],
            status=RuleStatus.ACTIVE,
        )

        # Rule 3: Archive low-access files
        low_access_condition = CompositeCondition(
            operator=LogicalOperator.AND,
            conditions=[
                MetricCondition(
                    metric_name="access_frequency",
                    operator=ComparisonOperator.LESS_THAN,
                    threshold=0.1,
                ),
                MetricCondition(
                    metric_name="age_days",
                    operator=ComparisonOperator.GREATER_THAN,
                    threshold=30,
                ),
            ],
        )

        archive_action = Action(action_type=ActionType.ARCHIVE, parameters=[])

        low_access_rule = AdaptationRule(
            rule_id="low_access",
            name="Low Access",
            description="Archive files with low access frequency",
            priority=RulePriority.LOW,
            trigger=TriggerType.METRIC,
            condition=low_access_condition,
            actions=[archive_action],
            status=RuleStatus.ACTIVE,
        )

        # Rule 4: Optimize high-access files
        high_access_condition = MetricCondition(
            metric_name="access_frequency",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=10.0,
        )

        optimize_action = Action(action_type=ActionType.OPTIMIZE, parameters=[])

        high_access_rule = AdaptationRule(
            rule_id="high_access",
            name="High Access",
            description="Optimize files with high access frequency",
            priority=RulePriority.HIGH,
            trigger=TriggerType.METRIC,
            condition=high_access_condition,
            actions=[optimize_action],
            status=RuleStatus.ACTIVE,
        )

        # Rule 5: Semantic reorganization
        semantic_condition = MetricCondition(
            metric_name="semantic_coherence",
            operator=ComparisonOperator.LESS_THAN,
            threshold=0.5,
        )

        semantic_action = Action(
            action_type=ActionType.TRANSFORM,
            parameters=[ActionParameter(name="operation", value="semantic_reorganize")],
        )

        semantic_rule = AdaptationRule(
            rule_id="semantic_drift",
            name="Semantic Drift",
            description="Reorganize files with low semantic coherence",
            priority=RulePriority.MEDIUM,
            trigger=TriggerType.METRIC,
            condition=semantic_condition,
            actions=[semantic_action],
            status=RuleStatus.ACTIVE,
        )

        # Register default rules
        self.rules_engine.register_rule(size_rule)
        self.rules_engine.register_rule(fragmentation_rule)
        self.rules_engine.register_rule(low_access_rule)
        self.rules_engine.register_rule(high_access_rule)
        self.rules_engine.register_rule(semantic_rule)

        # Load custom rules if provided
        if self.rules_path and self.rules_path.exists():
            try:
                with open(self.rules_path, "r") as f:
                    custom_rules = json.load(f)

                    for rule_data in custom_rules:
                        # Convert from old format to new format
                        if "condition" in rule_data and "action" in rule_data:
                            # Parse condition
                            condition_str = rule_data["condition"]

                            # Simple conversion of common conditions
                            if "size_bytes" in condition_str:
                                metric = "size_bytes"
                                if ">" in condition_str:
                                    operator = ComparisonOperator.GREATER_THAN
                                    threshold = float(
                                        condition_str.split(">")[1].strip()
                                    )
                                elif "<" in condition_str:
                                    operator = ComparisonOperator.LESS_THAN
                                    threshold = float(
                                        condition_str.split("<")[1].strip()
                                    )
                                else:
                                    continue

                                condition = MetricCondition(
                                    metric_name=metric,
                                    operator=operator,
                                    threshold=threshold,
                                )
                            elif "fragmentation" in condition_str:
                                metric = "fragmentation"
                                if ">" in condition_str:
                                    operator = ComparisonOperator.GREATER_THAN
                                    threshold = float(
                                        condition_str.split(">")[1].strip()
                                    )
                                elif "<" in condition_str:
                                    operator = ComparisonOperator.LESS_THAN
                                    threshold = float(
                                        condition_str.split("<")[1].strip()
                                    )
                                else:
                                    continue

                                condition = MetricCondition(
                                    metric_name=metric,
                                    operator=operator,
                                    threshold=threshold,
                                )
                            else:
                                # Skip complex conditions
                                continue

                            # Parse action
                            action_str = rule_data["action"]

                            if action_str == "split":
                                action_type = ActionType.SPLIT
                            elif action_str == "reorganize":
                                action_type = ActionType.REORGANIZE
                            elif action_str == "archive":
                                action_type = ActionType.ARCHIVE
                            elif action_str == "optimize_hot":
                                action_type = ActionType.OPTIMIZE
                            elif action_str == "semantic_reorganize":
                                action_type = ActionType.TRANSFORM
                            else:
                                action_type = ActionType.CUSTOM

                            action = Action(action_type=action_type, parameters=[])

                            # Create rule
                            rule = AdaptationRule(
                                rule_id=rule_data["rule_id"],
                                name=rule_data.get("name", rule_data["rule_id"]),
                                description=rule_data.get("description", ""),
                                priority=RulePriority(rule_data.get("priority", 50)),
                                trigger=TriggerType.METRIC,
                                condition=condition,
                                actions=[action],
                                status=RuleStatus.ACTIVE,
                            )

                            self.rules_engine.register_rule(rule)
            except Exception as e:
                logger.error(f"Error loading custom rules: {e}")

    def start_governance(self):
        """Start autonomous governance."""
        self._running = True
        self._governance_thread = threading.Thread(
            target=self._governance_loop, daemon=True
        )
        self._governance_thread.start()
        logger.info(f"Started enhanced self-governance for {self.maif_path}")

    def stop_governance(self):
        """Stop autonomous governance."""
        self._running = False
        if self._governance_thread:
            self._governance_thread.join()
        logger.info(f"Stopped enhanced self-governance for {self.maif_path}")

    def _governance_loop(self):
        """Main governance loop."""
        while self._running:
            try:
                # Update metrics
                self._update_metrics()

                # Evaluate rules
                self._evaluate_rules()

                # Wait before next evaluation
                time.sleep(60.0)  # Check every minute

            except Exception as e:
                logger.error(f"Governance error: {e}")
                time.sleep(300.0)  # Wait 5 minutes on error

    def _update_metrics(self):
        """Update MAIF metrics."""
        with self._lock:
            if not self.maif_path.exists():
                return

            # File metrics
            stat = self.maif_path.stat()
            self.metrics.size_bytes = stat.st_size
            self.metrics.last_accessed = stat.st_atime
            self.metrics.age_days = (time.time() - stat.st_ctime) / 86400

            # Access frequency (from optimizer)
            stats = self.optimizer.get_optimization_stats()
            total_accesses = (
                stats["metrics"]["total_reads"] + stats["metrics"]["total_writes"]
            )
            time_span = time.time() - stat.st_ctime
            self.metrics.access_frequency = (
                total_accesses / (time_span / 3600) if time_span > 0 else 0
            )

            # Fragmentation
            self.metrics.fragmentation = stats["metrics"]["fragmentation_ratio"]

            # Block count
            try:
                decoder = MAIFDecoder(str(self.maif_path))
                self.metrics.block_count = len(decoder.blocks)
            except:
                pass

            # Semantic coherence (simplified)
            self.metrics.semantic_coherence = 1.0 - self.metrics.fragmentation

    def _evaluate_rules(self) -> List[str]:
        """Evaluate adaptation rules and return actions executed."""
        with self._lock:
            # Create context for rule evaluation
            context = {
                "metrics": {
                    "size_bytes": self.metrics.size_bytes,
                    "block_count": self.metrics.block_count,
                    "access_frequency": self.metrics.access_frequency,
                    "last_accessed": self.metrics.last_accessed,
                    "compression_ratio": self.metrics.compression_ratio,
                    "fragmentation": self.metrics.fragmentation,
                    "age_days": self.metrics.age_days,
                    "semantic_coherence": self.metrics.semantic_coherence,
                },
                "current_time": time.time(),
                "maif_path": str(self.maif_path),
                "state": self.state.value,
            }

            # Evaluate rules
            triggered_rules = self.rules_engine.evaluate_rules(context)

            # Execute rules
            executed_actions = []
            for rule in triggered_rules:
                result = self.rules_engine.execute_rule(rule, context)

                if result.success:
                    executed_actions.append(rule.rule_id)

                    # Record in history
                    self.history.append(
                        {
                            "timestamp": time.time(),
                            "rule_id": rule.rule_id,
                            "action": rule.actions[0].action_type.value
                            if rule.actions
                            else "unknown",
                            "success": result.success,
                            "metrics": {
                                "size_bytes": self.metrics.size_bytes,
                                "fragmentation": self.metrics.fragmentation,
                                "access_frequency": self.metrics.access_frequency,
                            },
                        }
                    )

            return executed_actions

    # Action methods (reused from original implementation)
    def _action_split(self):
        """Split MAIF file."""
        with self._lock:
            self.state = MAIFLifecycleState.SPLITTING

            output_dir = self.maif_path.parent / f"{self.maif_path.stem}_split"

            # Split by size (100MB parts)
            parts = self.splitter.split(
                str(self.maif_path),
                str(output_dir),
                split_strategy="size",
                max_size_mb=100.0,
            )

            logger.info(f"Split {self.maif_path} into {len(parts)} parts")

            # Archive original
            archive_path = self.maif_path.with_suffix(".maif.archive")
            self.maif_path.rename(archive_path)

            self.state = MAIFLifecycleState.ARCHIVED

    def _action_reorganize(self):
        """Reorganize MAIF file."""
        with self._lock:
            self.state = MAIFLifecycleState.OPTIMIZING

            # Trigger reorganization
            self.optimizer._perform_reorganization()

            self.state = MAIFLifecycleState.ACTIVE

    def _action_archive(self):
        """Archive MAIF file."""
        with self._lock:
            self.state = MAIFLifecycleState.ARCHIVED

            # Compress and move to archive
            archive_dir = self.maif_path.parent / "archive"
            archive_dir.mkdir(exist_ok=True)

            archive_path = archive_dir / f"{self.maif_path.name}.gz"

            # Check if the MAIF file exists
            if not self.maif_path.exists():
                logger.info(
                    f"MAIF file {self.maif_path} does not exist, creating an empty file"
                )
                # Create an empty MAIF file using the encoder (v3 format)
                from .core import MAIFEncoder

                temp_encoder = MAIFEncoder(
                    str(self.maif_path), agent_id="lifecycle_manager"
                )
                temp_encoder.finalize()
                logger.info(f"Created empty MAIF file {self.maif_path}")

            import gzip

            with open(self.maif_path, "rb") as f_in:
                with gzip.open(archive_path, "wb") as f_out:
                    f_out.writelines(f_in)

            # Remove original
            self.maif_path.unlink()

            logger.info(f"Archived {self.maif_path} to {archive_path}")

    def _action_optimize_hot(self):
        """Optimize for high-frequency access."""
        with self._lock:
            self.state = MAIFLifecycleState.OPTIMIZING

            # Optimize for read-heavy workload
            self.optimizer.optimize_for_workload("read_heavy")

            self.state = MAIFLifecycleState.ACTIVE

    def _action_semantic_reorganize(self):
        """Reorganize based on semantic similarity."""
        with self._lock:
            self.state = MAIFLifecycleState.OPTIMIZING

            # Split by semantic clusters then merge
            temp_dir = self.maif_path.parent / "temp_semantic"

            parts = self.splitter.split(
                str(self.maif_path),
                str(temp_dir),
                split_strategy="semantic",
                num_clusters=5,
            )

            # Merge back in semantic order
            output_path = self.maif_path.with_suffix(".reorganized.maif")

            self.merger.merge(parts, str(output_path), merge_strategy="semantic")

            # Replace original
            self.maif_path.unlink()
            output_path.rename(self.maif_path)

            # Clean up
            import shutil

            shutil.rmtree(temp_dir)

            self.state = MAIFLifecycleState.ACTIVE

    def add_rule(self, rule: AdaptationRule):
        """Add an adaptation rule."""
        with self._lock:
            self.rules_engine.register_rule(rule)

    def get_governance_report(self) -> Dict[str, Any]:
        """Get governance status report."""
        with self._lock:
            return {
                "maif_path": str(self.maif_path),
                "state": self.state.value,
                "metrics": {
                    "size_mb": self.metrics.size_bytes / (1024 * 1024),
                    "block_count": self.metrics.block_count,
                    "access_frequency": self.metrics.access_frequency,
                    "fragmentation": self.metrics.fragmentation,
                    "age_days": self.metrics.age_days,
                    "semantic_coherence": self.metrics.semantic_coherence,
                },
                "active_rules": len(
                    self.rules_engine.evaluate_rules(
                        {"metrics": self.metrics.__dict__, "current_time": time.time()}
                    )
                ),
                "history": self.history[-10:],  # Last 10 actions
            }


class EnhancedMAIFLifecycleManager:
    """
    Enhanced lifecycle manager for multiple MAIF files.

    Uses the Adaptation Rules Engine for more sophisticated governance.
    """

    def __init__(self, workspace_dir: str):
        """
        Initialize the manager.

        Args:
            workspace_dir: Directory for managed MAIFs
        """
        self.workspace_dir = Path(workspace_dir)
        self.governed_maifs: Dict[str, EnhancedSelfGoverningMAIF] = {}
        self._lock = threading.Lock()

    def add_maif(self, maif_path: str, rules_path: Optional[str] = None):
        """Add a MAIF file to lifecycle management."""
        with self._lock:
            if maif_path not in self.governed_maifs:
                self.governed_maifs[maif_path] = EnhancedSelfGoverningMAIF(
                    maif_path, rules_path
                )
                logger.info(f"Added {maif_path} to enhanced lifecycle management")

    def remove_maif(self, maif_path: str):
        """Remove a MAIF file from lifecycle management."""
        with self._lock:
            if maif_path in self.governed_maifs:
                self.governed_maifs[maif_path].stop_governance()
                del self.governed_maifs[maif_path]
                logger.info(f"Removed {maif_path} from enhanced lifecycle management")

    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed MAIFs."""
        with self._lock:
            status = {}
            for path, governed in self.governed_maifs.items():
                status[path] = governed.get_governance_report()
            return status

    def merge_maifs(
        self, maif_paths: List[str], output_path: str, strategy: str = "semantic"
    ) -> Dict[str, Any]:
        """Merge multiple MAIFs."""
        merger = MAIFMerger()
        return merger.merge(maif_paths, output_path, strategy)

    def split_maif(
        self, maif_path: str, output_dir: str, strategy: str = "semantic", **kwargs
    ) -> List[str]:
        """Split a MAIF."""
        splitter = MAIFSplitter()
        return splitter.split(maif_path, output_dir, strategy, **kwargs)
