"""
Adaptation Rules Engine for MAIF
================================

Implements rules defining when/how MAIF can transition states.
This module provides a flexible rule-based system for managing
MAIF lifecycle and adaptation.
"""

import time
import json
import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import re
import copy
import uuid

from .lifecycle_management import MAIFLifecycleState, MAIFMetrics

logger = logging.getLogger(__name__)


class RulePriority(Enum):
    """Priority levels for adaptation rules."""

    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 10


class RuleStatus(Enum):
    """Status of a rule."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(Enum):
    """Types of actions that can be triggered by rules."""

    SPLIT = "split"
    MERGE = "merge"
    REORGANIZE = "reorganize"
    OPTIMIZE = "optimize"
    ARCHIVE = "archive"
    RESTORE = "restore"
    REPLICATE = "replicate"
    MIGRATE = "migrate"
    TRANSFORM = "transform"
    NOTIFY = "notify"
    CUSTOM = "custom"


class TriggerType(Enum):
    """Types of rule triggers."""

    METRIC = "metric"  # Triggered by metric threshold
    SCHEDULE = "schedule"  # Triggered by schedule
    EVENT = "event"  # Triggered by event
    DEPENDENCY = "dependency"  # Triggered by other rule
    MANUAL = "manual"  # Triggered manually
    COMPOSITE = "composite"  # Triggered by combination of conditions


class ComparisonOperator(Enum):
    """Comparison operators for metric conditions."""

    EQUAL = "eq"
    NOT_EQUAL = "ne"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex match


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""

    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class MetricCondition:
    """Condition based on a metric value."""

    metric_name: str
    operator: ComparisonOperator
    threshold: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate condition against metrics."""
        if self.metric_name not in metrics:
            return False

        value = metrics[self.metric_name]

        if self.operator == ComparisonOperator.EQUAL:
            return value == self.threshold
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return value != self.threshold
        elif self.operator == ComparisonOperator.GREATER_THAN:
            return value > self.threshold
        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            return value >= self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN:
            return value < self.threshold
        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            return value <= self.threshold
        elif self.operator == ComparisonOperator.CONTAINS:
            return self.threshold in value
        elif self.operator == ComparisonOperator.NOT_CONTAINS:
            return self.threshold not in value
        elif self.operator == ComparisonOperator.STARTS_WITH:
            return value.startswith(self.threshold)
        elif self.operator == ComparisonOperator.ENDS_WITH:
            return value.endswith(self.threshold)
        elif self.operator == ComparisonOperator.MATCHES:
            return bool(re.match(self.threshold, value))

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricCondition":
        """Create from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            operator=ComparisonOperator(data["operator"]),
            threshold=data["threshold"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ScheduleCondition:
    """Condition based on a schedule."""

    cron_expression: str
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, current_time: float) -> bool:
        """Evaluate condition against current time."""
        # Simple cron implementation
        # Format: minute hour day_of_month month day_of_week
        # e.g. "0 0 * * *" = daily at midnight

        import datetime

        try:
            import pytz

            dt = datetime.datetime.fromtimestamp(
                current_time, pytz.timezone(self.timezone)
            )
        except ImportError:
            # Fall back to local time if pytz not available
            dt = datetime.datetime.fromtimestamp(current_time)

        # Parse cron expression
        parts = self.cron_expression.split()
        if len(parts) != 5:
            logger.error(f"Invalid cron expression: {self.cron_expression}")
            return False

        minute, hour, day_of_month, month, day_of_week = parts

        # Check each component
        if minute != "*" and int(minute) != dt.minute:
            return False

        if hour != "*" and int(hour) != dt.hour:
            return False

        if day_of_month != "*" and int(day_of_month) != dt.day:
            return False

        if month != "*" and int(month) != dt.month:
            return False

        if day_of_week != "*" and int(day_of_week) != dt.weekday():
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cron_expression": self.cron_expression,
            "timezone": self.timezone,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScheduleCondition":
        """Create from dictionary."""
        return cls(
            cron_expression=data["cron_expression"],
            timezone=data.get("timezone", "UTC"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EventCondition:
    """Condition based on an event."""

    event_type: str
    event_source: Optional[str] = None
    event_properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, event: Dict[str, Any]) -> bool:
        """Evaluate condition against event."""
        if event.get("type") != self.event_type:
            return False

        if self.event_source and event.get("source") != self.event_source:
            return False

        # Check properties
        for key, value in self.event_properties.items():
            if key not in event or event[key] != value:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "event_source": self.event_source,
            "event_properties": self.event_properties,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventCondition":
        """Create from dictionary."""
        return cls(
            event_type=data["event_type"],
            event_source=data.get("event_source"),
            event_properties=data.get("event_properties", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CompositeCondition:
    """Composite condition combining multiple conditions."""

    operator: LogicalOperator
    conditions: List[Any]  # Can be any condition type
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate composite condition."""
        results = []

        for condition in self.conditions:
            if isinstance(condition, MetricCondition):
                result = condition.evaluate(context.get("metrics", {}))
            elif isinstance(condition, ScheduleCondition):
                result = condition.evaluate(context.get("current_time", time.time()))
            elif isinstance(condition, EventCondition):
                result = condition.evaluate(context.get("event", {}))
            elif isinstance(condition, CompositeCondition):
                result = condition.evaluate(context)
            elif isinstance(condition, dict):
                # Handle condition from dictionary
                if condition.get("type") == "metric":
                    result = MetricCondition.from_dict(condition).evaluate(
                        context.get("metrics", {})
                    )
                elif condition.get("type") == "schedule":
                    result = ScheduleCondition.from_dict(condition).evaluate(
                        context.get("current_time", time.time())
                    )
                elif condition.get("type") == "event":
                    result = EventCondition.from_dict(condition).evaluate(
                        context.get("event", {})
                    )
                elif condition.get("type") == "composite":
                    result = CompositeCondition.from_dict(condition).evaluate(context)
                else:
                    result = False
            else:
                result = False

            results.append(result)

        if self.operator == LogicalOperator.AND:
            return all(results)
        elif self.operator == LogicalOperator.OR:
            return any(results)
        elif self.operator == LogicalOperator.NOT:
            return not any(results)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": "composite",
            "operator": self.operator.value,
            "conditions": [
                condition.to_dict() if hasattr(condition, "to_dict") else condition
                for condition in self.conditions
            ],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompositeCondition":
        """Create from dictionary."""
        conditions = []

        for condition_data in data.get("conditions", []):
            if isinstance(condition_data, dict):
                if condition_data.get("type") == "metric":
                    conditions.append(MetricCondition.from_dict(condition_data))
                elif condition_data.get("type") == "schedule":
                    conditions.append(ScheduleCondition.from_dict(condition_data))
                elif condition_data.get("type") == "event":
                    conditions.append(EventCondition.from_dict(condition_data))
                elif condition_data.get("type") == "composite":
                    conditions.append(CompositeCondition.from_dict(condition_data))
                else:
                    conditions.append(condition_data)
            else:
                conditions.append(condition_data)

        return cls(
            operator=LogicalOperator(data["operator"]),
            conditions=conditions,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ActionParameter:
    """Parameter for an action."""

    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {"name": self.name, "value": self.value, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionParameter":
        """Create from dictionary."""
        return cls(
            name=data["name"], value=data["value"], metadata=data.get("metadata", {})
        )


@dataclass
class Action:
    """Action to be executed when a rule is triggered."""

    action_type: ActionType
    parameters: List[ActionParameter] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "parameters": [param.to_dict() for param in self.parameters],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create from dictionary."""
        return cls(
            action_type=ActionType(data["action_type"]),
            parameters=[
                ActionParameter.from_dict(param) for param in data.get("parameters", [])
            ],
            metadata=data.get("metadata", {}),
        )

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get parameter value by name."""
        for param in self.parameters:
            if param.name == name:
                return param.value
        return None

    def add_parameter(self, name: str, value: Any, metadata: Dict[str, Any] = None):
        """Add parameter."""
        self.parameters.append(
            ActionParameter(name=name, value=value, metadata=metadata or {})
        )


@dataclass
class AdaptationRule:
    """Rule defining when and how MAIF should adapt."""

    rule_id: str
    name: str
    description: str
    priority: RulePriority
    trigger: TriggerType
    condition: Any  # Can be any condition type
    actions: List[Action]
    status: RuleStatus = RuleStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition."""
        if self.status != RuleStatus.ACTIVE:
            return False

        if isinstance(self.condition, MetricCondition):
            return self.condition.evaluate(context.get("metrics", {}))
        elif isinstance(self.condition, ScheduleCondition):
            return self.condition.evaluate(context.get("current_time", time.time()))
        elif isinstance(self.condition, EventCondition):
            return self.condition.evaluate(context.get("event", {}))
        elif isinstance(self.condition, CompositeCondition):
            return self.condition.evaluate(context)
        elif isinstance(self.condition, dict):
            # Handle condition from dictionary
            if self.condition.get("type") == "metric":
                return MetricCondition.from_dict(self.condition).evaluate(
                    context.get("metrics", {})
                )
            elif self.condition.get("type") == "schedule":
                return ScheduleCondition.from_dict(self.condition).evaluate(
                    context.get("current_time", time.time())
                )
            elif self.condition.get("type") == "event":
                return EventCondition.from_dict(self.condition).evaluate(
                    context.get("event", {})
                )
            elif self.condition.get("type") == "composite":
                return CompositeCondition.from_dict(self.condition).evaluate(context)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "trigger": self.trigger.value,
            "condition": self.condition.to_dict()
            if hasattr(self.condition, "to_dict")
            else self.condition,
            "actions": [action.to_dict() for action in self.actions],
            "status": self.status.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptationRule":
        """Create from dictionary."""
        # Parse condition
        condition_data = data["condition"]
        condition = condition_data

        if isinstance(condition_data, dict):
            if condition_data.get("type") == "metric":
                condition = MetricCondition.from_dict(condition_data)
            elif condition_data.get("type") == "schedule":
                condition = ScheduleCondition.from_dict(condition_data)
            elif condition_data.get("type") == "event":
                condition = EventCondition.from_dict(condition_data)
            elif condition_data.get("type") == "composite":
                condition = CompositeCondition.from_dict(condition_data)

        # Parse actions
        actions = [
            Action.from_dict(action_data) for action_data in data.get("actions", [])
        ]

        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data["description"],
            priority=RulePriority(data["priority"]),
            trigger=TriggerType(data["trigger"]),
            condition=condition,
            actions=actions,
            status=RuleStatus(data["status"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RuleExecutionResult:
    """Result of rule execution."""

    rule_id: str
    success: bool
    timestamp: float
    actions_executed: List[str]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "success": self.success,
            "timestamp": self.timestamp,
            "actions_executed": self.actions_executed,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuleExecutionResult":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            success=data["success"],
            timestamp=data["timestamp"],
            actions_executed=data["actions_executed"],
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


class AdaptationRulesEngine:
    """
    Engine for managing and executing adaptation rules.

    Provides capabilities for defining, evaluating, and executing
    rules that govern MAIF lifecycle and adaptation.
    """

    def __init__(self):
        self.rules: Dict[str, AdaptationRule] = {}
        self.execution_history: List[RuleExecutionResult] = []
        self.action_handlers: Dict[ActionType, Callable] = {}
        self._lock = threading.RLock()

    def register_rule(self, rule: AdaptationRule) -> str:
        """
        Register a rule with the engine.

        Args:
            rule: Rule to register

        Returns:
            Rule ID
        """
        with self._lock:
            if not rule.rule_id:
                rule.rule_id = str(uuid.uuid4())

            self.rules[rule.rule_id] = rule
            return rule.rule_id

    def unregister_rule(self, rule_id: str) -> bool:
        """
        Unregister a rule from the engine.

        Args:
            rule_id: Rule ID

        Returns:
            True if rule was unregistered, False otherwise
        """
        with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                return True
            return False

    def get_rule(self, rule_id: str) -> Optional[AdaptationRule]:
        """
        Get rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Rule or None if not found
        """
        with self._lock:
            return self.rules.get(rule_id)

    def update_rule(self, rule: AdaptationRule) -> bool:
        """
        Update an existing rule using copy-on-write semantics.

        Args:
            rule: Updated rule

        Returns:
            True if rule was updated, False otherwise
        """
        with self._lock:
            if rule.rule_id in self.rules:
                # Copy-on-write: Check if the rule has actually changed
                existing_rule = self.rules[rule.rule_id]

                # Compare rule attributes to see if anything has changed
                if (
                    existing_rule.name == rule.name
                    and existing_rule.description == rule.description
                    and existing_rule.priority == rule.priority
                    and existing_rule.trigger == rule.trigger
                    and existing_rule.status == rule.status
                    and self._compare_conditions(
                        existing_rule.condition, rule.condition
                    )
                    and self._compare_actions(existing_rule.actions, rule.actions)
                    and existing_rule.metadata == rule.metadata
                ):
                    # Rule hasn't changed, no need to update
                    return True

                # Rule has changed, update it
                self.rules[rule.rule_id] = rule
                return True
            return False

    def _compare_conditions(self, condition1: Any, condition2: Any) -> bool:
        """Compare two conditions for equality."""
        # If they're the same object, they're equal
        if condition1 is condition2:
            return True

        # If they're different types, they're not equal
        if type(condition1) != type(condition2):
            return False

        # If they have to_dict methods, compare their dictionaries
        if hasattr(condition1, "to_dict") and hasattr(condition2, "to_dict"):
            return condition1.to_dict() == condition2.to_dict()

        # Otherwise, compare them directly
        return condition1 == condition2

    def _compare_actions(self, actions1: List[Action], actions2: List[Action]) -> bool:
        """Compare two lists of actions for equality."""
        # If they're different lengths, they're not equal
        if len(actions1) != len(actions2):
            return False

        # Compare each action
        for i in range(len(actions1)):
            action1 = actions1[i]
            action2 = actions2[i]

            # Compare action types
            if action1.action_type != action2.action_type:
                return False

            # Compare parameters
            if len(action1.parameters) != len(action2.parameters):
                return False

            # Create dictionaries of parameters for easier comparison
            params1 = {
                param.name: (param.value, param.metadata)
                for param in action1.parameters
            }
            params2 = {
                param.name: (param.value, param.metadata)
                for param in action2.parameters
            }

            if params1 != params2:
                return False

            # Compare metadata
            if action1.metadata != action2.metadata:
                return False

        # All actions are equal
        return True

    def register_action_handler(
        self, action_type: ActionType, handler: Callable[[Action, Dict[str, Any]], bool]
    ):
        """
        Register handler for action type.

        Args:
            action_type: Action type
            handler: Handler function
        """
        with self._lock:
            self.action_handlers[action_type] = handler

    def evaluate_rules(self, context: Dict[str, Any]) -> List[AdaptationRule]:
        """
        Evaluate all rules against context.

        Args:
            context: Evaluation context

        Returns:
            List of triggered rules
        """
        with self._lock:
            triggered_rules = []

            for rule in self.rules.values():
                if rule.evaluate(context):
                    triggered_rules.append(rule)

            # Sort by priority
            triggered_rules.sort(key=lambda r: r.priority.value, reverse=True)

            return triggered_rules

    def execute_rule(
        self, rule: AdaptationRule, context: Dict[str, Any]
    ) -> RuleExecutionResult:
        """
        Execute a rule's actions.

        Args:
            rule: Rule to execute
            context: Execution context

        Returns:
            Execution result
        """
        with self._lock:
            result = RuleExecutionResult(
                rule_id=rule.rule_id,
                success=True,
                timestamp=time.time(),
                actions_executed=[],
            )

            try:
                for action in rule.actions:
                    action_type = action.action_type

                    if action_type in self.action_handlers:
                        handler = self.action_handlers[action_type]
                        success = handler(action, context)

                        if success:
                            result.actions_executed.append(action_type.value)
                        else:
                            result.success = False
                            result.error_message = f"Action {action_type.value} failed"
                            break
                    else:
                        result.success = False
                        result.error_message = (
                            f"No handler for action type {action_type.value}"
                        )
                        break

            except Exception as e:
                result.success = False
                result.error_message = str(e)

            # Add to execution history
            self.execution_history.append(result)

            return result

    def process_event(self, event: Dict[str, Any]) -> List[RuleExecutionResult]:
        """
        Process event and execute triggered rules.

        Args:
            event: Event data

        Returns:
            List of execution results
        """
        with self._lock:
            # Create context
            context = {"event": event, "current_time": time.time()}

            # Evaluate rules
            triggered_rules = self.evaluate_rules(context)

            # Execute rules
            results = []

            for rule in triggered_rules:
                result = self.execute_rule(rule, context)
                results.append(result)

            return results

    def process_metrics(self, metrics: Dict[str, Any]) -> List[RuleExecutionResult]:
        """
        Process metrics and execute triggered rules.

        Args:
            metrics: Metric data

        Returns:
            List of execution results
        """
        with self._lock:
            # Create context
            context = {"metrics": metrics, "current_time": time.time()}

            # Evaluate rules
            triggered_rules = self.evaluate_rules(context)

            # Execute rules
            results = []

            for rule in triggered_rules:
                result = self.execute_rule(rule, context)
                results.append(result)

            return results

    def process_schedule(self) -> List[RuleExecutionResult]:
        """
        Process scheduled rules.

        Returns:
            List of execution results
        """
        with self._lock:
            # Create context
            context = {"current_time": time.time()}

            # Filter schedule-triggered rules
            schedule_rules = [
                rule
                for rule in self.rules.values()
                if rule.trigger == TriggerType.SCHEDULE
            ]

            # Evaluate rules
            triggered_rules = []

            for rule in schedule_rules:
                if rule.evaluate(context):
                    triggered_rules.append(rule)

            # Sort by priority
            triggered_rules.sort(key=lambda r: r.priority.value, reverse=True)

            # Execute rules
            results = []

            for rule in triggered_rules:
                result = self.execute_rule(rule, context)
                results.append(result)

            return results

    def to_dict(self) -> Dict[str, Any]:
        """Convert engine state to dictionary."""
        with self._lock:
            return {
                "rules": {
                    rule_id: rule.to_dict() for rule_id, rule in self.rules.items()
                },
                "execution_history": [
                    result.to_dict() for result in self.execution_history
                ],
            }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptationRulesEngine":
        """Create engine from dictionary."""
        engine = cls()

        for rule_id, rule_data in data.get("rules", {}).items():
            rule = AdaptationRule.from_dict(rule_data)
            engine.rules[rule_id] = rule

        for result_data in data.get("execution_history", []):
            result = RuleExecutionResult.from_dict(result_data)
            engine.execution_history.append(result)

        return engine

    def save(self, file_path: str):
        """Save engine state to file."""
        with self._lock:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str) -> "AdaptationRulesEngine":
        """Load engine state from file."""
        with open(file_path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)


class MAIFAdaptationManager:
    """
    Manager for MAIF adaptation rules.

    Provides high-level interface for managing MAIF adaptation
    based on rules and metrics.
    """

    def __init__(self, maif_path: str):
        self.maif_path = Path(maif_path)
        self.engine = AdaptationRulesEngine()
        self.metrics = MAIFMetrics()
        self._lock = threading.RLock()
        self._running = False
        self._thread = None

        # Register default action handlers
        self._register_default_handlers()

        # Register default rules
        self._register_default_rules()

    def _register_default_handlers(self):
        """Register default action handlers."""
        from .lifecycle_management import MAIFSplitter, MAIFMerger

        # Split handler
        def handle_split(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                output_dir = action.get_parameter("output_dir") or str(
                    self.maif_path.parent / "split"
                )
                strategy = action.get_parameter("strategy") or "size"

                # Create splitter
                splitter = MAIFSplitter()

                # Split MAIF
                splitter.split(str(self.maif_path), output_dir, split_strategy=strategy)

                return True
            except Exception as e:
                logger.error(f"Split action failed: {e}")
                return False

        # Merge handler
        def handle_merge(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                source_paths = action.get_parameter("source_paths")
                output_path = action.get_parameter("output_path") or str(
                    self.maif_path.parent / "merged.maif"
                )
                strategy = action.get_parameter("strategy") or "append"

                if not source_paths:
                    return False

                # Create merger
                merger = MAIFMerger()

                # Merge MAIFs
                merger.merge(source_paths, output_path, merge_strategy=strategy)

                return True
            except Exception as e:
                logger.error(f"Merge action failed: {e}")
                return False

        # Reorganize handler
        def handle_reorganize(action: Action, context: Dict[str, Any]) -> bool:
            try:
                from .self_optimizing import SelfOptimizingMAIF

                # Create optimizer
                optimizer = SelfOptimizingMAIF(str(self.maif_path))

                # Reorganize
                optimizer.reorganize()

                return True
            except Exception as e:
                logger.error(f"Reorganize action failed: {e}")
                return False

        # Optimize handler
        def handle_optimize(action: Action, context: Dict[str, Any]) -> bool:
            try:
                from .self_optimizing import SelfOptimizingMAIF

                # Get parameters
                strategy = action.get_parameter("strategy") or "default"

                # Create optimizer
                optimizer = SelfOptimizingMAIF(str(self.maif_path))

                # Optimize
                if strategy == "compression":
                    optimizer.optimize_compression()
                elif strategy == "access":
                    optimizer.optimize_access_patterns()
                else:
                    optimizer.optimize()

                return True
            except Exception as e:
                logger.error(f"Optimize action failed: {e}")
                return False

        # Archive handler
        def handle_archive(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                archive_path = action.get_parameter("archive_path") or str(
                    self.maif_path.with_suffix(".maif.archive")
                )

                # Archive file
                import shutil

                shutil.copy2(str(self.maif_path), archive_path)

                return True
            except Exception as e:
                logger.error(f"Archive action failed: {e}")
                return False

        # Restore handler
        def handle_restore(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                source_path = action.get_parameter("source_path")

                if not source_path:
                    return False

                # Restore file
                import shutil

                shutil.copy2(source_path, str(self.maif_path))

                return True
            except Exception as e:
                logger.error(f"Restore action failed: {e}")
                return False

        # Replicate handler
        def handle_replicate(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                target_paths = action.get_parameter("target_paths")

                if not target_paths:
                    return False

                # Replicate file
                import shutil

                for target_path in target_paths:
                    shutil.copy2(str(self.maif_path), target_path)

                return True
            except Exception as e:
                logger.error(f"Replicate action failed: {e}")
                return False

        # Migrate handler
        def handle_migrate(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                target_path = action.get_parameter("target_path")

                if not target_path:
                    return False

                # Migrate file
                import shutil

                shutil.move(str(self.maif_path), target_path)

                # Update path
                self.maif_path = Path(target_path)

                return True
            except Exception as e:
                logger.error(f"Migrate action failed: {e}")
                return False

        # Transform handler
        def handle_transform(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                transform_type = action.get_parameter("transform_type")
                output_path = action.get_parameter("output_path") or str(
                    self.maif_path.with_suffix(".transformed.maif")
                )

                if transform_type == "version_upgrade":
                    from .version_management import VersionManager, SchemaRegistry

                    # Get target version
                    target_version = action.get_parameter("target_version")

                    # Create registry and manager
                    registry = SchemaRegistry()
                    manager = VersionManager(registry)

                    # Upgrade file
                    manager.upgrade_file(
                        str(self.maif_path), output_path, target_version
                    )

                elif transform_type == "columnar":
                    from .columnar_storage import (
                        ColumnarFile,
                        ColumnType,
                        EncodingType,
                        CompressionType,
                    )
                    from .core import MAIFEncoder

                    # Convert to columnar format
                    output_path = (
                        action.get_parameter("output_path")
                        or str(self.maif_path) + ".columnar"
                    )
                    chunk_size = action.get_parameter("chunk_size") or 1000

                    # Open source MAIF file
                    encoder = MAIFEncoder(str(self.maif_path))

                    # Create columnar file
                    with ColumnarFile(output_path) as columnar_file:
                        # Analyze blocks to determine schema
                        schema = {}
                        sample_size = min(100, len(encoder.blocks))

                        for i, (block_id, block) in enumerate(encoder.blocks.items()):
                            if i >= sample_size:
                                break

                            # Extract metadata fields
                            metadata = block.get("metadata", {})
                            for key, value in metadata.items():
                                if key not in schema:
                                    # Infer column type from value
                                    if isinstance(value, bool):
                                        schema[key] = ColumnType.BOOLEAN
                                    elif isinstance(value, int):
                                        schema[key] = ColumnType.INT64
                                    elif isinstance(value, float):
                                        schema[key] = ColumnType.FLOAT64
                                    elif isinstance(value, str):
                                        schema[key] = ColumnType.STRING
                                    elif isinstance(value, bytes):
                                        schema[key] = ColumnType.BINARY
                                    elif isinstance(value, dict) or isinstance(
                                        value, list
                                    ):
                                        schema[key] = ColumnType.JSON
                                    else:
                                        schema[key] = (
                                            ColumnType.STRING
                                        )  # Default to string

                        # Add standard columns
                        schema["block_id"] = ColumnType.STRING
                        schema["block_type"] = ColumnType.STRING
                        schema["timestamp"] = ColumnType.TIMESTAMP
                        schema["data_size"] = ColumnType.INT64

                        # Add columns to columnar file
                        for column_name, column_type in schema.items():
                            columnar_file.add_column(column_name, column_type)

                        # Convert blocks to columnar format in chunks
                        data_buffer = {col: [] for col in schema}
                        blocks_processed = 0

                        for block_id, block in encoder.blocks.items():
                            # Extract block data
                            metadata = block.get("metadata", {})

                            # Add standard fields
                            data_buffer["block_id"].append(block_id)
                            data_buffer["block_type"].append(
                                block.get("type", "unknown")
                            )
                            data_buffer["timestamp"].append(
                                metadata.get("timestamp", time.time())
                            )
                            data_buffer["data_size"].append(len(block.get("data", b"")))

                            # Add metadata fields
                            for column_name in schema:
                                if column_name not in [
                                    "block_id",
                                    "block_type",
                                    "timestamp",
                                    "data_size",
                                ]:
                                    value = metadata.get(column_name, None)
                                    data_buffer[column_name].append(value)

                            blocks_processed += 1

                            # Write chunk when buffer is full
                            if blocks_processed % chunk_size == 0:
                                # Determine best encoding and compression for each column
                                encodings = {}
                                compressions = {}

                                for col_name, col_type in schema.items():
                                    # Use dictionary encoding for string columns with low cardinality
                                    if col_type == ColumnType.STRING:
                                        unique_values = len(
                                            set(
                                                v
                                                for v in data_buffer[col_name]
                                                if v is not None
                                            )
                                        )
                                        total_values = len(
                                            [
                                                v
                                                for v in data_buffer[col_name]
                                                if v is not None
                                            ]
                                        )
                                        if (
                                            total_values > 0
                                            and unique_values / total_values < 0.5
                                        ):
                                            encodings[col_name] = (
                                                EncodingType.DICTIONARY
                                            )
                                        else:
                                            encodings[col_name] = EncodingType.PLAIN

                                    # Use delta encoding for numeric/timestamp columns
                                    elif col_type in [
                                        ColumnType.INT32,
                                        ColumnType.INT64,
                                        ColumnType.FLOAT32,
                                        ColumnType.FLOAT64,
                                        ColumnType.TIMESTAMP,
                                    ]:
                                        encodings[col_name] = EncodingType.DELTA
                                    else:
                                        encodings[col_name] = EncodingType.PLAIN

                                    # Use compression for all columns
                                    compressions[col_name] = CompressionType.ZSTD

                                # Write batch
                                columnar_file.write_batch(
                                    data_buffer,
                                    row_group_size=chunk_size,
                                    encodings=encodings,
                                    compressions=compressions,
                                )

                                # Clear buffer
                                data_buffer = {col: [] for col in schema}

                        # Write remaining data
                        if any(len(values) > 0 for values in data_buffer.values()):
                            columnar_file.write_batch(
                                data_buffer, row_group_size=chunk_size
                            )

                    logger.info(
                        f"Converted {blocks_processed} blocks to columnar format: {output_path}"
                    )

                return True
            except Exception as e:
                logger.error(f"Transform action failed: {e}")
                return False

        # Notify handler
        def handle_notify(action: Action, context: Dict[str, Any]) -> bool:
            try:
                # Get parameters
                message = action.get_parameter("message") or "MAIF adaptation event"
                level = action.get_parameter("level") or "info"

                # Log notification
                if level == "debug":
                    logger.debug(message)
                elif level == "info":
                    logger.info(message)
                elif level == "warning":
                    logger.warning(message)
                elif level == "error":
                    logger.error(message)

                return True
            except Exception as e:
                logger.error(f"Notify action failed: {e}")
                return False

        # Register handlers
        self.engine.register_action_handler(ActionType.SPLIT, handle_split)
        self.engine.register_action_handler(ActionType.MERGE, handle_merge)
        self.engine.register_action_handler(ActionType.REORGANIZE, handle_reorganize)
        self.engine.register_action_handler(ActionType.OPTIMIZE, handle_optimize)
        self.engine.register_action_handler(ActionType.ARCHIVE, handle_archive)
        self.engine.register_action_handler(ActionType.RESTORE, handle_restore)
        self.engine.register_action_handler(ActionType.REPLICATE, handle_replicate)
        self.engine.register_action_handler(ActionType.MIGRATE, handle_migrate)
        self.engine.register_action_handler(ActionType.TRANSFORM, handle_transform)
        self.engine.register_action_handler(ActionType.NOTIFY, handle_notify)

    def _register_default_rules(self):
        """Register default adaptation rules."""
        # Size limit rule
        size_rule = AdaptationRule(
            rule_id="size_limit",
            name="Size Limit Rule",
            description="Split MAIF when it exceeds size threshold",
            priority=RulePriority.HIGH,
            trigger=TriggerType.METRIC,
            condition=MetricCondition(
                metric_name="size_bytes",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=1073741824,  # 1GB
            ),
            actions=[
                Action(
                    action_type=ActionType.SPLIT,
                    parameters=[
                        ActionParameter(name="strategy", value="size"),
                        ActionParameter(name="output_dir", value=None),
                    ],
                ),
                Action(
                    action_type=ActionType.NOTIFY,
                    parameters=[
                        ActionParameter(
                            name="message", value="MAIF split due to size limit"
                        ),
                        ActionParameter(name="level", value="info"),
                    ],
                ),
            ],
        )

        # Fragmentation rule
        fragmentation_rule = AdaptationRule(
            rule_id="fragmentation",
            name="Fragmentation Rule",
            description="Reorganize MAIF when fragmentation exceeds threshold",
            priority=RulePriority.MEDIUM,
            trigger=TriggerType.METRIC,
            condition=MetricCondition(
                metric_name="fragmentation",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=0.5,
            ),
            actions=[
                Action(action_type=ActionType.REORGANIZE),
                Action(
                    action_type=ActionType.NOTIFY,
                    parameters=[
                        ActionParameter(
                            name="message",
                            value="MAIF reorganized due to fragmentation",
                        ),
                        ActionParameter(name="level", value="info"),
                    ],
                ),
            ],
        )

        # Low access rule
        low_access_rule = AdaptationRule(
            rule_id="low_access",
            name="Low Access Rule",
            description="Archive MAIF when access frequency is low",
            priority=RulePriority.LOW,
            trigger=TriggerType.METRIC,
            condition=CompositeCondition(
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
            ),
            actions=[
                Action(action_type=ActionType.ARCHIVE),
                Action(
                    action_type=ActionType.NOTIFY,
                    parameters=[
                        ActionParameter(
                            name="message", value="MAIF archived due to low access"
                        ),
                        ActionParameter(name="level", value="info"),
                    ],
                ),
            ],
        )

        # High access rule
        high_access_rule = AdaptationRule(
            rule_id="high_access",
            name="High Access Rule",
            description="Optimize MAIF for access when frequency is high",
            priority=RulePriority.HIGH,
            trigger=TriggerType.METRIC,
            condition=MetricCondition(
                metric_name="access_frequency",
                operator=ComparisonOperator.GREATER_THAN,
                threshold=10.0,
            ),
            actions=[
                Action(
                    action_type=ActionType.OPTIMIZE,
                    parameters=[ActionParameter(name="strategy", value="access")],
                ),
                Action(
                    action_type=ActionType.NOTIFY,
                    parameters=[
                        ActionParameter(
                            name="message", value="MAIF optimized for high access"
                        ),
                        ActionParameter(name="level", value="info"),
                    ],
                ),
            ],
        )

        # Register rules
        self.engine.register_rule(size_rule)
        self.engine.register_rule(fragmentation_rule)
        self.engine.register_rule(low_access_rule)
        self.engine.register_rule(high_access_rule)

    def start(self):
        """Start adaptation manager."""
        with self._lock:
            if self._running:
                return

            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop adaptation manager."""
        with self._lock:
            self._running = False

            if self._thread:
                self._thread.join(timeout=5.0)
                self._thread = None

    def _run(self):
        """Run adaptation loop."""
        while self._running:
            try:
                # Update metrics
                self._update_metrics()

                # Process metrics
                self.engine.process_metrics(self.metrics.__dict__)

                # Process schedule
                self.engine.process_schedule()

                # Sleep
                time.sleep(60.0)  # Check every minute

            except Exception as e:
                logger.error(f"Adaptation error: {e}")
                time.sleep(300.0)  # Wait 5 minutes on error

    def _update_metrics(self):
        """Update MAIF metrics."""
        try:
            if not self.maif_path.exists():
                return

            # File metrics
            stat = self.maif_path.stat()
            self.metrics.size_bytes = stat.st_size
            self.metrics.last_accessed = stat.st_atime
            self.metrics.age_days = (time.time() - stat.st_ctime) / 86400

            # Get additional metrics from self-optimizing module
            try:
                from .self_optimizing import SelfOptimizingMAIF

                optimizer = SelfOptimizingMAIF(str(self.maif_path))
                stats = optimizer.get_optimization_stats()

                # Access frequency
                total_accesses = (
                    stats["metrics"]["total_reads"] + stats["metrics"]["total_writes"]
                )
                time_span = time.time() - stat.st_ctime
                self.metrics.access_frequency = (
                    total_accesses / (time_span / 3600) if time_span > 0 else 0
                )

                # Fragmentation
                self.metrics.fragmentation = stats["metrics"]["fragmentation_ratio"]

                # Compression ratio
                self.metrics.compression_ratio = stats["metrics"].get(
                    "compression_ratio", 1.0
                )

                # Semantic coherence
                self.metrics.semantic_coherence = 1.0 - self.metrics.fragmentation

            except Exception as e:
                logger.warning(f"Error getting optimization stats: {e}")

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")

    def process_event(self, event: Dict[str, Any]) -> List[RuleExecutionResult]:
        """
        Process event and execute triggered rules.

        Args:
            event: Event data

        Returns:
            List of execution results
        """
        with self._lock:
            # Update metrics
            self._update_metrics()

            # Process event
            return self.engine.process_event(event)

    def add_rule(self, rule: AdaptationRule) -> str:
        """
        Add rule to engine.

        Args:
            rule: Rule to add

        Returns:
            Rule ID
        """
        with self._lock:
            return self.engine.register_rule(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove rule from engine.

        Args:
            rule_id: Rule ID

        Returns:
            True if rule was removed, False otherwise
        """
        with self._lock:
            return self.engine.unregister_rule(rule_id)

    def get_rule(self, rule_id: str) -> Optional[AdaptationRule]:
        """
        Get rule by ID.

        Args:
            rule_id: Rule ID

        Returns:
            Rule or None if not found
        """
        with self._lock:
            return self.engine.get_rule(rule_id)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            return self.metrics.__dict__

    def get_execution_history(self) -> List[RuleExecutionResult]:
        """
        Get rule execution history.

        Returns:
            List of execution results
        """
        with self._lock:
            return self.engine.execution_history
