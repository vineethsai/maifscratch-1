"""
MAIF Enhanced Integration Module
================================

This module integrates the core MAIF functionality with the newly implemented features:
- Event Sourcing
- Columnar Storage
- Dynamic Version Management
- Adaptation Rules Engine

It provides a cohesive interface for using all these features together in a
production-ready environment.
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path

# Import core MAIF components
from .core import MAIFEncoder, MAIFDecoder, MAIFBlock, BlockType
from .security import MAIFSigner, ProvenanceEntry
from .validation import MAIFValidator

# Import newly implemented features
from .event_sourcing import EventLog, MaterializedView, EventSourcedMAIF, EventType, Event
from .columnar_storage import ColumnarFile, ColumnType, EncodingType, CompressionType
from .version_management import SchemaRegistry, VersionManager, Schema, SchemaField, DataTransformer
from .adaptation_rules import (
    AdaptationRulesEngine, AdaptationRule, RulePriority, RuleStatus,
    ActionType, TriggerType, ComparisonOperator, LogicalOperator,
    MetricCondition, ScheduleCondition, EventCondition, CompositeCondition,
    Action, ActionParameter
)

# Import lifecycle management
from .lifecycle_management import MAIFLifecycleState, MAIFMetrics, SelfGoverningMAIF

logger = logging.getLogger(__name__)


class EnhancedMAIF:
    """
    Enhanced MAIF implementation that integrates all advanced features.
    
    This class provides a unified interface for working with MAIF files
    with full support for event sourcing, columnar storage, dynamic version
    management, and adaptation rules.
    """
    
    def __init__(self, maif_path: str, agent_id: Optional[str] = None,
                 enable_event_sourcing: bool = True,
                 enable_columnar_storage: bool = True,
                 enable_version_management: bool = True,
                 enable_adaptation_rules: bool = True):
        """
        Initialize an enhanced MAIF instance.
        
        Args:
            maif_path: Path to the MAIF file
            agent_id: ID of the agent using this MAIF
            enable_event_sourcing: Whether to enable event sourcing
            enable_columnar_storage: Whether to enable columnar storage
            enable_version_management: Whether to enable version management
            enable_adaptation_rules: Whether to enable adaptation rules
        """
        self.maif_path = Path(maif_path)
        self.agent_id = agent_id or f"agent-{int(time.time())}"
        
        # Core MAIF components (v3 format - self-contained)
        self.encoder = MAIFEncoder(str(self.maif_path), agent_id=self.agent_id)
        self.manifest_path = self.maif_path  # v3 is self-contained, no separate manifest
        
        # Initialize components based on enabled features
        self._init_event_sourcing() if enable_event_sourcing else None
        self._init_columnar_storage() if enable_columnar_storage else None
        self._init_version_management() if enable_version_management else None
        self._init_adaptation_rules() if enable_adaptation_rules else None
        
        # Lifecycle management
        self.metrics = MAIFMetrics()
        self.state = MAIFLifecycleState.CREATED
        
        # Threading
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced MAIF initialized at {maif_path}")
    
    def _init_event_sourcing(self):
        """Initialize event sourcing components."""
        event_log_path = self.maif_path.with_suffix('.events')
        self.event_log = EventLog(str(event_log_path))
        self.event_sourced_maif = EventSourcedMAIF(
            maif_id=self.maif_path.stem,
            event_log=self.event_log,
            agent_id=self.agent_id
        )
        logger.info("Event sourcing initialized")
    
    def _init_columnar_storage(self):
        """Initialize columnar storage components."""
        columnar_path = self.maif_path.with_suffix('.columnar')
        self.columnar_file = ColumnarFile(str(columnar_path))
        
        # Initialize schema with required columns
        self.columnar_file.schema = {
            "content": ColumnType.STRING,
            "block_id": ColumnType.STRING,
            "timestamp": ColumnType.FLOAT64
        }
        
        logger.info("Columnar storage initialized")
    
    def _init_version_management(self):
        """Initialize version management components."""
        registry_path = self.maif_path.with_suffix('.schema')
        
        # Create default schema if not exists
        if not registry_path.exists():
            self._create_default_schema(registry_path)
        
        self.schema_registry = SchemaRegistry.load(str(registry_path))
        self.version_manager = VersionManager(self.schema_registry)
        self.data_transformer = DataTransformer(self.schema_registry)
        logger.info("Version management initialized")
    
    def _init_adaptation_rules(self):
        """Initialize adaptation rules engine."""
        self.rules_engine = AdaptationRulesEngine()
        
        # Register default action handlers
        self._register_default_handlers()
        
        # Register default rules
        self._register_default_rules()
        
        logger.info("Adaptation rules engine initialized")
    
    def _create_default_schema(self, registry_path: Path):
        """Create default schema registry."""
        registry = SchemaRegistry()
        
        # Define initial schema
        initial_schema = Schema(
            version="1.0.0",
            fields=[
                SchemaField(name="id", field_type="string", required=True),
                SchemaField(name="type", field_type="string", required=True),
                SchemaField(name="content", field_type="string", required=True),
                SchemaField(name="metadata", field_type="json", required=False)
            ],
            metadata={"created_at": time.time()}
        )
        
        registry.register_schema(initial_schema)
        registry.save(str(registry_path))
    
    def _register_default_handlers(self):
        """Register default action handlers for adaptation rules."""
        
        def handle_split(action: Action, context: Dict[str, Any]) -> bool:
            """Handle split action."""
            try:
                # Get parameters
                output_dir = action.get_parameter("output_dir") or str(self.maif_path.parent / "split")
                strategy = action.get_parameter("strategy") or "size"
                
                # Create SelfGoverningMAIF and split
                gov_maif = SelfGoverningMAIF(str(self.maif_path))
                gov_maif._action_split()
                
                return True
            except Exception as e:
                logger.error(f"Error handling split action: {e}")
                return False
        
        def handle_optimize(action: Action, context: Dict[str, Any]) -> bool:
            """Handle optimize action."""
            try:
                # Create SelfGoverningMAIF and optimize
                gov_maif = SelfGoverningMAIF(str(self.maif_path))
                gov_maif._action_optimize_hot()
                
                return True
            except Exception as e:
                logger.error(f"Error handling optimize action: {e}")
                return False
        
        def handle_archive(action: Action, context: Dict[str, Any]) -> bool:
            """Handle archive action."""
            try:
                # Debug logging
                logger.info(f"Archive action parameters: {[f'{p.name}={p.value}' for p in action.parameters]}")
                logger.info(f"Archive action context keys: {list(context.keys())}")
                
                # Create SelfGoverningMAIF and archive
                # Use enhanced version if available
                try:
                    from .lifecycle_management_enhanced import EnhancedSelfGoverningMAIF
                    logger.info("Importing EnhancedSelfGoverningMAIF succeeded")
                    gov_maif = EnhancedSelfGoverningMAIF(str(self.maif_path))
                    logger.info("Created EnhancedSelfGoverningMAIF instance")
                except (ImportError, AttributeError) as e:
                    logger.info(f"Failed to import EnhancedSelfGoverningMAIF: {e}")
                    from .lifecycle_management import SelfGoverningMAIF
                    gov_maif = SelfGoverningMAIF(str(self.maif_path))
                    logger.info("Using SelfGoverningMAIF for archive action")
                    
                gov_maif._action_archive()
                logger.info("Archive action completed successfully")
                
                return True
            except Exception as e:
                logger.error(f"Error handling archive action: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error args: {e.args}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return False
        
        # Register handlers
        self.rules_engine.register_action_handler(ActionType.SPLIT, handle_split)
        self.rules_engine.register_action_handler(ActionType.OPTIMIZE, handle_optimize)
        self.rules_engine.register_action_handler(ActionType.ARCHIVE, handle_archive)
    
    def _register_default_rules(self):
        """Register default adaptation rules."""
        
        # Rule 1: Split large files
        split_condition = MetricCondition(
            metric_name="size_bytes",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=100 * 1024 * 1024  # 100 MB
        )
        
        split_action = Action(
            action_type=ActionType.SPLIT,
            parameters=[
                ActionParameter(name="strategy", value="size"),
                ActionParameter(name="max_size_mb", value=50.0)
            ]
        )
        
        split_rule = AdaptationRule(
            rule_id="split_large_files",
            name="Split Large Files",
            description="Split files larger than 100 MB",
            priority=RulePriority.MEDIUM,
            trigger=TriggerType.METRIC,
            condition=split_condition,
            actions=[split_action],
            status=RuleStatus.ACTIVE
        )
        
        # Rule 2: Optimize frequently accessed files
        optimize_condition = MetricCondition(
            metric_name="access_frequency",
            operator=ComparisonOperator.GREATER_THAN,
            threshold=10.0  # 10 accesses per minute
        )
        
        optimize_action = Action(
            action_type=ActionType.OPTIMIZE,
            parameters=[]
        )
        
        optimize_rule = AdaptationRule(
            rule_id="optimize_hot_files",
            name="Optimize Hot Files",
            description="Optimize frequently accessed files",
            priority=RulePriority.HIGH,
            trigger=TriggerType.METRIC,
            condition=optimize_condition,
            actions=[optimize_action],
            status=RuleStatus.ACTIVE
        )
        
        # Rule 3: Archive old files
        archive_condition = MetricCondition(
            metric_name="last_accessed",
            operator=ComparisonOperator.LESS_THAN,
            threshold=time.time() - (30 * 24 * 60 * 60)  # 30 days ago
        )
        
        archive_action = Action(
            action_type=ActionType.ARCHIVE,
            parameters=[]
        )
        
        archive_rule = AdaptationRule(
            rule_id="archive_old_files",
            name="Archive Old Files",
            description="Archive files not accessed in 30 days",
            priority=RulePriority.LOW,
            trigger=TriggerType.METRIC,
            condition=archive_condition,
            actions=[archive_action],
            status=RuleStatus.ACTIVE
        )
        
        # Register rules
        self.rules_engine.register_rule(split_rule)
        self.rules_engine.register_rule(optimize_rule)
        self.rules_engine.register_rule(archive_rule)
    
    def add_text_block(self, text: str, metadata: Optional[Dict] = None) -> str:
        """
        Add a text block to the MAIF.
        
        Args:
            text: Text content
            metadata: Block metadata
            
        Returns:
            Block ID
        """
        with self._lock:
            # Add to core MAIF
            block_id = self.encoder.add_text_block(text, metadata)
            
            # Record event if event sourcing enabled
            if hasattr(self, 'event_sourced_maif'):
                self.event_sourced_maif.add_block(
                    block_id=block_id,
                    block_type="text",
                    data=text.encode('utf-8'),
                    metadata=metadata
                )
            
            # Add to columnar storage if enabled
            if hasattr(self, 'columnar_file'):
                # Convert text to columnar format
                data = {
                    "content": [text],
                    "block_id": [block_id],
                    "timestamp": [time.time()]
                }
                
                # Add metadata as columns
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            data[key] = [value]
                            
                            # Check if column exists in schema, if not add it
                            if key not in self.columnar_file.schema:
                                # Determine column type
                                if isinstance(value, bool):
                                    col_type = ColumnType.BOOLEAN
                                elif isinstance(value, int):
                                    col_type = ColumnType.INT64
                                elif isinstance(value, float):
                                    col_type = ColumnType.FLOAT64
                                else:
                                    col_type = ColumnType.STRING
                                
                                self.columnar_file.add_column(key, col_type)
                
                self.columnar_file.write_batch(data)
            
            return block_id
    
    def add_binary_block(self, data: bytes, 
                        metadata: Optional[Dict] = None) -> str:
        """
        Add a binary block to the MAIF.
        
        Args:
            data: Binary data
            metadata: Block metadata
            
        Returns:
            Block ID
        """
        with self._lock:
            # Add to core MAIF (block_type defaults to BINARY)
            block_id = self.encoder.add_binary_block(data, metadata=metadata)
            
            # Record event if event sourcing enabled
            if hasattr(self, 'event_sourced_maif'):
                self.event_sourced_maif.add_block(
                    block_id=block_id,
                    block_type="binary",
                    data=data,
                    metadata=metadata
                )
            
            return block_id
    
    def update_metrics(self):
        """Update MAIF metrics for adaptation rules."""
        with self._lock:
            if not self.maif_path.exists():
                return
            
            # Update basic metrics
            self.metrics.size_bytes = self.maif_path.stat().st_size
            
            # Load MAIF to get block count if needed
            if self.metrics.block_count == 0:
                try:
                    decoder = MAIFDecoder(str(self.maif_path))
                    self.metrics.block_count = len(decoder.blocks)
                except Exception as e:
                    logger.error(f"Error loading MAIF for metrics update: {e}")
            
            # Update last accessed time
            self.metrics.last_accessed = time.time()
    
    def evaluate_rules(self) -> List[str]:
        """
        Evaluate adaptation rules and return actions to execute.
        
        Returns:
            List of action IDs to execute
        """
        with self._lock:
            if not hasattr(self, 'rules_engine'):
                return []
            
            # Update metrics
            self.update_metrics()
            
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
                    "semantic_coherence": self.metrics.semantic_coherence
                },
                "current_time": time.time(),
                "maif_path": str(self.maif_path),
                "agent_id": self.agent_id
            }
            
            # Evaluate rules
            triggered_rules = self.rules_engine.evaluate_rules(context)
            
            # Execute rules
            results = []
            for rule in triggered_rules:
                result = self.rules_engine.execute_rule(rule, context)
                if result.success:
                    results.append(result.rule_id)
            
            return results
    
    def save(self):
        """Save MAIF to disk (v3 format - finalize)."""
        with self._lock:
            # Finalize the MAIF file (v3 self-contained format)
            if not self.encoder.finalized:
                self.encoder.finalize()
            
            # Save columnar file if enabled
            if hasattr(self, 'columnar_file'):
                self.columnar_file.close()
            
            logger.info(f"Enhanced MAIF saved to {self.maif_path}")
    
    def get_history(self) -> List[Event]:
        """
        Get event history if event sourcing is enabled.
        
        Returns:
            List of events
        """
        if hasattr(self, 'event_sourced_maif'):
            return self.event_sourced_maif.get_history()
        return []
    
    def get_schema_version(self) -> str:
        """
        Get current schema version if version management is enabled.
        
        Returns:
            Schema version
        """
        if hasattr(self, 'schema_registry'):
            return self.schema_registry.get_latest_version()
        return "unknown"
    
    def get_columnar_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get columnar storage statistics if enabled.
        
        Returns:
            Statistics by column
        """
        if hasattr(self, 'columnar_file'):
            return self.columnar_file.get_statistics()
        return {}


class EnhancedMAIFManager:
    """
    Manager for multiple Enhanced MAIF instances.
    
    Provides centralized management of multiple MAIF files with
    integrated event sourcing, columnar storage, version management,
    and adaptation rules.
    """
    
    def __init__(self, workspace_dir: str):
        """
        Initialize the manager.
        
        Args:
            workspace_dir: Directory for managed MAIFs
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.maifs: Dict[str, EnhancedMAIF] = {}
        self._lock = threading.RLock()
        
        logger.info(f"Enhanced MAIF Manager initialized at {workspace_dir}")
    
    def create_maif(self, name: str, agent_id: Optional[str] = None,
                   enable_event_sourcing: bool = True,
                   enable_columnar_storage: bool = True,
                   enable_version_management: bool = True,
                   enable_adaptation_rules: bool = True) -> EnhancedMAIF:
        """
        Create a new Enhanced MAIF.
        
        Args:
            name: MAIF name
            agent_id: Agent ID
            enable_event_sourcing: Whether to enable event sourcing
            enable_columnar_storage: Whether to enable columnar storage
            enable_version_management: Whether to enable version management
            enable_adaptation_rules: Whether to enable adaptation rules
            
        Returns:
            Enhanced MAIF instance
        """
        with self._lock:
            maif_path = self.workspace_dir / f"{name}.maif"
            
            maif = EnhancedMAIF(
                str(maif_path),
                agent_id=agent_id,
                enable_event_sourcing=enable_event_sourcing,
                enable_columnar_storage=enable_columnar_storage,
                enable_version_management=enable_version_management,
                enable_adaptation_rules=enable_adaptation_rules
            )
            
            self.maifs[name] = maif
            return maif
    
    def get_maif(self, name: str) -> Optional[EnhancedMAIF]:
        """
        Get an Enhanced MAIF by name.
        
        Args:
            name: MAIF name
            
        Returns:
            Enhanced MAIF instance or None if not found
        """
        with self._lock:
            return self.maifs.get(name)
    
    def load_maif(self, path: str, name: Optional[str] = None,
                 agent_id: Optional[str] = None,
                 enable_event_sourcing: bool = True,
                 enable_columnar_storage: bool = True,
                 enable_version_management: bool = True,
                 enable_adaptation_rules: bool = True) -> EnhancedMAIF:
        """
        Load an existing MAIF file.
        
        Args:
            path: Path to MAIF file
            name: Name to use for the MAIF (defaults to filename)
            agent_id: Agent ID
            enable_event_sourcing: Whether to enable event sourcing
            enable_columnar_storage: Whether to enable columnar storage
            enable_version_management: Whether to enable version management
            enable_adaptation_rules: Whether to enable adaptation rules
            
        Returns:
            Enhanced MAIF instance
        """
        with self._lock:
            path = Path(path)
            name = name or path.stem
            
            # Copy to workspace if outside
            if not path.is_relative_to(self.workspace_dir):
                target_path = self.workspace_dir / path.name
                import shutil
                shutil.copy2(path, target_path)
                path = target_path
            
            maif = EnhancedMAIF(
                str(path),
                agent_id=agent_id,
                enable_event_sourcing=enable_event_sourcing,
                enable_columnar_storage=enable_columnar_storage,
                enable_version_management=enable_version_management,
                enable_adaptation_rules=enable_adaptation_rules
            )
            
            self.maifs[name] = maif
            return maif
    
    def evaluate_all_rules(self) -> Dict[str, List[str]]:
        """
        Evaluate adaptation rules for all managed MAIFs.
        
        Returns:
            Dictionary mapping MAIF names to lists of executed actions
        """
        with self._lock:
            results = {}
            
            for name, maif in self.maifs.items():
                actions = maif.evaluate_rules()
                results[name] = actions
            
            return results
    
    def save_all(self):
        """Save all managed MAIFs."""
        with self._lock:
            for maif in self.maifs.values():
                maif.save()
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all managed MAIFs.
        
        Returns:
            Dictionary mapping MAIF names to status reports
        """
        with self._lock:
            status = {}
            
            for name, maif in self.maifs.items():
                status[name] = {
                    "path": str(maif.maif_path),
                    "agent_id": maif.agent_id,
                    "state": maif.state.value,
                    "metrics": {
                        "size_bytes": maif.metrics.size_bytes,
                        "block_count": maif.metrics.block_count,
                        "access_frequency": maif.metrics.access_frequency,
                        "last_accessed": maif.metrics.last_accessed
                    },
                    "features": {
                        "event_sourcing": hasattr(maif, 'event_sourced_maif'),
                        "columnar_storage": hasattr(maif, 'columnar_file'),
                        "version_management": hasattr(maif, 'version_manager'),
                        "adaptation_rules": hasattr(maif, 'rules_engine')
                    }
                }
            
            return status

class ConversionResult:
    """
    Result of a conversion operation performed by EnhancedMAIFProcessor.
    
    Contains information about the conversion process, including success status,
    input and output paths, and any additional metadata.
    """
    
    def __init__(self, success: bool, input_path: str, output_path: str, 
                 metadata: Optional[Dict[str, Any]] = None,
                 warnings: Optional[List[str]] = None):
        """
        Initialize a conversion result.
        
        Args:
            success: Whether the conversion was successful
            input_path: Path to the input file
            output_path: Path to the output file
            metadata: Additional metadata about the conversion
            warnings: List of warnings generated during conversion
        """
        self.success = success
        self.input_path = input_path
        self.output_path = output_path
        self.metadata = metadata or {}
        self.warnings = warnings or []
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversionResult':
        """Create from dictionary representation."""
        result = cls(
            success=data.get("success", False),
            input_path=data.get("input_path", ""),
            output_path=data.get("output_path", ""),
            metadata=data.get("metadata", {})
        )
        result.timestamp = data.get("timestamp", time.time())
        return result


class EnhancedMAIFProcessor:
    """
    Processor for converting between MAIF and other formats.
    
    Provides functionality for converting various file formats to MAIF
    and extracting content from MAIF files to other formats. Integrates
    with all enhanced features including event sourcing, columnar storage,
    version management, and adaptation rules.
    """
    
    def __init__(self, workspace_dir: str, agent_id: Optional[str] = None):
        """
        Initialize the processor.
        
        Args:
            workspace_dir: Directory for processing files
            agent_id: ID of the agent using this processor
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent_id = agent_id or f"processor-{int(time.time())}"
        self.manager = EnhancedMAIFManager(workspace_dir)
        
        self._lock = threading.RLock()
        self.conversion_history: List[ConversionResult] = []
        
        logger.info(f"Enhanced MAIF Processor initialized at {workspace_dir}")
    
    def convert_to_maif(self, input_path: str, output_name: Optional[str] = None,
                       enable_event_sourcing: bool = True,
                       enable_columnar_storage: bool = True,
                       enable_version_management: bool = True,
                       enable_adaptation_rules: bool = True) -> ConversionResult:
        """
        Convert a file to MAIF format.
        
        Args:
            input_path: Path to the input file
            output_name: Name for the output MAIF (defaults to input filename)
            enable_event_sourcing: Whether to enable event sourcing
            enable_columnar_storage: Whether to enable columnar storage
            enable_version_management: Whether to enable version management
            enable_adaptation_rules: Whether to enable adaptation rules
            
        Returns:
            Conversion result
        """
        with self._lock:
            input_path = Path(input_path)
            
            if not input_path.exists():
                return ConversionResult(
                    success=False,
                    input_path=str(input_path),
                    output_path="",
                    metadata={"error": "Input file does not exist"}
                )
            
            # Determine output name
            if output_name is None:
                output_name = input_path.stem
            
            # Create enhanced MAIF
            maif = self.manager.create_maif(
                name=output_name,
                agent_id=self.agent_id,
                enable_event_sourcing=enable_event_sourcing,
                enable_columnar_storage=enable_columnar_storage,
                enable_version_management=enable_version_management,
                enable_adaptation_rules=enable_adaptation_rules
            )
            
            # Process based on file type
            try:
                if input_path.suffix.lower() in ['.txt', '.md', '.csv', '.json', '.xml', '.html']:
                    self._convert_text_file(input_path, maif)
                elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    self._convert_image_file(input_path, maif)
                elif input_path.suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac']:
                    self._convert_audio_file(input_path, maif)
                elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    self._convert_video_file(input_path, maif)
                else:
                    self._convert_binary_file(input_path, maif)
                
                # Save MAIF
                maif.save()
                
                # Determine format
                file_format = "binary"
                if input_path.suffix.lower() in ['.txt', '.md', '.csv', '.json', '.xml', '.html']:
                    file_format = "text"
                elif input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    file_format = "image"
                elif input_path.suffix.lower() in ['.mp3', '.wav', '.ogg', '.flac']:
                    file_format = "audio"
                elif input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    file_format = "video"

                result = ConversionResult(
                    success=True,
                    input_path=str(input_path),
                    output_path=str(maif.maif_path),
                    metadata={
                        "file_type": input_path.suffix.lower(),
                        "format": file_format,
                        "size_bytes": input_path.stat().st_size,
                        "maif_size_bytes": maif.metrics.size_bytes,
                        "block_count": maif.metrics.block_count
                    }
                )
                
                self.conversion_history.append(result)
                return result
                
            except Exception as e:
                logger.error(f"Error converting {input_path}: {e}")
                return ConversionResult(
                    success=False,
                    input_path=str(input_path),
                    output_path=str(maif.maif_path) if hasattr(maif, 'maif_path') else "",
                    metadata={"error": str(e)}
                )
    
    def extract_from_maif(self, maif_name: str, output_dir: Optional[str] = None,
                         extract_type: str = "all") -> ConversionResult:
        """
        Extract content from a MAIF file.
        
        Args:
            maif_name: Name of the MAIF to extract from
            output_dir: Directory for extracted files
            extract_type: Type of content to extract (all, text, images, audio, video)
            
        Returns:
            Conversion result
        """
        with self._lock:
            # Get MAIF
            maif = self.manager.get_maif(maif_name)
            if maif is None:
                return ConversionResult(
                    success=False,
                    input_path=maif_name,
                    output_path="",
                    metadata={"error": f"MAIF {maif_name} not found"}
                )
            
            # Determine output directory
            if output_dir is None:
                output_dir = self.workspace_dir / f"{maif_name}_extracted"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load MAIF (v3 format - self-contained)
                decoder = MAIFDecoder(str(maif.maif_path))
                
                # Extract content
                extracted_files = []
                
                for block in decoder.blocks:
                    if extract_type != "all" and block.block_type != extract_type:
                        continue
                    
                    output_path = self._extract_block(block, output_dir)
                    if output_path:
                        extracted_files.append(output_path)
                
                result = ConversionResult(
                    success=True,
                    input_path=str(maif.maif_path),
                    output_path=str(output_dir),
                    metadata={
                        "extracted_files": extracted_files,
                        "file_count": len(extracted_files)
                    }
                )
                
                self.conversion_history.append(result)
                return result
                
            except Exception as e:
                logger.error(f"Error extracting from {maif_name}: {e}")
                return ConversionResult(
                    success=False,
                    input_path=str(maif.maif_path) if hasattr(maif, 'maif_path') else "",
                    output_path=str(output_dir),
                    metadata={"error": str(e)}
                )
    
    def _convert_text_file(self, input_path: Path, maif: EnhancedMAIF):
        """Convert a text file to MAIF."""
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "filename": input_path.name,
            "file_type": input_path.suffix.lower(),
            "size_bytes": input_path.stat().st_size,
            "created": input_path.stat().st_ctime,
            "modified": input_path.stat().st_mtime
        }
        
        maif.add_text_block(content, metadata)
    
    def _convert_image_file(self, input_path: Path, maif: EnhancedMAIF):
        """Convert an image file to MAIF."""
        with open(input_path, 'rb') as f:
            content = f.read()
        
        metadata = {
            "filename": input_path.name,
            "file_type": input_path.suffix.lower(),
            "size_bytes": input_path.stat().st_size,
            "created": input_path.stat().st_ctime,
            "modified": input_path.stat().st_mtime
        }
        
        maif.add_binary_block(content, "image", metadata)
    
    def _convert_audio_file(self, input_path: Path, maif: EnhancedMAIF):
        """Convert an audio file to MAIF."""
        with open(input_path, 'rb') as f:
            content = f.read()
        
        metadata = {
            "filename": input_path.name,
            "file_type": input_path.suffix.lower(),
            "size_bytes": input_path.stat().st_size,
            "created": input_path.stat().st_ctime,
            "modified": input_path.stat().st_mtime
        }
        
        maif.add_binary_block(content, "audio", metadata)
    
    def _convert_video_file(self, input_path: Path, maif: EnhancedMAIF):
        """Convert a video file to MAIF."""
        with open(input_path, 'rb') as f:
            content = f.read()
        
        metadata = {
            "filename": input_path.name,
            "file_type": input_path.suffix.lower(),
            "size_bytes": input_path.stat().st_size,
            "created": input_path.stat().st_ctime,
            "modified": input_path.stat().st_mtime
        }
        
        maif.add_binary_block(content, "video", metadata)
    
    def _convert_binary_file(self, input_path: Path, maif: EnhancedMAIF):
        """Convert a binary file to MAIF."""
        with open(input_path, 'rb') as f:
            content = f.read()
        
        metadata = {
            "filename": input_path.name,
            "file_type": input_path.suffix.lower(),
            "size_bytes": input_path.stat().st_size,
            "created": input_path.stat().st_ctime,
            "modified": input_path.stat().st_mtime
        }
        
        maif.add_binary_block(content, "binary", metadata)
    
    def _extract_block(self, block: MAIFBlock, output_dir: Path) -> Optional[str]:
        """Extract a block to a file."""
        try:
            # Determine filename
            if block.metadata and "filename" in block.metadata:
                filename = block.metadata["filename"]
            else:
                if block.block_type == "text":
                    filename = f"{block.block_id}.txt"
                elif block.block_type == "image":
                    filename = f"{block.block_id}.png"
                elif block.block_type == "audio":
                    filename = f"{block.block_id}.mp3"
                elif block.block_type == "video":
                    filename = f"{block.block_id}.mp4"
                else:
                    filename = f"{block.block_id}.bin"
            
            output_path = output_dir / filename
            
            # Write content
            if block.block_type == "text":
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(block.content)
            else:
                with open(output_path, 'wb') as f:
                    f.write(block.content)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error extracting block {block.block_id}: {e}")
            return None
    
    def get_conversion_history(self) -> List[ConversionResult]:
        """Get history of conversions."""
        with self._lock:
            return self.conversion_history.copy()