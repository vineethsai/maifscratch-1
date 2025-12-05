"""
MAIF Lifecycle Management
Implements merging/splitting operations and self-governing data fabric.
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

logger = logging.getLogger(__name__)

# Block type constants (match SecureBlockType enum values)
BLOCK_TYPE_TEXT = 1
BLOCK_TYPE_EMBEDDINGS = 2
BLOCK_TYPE_BINARY = 3


def get_block_type_name(block_type) -> str:
    """Get block type name from int or enum."""
    if hasattr(block_type, "name"):
        return get_block_type_name(block_type)
    # Handle int values
    type_map = {1: "TEXT", 2: "EMBEDDINGS", 3: "BINARY", 4: "METADATA", 5: "INDEX"}
    return type_map.get(int(block_type), f"UNKNOWN_{block_type}")


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


# Governance Rules
@dataclass
class GovernanceRule:
    """Rule for MAIF self-governance."""

    rule_id: str
    condition: str  # Python expression to evaluate
    action: str  # Action to take when condition is met
    priority: int = 0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


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


class MAIFMerger:
    """Handles merging of multiple MAIF files."""

    def __init__(self):
        self.embedder = OptimizedSemanticEmbedder()
        self.csb = CryptographicSemanticBinding()
        self.validator = MAIFValidator()

    def merge(
        self,
        maif_paths: List[str],
        output_path: str,
        merge_strategy: str = "append",
        deduplication: bool = True,
    ) -> Dict[str, Any]:
        """
        Merge multiple MAIF files into one.

        Args:
            maif_paths: List of MAIF file paths to merge
            output_path: Output path for merged MAIF
            merge_strategy: "append", "semantic", or "temporal"
            deduplication: Whether to remove duplicate blocks

        Returns:
            Merge statistics
        """
        logger.info(
            f"Merging {len(maif_paths)} MAIF files using {merge_strategy} strategy"
        )

        # Track merge statistics
        stats = {
            "total_blocks": 0,
            "merged_blocks": 0,
            "duplicate_blocks": 0,
            "conflicts_resolved": 0,
            "merge_time": 0.0,
        }

        start_time = time.time()

        # Block deduplication tracking
        seen_hashes = set()
        block_groups = []

        # Load all MAIF files (v3 format - self-contained)
        for maif_path in maif_paths:
            decoder = MAIFDecoder(maif_path)
            decoder.load()

            blocks = []
            for block in decoder.blocks:
                stats["total_blocks"] += 1

                # Get block data and hash
                block_data = block.data
                block_hash = hashlib.sha256(block_data).hexdigest()

                # Check for duplicates
                if deduplication and block_hash in seen_hashes:
                    stats["duplicate_blocks"] += 1
                    continue

                seen_hashes.add(block_hash)
                blocks.append(
                    {
                        "block": block,
                        "data": block_data,
                        "hash": block_hash,
                        "source": maif_path,
                    }
                )

            block_groups.append(blocks)

        # Apply merge strategy
        if merge_strategy == "append":
            merged_blocks = self._merge_append(block_groups)
        elif merge_strategy == "semantic":
            merged_blocks = self._merge_semantic(block_groups)
        elif merge_strategy == "temporal":
            merged_blocks = self._merge_temporal(block_groups)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

        # Create output encoder (v3 format)
        output_encoder = MAIFEncoder(output_path, agent_id="maif_merger")

        # Write merged blocks
        for block_info in merged_blocks:
            block = block_info["block"]
            data = block_info["data"]
            block_type = block.header.block_type

            # Add to output MAIF
            if get_block_type_name(block_type) == "TEXT":
                output_encoder.add_text_block(
                    data.decode("utf-8"), metadata=block.metadata
                )
            else:
                output_encoder.add_binary_block(
                    data, block_type, metadata=block.metadata
                )

            stats["merged_blocks"] += 1

        # Finalize output MAIF (v3 format)
        output_encoder.finalize()

        # Validate merged MAIF
        validation_result = self.validator.validate(output_path)

        stats["merge_time"] = time.time() - start_time
        stats["is_valid"] = validation_result.is_valid
        stats["validation_errors"] = validation_result.errors

        logger.info(
            f"Merge completed: {stats['merged_blocks']} blocks merged, "
            f"{stats['duplicate_blocks']} duplicates removed"
        )

        return stats

    def _merge_append(self, block_groups: List[List[Dict]]) -> List[Dict]:
        """Simple append merge - concatenate all blocks."""
        merged = []
        for group in block_groups:
            merged.extend(group)
        return merged

    def _merge_semantic(self, block_groups: List[List[Dict]]) -> List[Dict]:
        """Semantic merge - group similar blocks together."""
        all_blocks = []
        for group in block_groups:
            all_blocks.extend(group)

        # Extract embeddings for text blocks
        text_blocks = []
        other_blocks = []

        for block_info in all_blocks:
            block_type = block_info["block"].header.block_type
            if get_block_type_name(block_type) == "TEXT":
                text_blocks.append(block_info)
            else:
                other_blocks.append(block_info)

        # Sort text blocks by semantic similarity
        if text_blocks:
            texts = [b["data"].decode("utf-8") for b in text_blocks]
            embeddings = self.embedder.embed_texts(texts)

            # Cluster similar texts
            from sklearn.cluster import DBSCAN

            vectors = np.array([e.vector for e in embeddings])
            clustering = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
            labels = clustering.fit_predict(vectors)

            # Group by cluster
            clustered_blocks = []
            for label in set(labels):
                cluster = [text_blocks[i] for i, l in enumerate(labels) if l == label]
                clustered_blocks.extend(cluster)

            return clustered_blocks + other_blocks

        return other_blocks

    def _merge_temporal(self, block_groups: List[List[Dict]]) -> List[Dict]:
        """Temporal merge - sort blocks by timestamp."""
        all_blocks = []
        for group in block_groups:
            all_blocks.extend(group)

        # Sort by timestamp in metadata
        def get_timestamp(block_info):
            metadata = block_info["block"].metadata or {}
            return metadata.get("timestamp", 0)

        return sorted(all_blocks, key=get_timestamp)


class MAIFSplitter:
    """Handles splitting of MAIF files."""

    def __init__(self):
        self.validator = MAIFValidator()

    def split(
        self, maif_path: str, output_dir: str, split_strategy: str = "size", **kwargs
    ) -> List[str]:
        """
        Split a MAIF file into multiple files.

        Args:
            maif_path: Path to MAIF file to split
            output_dir: Directory for output files
            split_strategy: "size", "count", "type", or "semantic"
            **kwargs: Strategy-specific parameters

        Returns:
            List of output file paths
        """
        logger.info(f"Splitting {maif_path} using {split_strategy} strategy")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load MAIF (v3 format - self-contained)
        decoder = MAIFDecoder(maif_path)
        decoder.load()

        # Apply split strategy
        if split_strategy == "size":
            return self._split_by_size(decoder, output_dir, **kwargs)
        elif split_strategy == "count":
            return self._split_by_count(decoder, output_dir, **kwargs)
        elif split_strategy == "type":
            return self._split_by_type(decoder, output_dir)
        elif split_strategy == "semantic":
            return self._split_by_semantic(decoder, output_dir, **kwargs)
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")

    def _split_by_size(
        self, decoder: MAIFDecoder, output_dir: Path, max_size_mb: float = 100.0
    ) -> List[str]:
        """Split by file size."""
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        output_paths = []

        current_size = 0
        part_num = 1
        output_path = output_dir / f"part_{part_num}.maif"
        current_encoder = MAIFEncoder(str(output_path), agent_id="maif_splitter")

        for block in decoder.blocks:
            block_data = block.data
            block_size = len(block_data)
            block_type = block.header.block_type

            # Check if adding this block would exceed size limit
            if current_size + block_size > max_size_bytes and current_size > 0:
                # Finalize current part (v3 format)
                current_encoder.finalize()
                output_paths.append(str(output_path))

                # Start new part
                part_num += 1
                output_path = output_dir / f"part_{part_num}.maif"
                current_encoder = MAIFEncoder(
                    str(output_path), agent_id="maif_splitter"
                )
                current_size = 0

            # Add block to current part
            if get_block_type_name(block_type) == "TEXT":
                current_encoder.add_text_block(
                    block_data.decode("utf-8"), metadata=block.metadata
                )
            else:
                current_encoder.add_binary_block(
                    block_data, block_type, metadata=block.metadata
                )

            current_size += block_size

        # Finalize final part (v3 format)
        if current_size > 0:
            current_encoder.finalize()
            output_paths.append(str(output_path))

        logger.info(f"Split into {len(output_paths)} parts")
        return output_paths

    def _split_by_count(
        self, decoder: MAIFDecoder, output_dir: Path, blocks_per_file: int = 100
    ) -> List[str]:
        """Split by block count."""
        output_paths = []

        blocks = list(decoder.blocks)
        total_blocks = len(blocks)

        for i in range(0, total_blocks, blocks_per_file):
            part_blocks = blocks[i : i + blocks_per_file]
            part_num = (i // blocks_per_file) + 1

            output_path = output_dir / f"part_{part_num}.maif"
            encoder = MAIFEncoder(str(output_path), agent_id="maif_splitter")

            for block in part_blocks:
                block_data = block.data
                block_type = block.header.block_type

                if get_block_type_name(block_type) == "TEXT":
                    encoder.add_text_block(
                        block_data.decode("utf-8"), metadata=block.metadata
                    )
                else:
                    encoder.add_binary_block(
                        block_data, block_type, metadata=block.metadata
                    )

            # Finalize (v3 format)
            encoder.finalize()
            output_paths.append(str(output_path))

        logger.info(f"Split into {len(output_paths)} parts")
        return output_paths

    def _split_by_type(self, decoder: MAIFDecoder, output_dir: Path) -> List[str]:
        """Split by block type."""
        type_encoders = {}
        type_paths = {}

        for block in decoder.blocks:
            block_type = block.header.block_type
            type_name = get_block_type_name(block_type)

            if type_name not in type_encoders:
                output_path = output_dir / f"{type_name}.maif"
                type_paths[type_name] = output_path
                type_encoders[type_name] = MAIFEncoder(
                    str(output_path), agent_id="maif_splitter"
                )

            # Get block data
            block_data = block.data or b""

            if type_name == "TEXT":
                try:
                    text_content = block_data.decode("utf-8") if block_data else ""
                    type_encoders[type_name].add_text_block(
                        text_content, metadata=block.metadata
                    )
                except UnicodeDecodeError:
                    type_encoders[type_name].add_binary_block(
                        block_data, block_type, metadata=block.metadata
                    )
            else:
                type_encoders[type_name].add_binary_block(
                    block_data, block_type, metadata=block.metadata
                )

        output_paths = []

        for type_name, encoder in type_encoders.items():
            encoder.finalize()
            output_paths.append(str(type_paths[type_name]))

        logger.info(f"Split into {len(output_paths)} type-specific files")
        return output_paths

    def _split_by_semantic(
        self, decoder: MAIFDecoder, output_dir: Path, num_clusters: int = 5
    ) -> List[str]:
        """Split by semantic similarity."""
        from .semantic_optimized import OptimizedSemanticEmbedder

        embedder = OptimizedSemanticEmbedder()

        # Extract text blocks
        text_blocks = []
        other_blocks = []

        for block in decoder.blocks:
            block_type = block.header.block_type
            if get_block_type_name(block_type) == "TEXT":
                text_blocks.append(block)
            else:
                other_blocks.append(block)

        if not text_blocks:
            # No text blocks to cluster
            return self._split_by_count(decoder, output_dir)

        # Get embeddings
        texts = []
        for block in text_blocks:
            data = block.data.decode("utf-8")
            texts.append(data)

        embeddings = embedder.embed_texts(texts)
        vectors = np.array([e.vector for e in embeddings])

        # Cluster
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=min(num_clusters, len(text_blocks)), n_init=10)
        labels = kmeans.fit_predict(vectors)

        # Create encoders for each cluster
        cluster_encoders = {}
        cluster_paths = {}

        for i, (block, label) in enumerate(zip(text_blocks, labels)):
            if label not in cluster_encoders:
                output_path = output_dir / f"cluster_{label}.maif"
                cluster_paths[label] = output_path
                cluster_encoders[label] = MAIFEncoder(
                    str(output_path), agent_id="maif_splitter"
                )

            block_data = block.data
            cluster_encoders[label].add_text_block(
                block_data.decode("utf-8"), metadata=block.metadata
            )

        # Add non-text blocks to first cluster
        if 0 in cluster_encoders and other_blocks:
            for block in other_blocks:
                block_data = block.data
                block_type = block.header.block_type
                cluster_encoders[0].add_binary_block(
                    block_data, block_type, metadata=block.metadata
                )

        # Finalize clusters (v3 format)
        output_paths = []

        for cluster_id, encoder in cluster_encoders.items():
            encoder.finalize()
            output_paths.append(str(cluster_paths[cluster_id]))

        logger.info(f"Split into {len(output_paths)} semantic clusters")
        return output_paths


class SelfGoverningMAIF:
    """
    Self-governing data fabric for MAIF files.
    Implements autonomous lifecycle management based on rules and metrics.
    """

    def __init__(self, maif_path: str, rules_path: Optional[str] = None):
        self.maif_path = Path(maif_path)
        self.rules_path = Path(rules_path) if rules_path else None

        # Components
        self.optimizer = SelfOptimizingMAIF(str(maif_path))
        self.merger = MAIFMerger()
        self.splitter = MAIFSplitter()

        # Governance state
        self.state = MAIFLifecycleState.CREATED
        self.rules: List[GovernanceRule] = []
        self.metrics = MAIFMetrics()
        self.history: List[Dict[str, Any]] = []

        # Threading
        self._lock = threading.RLock()
        self._governance_thread = None
        self._running = False

        # Load rules
        self._load_rules()

        # Start governance
        self.start_governance()

    def _load_rules(self):
        """Load governance rules."""
        # Default rules
        self.rules = [
            GovernanceRule(
                rule_id="size_limit",
                condition="metrics.size_bytes > 1073741824",  # 1GB
                action="split",
                priority=10,
            ),
            GovernanceRule(
                rule_id="fragmentation",
                condition="metrics.fragmentation > 0.5",
                action="reorganize",
                priority=8,
            ),
            GovernanceRule(
                rule_id="low_access",
                condition="metrics.access_frequency < 0.1 and metrics.age_days > 30",
                action="archive",
                priority=5,
            ),
            GovernanceRule(
                rule_id="high_access",
                condition="metrics.access_frequency > 10.0",
                action="optimize_hot",
                priority=9,
            ),
            GovernanceRule(
                rule_id="semantic_drift",
                condition="metrics.semantic_coherence < 0.5",
                action="semantic_reorganize",
                priority=7,
            ),
        ]

        # Load custom rules if provided
        if self.rules_path and self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                custom_rules = json.load(f)
                for rule_data in custom_rules:
                    self.rules.append(GovernanceRule(**rule_data))

        # Sort by priority
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def start_governance(self):
        """Start autonomous governance."""
        self._running = True
        self._governance_thread = threading.Thread(
            target=self._governance_loop, daemon=True
        )
        self._governance_thread.start()
        logger.info(f"Started self-governance for {self.maif_path}")

    def stop_governance(self):
        """Stop autonomous governance."""
        self._running = False
        if self._governance_thread:
            self._governance_thread.join()
        logger.info(f"Stopped self-governance for {self.maif_path}")

    def _governance_loop(self):
        """Main governance loop."""
        while self._running:
            try:
                # Update metrics
                self._update_metrics()

                # Evaluate rules
                actions = self._evaluate_rules()

                # Execute actions
                for action in actions:
                    self._execute_action(action)

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
                decoder.load()
                self.metrics.block_count = len(decoder.blocks)
            except:
                pass

            # Semantic coherence (simplified)
            self.metrics.semantic_coherence = 1.0 - self.metrics.fragmentation

    def _evaluate_rules(self) -> List[str]:
        """Evaluate governance rules and return actions to execute."""
        actions = []

        with self._lock:
            for rule in self.rules:
                if not rule.enabled:
                    continue

                try:
                    # Evaluate condition
                    # Create safe evaluation context
                    eval_context = {
                        "metrics": self.metrics,
                        "state": self.state,
                        "age_days": self.metrics.age_days,
                        "size_mb": self.metrics.size_bytes / (1024 * 1024),
                    }

                    if eval(rule.condition, {"__builtins__": {}}, eval_context):
                        logger.info(f"Rule {rule.rule_id} triggered: {rule.action}")
                        actions.append(rule.action)

                        # Record in history
                        self.history.append(
                            {
                                "timestamp": time.time(),
                                "rule_id": rule.rule_id,
                                "action": rule.action,
                                "metrics": {
                                    "size_bytes": self.metrics.size_bytes,
                                    "fragmentation": self.metrics.fragmentation,
                                    "access_frequency": self.metrics.access_frequency,
                                },
                            }
                        )

                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.rule_id}: {e}")

        return actions

    def _execute_action(self, action: str):
        """Execute a governance action."""
        logger.info(f"Executing action: {action}")

        try:
            if action == "split":
                self._action_split()
            elif action == "reorganize":
                self._action_reorganize()
            elif action == "archive":
                self._action_archive()
            elif action == "optimize_hot":
                self._action_optimize_hot()
            elif action == "semantic_reorganize":
                self._action_semantic_reorganize()
            else:
                logger.warning(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")

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

    def add_rule(self, rule: GovernanceRule):
        """Add a governance rule."""
        with self._lock:
            self.rules.append(rule)
            self.rules.sort(key=lambda r: r.priority, reverse=True)

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
                "active_rules": len([r for r in self.rules if r.enabled]),
                "history": self.history[-10:],  # Last 10 actions
            }


# Lifecycle Manager for multiple MAIFs
class MAIFLifecycleManager:
    """Manages lifecycle of multiple MAIF files."""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.governed_maifs: Dict[str, SelfGoverningMAIF] = {}
        self._lock = threading.Lock()

    def add_maif(self, maif_path: str, rules_path: Optional[str] = None):
        """Add a MAIF file to lifecycle management."""
        with self._lock:
            if maif_path not in self.governed_maifs:
                self.governed_maifs[maif_path] = SelfGoverningMAIF(
                    maif_path, rules_path
                )
                logger.info(f"Added {maif_path} to lifecycle management")

    def remove_maif(self, maif_path: str):
        """Remove a MAIF file from lifecycle management."""
        with self._lock:
            if maif_path in self.governed_maifs:
                self.governed_maifs[maif_path].stop_governance()
                del self.governed_maifs[maif_path]
                logger.info(f"Removed {maif_path} from lifecycle management")

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
