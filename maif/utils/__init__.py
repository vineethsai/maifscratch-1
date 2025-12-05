"""
MAIF Utils Module

Contains utility features:
- Configuration management
- Error handling
- Validation
- Debugging tools
- Metadata management
- Migration tools
- Metrics and monitoring
- Rate limiting
- Health checks
- Cost tracking
- Batch processing
"""

# Configuration
try:
    from .config import MAIFConfig, get_config
except ImportError:
    MAIFConfig = None
    get_config = None

# Error handling
try:
    from .error_handling import (
        MAIFError,
        ValidationError,
        IntegrityError,
        EncryptionError,
    )
except ImportError:
    MAIFError = None
    ValidationError = None
    IntegrityError = None
    EncryptionError = None

# Validation
try:
    from .validation import (
        MAIFValidator,
        MAIFRepairTool,
        ValidationResult,
        validate_maif,
        get_validation_report,
    )
except ImportError:
    MAIFValidator = None
    MAIFRepairTool = None
    ValidationResult = None
    validate_maif = None
    get_validation_report = None

# Debug tools
try:
    from .debug_tools import MAIFDebugger, debug_maif
except ImportError:
    MAIFDebugger = None
    debug_maif = None

# Metadata
try:
    from .metadata import MAIFMetadataManager
except ImportError:
    MAIFMetadataManager = None

MetadataSchema = None  # Not implemented

# Version management
try:
    from .version_management import VersionManager, VersionInfo
except ImportError:
    VersionManager = None
    VersionInfo = None

# Migration tools
try:
    from .migration_tools import VectorDBMigrator, migrate_to_maif
except ImportError:
    VectorDBMigrator = None
    migrate_to_maif = None

try:
    from .migration_utils import MigrationHelper
except ImportError:
    MigrationHelper = None

# Metrics
try:
    from .metrics_aggregator import (
        MetricsAggregator,
        MAIFMetrics,
        initialize_metrics,
        get_metrics,
    )
except ImportError:
    MetricsAggregator = None
    MAIFMetrics = None
    initialize_metrics = None
    get_metrics = None

# Rate limiting
try:
    from .rate_limiter import (
        RateLimiter,
        RateLimitConfig,
        CostBasedRateLimiter,
        rate_limit,
    )
except ImportError:
    RateLimiter = None
    RateLimitConfig = None
    CostBasedRateLimiter = None
    rate_limit = None

try:
    from .rate_limiter_sync import SyncRateLimiter
except ImportError:
    SyncRateLimiter = None

# Health checks
try:
    from .health_check import HealthChecker, HealthStatus
except ImportError:
    HealthChecker = None
    HealthStatus = None

# Cost tracking
try:
    from .cost_tracker import (
        CostTracker,
        Budget,
        BudgetExceededException,
        initialize_cost_tracking,
        get_cost_tracker,
        with_cost_tracking,
    )
except ImportError:
    CostTracker = None
    Budget = None
    BudgetExceededException = None
    initialize_cost_tracking = None
    get_cost_tracker = None
    with_cost_tracking = None

# Batch processing
try:
    from .batch_processor import (
        BatchProcessor,
        StreamBatchProcessor,
        DistributedBatchProcessor,
        batch_process,
    )
except ImportError:
    BatchProcessor = None
    StreamBatchProcessor = None
    DistributedBatchProcessor = None
    batch_process = None

# Convenience API
try:
    from .convenience_api import SimpleMAIFAgent, create_agent
except ImportError:
    SimpleMAIFAgent = None
    create_agent = None

__all__ = [
    # Config
    "MAIFConfig",
    "get_config",
    # Errors
    "MAIFError",
    "ValidationError",
    "IntegrityError",
    "EncryptionError",
    # Validation
    "MAIFValidator",
    "MAIFRepairTool",
    "ValidationResult",
    "validate_maif",
    "get_validation_report",
    # Debug
    "MAIFDebugger",
    "debug_maif",
    # Metadata
    "MAIFMetadataManager",
    "MetadataSchema",
    # Version
    "VersionManager",
    "VersionInfo",
    # Migration
    "VectorDBMigrator",
    "migrate_to_maif",
    "MigrationHelper",
    # Metrics
    "MetricsAggregator",
    "MAIFMetrics",
    "initialize_metrics",
    "get_metrics",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "CostBasedRateLimiter",
    "rate_limit",
    "SyncRateLimiter",
    # Health
    "HealthChecker",
    "HealthStatus",
    # Cost
    "CostTracker",
    "Budget",
    "BudgetExceededException",
    "initialize_cost_tracking",
    "get_cost_tracker",
    "with_cost_tracking",
    # Batch
    "BatchProcessor",
    "StreamBatchProcessor",
    "DistributedBatchProcessor",
    "batch_process",
    # Convenience
    "SimpleMAIFAgent",
    "create_agent",
]

