"""
MAIF (Multimodal Artifact File Format) Library

A comprehensive library for creating, managing, and analyzing MAIF files.
MAIF is an AI-native file format designed for multimodal content with
embedded security (Ed25519 signatures), semantics, and provenance tracking.

Version 3.0 - Secure Format
- Self-contained binary files (no external manifest)
- Ed25519 cryptographic signatures
- Immutable blocks with tamper detection
- Embedded provenance chain

Quick Start:
    from maif import MAIFEncoder, MAIFDecoder

    # Create a MAIF file
    encoder = MAIFEncoder("output.maif", agent_id="my-agent")
    encoder.add_text_block("Hello, world!")
    encoder.finalize()

    # Read and verify a MAIF file
    decoder = MAIFDecoder("output.maif")
    is_valid, errors = decoder.verify_integrity()
    if is_valid:
        blocks = decoder.get_blocks()
"""

# =============================================================================
# Core API - Main encoding/decoding functionality
# =============================================================================

from .core import (
    # Primary classes
    MAIFEncoder,
    MAIFDecoder,
    MAIFParser,
    BlockType,
    # Data classes
    MAIFBlock,
    MAIFVersion,
    MAIFHeader,
    # Secure format structures
    SecureBlock,
    SecureBlockHeader,
    SecureFileHeader,
    ProvenanceEntry,
    FileFooter,
    BlockFlags,
    FileFlags,
    # Convenience functions
    create_maif,
    verify_maif,
    quick_create,
    quick_verify,
    quick_read,
    # Constants
    MAGIC_HEADER,
    MAGIC_FOOTER,
    FORMAT_VERSION_MAJOR,
    FORMAT_VERSION_MINOR,
)

# Legacy aliases for backwards compatibility
SecureMAIFWriter = MAIFEncoder
SecureMAIFReader = MAIFDecoder
SecureBlockType = BlockType

# =============================================================================
# Security - Signing and verification
# =============================================================================

try:
    from .security import MAIFSigner, MAIFVerifier
except ImportError:
    MAIFSigner = None
    MAIFVerifier = None

# =============================================================================
# Privacy - Privacy controls and encryption
# =============================================================================

try:
    from .privacy import (
        PrivacyEngine,
        PrivacyPolicy,
        PrivacyLevel,
        EncryptionMode,
        AccessRule,
        DifferentialPrivacy,
        SecureMultipartyComputation,
        ZeroKnowledgeProof,
    )
except ImportError:
    PrivacyEngine = None

# =============================================================================
# Semantics - Embeddings and knowledge graphs
# =============================================================================

try:
    from .semantic import (
        SemanticEmbedder,
        SemanticEmbedding,
        KnowledgeTriple,
        CrossModalAttention,
        HierarchicalSemanticCompression,
        CryptographicSemanticBinding,
        DeepSemanticUnderstanding,
        KnowledgeGraphBuilder,
    )
except ImportError:
    SemanticEmbedder = None

# Enhanced algorithms
try:
    from .semantic_optimized import (
        AdaptiveCrossModalAttention,
        HierarchicalSemanticCompression as EnhancedHierarchicalSemanticCompression,
        CryptographicSemanticBinding as EnhancedCryptographicSemanticBinding,
        AttentionWeights,
    )

    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ENHANCED_ALGORITHMS_AVAILABLE = False

# =============================================================================
# Forensics and Validation
# =============================================================================

try:
    from .forensics import ForensicAnalyzer, ForensicEvidence
except ImportError:
    ForensicAnalyzer = None

try:
    from .validation import MAIFValidator, MAIFRepairTool
except ImportError:
    MAIFValidator = None

# =============================================================================
# Compression
# =============================================================================

try:
    from .compression_manager import CompressionManager
    from .compression import CompressionMetadata
except ImportError:
    CompressionManager = None

# =============================================================================
# Streaming
# =============================================================================

try:
    from .streaming import MAIFStreamReader, MAIFStreamWriter
except ImportError:
    MAIFStreamReader = None
    MAIFStreamWriter = None

# =============================================================================
# Metadata
# =============================================================================

try:
    from .metadata import MAIFMetadataManager
except ImportError:
    MAIFMetadataManager = None

# =============================================================================
# Agent Framework
# =============================================================================

try:
    from .agentic_framework import (
        MAIFAgent,
        PerceptionSystem,
        ReasoningSystem,
        ExecutionSystem,
    )
except ImportError:
    MAIFAgent = None

# =============================================================================
# Production Features
# =============================================================================

try:
    from .health_check import HealthChecker, HealthStatus
except ImportError:
    HealthChecker = None

try:
    from .rate_limiter import (
        RateLimiter,
        RateLimitConfig,
        CostBasedRateLimiter,
        rate_limit,
    )
except ImportError:
    RateLimiter = None

try:
    from .metrics_aggregator import (
        MetricsAggregator,
        MAIFMetrics,
        initialize_metrics,
        get_metrics,
    )
except ImportError:
    MetricsAggregator = None

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

try:
    from .batch_processor import (
        BatchProcessor,
        StreamBatchProcessor,
        DistributedBatchProcessor,
        batch_process,
    )
except ImportError:
    BatchProcessor = None

# =============================================================================
# Integration
# =============================================================================

try:
    from .integration_enhanced import EnhancedMAIFProcessor, ConversionResult
except ImportError:
    EnhancedMAIFProcessor = None

# =============================================================================
# AWS Integrations (optional)
# =============================================================================

AWS_IMPORTS_AVAILABLE = False
try:
    from .aws_lambda_integration import AWSLambdaIntegration
    from .aws_stepfunctions_integration import AWSStepFunctionsIntegration
    from .aws_xray_integration import MAIFXRayIntegration, xray_trace, xray_subsegment
    from .aws_deployment import (
        DeploymentManager,
        CloudFormationGenerator,
        LambdaPackager,
        DockerfileGenerator,
    )

    AWS_IMPORTS_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# Convenience API
# =============================================================================

try:
    from .convenience_api import SimpleMAIFAgent, create_agent

    CONVENIENCE_API_AVAILABLE = True
except ImportError:
    CONVENIENCE_API_AVAILABLE = False

try:
    from .migration_tools import VectorDBMigrator, migrate_to_maif

    MIGRATION_TOOLS_AVAILABLE = True
except ImportError:
    MIGRATION_TOOLS_AVAILABLE = False

try:
    from .debug_tools import MAIFDebugger, debug_maif

    DEBUG_TOOLS_AVAILABLE = True
except ImportError:
    DEBUG_TOOLS_AVAILABLE = False

# =============================================================================
# Module Info
# =============================================================================

__version__ = "3.0.0"
__author__ = "MAIF Development Team"
__license__ = "MIT"

__all__ = [
    # Core API
    "MAIFEncoder",
    "MAIFDecoder",
    "MAIFParser",
    "BlockType",
    "MAIFBlock",
    "MAIFVersion",
    "MAIFHeader",
    # Secure format
    "SecureBlock",
    "SecureBlockHeader",
    "SecureFileHeader",
    "ProvenanceEntry",
    "FileFooter",
    "BlockFlags",
    "FileFlags",
    # Legacy aliases
    "SecureMAIFWriter",
    "SecureMAIFReader",
    "SecureBlockType",
    # Convenience functions
    "create_maif",
    "verify_maif",
    "quick_create",
    "quick_verify",
    "quick_read",
    # Security
    "MAIFSigner",
    "MAIFVerifier",
    # Privacy
    "PrivacyEngine",
    "PrivacyPolicy",
    "PrivacyLevel",
    "EncryptionMode",
    "AccessRule",
    # Semantics
    "SemanticEmbedder",
    "SemanticEmbedding",
    "KnowledgeTriple",
    "KnowledgeGraphBuilder",
    # Forensics & Validation
    "ForensicAnalyzer",
    "MAIFValidator",
    # Streaming
    "MAIFStreamReader",
    "MAIFStreamWriter",
    # Agent
    "MAIFAgent",
    # Production
    "HealthChecker",
    "RateLimiter",
    "MetricsAggregator",
    "CostTracker",
    "BatchProcessor",
    # Constants
    "MAGIC_HEADER",
    "MAGIC_FOOTER",
    "FORMAT_VERSION_MAJOR",
    "FORMAT_VERSION_MINOR",
    # Feature flags
    "ENHANCED_ALGORITHMS_AVAILABLE",
    "AWS_IMPORTS_AVAILABLE",
    "CONVENIENCE_API_AVAILABLE",
]
