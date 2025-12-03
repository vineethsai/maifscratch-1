"""
MAIF (Multimodal Artifact File Format) Library

A comprehensive library for creating, managing, and analyzing MAIF files.
MAIF is an AI-native file format designed for multimodal content with
embedded security, semantics, and provenance tracking.

Production-ready with seamless AWS integration.
"""

from .core import MAIFEncoder, MAIFDecoder, MAIFParser, MAIFBlock, MAIFVersion
from .security import MAIFSigner, MAIFVerifier
from .privacy import PrivacyEngine, PrivacyPolicy, PrivacyLevel, EncryptionMode, AccessRule, DifferentialPrivacy, SecureMultipartyComputation, ZeroKnowledgeProof
from .semantic import (
    SemanticEmbedder, SemanticEmbedding, KnowledgeTriple,
    CrossModalAttention, HierarchicalSemanticCompression,
    CryptographicSemanticBinding, DeepSemanticUnderstanding,
    KnowledgeGraphBuilder
)

# Import enhanced algorithms from semantic_optimized
try:
    from .semantic_optimized import (
        AdaptiveCrossModalAttention,
        HierarchicalSemanticCompression as EnhancedHierarchicalSemanticCompression,
        CryptographicSemanticBinding as EnhancedCryptographicSemanticBinding,
        AttentionWeights
    )
    ENHANCED_ALGORITHMS_AVAILABLE = True
except ImportError:
    ENHANCED_ALGORITHMS_AVAILABLE = False

from .forensics import ForensicAnalyzer, ForensicEvidence
from .compression_manager import CompressionManager
from .compression import CompressionMetadata
from .binary_format import MAIFBinaryParser, MAIFBinaryWriter
from .validation import MAIFValidator, MAIFRepairTool
from .metadata import MAIFMetadataManager
from .streaming import MAIFStreamReader, MAIFStreamWriter
from .integration_enhanced import EnhancedMAIFProcessor, ConversionResult

# Agent Framework
from .agentic_framework import MAIFAgent, PerceptionSystem, ReasoningSystem, ExecutionSystem

# AWS Integrations removed - core library now AWS-independent
# AWS integrations - imported conditionally to avoid errors when AWS is not needed
try:
    from .aws_lambda_integration import AWSLambdaIntegration
    from .aws_stepfunctions_integration import AWSStepFunctionsIntegration
    from .aws_xray_integration import MAIFXRayIntegration, xray_trace, xray_subsegment
    from .aws_deployment import DeploymentManager, CloudFormationGenerator, LambdaPackager, DockerfileGenerator
    AWS_IMPORTS_AVAILABLE = True
except ImportError:
    # AWS features not available
    AWS_IMPORTS_AVAILABLE = False

# Production Features
from .health_check import HealthChecker, HealthStatus
from .rate_limiter import RateLimiter, RateLimitConfig, CostBasedRateLimiter, rate_limit
from .metrics_aggregator import MetricsAggregator, MAIFMetrics, initialize_metrics, get_metrics
from .cost_tracker import CostTracker, Budget, BudgetExceededException, initialize_cost_tracking, get_cost_tracker, with_cost_tracking
from .batch_processor import BatchProcessor, StreamBatchProcessor, DistributedBatchProcessor, batch_process
# API Gateway integration removed (was AWS-dependent)

# Advanced Features
# from .multi_agent import MAIFAgentConsortium  # Disabled due to import issues

# Import simple API for easy access
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from maif_api import MAIF, create_maif, load_maif, quick_text_maif, quick_multimodal_maif
    SIMPLE_API_AVAILABLE = True
except ImportError:
    SIMPLE_API_AVAILABLE = False

# Import convenience features
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

__version__ = "2.0.0"
__author__ = "MAIF Development Team"
__license__ = "MIT"

__all__ = [
    # Core functionality
    'MAIFEncoder',
    'MAIFDecoder',
    'MAIFParser',
    'MAIFBlock',
    'MAIFVersion',
    
    # Security
    'MAIFSigner',
    'MAIFVerifier',
    
    # Privacy
    'PrivacyEngine',
    'PrivacyPolicy',
    'PrivacyLevel',
    'EncryptionMode',
    'AccessRule',
    'DifferentialPrivacy',
    'SecureMultipartyComputation',
    'ZeroKnowledgeProof',
    
    # Semantics
    'SemanticEmbedder',
    'SemanticEmbedding',
    'KnowledgeTriple',
    'CrossModalAttention',
    'HierarchicalSemanticCompression',
    'CryptographicSemanticBinding',
    'DeepSemanticUnderstanding',
    'KnowledgeGraphBuilder',
    
    # Enhanced Novel Algorithms (if available)
    'AdaptiveCrossModalAttention',
    'EnhancedHierarchicalSemanticCompression',
    'EnhancedCryptographicSemanticBinding',
    'AttentionWeights',
    'ENHANCED_ALGORITHMS_AVAILABLE',
    
    # Forensics
    'ForensicAnalyzer',
    'ForensicEvidence',
    
    # Compression
    'CompressionManager',
    'CompressionMetadata',
    
    # Binary Format
    'MAIFBinaryParser',
    'MAIFBinaryWriter',
    
    # Validation
    'MAIFValidator',
    'MAIFRepairTool',
    
    # Metadata
    'MAIFMetadataManager',
    
    # Streaming
    'MAIFStreamReader',
    'MAIFStreamWriter',
    
    # Integration
    'EnhancedMAIFProcessor',
    'ConversionResult',
    
    # Agent Framework
    'MAIFAgent',
    'PerceptionSystem',
    'ReasoningSystem',
    'ExecutionSystem',
    
    # AWS Decorators
    'maif_agent',
    'aws_agent',
    'aws_bedrock',
    'aws_kms',
    'aws_s3',
    'aws_dynamodb',
    'aws_lambda',
    'aws_stepfunctions',
    
    # AWS Integrations
    'AWSTrustEngine',
    'AWSKMSIntegration',
    'AWSS3Integration',
    'AWSDynamoDBIntegration',
    'AWSLambdaIntegration',
    'AWSStepFunctionsIntegration',
    'MAIFXRayIntegration',
    'xray_trace',
    'xray_subsegment',
    
    # Deployment Tools
    'DeploymentManager',
    'CloudFormationGenerator',
    'LambdaPackager',
    'DockerfileGenerator',
    
    # Production Features
    'HealthChecker',
    'HealthStatus',
    'ComponentHealth',
    'RateLimiter',
    'RateLimitConfig',
    'CostBasedRateLimiter',
    'rate_limit',
    'MetricsAggregator',
    'MAIFMetrics',
    'initialize_metrics',
    'get_metrics',
    'CostTracker',
    'Budget',
    'BudgetExceededException',
    'initialize_cost_tracking',
    'get_cost_tracker',
    'with_cost_tracking',
    'BatchProcessor',
    'StreamBatchProcessor',
    'DistributedBatchProcessor',
    'batch_process',
    'APIGatewayIntegration',
    'APIGatewayHandler',
    'api_endpoint',
    
    # Advanced Features
    'BedrockAgentSwarm',
    'BedrockModelProvider',
    'MAIFAgentConsortium',
    
    # Simple API (if available)
    'MAIF',
    'create_maif',
    'load_maif',
    'quick_text_maif',
    'quick_multimodal_maif',
    'SIMPLE_API_AVAILABLE',
]

# Convenience functions for production use
def create_production_agent(name: str, use_aws: bool = True, **kwargs):
    """Create a production-ready MAIF agent with AWS integration."""
    @maif_agent(use_aws=use_aws, **kwargs)
    class ProductionAgent(MAIFAgent):
        pass
    
    return ProductionAgent(agent_id=name)

def initialize_production_monitoring(namespace: str = "MAIF/Production"):
    """Initialize all production monitoring systems."""
    metrics = initialize_metrics(namespace=namespace)
    tracker = initialize_cost_tracking()
    return metrics, tracker