# MAIF (Multimodal Artifact File Format)

## Overview

MAIF is a container format and SDK for AI agent data persistence. It provides cryptographically-secure, auditable data structures for multi-agent AI systems, designed to support compliance with security standards such as FIPS 140-2 and DISA STIG.

[![Implementation Status](https://img.shields.io/badge/Status-Active-green.svg)](https://github.com/vineethsai/maifscratch-1)
[![Security Model](https://img.shields.io/badge/Security-FIPS%20140--2%20Compliant-red.svg)](#security-features)
[![AWS Integration](https://img.shields.io/badge/AWS-KMS%20%26%20CloudWatch-yellow.svg)](#aws-integration)

## Technical Architecture

### Core Components

1. **Container Format** ([`maif/core.py`](maif/core.py), [`maif/binary_format.py`](maif/binary_format.py))
   - ISO BMFF-inspired hierarchical block structure
   - FourCC block type identifiers
   - Memory-mapped I/O for efficient access
   - Progressive loading with streaming support

2. **Security Layer** ([`maif/security.py`](maif/security.py))
   - AWS KMS integration for key management
   - FIPS 140-2 compliant encryption (AES-256-GCM)
   - Digital signatures using RSA/ECDSA
   - Cryptographic provenance chains
   - Mandatory encryption (no plaintext fallback)

3. **Compliance Logging** ([`maif/compliance_logging.py`](maif/compliance_logging.py))
   - STIG/FIPS validation framework
   - SIEM integration (CloudWatch, Splunk, Elasticsearch)
   - Tamper-evident audit trails using hash chains
   - Support for HIPAA, FISMA, PCI-DSS compliance frameworks

### Performance Characteristics

- **Semantic Search**: ~30ms response time for 1M+ vectors
- **Compression Ratio**: Up to 64× using hierarchical semantic compression
- **Hash Verification**: >500 MB/s throughput
- **Memory Efficiency**: 64KB minimum buffer with streaming

## Installation

```bash
# Basic installation
pip install -e .

# With AWS support
pip install -e ".[aws]"

# Full installation with all dependencies
pip install -e ".[full]"
```

## Quick Start

### Basic File Operations

```python
from maif.core import MAIFEncoder, MAIFDecoder
from maif.block_types import BlockType

# Create MAIF file
encoder = MAIFEncoder()
encoder.add_block(
    block_type=BlockType.TEXT,
    data=b"Agent conversation data",
    metadata={"agent_id": "agent-001", "timestamp": 1234567890}
)
encoder.save("agent_data.maif")

# Read MAIF file
decoder = MAIFDecoder("agent_data.maif")
for block in decoder.read_blocks():
    print(f"Type: {block.block_type}, Size: {len(block.data)}")
```

### Security Configuration

```python
from maif.security import SecurityManager

# Initialize with KMS
security = SecurityManager(
    use_kms=True,
    kms_key_id="arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
    region_name="us-east-1",
    require_encryption=True  # Fail if encryption unavailable
)

# Encrypt data
encrypted = security.encrypt_data(b"sensitive data")

# Decrypt data
decrypted = security.decrypt_data(encrypted)
```

### Compliance Logging

```python
from maif.compliance_logging import EnhancedComplianceLogger, ComplianceLevel, ComplianceFramework
from maif.compliance_logging import AuditEventType

# Configure compliance logger
logger = EnhancedComplianceLogger(
    compliance_level=ComplianceLevel.FIPS_140_2,
    frameworks=[ComplianceFramework.FISMA, ComplianceFramework.DISA_STIG],
    siem_config={
        'provider': 'cloudwatch',
        'log_group': '/aws/maif/compliance',
        'region': 'us-east-1'
    }
)

# Log security event
event = logger.log_event(
    event_type=AuditEventType.DATA_ACCESS,
    action="read_classified_data",
    user_id="analyst-001",
    resource_id="doc-456",
    classification="SECRET",
    success=True
)
```

## Core Features

### Security & Compliance

- **FIPS 140-2 Compliant**: Uses FIPS-approved cryptographic algorithms.
- **AWS KMS Integration**: Key management via AWS KMS.
- **Digital Signatures**: RSA/ECDSA with provenance chains.
- **Access Control**: Permissions with expiry.
- **Audit Trails**: Operation history with SIEM integration.

### AI Algorithms

- **ACAM**: Adaptive Cross-Modal Attention.
- **HSC**: Hierarchical Semantic Compression.
- **CSB**: Cryptographic Semantic Binding.

### Architecture Features

- **Multimodal Support**: Text, embeddings, images, video, knowledge graphs.
- **Streaming Architecture**: Memory-mapped access with progressive loading.
- **Block-Level Versioning**: Append-only architecture with version tracking.
- **Self-Describing Format**: Metadata for interpretation.

## Advanced Usage

### Semantic Operations

```python
from maif.semantic_optimized import HierarchicalSemanticCompression, AdaptiveCrossModalAttention

# Semantic compression
hsc = HierarchicalSemanticCompression(compression_levels=3)
compressed = hsc.compress_embeddings(embeddings)
print(f"Compression ratio: {compressed.compression_ratio:.2f}x")

# Cross-modal attention
acam = AdaptiveCrossModalAttention(
    modalities=['text', 'image', 'audio'],
    attention_heads=8
)
attended = acam.forward(multimodal_data)
```

### Block Storage

```python
from maif.block_storage import BlockStorage, BlockType

# Create block storage
storage = BlockStorage("data.maif", enable_mmap=True)

# Add blocks with versioning
block_id = storage.add_block(
    block_type=BlockType.EMBEDDINGS,
    data=embeddings_data,
    metadata={"model": "text-embedding-ada-002"}
)

# Query blocks
blocks = storage.query_blocks(
    block_type=BlockType.EMBEDDINGS,
    metadata_filter=lambda m: m.get("model") == "text-embedding-ada-002"
)
```

## Agentic Framework

MAIF includes a framework for building autonomous AI agents, with optional AWS integration.

### Core Agent Architecture

```python
from maif.agentic_framework import MAIFAgent

# Create an autonomous agent
agent = MAIFAgent(
    agent_id="medical-assistant-001",
    workspace_path="/data/agents/medical",
    use_aws=True,
    config={
        "memory_size": 10000,
        "learning_rate": 0.1
    }
)

await agent.initialize()
```

### Multi-Agent Orchestration

Coordinate multiple agents:

```python
from maif.agentic_framework import MultiAgentOrchestrator

# Create orchestrator
orchestrator = MultiAgentOrchestrator(agent)

# Define multi-agent task
task = {
    'id': 'diagnosis-001',
    'description': 'Collaborative medical diagnosis',
    'requirements': ['medical-knowledge', 'diagnostic-skills'],
    'priority': 10,
    'strategy': 'consensus'
}

# Execute
result = await orchestrator.execute_multi_agent_task(task)
```

### AWS Bedrock Swarm Integration

When AWS is enabled, the framework can leverage Bedrock:

```python
from maif.bedrock_swarm import BedrockAgentSwarm

swarm = BedrockAgentSwarm(
    max_agents=10,
    enable_consensus=True,
    enable_security=True,
    enable_compliance_logging=True
)

# Execute task across multiple Bedrock models
result = swarm.execute_task(
    task="Analyze patient data for treatment options",
    strategy="consensus",
    agents=["Claude", "Titan", "AI21"]
)
```

## AWS Integration

### Credential Management

MAIF uses a centralized credential system:

```python
from maif.aws_config import configure_aws

# Configure globally
configure_aws(
    environment="production",
    profile="my-aws-profile",
    region="us-east-1"
)
```

Once configured, services like `BedrockClient`, `S3Client`, `DynamoDBClient`, and `KMSClient` automatically use these credentials.

### S3 Integration

```python
from maif.aws_s3_integration import S3Client

s3_client = S3Client()

# Upload MAIF file
s3_client.upload_file(
    local_path="agent_data.maif",
    bucket="my-bucket",
    key="data/agent_data.maif"
)
```

## Security Considerations

1. **Encryption**: Data encrypted using FIPS-approved algorithms.
2. **Key Management**: AWS KMS integration.
3. **Access Control**: IAM-based permissions.
4. **Audit Trail**: Operations logged with cryptographic integrity.
5. **Data Classification**: Support for classification levels.

## Compliance

- **FIPS 140-2**: Uses FIPS-approved algorithms.
- **DISA STIG**: Implements security controls and audit logging.
- **FISMA**: Supports control families and compliance reporting.

## Limitations

1. **Performance**: Semantic search achieves ~30ms.
2. **Compression**: Maximum compression ratio of 64×.
3. **Streaming**: Limited to sequential access patterns.
4. **Scalability**: Single-file format.
5. **Block Size**: Maximum block size of 2GB.

## Contributing

Please ensure all contributions:
1. Include comprehensive unit tests.
2. Pass FIPS compliance validation.
3. Include security impact analysis.
4. Update technical documentation.
5. Follow PEP 8 style guidelines.

## References

- [FIPS 140-2 Standards](https://csrc.nist.gov/publications/detail/fips/140/2/final)
- [DISA STIG Requirements](https://public.cyber.mil/stigs/)
- [NIST 800-53 Controls](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- [ISO BMFF Specification](https://www.iso.org/standard/68960.html)

## License

MIT License - See LICENSE file for details
