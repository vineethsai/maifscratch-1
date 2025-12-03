# Advanced MAIF Examples

Complex examples demonstrating advanced features including multi-agent systems, lifecycle management, and novel algorithms.

## Overview

These examples showcase:
- Multi-agent collaboration patterns
- Lifecycle management (merge, split, archive)
- Novel AI algorithms (ACAM, HSC, CSB)
- State restoration and persistence
- Video and multimodal processing
- Version management
- Cross-backend operations

## Examples

### multi_agent_consortium_demo.py

Complete multi-agent collaboration system with specialized agents working together.

**Scenario**: Multiple agents (Geography, Cultural, Logistics, Safety, Coordinator) collaborate to answer complex queries.

**Features:**
- Agent specialization and role assignment
- Shared MAIF artifact for coordination
- Dependency management between agents
- Iterative refinement across agents
- Forensic analysis of collaboration
- Version tracking per agent contribution

**Run:**
```bash
python3 multi_agent_consortium_demo.py
```

**Key concepts:**
```python
# Each agent contributes to shared MAIF
class GeographyAgent(BaseAgent):
    def contribute(self, query, context):
        analysis = self.analyze(query)
        self.maif.add_text(analysis, metadata={
            "agent": self.agent_id,
            "contribution_type": "geography"
        })
        return contribution

# Coordinator synthesizes all contributions
coordinator.synthesize_contributions(all_agent_outputs)
```

### lifecycle_management_demo.py

Demonstrates MAIF artifact lifecycle operations.

**Features:**
- Merging multiple MAIF files
- Splitting large artifacts
- Archiving old data
- Self-governance rules
- Optimization based on access patterns
- Deduplication strategies

**Run:**
```bash
python3 lifecycle_management_demo.py
```

**Key operations:**
```python
from maif.lifecycle_management import MAIFMerger, MAIFSplitter

# Merge multiple artifacts
merger = MAIFMerger()
stats = merger.merge(
    maif_paths=["file1.maif", "file2.maif"],
    output_path="merged.maif",
    merge_strategy="semantic",  # or "append", "temporal"
    deduplication=True
)

# Split large artifact
splitter = MAIFSplitter()
split_files = splitter.split(
    "large.maif",
    strategy="size",  # or "semantic", "temporal"
    max_size_mb=100
)
```

### novel_algorithms_demo.py

Demonstrates three novel AI algorithms implemented in MAIF.

**Algorithms:**

1. **ACAM (Adaptive Cross-Modal Attention)**
   - Computes attention weights across modalities
   - Adaptive weighting based on content
   - Supports text, image, audio, video

2. **HSC (Hierarchical Semantic Compression)**
   - 3-tier compression (coarse, medium, fine)
   - Achieves 60% compression with 95% fidelity
   - Preserves semantic relationships

3. **CSB (Cryptographic Semantic Binding)**
   - Binds embeddings to source data cryptographically
   - Prevents embedding manipulation
   - Verifiable semantic commitments

**Run:**
```bash
python3 novel_algorithms_demo.py
```

**Example usage:**
```python
from maif.semantic_optimized import (
    AdaptiveCrossModalAttention,
    HierarchicalSemanticCompression,
    CryptographicSemanticBinding
)

# ACAM
acam = AdaptiveCrossModalAttention(embedding_dim=384)
weights = acam.compute_attention_weights({
    'text': text_embeddings,
    'image': image_embeddings
})

# HSC
hsc = HierarchicalSemanticCompression(target_compression_ratio=0.4)
compressed = hsc.compress_embeddings(embeddings)

# CSB
csb = CryptographicSemanticBinding()
binding = csb.create_semantic_commitment(embedding, source_data)
verified = csb.verify_semantic_binding(embedding, source_data, binding)
```

### agent_state_restoration_demo.py

Shows how to save and restore complete agent state.

**Features:**
- Full agent state serialization
- State restoration from MAIF
- Checkpoint creation
- Recovery from failures
- State versioning

**Run:**
```bash
python3 agent_state_restoration_demo.py
```

### video_demo.py

Video processing with semantic analysis.

**Features:**
- Video metadata extraction
- Frame analysis
- Semantic embeddings for video content
- Efficient storage with compression
- Searchable video content

**Run:**
```bash
python3 video_demo.py
```

### versioning_demo.py

Version management and history tracking.

**Features:**
- Block-level versioning
- Version history queries
- Rollback capabilities
- Change tracking per agent
- Forensic analysis of changes

**Run:**
```bash
python3 versioning_demo.py
```

## Advanced Concepts

### Multi-Agent Coordination

Agents coordinate via shared MAIF artifacts:

```python
# Agent 1 writes
agent1_maif.add_text(result, metadata={"agent": "agent_1"})

# Agent 2 reads and extends
agent2_maif = load_maif("shared.maif")
previous_results = agent2_maif.get_blocks_by_agent("agent_1")
agent2_maif.add_text(extended_result, metadata={"agent": "agent_2"})
```

### Lifecycle Policies

Define self-governance rules:

```python
from maif.lifecycle_management import GovernanceRule

rule = GovernanceRule(
    rule_id="archive_old_data",
    condition="age_days > 90 and access_frequency < 0.1",
    action="archive",
    priority=10
)

lifecycle_manager.add_rule(rule)
lifecycle_manager.apply_policies()
```

### Semantic Compression

Reduce storage while preserving meaning:

```python
# Original: 1000 embeddings × 384 dims = 384,000 floats
hsc = HierarchicalSemanticCompression(compression_levels=3)
compressed = hsc.compress_embeddings(embeddings)

# Result: ~150,000 floats (60% reduction)
# Fidelity: 95% (measured by cosine similarity)
```

## Performance Considerations

### Memory Usage
- Video processing: ~500MB per video
- Large merges: 2x source file size temporarily
- Compression: 1.5x input size during processing

### Processing Time
- ACAM computation: O(n²) where n = number of modalities
- HSC compression: ~1s per 1000 embeddings
- CSB binding: ~100ms per embedding
- Merge operation: ~1s per MB of data

### Optimization Tips
1. Use streaming for large files
2. Enable compression for embeddings
3. Batch operations when possible
4. Use memory-mapped I/O for reads

## Requirements

### Core Dependencies
```bash
pip install numpy cryptography pydantic
```

### Optional Dependencies
```bash
# For semantic features
pip install sentence-transformers scipy

# For video processing
pip install opencv-python pillow

# For compression
pip install brotli zstandard lz4

# For performance
pip install numba xxhash
```

Install all:
```bash
pip install -e ../../[full]
```

## Testing

Run advanced feature tests:
```bash
cd ../../tests
pytest test_lifecycle.py test_semantic.py test_multi_agent.py -v
```

## Production Considerations

### Multi-Agent Systems
- Use distributed coordination for >10 agents
- Implement conflict resolution strategies
- Monitor agent performance metrics
- Set timeouts for agent operations

### Lifecycle Management
- Schedule regular optimization runs
- Monitor artifact sizes
- Implement retention policies
- Test restore procedures

### Algorithm Usage
- Profile performance for your data
- Tune compression ratios
- Validate semantic fidelity
- Monitor memory usage

## Troubleshooting

### High Memory Usage
- Enable streaming mode
- Reduce batch sizes
- Use compression
- Process in chunks

### Slow Performance
- Check for unnecessary copies
- Enable memory-mapped I/O
- Use batch operations
- Profile with `cProfile`

### Merge Conflicts
- Use semantic merge strategy
- Implement conflict resolution
- Validate merged output
- Keep backup copies

## Next Steps

After exploring these examples:

1. **Integrate into your system** - Adapt patterns to your use case
2. **Optimize for your data** - Tune parameters and strategies
3. **Add monitoring** - Track performance and errors
4. **Deploy to production** - Follow deployment guides in docs

## Additional Resources

- Architecture Guide: `../../docs/guide/architecture.md`
- Performance Guide: `../../docs/guide/performance.md`
- Distributed Systems: `../../docs/guide/distributed.md`
- Algorithm Details: `../../docs/NOVEL_ALGORITHMS_IMPLEMENTATION.md`
- API Reference: `../../docs/api/`

## Contributing

To add new advanced examples:
1. Follow existing example structure
2. Include comprehensive docstrings
3. Add error handling
4. Provide usage examples
5. Update this README

