# Semantic Understanding

MAIF provides semantic processing capabilities through embeddings, cross-modal attention, and knowledge graph features. This guide covers MAIF's semantic processing features.

## Overview

MAIF's semantic capabilities include:

- **Embeddings**: Vector representations for content
- **Cross-Modal Attention**: ACAM algorithm for multimodal understanding
- **Semantic Compression**: HSC for efficient storage
- **Knowledge Graphs**: Relationship extraction and storage

## Semantic Embedder

Generate embeddings for text and other content:

```python
from maif.semantic import SemanticEmbedder, SemanticEmbedding

# Create embedder
embedder = SemanticEmbedder()

# Generate embedding for text
text = "Machine learning is transforming industries"
embedding = embedder.embed(text)

print(f"Embedding dimension: {len(embedding.vector)}")
```

### Storing Embeddings

```python
from maif.core import MAIFEncoder

encoder = MAIFEncoder(agent_id="semantic-demo")

# Add text with embeddings
text = "Important document content"
encoder.add_text_block(text)

# Add pre-computed embeddings
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
encoder.add_embeddings_block(
    embeddings,
    metadata={"model": "custom", "dimension": 3}
)

encoder.save("semantic.maif")
```

### Using the Simple API

```python
from maif_api import create_maif

maif = create_maif("embedding-agent")

# Add embeddings
embeddings = [
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8]
]
maif.add_embeddings(embeddings, model_name="my-model", compress=True)

maif.save("embeddings.maif")
```

## Cross-Modal Attention (ACAM)

The Adaptive Cross-Modal Attention Mechanism enables understanding across different data types:

```python
from maif.semantic import CrossModalAttention

# Create attention mechanism
acam = CrossModalAttention()

# Process multimodal content
# ACAM computes attention weights between modalities
```

### Enhanced ACAM

```python
from maif.semantic_optimized import AdaptiveCrossModalAttention, AttentionWeights

# Enhanced ACAM with adaptive features
acam = AdaptiveCrossModalAttention(
    num_heads=8,
    temperature=0.1
)

# Get attention weights
weights = acam.compute_attention(query_embeddings, key_embeddings)
```

### Multimodal Content with ACAM

```python
from maif_api import create_maif

maif = create_maif("multimodal-agent")

# Add multimodal content with ACAM processing
maif.add_multimodal({
    "text": "A sunset over mountains",
    "description": "Nature photography",
    "tags": ["sunset", "mountains"]
}, title="Sunset Scene", use_acam=True)

maif.save("multimodal.maif")
```

## Semantic Compression (HSC)

Hierarchical Semantic Compression reduces embedding storage:

```python
from maif.semantic import HierarchicalSemanticCompression

# Create compressor
hsc = HierarchicalSemanticCompression()

# Compress embeddings
original_embeddings = [[0.1, 0.2, ...], ...]
compressed = hsc.compress(original_embeddings)

# Decompress when needed
decompressed = hsc.decompress(compressed)
```

### Enhanced HSC

```python
from maif.semantic_optimized import HierarchicalSemanticCompression as EnhancedHSC

# Enhanced compression with better semantic preservation
hsc = EnhancedHSC(
    compression_ratio=0.1,
    preserve_semantics=True
)
```

### Compressed Embeddings in MAIF

```python
from maif_api import create_maif

maif = create_maif("compressed-agent")

# Store compressed embeddings
embeddings = [[0.1, 0.2, 0.3] for _ in range(100)]
maif.add_embeddings(embeddings, model_name="bert", compress=True)

maif.save("compressed.maif")
```

## Knowledge Graphs

Build and query knowledge graphs from content:

```python
from maif.semantic import KnowledgeGraphBuilder, KnowledgeTriple

# Build knowledge graph
kg_builder = KnowledgeGraphBuilder()

# Add knowledge triples
triple = KnowledgeTriple(
    subject="Python",
    predicate="is_a",
    object="Programming Language"
)
kg_builder.add_triple(triple)

# Query relationships
related = kg_builder.get_related("Python")
```

## Cryptographic Semantic Binding (CSB)

Bind semantic content with cryptographic verification:

```python
from maif.semantic import CryptographicSemanticBinding

# Create CSB instance
csb = CryptographicSemanticBinding()

# Bind content with cryptographic proof
bound_content = csb.bind(content, embedding)

# Verify binding
is_valid = csb.verify(bound_content)
```

### Enhanced CSB

```python
from maif.semantic_optimized import CryptographicSemanticBinding as EnhancedCSB

csb = EnhancedCSB(
    security_level="high"
)
```

## Deep Semantic Understanding

Advanced semantic analysis:

```python
from maif.semantic import DeepSemanticUnderstanding

# Create deep semantic analyzer
dsu = DeepSemanticUnderstanding()

# Analyze content
analysis = dsu.analyze("Complex technical document...")

# Get semantic features
features = analysis.features
concepts = analysis.concepts
relationships = analysis.relationships
```

## Working with Embeddings

### Reading Embeddings

```python
from maif.core import MAIFDecoder

decoder = MAIFDecoder("embeddings.maif")

# Get embedding blocks
embedding_blocks = decoder.get_blocks_by_type("EMBD")
for block in embedding_blocks:
    print(f"Block: {block.block_id}")
    print(f"Metadata: {block.metadata}")
```

### Searching with Embeddings

```python
from maif_api import load_maif

maif = load_maif("content.maif")

# Search content (uses embeddings if available)
results = maif.search("machine learning", top_k=5)
for result in results:
    print(f"Score: {result.get('score', 0):.3f}")
    print(f"Text: {result['text'][:100]}...")
```

## Complete Example

```python
from maif.core import MAIFEncoder
from maif.semantic import (
    SemanticEmbedder,
    CrossModalAttention,
    KnowledgeGraphBuilder,
    KnowledgeTriple
)

# Create encoder
encoder = MAIFEncoder(agent_id="semantic-demo")

# Create semantic components
embedder = SemanticEmbedder()
kg_builder = KnowledgeGraphBuilder()

# Add text with embeddings
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks are part of deep learning"
]

for doc in documents:
    # Add text block
    encoder.add_text_block(doc)

# Add knowledge triples
kg_builder.add_triple(KnowledgeTriple(
    "Python", "is_a", "Language"
))
kg_builder.add_triple(KnowledgeTriple(
    "Machine Learning", "uses", "Algorithms"
))

# Save
encoder.save("semantic_demo.maif")
print("Semantic artifact created!")
```

## Available Semantic Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `SemanticEmbedder` | `maif.semantic` | Generate embeddings |
| `SemanticEmbedding` | `maif.semantic` | Embedding data structure |
| `CrossModalAttention` | `maif.semantic` | Multi-modal attention |
| `HierarchicalSemanticCompression` | `maif.semantic` | Embedding compression |
| `CryptographicSemanticBinding` | `maif.semantic` | Secure semantic binding |
| `DeepSemanticUnderstanding` | `maif.semantic` | Advanced analysis |
| `KnowledgeGraphBuilder` | `maif.semantic` | Knowledge graph construction |
| `KnowledgeTriple` | `maif.semantic` | Knowledge representation |

### Enhanced Components

| Component | Module | Purpose |
|-----------|--------|---------|
| `AdaptiveCrossModalAttention` | `maif.semantic_optimized` | Enhanced ACAM |
| `AttentionWeights` | `maif.semantic_optimized` | Attention data |
| `EnhancedHSC` | `maif.semantic_optimized` | Improved compression |
| `EnhancedCSB` | `maif.semantic_optimized` | Enhanced binding |

## Best Practices

1. **Use compression for large embedding sets** to reduce storage
2. **Enable ACAM for multimodal content** for better cross-modal understanding
3. **Build knowledge graphs** for structured relationship storage
4. **Verify CSB bindings** when security is critical

## Next Steps

- **[Multimodal Data →](/guide/multimodal)** - Working with multiple data types
- **[Performance →](/guide/performance)** - Optimization techniques
- **[API Reference →](/api/)** - Complete API documentation
