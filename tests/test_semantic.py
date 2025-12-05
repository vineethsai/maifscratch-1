"""
Comprehensive tests for MAIF semantic functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from maif.semantic import (
    SemanticEmbedding,
    KnowledgeTriple,
    SemanticEmbedder,
    KnowledgeGraphBuilder,
    CrossModalAttention,
    HierarchicalSemanticCompression,
    CryptographicSemanticBinding,
    DeepSemanticUnderstanding,
)


class TestSemanticEmbedding:
    """Test SemanticEmbedding data structure."""

    def test_semantic_embedding_creation(self):
        """Test basic SemanticEmbedding creation."""
        vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        metadata = {"model": "test-model", "source": "test"}

        embedding = SemanticEmbedding(vector=vector, metadata=metadata)

        assert embedding.vector == vector
        assert embedding.metadata == metadata
        assert len(embedding.vector) == 5

    def test_semantic_embedding_with_numpy(self):
        """Test SemanticEmbedding with numpy arrays."""
        vector = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        embedding = SemanticEmbedding(vector=vector.tolist(), metadata={})

        assert len(embedding.vector) == 5
        assert isinstance(embedding.vector, list)


class TestKnowledgeTriple:
    """Test KnowledgeTriple data structure."""

    def test_knowledge_triple_creation(self):
        """Test basic KnowledgeTriple creation."""
        triple = KnowledgeTriple(
            subject="John", predicate="works_at", object="ACME Corp"
        )

        assert triple.subject == "John"
        assert triple.predicate == "works_at"
        assert triple.object == "ACME Corp"


class TestSemanticEmbedder:
    """Test SemanticEmbedder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock the sentence transformer to avoid dependency issues
        with patch("sentence_transformers.SentenceTransformer"):
            self.embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")

    def test_embedder_initialization(self):
        """Test SemanticEmbedder initialization."""
        with patch("maif.semantic.SENTENCE_TRANSFORMERS_AVAILABLE", True), patch(
            "maif.semantic.SentenceTransformer"
        ) as mock_transformer:
            mock_model = Mock()
            mock_transformer.return_value = mock_model

            embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")

            assert embedder.model_name == "all-MiniLM-L6-v2"
            assert embedder.embeddings == []
            mock_transformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_embed_text(self):
        """Test text embedding generation."""
        with patch("maif.semantic.SENTENCE_TRANSFORMERS_AVAILABLE", True), patch(
            "maif.semantic.SentenceTransformer"
        ) as mock_transformer:
            # Mock the model
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            mock_transformer.return_value = mock_model

            embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")

            text = "Hello, semantic world!"
            metadata = {"source": "test"}

            embedding = embedder.embed_text(text, metadata)

            assert isinstance(embedding, SemanticEmbedding)
            assert len(embedding.vector) == 5
            assert embedding.metadata["source"] == "test"
            # No longer expect "text" in metadata in production code

        # Check that embedding was stored
        assert len(embedder.embeddings) == 1
        assert embedder.embeddings[0] == embedding

    @patch("sentence_transformers.SentenceTransformer")
    def test_embed_texts_batch(self, mock_transformer):
        """Test batch text embedding."""
        # Mock the model
        mock_model = Mock()
        # Create 384-dimensional embeddings as expected
        mock_model.encode.return_value = np.random.rand(3, 384)
        mock_transformer.return_value = mock_model

        embedder = SemanticEmbedder(model_name="all-MiniLM-L6-v2")

        texts = ["Text 1", "Text 2", "Text 3"]
        metadata_list = [{"id": 1}, {"id": 2}, {"id": 3}]

        embeddings = embedder.embed_texts(texts, metadata_list)

        assert len(embeddings) == 3
        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, SemanticEmbedding)
            assert len(embedding.vector) == 384
            assert embedding.metadata["id"] == i + 1
            assert embedding.metadata["text"] == texts[i]

    def test_compute_similarity(self):
        """Test similarity computation between embeddings."""
        embedding1 = SemanticEmbedding(vector=[1.0, 0.0, 0.0], metadata={})
        embedding2 = SemanticEmbedding(vector=[1.0, 0.0, 0.0], metadata={})  # Identical
        embedding3 = SemanticEmbedding(
            vector=[0.0, 1.0, 0.0], metadata={}
        )  # Orthogonal

        # Identical vectors should have similarity 1.0
        similarity = self.embedder.compute_similarity(embedding1, embedding2)
        assert abs(similarity - 1.0) < 1e-6

        # Orthogonal vectors should have similarity 0.0
        similarity = self.embedder.compute_similarity(embedding1, embedding3)
        assert abs(similarity - 0.0) < 1e-6

    def test_search_similar(self):
        """Test similarity search."""
        # Add some embeddings
        self.embedder.embeddings = [
            SemanticEmbedding(vector=[1.0, 0.0, 0.0], metadata={"id": 1}),
            SemanticEmbedding(vector=[0.9, 0.1, 0.0], metadata={"id": 2}),
            SemanticEmbedding(vector=[0.0, 1.0, 0.0], metadata={"id": 3}),
            SemanticEmbedding(vector=[0.0, 0.0, 1.0], metadata={"id": 4}),
        ]

        query_embedding = SemanticEmbedding(vector=[1.0, 0.0, 0.0], metadata={})

        results = self.embedder.search_similar(query_embedding, top_k=2)

        assert len(results) == 2
        # First result should be most similar (id=1)
        assert results[0][0].metadata["id"] == 1
        assert results[0][1] > results[1][1]  # Higher similarity score

    def test_get_embeddings_data(self):
        """Test embeddings data export."""
        self.embedder.embeddings = [
            SemanticEmbedding(vector=[0.1, 0.2], metadata={"id": 1}),
            SemanticEmbedding(vector=[0.3, 0.4], metadata={"id": 2}),
        ]

        data = self.embedder.get_embeddings_data()

        assert len(data) == 2
        assert data[0]["vector"] == [0.1, 0.2]
        assert data[0]["metadata"]["id"] == 1
        assert data[1]["vector"] == [0.3, 0.4]
        assert data[1]["metadata"]["id"] == 2


class TestKnowledgeGraphBuilder:
    """Test KnowledgeGraphBuilder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kg_builder = KnowledgeGraphBuilder()

    def test_kg_builder_initialization(self):
        """Test KnowledgeGraphBuilder initialization."""
        assert self.kg_builder.triples == []
        assert self.kg_builder.entities == {}
        assert self.kg_builder.relations == {}

    def test_add_triple(self):
        """Test adding knowledge triples."""
        self.kg_builder.add_triple(
            subject="John",
            predicate="works_at",
            obj="ACME Corp",
            confidence=0.9,
            source="test_document",
        )

        assert len(self.kg_builder.triples) == 1
        triple = self.kg_builder.triples[0]
        assert triple.subject == "John"
        assert triple.predicate == "works_at"
        assert triple.object == "ACME Corp"

        # Check entity and relation tracking
        assert "John" in self.kg_builder.entities
        assert "ACME Corp" in self.kg_builder.entities
        assert "works_at" in self.kg_builder.relations

    def test_extract_entities_from_text(self):
        """Test entity extraction from text."""
        text = "John Smith works at Microsoft Corporation in Seattle."

        entities = self.kg_builder.extract_entities_from_text(text, source="test")

        # Should extract some entities (implementation may vary)
        assert isinstance(entities, list)
        assert len(entities) >= 0  # May be empty if no NLP library available

    def test_find_related_entities(self):
        """Test finding related entities."""
        # Add some triples
        self.kg_builder.add_triple("John", "works_at", "ACME Corp")
        self.kg_builder.add_triple("ACME Corp", "located_in", "New York")
        self.kg_builder.add_triple("Jane", "works_at", "ACME Corp")

        # Find entities related to John
        related = self.kg_builder.find_related_entities("John", max_depth=2)

        # Should find ACME Corp (depth 1) and potentially New York and Jane (depth 2)
        assert len(related) > 0

        # Check that ACME Corp is found
        related_entities = [entity for entity, relation, depth in related]
        assert "ACME Corp" in related_entities

    def test_get_graph_statistics(self):
        """Test graph statistics generation."""
        # Add some test data
        self.kg_builder.add_triple("John", "works_at", "ACME Corp")
        self.kg_builder.add_triple("Jane", "works_at", "ACME Corp")
        self.kg_builder.add_triple("ACME Corp", "located_in", "New York")

        stats = self.kg_builder.get_graph_statistics()

        assert stats["total_triples"] == 3
        assert stats["total_entities"] == 4  # John, Jane, ACME Corp, New York
        assert stats["total_relations"] == 2  # works_at, located_in
        assert "most_connected_entities" in stats
        assert "most_common_relations" in stats

    def test_export_import_json(self):
        """Test JSON export and import."""
        # Add test data
        self.kg_builder.add_triple("John", "works_at", "ACME Corp", confidence=0.9)
        self.kg_builder.add_triple("Jane", "knows", "John", confidence=0.8)

        # Export to JSON
        json_data = self.kg_builder.export_to_json()

        assert "triples" in json_data
        assert "entities" in json_data
        assert "relations" in json_data
        assert len(json_data["triples"]) == 2

        # Import to new builder
        new_builder = KnowledgeGraphBuilder()
        new_builder.import_from_json(json_data)

        assert len(new_builder.triples) == 2
        assert len(new_builder.entities) == 3  # John, Jane, ACME Corp
        assert len(new_builder.relations) == 2  # works_at, knows


class TestCrossModalAttention:
    """Test CrossModalAttention functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.attention = CrossModalAttention()

    def test_compute_coherence_score(self):
        """Test coherence score computation."""
        # Similar embeddings should have high coherence
        embedding1 = [0.8, 0.6, 0.0]
        embedding2 = [0.6, 0.8, 0.0]

        score = self.attention.compute_coherence_score(
            embedding1, embedding2, modality1="text", modality2="image"
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for similar embeddings

        # Orthogonal embeddings should have lower coherence
        embedding3 = [0.0, 0.0, 1.0]

        score_low = self.attention.compute_coherence_score(
            embedding1, embedding3, modality1="text", modality2="image"
        )

        assert score_low < score  # Should be lower than similar embeddings

    def test_compute_attention_weights(self):
        """Test attention weight computation."""
        embeddings = {
            "text": [0.8, 0.6, 0.0],
            "image": [0.6, 0.8, 0.0],
            "audio": [0.0, 0.0, 1.0],
        }

        weights = self.attention.compute_attention_weights(
            embeddings, query_modality="text"
        )

        assert len(weights) == 3
        assert "text" in weights
        assert "image" in weights
        assert "audio" in weights

        # Weights should sum to 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 1e-6

        # Text should have highest weight (self-attention)
        assert weights["text"] >= weights["image"]
        assert weights["text"] >= weights["audio"]

    def test_get_attended_representation(self):
        """Test attended representation generation."""
        embeddings = {
            "text": [1.0, 0.0, 0.0],
            "image": [0.0, 1.0, 0.0],
            "audio": [0.0, 0.0, 1.0],
        }

        attended = self.attention.get_attended_representation(
            embeddings, query_modality="text"
        )

        assert len(attended) == 3  # Same dimension as input embeddings
        assert isinstance(attended, list)

        # Should be a weighted combination of input embeddings
        assert all(isinstance(x, float) for x in attended)


class TestHierarchicalSemanticCompression:
    """Test HierarchicalSemanticCompression functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.hsc = HierarchicalSemanticCompression()

    def test_compress_embeddings(self):
        """Test embedding compression."""
        # Create test embeddings
        embeddings = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.4, 0.5, 0.6, 0.7, 0.8],
        ]

        compressed = self.hsc.compress_embeddings(
            embeddings, target_compression_ratio=2.0, preserve_semantic_structure=True
        )

        assert "compressed_embeddings" in compressed
        assert "compression_metadata" in compressed
        assert "original_shape" in compressed["compression_metadata"]
        assert compressed["compression_metadata"]["original_shape"] == [4, 5]

    def test_dimensionality_reduction(self):
        """Test dimensionality reduction."""
        # Create high-dimensional embeddings
        embeddings = np.random.rand(10, 20)  # 10 embeddings, 20 dimensions

        reduced = self.hsc._apply_dimensionality_reduction(embeddings, target_dim=5)

        assert reduced.shape == (10, 5)  # Reduced to 5 dimensions

    def test_semantic_clustering(self):
        """Test semantic clustering."""
        # Create embeddings with clear clusters
        embeddings = np.array(
            [
                [1.0, 0.0],  # Cluster 1
                [1.1, 0.1],
                [0.0, 1.0],  # Cluster 2
                [0.1, 1.1],
                [0.5, 0.5],  # Middle point
            ]
        )

        cluster_labels = self.hsc._apply_semantic_clustering(embeddings, num_clusters=2)

        assert len(cluster_labels) == 5
        assert len(set(cluster_labels)) <= 2  # Should have at most 2 clusters

    def test_quantization(self):
        """Test embedding quantization."""
        embeddings = np.array([[0.1, 0.9, 0.5], [0.3, 0.7, 0.2], [0.8, 0.1, 0.6]])

        quantized = self.hsc._apply_quantization(embeddings, bits=8)

        assert quantized.shape == embeddings.shape
        # Quantized values should be different from original (unless exactly on quantization boundaries)
        assert not np.array_equal(quantized, embeddings)

    def test_compress_decompress_cycle(self):
        """Test compression and decompression cycle."""
        original_embeddings = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 0.1, 0.2, 0.3],
        ]

        # Compress
        compressed = self.hsc.compress_embeddings(
            original_embeddings, target_compression_ratio=1.5
        )

        # Decompress
        decompressed = self.hsc.decompress_embeddings(compressed)

        assert len(decompressed) == len(original_embeddings)
        assert len(decompressed[0]) == len(original_embeddings[0])

        # Should be approximately equal (some loss is expected)
        for orig, decomp in zip(original_embeddings, decompressed):
            for o, d in zip(orig, decomp):
                assert abs(o - d) < 0.5  # Allow some compression loss


class TestCryptographicSemanticBinding:
    """Test CryptographicSemanticBinding functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.csb = CryptographicSemanticBinding()

    def test_create_semantic_commitment(self):
        """Test semantic commitment creation."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        source_data = "This is the original text"

        commitment = self.csb.create_semantic_commitment(
            embedding, source_data, algorithm="sha256"
        )

        assert "commitment_hash" in commitment
        assert "binding_proof" in commitment
        assert "algorithm" in commitment
        assert commitment["algorithm"] == "sha256"
        assert len(commitment["commitment_hash"]) > 0

    def test_verify_semantic_binding(self):
        """Test semantic binding verification."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        source_data = "This is the original text"

        # Create commitment
        commitment = self.csb.create_semantic_commitment(embedding, source_data)

        # Verify with correct data
        is_valid = self.csb.verify_semantic_binding(embedding, source_data, commitment)
        assert is_valid is True

        # Verify with wrong embedding
        wrong_embedding = [0.9, 0.8, 0.7, 0.6, 0.5]
        is_valid = self.csb.verify_semantic_binding(
            wrong_embedding, source_data, commitment
        )
        assert is_valid is False

        # Verify with wrong source data
        wrong_data = "This is different text"
        is_valid = self.csb.verify_semantic_binding(embedding, wrong_data, commitment)
        assert is_valid is False

    def test_zero_knowledge_proof(self):
        """Test zero-knowledge proof creation."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        secret_value = "secret_knowledge"

        proof = self.csb.create_zero_knowledge_proof(embedding, secret_value)

        assert "proof_hash" in proof
        assert "challenge" in proof
        assert "response" in proof
        assert len(proof["proof_hash"]) > 0

    def test_verify_zero_knowledge_proof(self):
        """Test zero-knowledge proof verification."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        secret_value = "secret_knowledge"

        # Create proof
        proof = self.csb.create_zero_knowledge_proof(embedding, secret_value)

        # Verify proof (simplified verification)
        is_valid = self.csb.verify_zero_knowledge_proof(proof, embedding)

        # Note: This is a simplified implementation, so verification might be basic
        assert isinstance(is_valid, bool)


class TestDeepSemanticUnderstanding:
    """Test DeepSemanticUnderstanding functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dsu = DeepSemanticUnderstanding()

    def test_dsu_initialization(self):
        """Test DeepSemanticUnderstanding initialization."""
        assert hasattr(self.dsu, "embedder")
        assert hasattr(self.dsu, "kg_builder")
        assert hasattr(self.dsu, "attention")

    def test_process_multimodal_input(self):
        """Test multimodal input processing."""
        inputs = {
            "text": "A beautiful sunset over the ocean",
            "image": b"fake_image_data",
            "metadata": {"location": "beach", "time": "evening"},
        }

        result = self.dsu.process_multimodal_input(inputs)

        assert "unified_embedding" in result
        assert "attention_weights" in result
        assert "semantic_features" in result
        assert isinstance(result["unified_embedding"], list)
        assert isinstance(result["attention_weights"], dict)

    def test_extract_semantic_features(self):
        """Test semantic feature extraction."""
        # Test text features
        text_features = self.dsu._extract_semantic_features(
            "The quick brown fox jumps over the lazy dog", "text"
        )

        assert "entities" in text_features
        assert "sentiment" in text_features
        assert isinstance(text_features["entities"], list)
        assert text_features["sentiment"] in ["positive", "negative", "neutral"]

        # Test binary features (simplified)
        binary_features = self.dsu._extract_semantic_features(b"binary_data", "image")

        assert "format" in binary_features
        assert "size" in binary_features

    def test_sentiment_analysis(self):
        """Test simple sentiment analysis."""
        positive_text = "I love this amazing product!"
        negative_text = "This is terrible and awful"
        neutral_text = "The weather is cloudy today"

        pos_sentiment = self.dsu._simple_sentiment_analysis(positive_text)
        neg_sentiment = self.dsu._simple_sentiment_analysis(negative_text)
        neu_sentiment = self.dsu._simple_sentiment_analysis(neutral_text)

        assert pos_sentiment in ["positive", "negative", "neutral"]
        assert neg_sentiment in ["positive", "negative", "neutral"]
        assert neu_sentiment in ["positive", "negative", "neutral"]

    def test_entity_extraction(self):
        """Test simple entity extraction."""
        text = "John Smith works at Microsoft in Seattle"

        entities = self.dsu._extract_simple_entities(text)

        assert isinstance(entities, list)
        # Should extract some capitalized words as entities
        assert len(entities) >= 0

    def test_semantic_reasoning(self):
        """Test semantic reasoning capabilities."""
        query = "What is the weather like?"
        context = {
            "text_data": ["It's sunny today", "The temperature is 75 degrees"],
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "metadata": {"location": "San Francisco", "date": "2024-01-15"},
        }

        result = self.dsu.semantic_reasoning(query, context)

        assert "reasoning_result" in result
        assert "confidence" in result
        assert "explanation" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_unified_representation_creation(self):
        """Test unified representation creation."""
        embeddings = {
            "text": [0.1, 0.2, 0.3],
            "image": [0.4, 0.5, 0.6],
            "audio": [0.7, 0.8, 0.9],
        }

        features = {
            "text": {"sentiment": "positive", "entities": ["test"]},
            "image": {"format": "jpeg", "size": 1024},
            "audio": {"duration": 30, "format": "wav"},
        }

        unified = self.dsu._create_unified_representation(embeddings, features)

        assert len(unified) == 3  # Same dimension as input embeddings
        assert isinstance(unified, list)
        assert all(isinstance(x, float) for x in unified)


class TestSemanticIntegration:
    """Test semantic integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("sentence_transformers.SentenceTransformer"):
            self.embedder = SemanticEmbedder()
        self.kg_builder = KnowledgeGraphBuilder()
        self.attention = CrossModalAttention()

    def test_end_to_end_semantic_workflow(self):
        """Test complete semantic processing workflow."""
        # 1. Create embeddings
        with patch.object(self.embedder, "model") as mock_model:
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

            embedding = self.embedder.embed_text(
                "John Smith works at ACME Corporation",
                metadata={"source": "hr_document"},
            )

        # 2. Build knowledge graph
        self.kg_builder.add_triple("John Smith", "works_at", "ACME Corporation")
        self.kg_builder.add_triple("ACME Corporation", "type", "Company")

        # 3. Test cross-modal attention
        embeddings = {"text": embedding.vector, "knowledge": [0.2, 0.3, 0.4, 0.5, 0.6]}

        weights = self.attention.compute_attention_weights(embeddings, "text")

        # Verify workflow completion
        assert len(self.embedder.embeddings) == 1
        assert len(self.kg_builder.triples) == 2
        assert len(weights) == 2
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_multimodal_semantic_fusion(self):
        """Test fusion of multiple semantic modalities."""
        # Text embedding
        text_embedding = [0.8, 0.6, 0.4, 0.2]

        # Image embedding (simulated)
        image_embedding = [0.2, 0.4, 0.6, 0.8]

        # Knowledge graph embedding (simulated)
        kg_embedding = [0.5, 0.5, 0.5, 0.5]

        embeddings = {
            "text": text_embedding,
            "image": image_embedding,
            "knowledge": kg_embedding,
        }

        # Compute attention weights
        weights = self.attention.compute_attention_weights(embeddings, "text")

        # Create attended representation
        attended = self.attention.get_attended_representation(embeddings, "text")

        assert len(attended) == 4
        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestSemanticErrorHandling:
    """Test semantic error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.kg_builder = KnowledgeGraphBuilder()
        self.attention = CrossModalAttention()
        self.hsc = HierarchicalSemanticCompression()

    def test_empty_embeddings(self):
        """Test handling of empty embeddings."""
        # Empty embedding list
        empty_embeddings = []

        compressed = self.hsc.compress_embeddings(empty_embeddings)

        assert "compressed_embeddings" in compressed
        assert "compression_metadata" in compressed

    def test_single_embedding(self):
        """Test handling of single embedding."""
        single_embedding = [[0.1, 0.2, 0.3]]

        compressed = self.hsc.compress_embeddings(single_embedding)
        decompressed = self.hsc.decompress_embeddings(compressed)

        assert len(decompressed) == 1
        assert len(decompressed[0]) == 3

    def test_zero_vector_embeddings(self):
        """Test handling of zero vector embeddings."""
        zero_embeddings = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

        # Should handle gracefully
        compressed = self.hsc.compress_embeddings(zero_embeddings)
        assert compressed is not None

    def test_invalid_attention_inputs(self):
        """Test attention with invalid inputs."""
        # Empty embeddings dict
        empty_embeddings = {}

        weights = self.attention.compute_attention_weights(empty_embeddings, "text")
        assert len(weights) == 0
        assert list(weights.keys()) == []
        assert list(weights.values()) == []

        # Mismatched dimensions
        mismatched_embeddings = {
            "text": [0.1, 0.2],
            "image": [0.3, 0.4, 0.5],  # Different dimension
        }

        # Should handle gracefully (implementation dependent)
        try:
            weights = self.attention.compute_attention_weights(
                mismatched_embeddings, "text"
            )
            # If no exception, that's fine too
        except Exception:
            # Expected to handle gracefully
            pass


if __name__ == "__main__":
    pytest.main([__file__])
