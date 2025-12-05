"""
Test PyTorch model storage and complete recovery using MAIF v3 format.
Demonstrates full model lifecycle: save -> store in MAIF -> recover -> verify.
"""

import pytest
import tempfile
import os
import json
import time
import random
import shutil
from maif import MAIFEncoder, MAIFDecoder, BlockType


class TestPyTorchModelRecovery:
    """Test complete PyTorch model storage and recovery with MAIF v3."""

    def setup_method(self):
        """Set up test fixtures."""
        random.seed(42)
        self.temp_dir = tempfile.mkdtemp()
        self.maif_path = os.path.join(self.temp_dir, "model.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_neural_network_model(self):
        """Create a complete neural network model with all components."""

        # 1. Network Architecture
        architecture = {
            "model_name": "SimpleClassifier",
            "framework": "pytorch",
            "version": "1.0",
            "layers": [
                {
                    "name": "fc1",
                    "type": "linear",
                    "input_size": 784,
                    "output_size": 128,
                    "activation": "relu",
                    "dropout": 0.2,
                },
                {
                    "name": "fc2",
                    "type": "linear",
                    "input_size": 128,
                    "output_size": 64,
                    "activation": "relu",
                    "dropout": 0.3,
                },
                {
                    "name": "fc3",
                    "type": "linear",
                    "input_size": 64,
                    "output_size": 10,
                    "activation": "softmax",
                },
            ],
            "loss_function": "cross_entropy",
            "optimizer": {
                "type": "adam",
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "weight_decay": 1e-4,
            },
        }

        # 2. Model Weights (simulate trained weights)
        weights = {}
        total_params = 0

        for layer in architecture["layers"]:
            layer_name = layer["name"]
            input_size = layer["input_size"]
            output_size = layer["output_size"]

            limit = (6.0 / (input_size + output_size)) ** 0.5
            layer_weights = [
                [random.uniform(-limit, limit) for _ in range(input_size)]
                for _ in range(output_size)
            ]
            layer_biases = [random.uniform(-0.1, 0.1) for _ in range(output_size)]

            weights[layer_name] = {
                "weight": layer_weights,
                "bias": layer_biases,
                "shape": [output_size, input_size],
            }

            total_params += (input_size * output_size) + output_size

        # 3. Training Configuration
        training_config = {
            "batch_size": 32,
            "epochs": 50,
            "learning_rate_schedule": "cosine_annealing",
        }

        # 4. Training History
        training_history = []
        for epoch in range(10):
            train_loss = 2.3 - (epoch * 0.08) + random.uniform(-0.1, 0.1)
            train_acc = 0.1 + (epoch * 0.03) + random.uniform(-0.02, 0.02)

            training_history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(max(0.1, train_loss), 4),
                    "train_accuracy": round(min(0.99, max(0.1, train_acc)), 4),
                }
            )

        # 5. Model Metadata
        model_metadata = {
            "created_at": time.time(),
            "total_parameters": total_params,
            "dataset": {"name": "MNIST", "num_classes": 10},
        }

        return {
            "architecture": architecture,
            "weights": weights,
            "training_config": training_config,
            "training_history": training_history,
            "model_metadata": model_metadata,
        }

    def test_complete_model_storage_and_recovery(self):
        """Test storing and recovering a complete PyTorch model."""

        # Step 1: Create complete model
        model_data = self.create_neural_network_model()

        assert "architecture" in model_data
        assert "weights" in model_data
        assert "training_config" in model_data

        # Step 2: Store model in MAIF
        encoder = MAIFEncoder(self.maif_path, agent_id="pytorch-model-storage")

        # Store architecture
        arch_json = json.dumps(model_data["architecture"], indent=2)
        encoder.add_text_block(arch_json, metadata={"type": "model_architecture"})

        # Store weights as binary
        weights_json = json.dumps(model_data["weights"]).encode("utf-8")
        encoder.add_binary_block(
            weights_json, BlockType.BINARY, metadata={"type": "model_weights"}
        )

        # Store training config
        config_json = json.dumps(model_data["training_config"])
        encoder.add_text_block(config_json, metadata={"type": "training_config"})

        # Store training history
        history_json = json.dumps(model_data["training_history"])
        encoder.add_text_block(history_json, metadata={"type": "training_history"})

        # Store metadata
        metadata_json = json.dumps(model_data["model_metadata"])
        encoder.add_text_block(metadata_json, metadata={"type": "model_metadata"})

        # Finalize
        encoder.finalize()

        # Step 3: Verify file integrity
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        is_valid, errors = decoder.verify_integrity()
        assert is_valid, f"Integrity check failed: {errors}"

        # Step 4: Recover model
        recovered = {}
        for block in decoder.blocks:
            block_type = block.metadata.get("type") if block.metadata else None
            if block_type == "model_architecture":
                recovered["architecture"] = json.loads(block.data.decode("utf-8"))
            elif block_type == "model_weights":
                recovered["weights"] = json.loads(block.data.decode("utf-8"))
            elif block_type == "training_config":
                recovered["training_config"] = json.loads(block.data.decode("utf-8"))
            elif block_type == "training_history":
                recovered["training_history"] = json.loads(block.data.decode("utf-8"))
            elif block_type == "model_metadata":
                recovered["model_metadata"] = json.loads(block.data.decode("utf-8"))

        # Step 5: Verify recovery
        assert (
            recovered["architecture"]["model_name"]
            == model_data["architecture"]["model_name"]
        )
        assert len(recovered["weights"]) == len(model_data["weights"])
        assert len(recovered["training_history"]) == len(model_data["training_history"])

    def test_model_versioning_and_comparison(self):
        """Test storing multiple model versions."""

        encoder = MAIFEncoder(self.maif_path, agent_id="model-versioning")

        # Create and store model v1.0
        model_v1 = self.create_neural_network_model()
        model_v1["architecture"]["version"] = "1.0"

        v1_json = json.dumps(model_v1)
        encoder.add_text_block(
            v1_json, metadata={"type": "complete_model", "version": "1.0"}
        )

        # Create improved model v2.0
        model_v2 = self.create_neural_network_model()
        model_v2["architecture"]["version"] = "2.0"
        model_v2["architecture"]["layers"].append(
            {
                "name": "fc4",
                "type": "linear",
                "input_size": 10,
                "output_size": 10,
                "activation": "softmax",
            }
        )

        v2_json = json.dumps(model_v2)
        encoder.add_text_block(
            v2_json, metadata={"type": "complete_model", "version": "2.0"}
        )

        encoder.finalize()

        # Load and verify
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        model_blocks = [
            b
            for b in decoder.blocks
            if b.metadata and b.metadata.get("type") == "complete_model"
        ]
        assert len(model_blocks) == 2

        # Extract and compare
        versions = {}
        for block in model_blocks:
            data = json.loads(block.data.decode("utf-8"))
            version = data["architecture"]["version"]
            versions[version] = data

        assert "1.0" in versions and "2.0" in versions
        assert len(versions["2.0"]["architecture"]["layers"]) > len(
            versions["1.0"]["architecture"]["layers"]
        )

    def test_model_component_extraction(self):
        """Test extracting specific model components."""

        model_data = self.create_neural_network_model()
        encoder = MAIFEncoder(self.maif_path, agent_id="model-components")

        # Store each component
        encoder.add_text_block(
            json.dumps(model_data["architecture"]),
            metadata={"component": "architecture"},
        )
        encoder.add_text_block(
            json.dumps(model_data["weights"]), metadata={"component": "weights"}
        )
        encoder.add_text_block(
            json.dumps(model_data["training_history"]),
            metadata={"component": "history"},
        )

        encoder.finalize()

        # Load and extract selectively
        decoder = MAIFDecoder(self.maif_path)
        decoder.load()

        # Extract only architecture
        arch_block = next(
            b
            for b in decoder.blocks
            if b.metadata and b.metadata.get("component") == "architecture"
        )
        architecture = json.loads(arch_block.data.decode("utf-8"))

        assert architecture["model_name"] == model_data["architecture"]["model_name"]
        assert len(architecture["layers"]) == len(model_data["architecture"]["layers"])

        # Extract only weights
        weights_block = next(
            b
            for b in decoder.blocks
            if b.metadata and b.metadata.get("component") == "weights"
        )
        weights = json.loads(weights_block.data.decode("utf-8"))

        assert set(weights.keys()) == set(model_data["weights"].keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
