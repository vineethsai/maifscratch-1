"""
Tests for MAIF streaming functionality (v3 format).
"""

import pytest
import tempfile
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from maif import MAIFEncoder, MAIFDecoder, BlockType
from maif.streaming import (
    MAIFStreamReader,
    MAIFStreamWriter,
    StreamingConfig,
    PerformanceProfiler,
)


class TestMAIFStreamReader:
    """Test MAIFStreamReader functionality."""

    @pytest.fixture
    def sample_maif(self, tmp_path):
        """Create a sample MAIF file for testing."""
        maif_path = tmp_path / "test.maif"

        encoder = MAIFEncoder(str(maif_path), agent_id="test-agent")
        encoder.add_text_block("Test block 1", metadata={"id": 1})
        encoder.add_text_block("Test block 2", metadata={"id": 2})
        encoder.add_text_block("Test block 3", metadata={"id": 3})
        encoder.finalize()

        return str(maif_path)

    def test_stream_reader_initialization(self, sample_maif):
        """Test MAIFStreamReader initialization."""
        config = StreamingConfig(chunk_size=4096)

        with MAIFStreamReader(sample_maif, config) as reader:
            assert reader.maif_path == sample_maif
            assert reader.config.chunk_size == 4096

    def test_stream_blocks_basic(self, sample_maif):
        """Test basic block streaming."""
        with MAIFStreamReader(sample_maif) as reader:
            blocks = list(reader.stream_blocks())

            assert len(blocks) >= 1
            for block_type, block_data in blocks:
                assert block_type is not None
                assert isinstance(block_data, bytes)

    def test_stream_blocks_parallel(self, sample_maif):
        """Test parallel block streaming."""
        config = StreamingConfig(max_workers=4)

        with MAIFStreamReader(sample_maif, config) as reader:
            blocks = list(reader.stream_blocks_parallel())

            assert len(blocks) >= 1

    def test_get_performance_stats(self, sample_maif):
        """Test performance statistics collection."""
        with MAIFStreamReader(sample_maif) as reader:
            # Read some blocks
            for _ in reader.stream_blocks():
                pass

            stats = reader.get_performance_stats()

            assert "total_blocks_read" in stats
            assert "total_bytes_read" in stats
            assert "file_size" in stats

    def test_blocks_property(self, sample_maif):
        """Test blocks property."""
        with MAIFStreamReader(sample_maif) as reader:
            blocks = reader.blocks
            assert len(blocks) == 3


class TestMAIFStreamWriter:
    """Test MAIFStreamWriter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = os.path.join(self.temp_dir, "test_output.maif")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_stream_writer_initialization(self):
        """Test MAIFStreamWriter initialization."""
        config = StreamingConfig(buffer_size=8192)

        with MAIFStreamWriter(self.output_path, config) as writer:
            assert writer.output_path == self.output_path
            assert writer.config.buffer_size == 8192

    def test_write_block(self):
        """Test writing blocks."""
        with MAIFStreamWriter(self.output_path) as writer:
            block_id = writer.write_block(b"Test data", "text")
            assert block_id is not None

        # Verify file was created
        assert os.path.exists(self.output_path)
        assert os.path.getsize(self.output_path) > 0

    def test_write_block_stream(self):
        """Test writing block streams."""
        test_data_chunks = [
            b"First chunk of data",
            b"Second chunk of data",
            b"Third chunk of data",
        ]

        with MAIFStreamWriter(self.output_path) as writer:
            block_id = writer.write_block_stream("text", iter(test_data_chunks))
            assert block_id is not None

        # Verify file was created
        assert os.path.exists(self.output_path)
        assert os.path.getsize(self.output_path) > 0

    def test_context_manager_writer(self):
        """Test writer context manager functionality."""
        writer = MAIFStreamWriter(self.output_path)

        # Test __enter__
        entered_writer = writer.__enter__()
        assert entered_writer is writer

        # Write something
        writer.write_block(b"test", "text")

        # Test __exit__
        writer.__exit__(None, None, None)

        # File should be finalized
        assert os.path.exists(self.output_path)

    def test_large_stream_writing(self):
        """Test writing large streams."""

        def large_data_generator():
            for i in range(100):
                yield f"Data chunk {i} with some content".encode("utf-8")

        with MAIFStreamWriter(self.output_path) as writer:
            block_id = writer.write_block_stream("text", large_data_generator())
            assert block_id is not None

        # Verify file was created
        assert os.path.exists(self.output_path)
        assert os.path.getsize(self.output_path) > 0

    def test_concurrent_writing(self):
        """Test concurrent writing scenarios."""
        config = StreamingConfig(max_workers=2)

        def write_data(writer_id):
            output_path = os.path.join(self.temp_dir, f"concurrent_{writer_id}.maif")
            data_chunks = [
                f"Writer {writer_id} chunk {i}".encode("utf-8") for i in range(10)
            ]

            with MAIFStreamWriter(output_path, config) as writer:
                block_id = writer.write_block_stream("text", iter(data_chunks))
                return block_id

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(write_data, i) for i in range(3)]
            results = [future.result() for future in futures]

        assert len(results) == 3
        assert all(result is not None for result in results)

        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"concurrent_{i}.maif")
            assert os.path.exists(file_path)

    def test_get_stats(self):
        """Test getting writing statistics."""
        with MAIFStreamWriter(self.output_path) as writer:
            writer.write_block(b"test data 1", "text")
            writer.write_block(b"test data 2", "text")

            stats = writer.get_stats()

            assert stats["blocks_written"] == 2
            assert stats["bytes_written"] > 0


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()

    def test_profiler_initialization(self):
        """Test PerformanceProfiler initialization."""
        assert self.profiler.timings == {}
        assert self.profiler.operation_counts == {}

    def test_start_end_timer(self):
        """Test timer start and end."""
        operation_name = "test_operation"

        self.profiler.start_timer(operation_name)
        time.sleep(0.01)  # 10ms
        elapsed = self.profiler.end_timer(operation_name)

        assert elapsed >= 0.01
        assert operation_name in self.profiler.timings

    def test_start_timing_alias(self):
        """Test start_timing alias."""
        self.profiler.start_timing("test_op")
        time.sleep(0.005)
        elapsed = self.profiler.end_timer("test_op")

        assert elapsed >= 0.005

    def test_get_stats(self):
        """Test getting profiler stats."""
        self.profiler.start_timing("op1")
        time.sleep(0.01)
        self.profiler.end_timer("op1")

        stats = self.profiler.get_stats()
        assert "op1" in stats

    def test_context_timer(self):
        """Test context manager timer."""
        with self.profiler.context_timer("ctx_op"):
            time.sleep(0.01)

        assert "ctx_op" in self.profiler.timings

    def test_concurrent_profiling(self):
        """Test profiler thread safety."""

        def profile_operation(op_id):
            operation_name = f"concurrent_op_{op_id}"
            self.profiler.start_timing(operation_name)
            time.sleep(0.01)
            self.profiler.end_timer(operation_name)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=profile_operation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(self.profiler.timings) == 5


class TestStreamingIntegration:
    """Test streaming integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_write_cycle(self):
        """Test complete read-write streaming cycle."""
        # Create original MAIF file
        original_path = os.path.join(self.temp_dir, "original.maif")

        encoder = MAIFEncoder(original_path, agent_id="test-agent")
        encoder.add_text_block("Original text block", metadata={"id": 1})
        encoder.add_binary_block(
            b"original_binary_data", BlockType.BINARY, metadata={"id": 2}
        )
        encoder.finalize()

        # Stream read and write to new file
        new_path = os.path.join(self.temp_dir, "streamed.maif")

        with MAIFStreamReader(original_path) as reader:
            with MAIFStreamWriter(new_path) as writer:
                for block_type, block_data in reader.stream_blocks():
                    writer.write_block(block_data, str(block_type))

        # Verify new file was created
        assert os.path.exists(new_path)
        assert os.path.getsize(new_path) > 0

    def test_streaming_with_compression(self):
        """Test streaming with compression enabled."""
        config = StreamingConfig(
            enable_compression=True, compression_level=6, buffer_size=4096
        )

        # Create test file
        original_path = os.path.join(self.temp_dir, "compress_test.maif")
        encoder = MAIFEncoder(original_path, agent_id="test")
        encoder.add_text_block("Compress this text " * 100)
        encoder.finalize()

        # Read with config
        with MAIFStreamReader(original_path, config) as reader:
            blocks = list(reader.stream_blocks())
            assert len(blocks) >= 1

    def test_streaming_performance_monitoring(self):
        """Test performance monitoring during streaming."""
        # Create test file
        test_path = os.path.join(self.temp_dir, "perf_test.maif")
        encoder = MAIFEncoder(test_path, agent_id="test")
        for i in range(10):
            encoder.add_text_block(f"Block {i} " * 100)
        encoder.finalize()

        config = StreamingConfig(max_workers=4)

        with MAIFStreamReader(test_path, config) as reader:
            _blocks = list(reader.stream_blocks())  # noqa: F841
            stats = reader.get_performance_stats()

        assert stats["total_blocks_read"] >= 0
        assert stats["file_size"] > 0

    def test_large_file_streaming(self):
        """Test streaming of larger files."""
        large_path = os.path.join(self.temp_dir, "large_test.maif")

        encoder = MAIFEncoder(large_path, agent_id="test")
        for i in range(50):
            encoder.add_text_block(f"Large block content {i} " * 100)
        encoder.finalize()

        block_count = 0
        with MAIFStreamReader(large_path) as reader:
            for block_type, block_data in reader.stream_blocks():
                block_count += 1

        assert block_count == 50


class TestStreamingConfig:
    """Test StreamingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.chunk_size > 0
        assert config.max_workers > 0
        assert config.buffer_size > 0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StreamingConfig(
            chunk_size=8192, max_workers=16, buffer_size=1024 * 1024
        )

        assert config.chunk_size == 8192
        assert config.max_workers == 16
        assert config.buffer_size == 1024 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
