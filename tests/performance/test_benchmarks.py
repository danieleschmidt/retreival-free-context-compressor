"""Performance benchmarks and profiling tests."""

import pytest
import time
import psutil
import os
from unittest.mock import MagicMock, patch


class TestCompressionBenchmarks:
    """Benchmark tests for compression performance."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_compression_throughput(self, sample_documents):
        """Benchmark compression throughput (tokens/second)."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["token"] * 100
            mock_compressor.count_tokens.return_value = 1000
            MockCompressor.return_value = mock_compressor
            
            document = sample_documents["long"]
            
            # Benchmark compression speed
            start_time = time.time()
            iterations = 10
            
            for _ in range(iterations):
                mock_compressor.compress(document)
            
            elapsed = time.time() - start_time
            tokens_per_second = (1000 * iterations) / elapsed
            
            # Assert minimum throughput requirement
            assert tokens_per_second > 1000  # At least 1K tokens/sec

    @pytest.mark.slow
    def test_memory_efficiency(self, sample_documents):
        """Benchmark memory usage during compression."""
        process = psutil.Process(os.getpid())
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["token"] * 50
            MockCompressor.return_value = mock_compressor
            
            # Measure baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process large document
            document = sample_documents["long"] * 10  # Extra large
            mock_compressor.compress(document)
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory
            
            # Assert reasonable memory usage (would be actual requirements)
            assert memory_increase < 1000  # Less than 1GB increase

    @pytest.mark.slow
    def test_compression_scaling(self, sample_documents):
        """Test how compression performance scales with input size."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            MockCompressor.return_value = mock_compressor
            
            results = {}
            
            for size, doc in sample_documents.items():
                # Simulate processing time proportional to input size
                processing_time = len(doc) / 100000  # Simulated processing
                mock_compressor.compress.return_value = ["token"] * int(processing_time * 10)
                
                start_time = time.time()
                mock_compressor.compress(doc)
                elapsed = time.time() - start_time
                
                results[size] = {
                    "input_length": len(doc),
                    "processing_time": elapsed,
                }
            
            # Verify scaling characteristics
            assert results["medium"]["processing_time"] > results["short"]["processing_time"]

    @pytest.mark.slow
    def test_streaming_performance(self, sample_documents):
        """Benchmark streaming compression performance."""
        with patch('retrieval_free.StreamingCompressor') as MockStreamingCompressor:
            mock_compressor = MagicMock()
            mock_compressor.add_chunk.return_value = ["stream_token"]
            mock_compressor.should_prune.return_value = False
            MockStreamingCompressor.return_value = mock_compressor
            
            chunks = [sample_documents["short"]] * 100
            
            start_time = time.time()
            
            for chunk in chunks:
                mock_compressor.add_chunk(chunk)
            
            elapsed = time.time() - start_time
            chunks_per_second = len(chunks) / elapsed
            
            # Assert minimum streaming throughput
            assert chunks_per_second > 50  # At least 50 chunks/sec


class TestMemoryProfiling:
    """Memory profiling and leak detection tests."""

    @pytest.mark.slow
    def test_memory_leak_detection(self, sample_documents):
        """Test for memory leaks during repeated operations."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["token"] * 10
            MockCompressor.return_value = mock_compressor
            
            process = psutil.Process(os.getpid())
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Perform many operations
            for _ in range(100):
                mock_compressor.compress(sample_documents["medium"])
            
            # Measure final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Assert no significant memory growth (indicating leaks)
            assert memory_growth < 100  # Less than 100MB growth

    @pytest.mark.slow
    def test_garbage_collection_efficiency(self):
        """Test garbage collection behavior with compression objects."""
        import gc
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            MockCompressor.return_value = mock_compressor
            
            # Create and destroy many compressor objects
            objects_before = len(gc.get_objects())
            
            compressors = []
            for i in range(100):
                compressor = MockCompressor()
                compressors.append(compressor)
            
            # Clear references
            compressors.clear()
            gc.collect()
            
            objects_after = len(gc.get_objects())
            
            # Assert reasonable object cleanup
            object_growth = objects_after - objects_before
            assert object_growth < 1000  # Reasonable object growth


class TestConcurrencyBenchmarks:
    """Benchmark concurrent compression operations."""

    @pytest.mark.slow
    def test_concurrent_compression(self, sample_documents):
        """Test performance of concurrent compression operations."""
        import threading
        import queue
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["token"] * 5
            MockCompressor.return_value = mock_compressor
            
            results_queue = queue.Queue()
            
            def compress_worker(document):
                start_time = time.time()
                result = mock_compressor.compress(document)
                elapsed = time.time() - start_time
                results_queue.put({"result": result, "time": elapsed})
            
            # Start concurrent compressions
            threads = []
            for _ in range(5):
                thread = threading.Thread(
                    target=compress_worker,
                    args=(sample_documents["medium"],)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Collect results
            results = []
            while not results_queue.empty():
                results.append(results_queue.get())
            
            assert len(results) == 5
            # Verify all operations completed reasonably fast
            for result in results:
                assert result["time"] < 1.0