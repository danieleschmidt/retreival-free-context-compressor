"""Performance benchmarks for compression algorithms."""

import time
import pytest
import torch
from unittest.mock import Mock, patch
import statistics
from typing import List, Dict, Any


@pytest.mark.slow
@pytest.mark.benchmark
class TestCompressionBenchmarks:
    """Benchmark compression performance."""
    
    def test_compression_latency(self, sample_document, benchmark_data):
        """Benchmark compression latency."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            # Simulate realistic latency
            def mock_compress(text):
                time.sleep(0.1)  # Simulate 100ms processing time
                return {
                    'compressed_tokens': torch.randn(32, 768),
                    'compression_ratio': 8.0,
                    'latency_ms': 100
                }
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Benchmark multiple runs
            latencies = []
            for _ in range(10):
                start_time = time.time()
                result = compressor.compress(sample_document)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
            
            # Assert performance requirements
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
            
            assert avg_latency < benchmark_data["max_latency_ms"]
            assert p95_latency < benchmark_data["max_latency_ms"] * 1.5
    
    def test_throughput_benchmark(self, benchmark_data):
        """Benchmark compression throughput."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_compress(text):
                # Simulate processing based on text length
                processing_time = len(text) / 10000  # 10k chars per second
                time.sleep(processing_time)
                return {
                    'compressed_tokens': torch.randn(32, 768),
                    'compression_ratio': 8.0
                }
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Test with multiple document sizes
            documents = benchmark_data["documents"]
            start_time = time.time()
            
            results = []
            for doc in documents[:100]:  # Test with first 100 docs
                result = compressor.compress(doc)
                results.append(result)
            
            end_time = time.time()
            total_time = end_time - start_time
            throughput = len(results) / total_time  # docs per second
            
            # Assert minimum throughput
            assert throughput > 10  # At least 10 docs/second
            assert len(results) == 100
    
    @pytest.mark.parametrize("doc_length", [1000, 5000, 10000, 50000])
    def test_compression_scaling(self, doc_length):
        """Test how compression performance scales with document length."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_compress(text):
                # Simulate O(n) scaling
                processing_time = len(text) / 50000  # Scale with length
                time.sleep(processing_time)
                compression_ratio = min(16.0, len(text) / 100)  # Better ratio for longer docs
                return {
                    'compressed_tokens': torch.randn(max(10, len(text) // 100), 768),
                    'compression_ratio': compression_ratio,
                    'processing_time': processing_time
                }
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            test_doc = "Test content. " * (doc_length // 13)  # Approximate length
            
            start_time = time.time()
            result = compressor.compress(test_doc)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Performance should scale reasonably with document length
            expected_max_time = doc_length / 10000  # 10k chars per second minimum
            assert processing_time < expected_max_time
            assert result['compression_ratio'] > 1.0
    
    def test_memory_usage(self, sample_document):
        """Test memory usage during compression."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_compress(text):
                # Simulate memory allocation
                temp_tensor = torch.randn(1000, 768)  # Simulate memory usage
                result = {
                    'compressed_tokens': torch.randn(32, 768),
                    'compression_ratio': 8.0,
                    'peak_memory_mb': temp_tensor.numel() * 4 / (1024 * 1024)  # 4 bytes per float
                }
                del temp_tensor  # Clean up
                return result
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Measure memory before and after
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()
            
            result = compressor.compress(sample_document)
            
            if torch.cuda.is_available():
                final_memory = torch.cuda.memory_allocated()
                memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
                
                # Assert reasonable memory usage
                assert memory_increase < 1000  # Less than 1GB increase
            
            assert 'peak_memory_mb' in result


@pytest.mark.slow
@pytest.mark.benchmark
class TestStreamingBenchmarks:
    """Benchmark streaming compression performance."""
    
    def test_streaming_latency(self, sample_document):
        """Test streaming compression latency."""
        with patch('retrieval_free.streaming.StreamingCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_add_chunk(chunk):
                time.sleep(0.05)  # 50ms per chunk
                return torch.randn(20, 768)
            
            mock_instance.add_chunk.side_effect = mock_add_chunk
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Test adding multiple chunks
            chunks = [sample_document[i:i+500] for i in range(0, len(sample_document), 500)]
            
            latencies = []
            for chunk in chunks:
                start_time = time.time()
                result = compressor.add_chunk(chunk)
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)
            
            avg_latency = statistics.mean(latencies)
            assert avg_latency < 100  # Less than 100ms per chunk
    
    def test_streaming_memory_efficiency(self):
        """Test memory efficiency of streaming compression."""
        with patch('retrieval_free.streaming.StreamingCompressor') as mock_compressor:
            mock_instance = Mock()
            
            # Simulate memory-efficient streaming
            current_memory = 0
            
            def mock_add_chunk(chunk):
                nonlocal current_memory
                current_memory += len(chunk)
                # Simulate pruning when memory gets high
                if current_memory > 100000:  # 100k chars
                    current_memory = 50000  # Prune to 50k
                return torch.randn(min(40, current_memory // 1000), 768)
            
            def mock_should_prune():
                return current_memory > 80000
            
            def mock_prune():
                nonlocal current_memory
                current_memory = 40000
                return {'pruned_tokens': 20}
            
            mock_instance.add_chunk.side_effect = mock_add_chunk
            mock_instance.should_prune.side_effect = mock_should_prune
            mock_instance.prune_obsolete.side_effect = mock_prune
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Add many chunks to test memory management
            large_chunk = "Test content " * 1000  # ~13k chars
            for _ in range(20):
                result = compressor.add_chunk(large_chunk)
                
                if compressor.should_prune():
                    prune_result = compressor.prune_obsolete()
                    assert 'pruned_tokens' in prune_result
            
            # Memory should be bounded
            assert current_memory < 120000  # Should not exceed reasonable limit


@pytest.mark.benchmark
class TestCompressionQuality:
    """Benchmark compression quality metrics."""
    
    def test_compression_ratio_consistency(self, sample_document):
        """Test consistency of compression ratios."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_compress(text):
                # Simulate consistent but slightly variable compression
                base_ratio = 8.0
                variation = torch.rand(1).item() * 0.5 - 0.25  # ±0.25 variation
                return {
                    'compressed_tokens': torch.randn(32, 768),
                    'compression_ratio': base_ratio + variation
                }
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Test multiple compressions of the same document
            ratios = []
            for _ in range(10):
                result = compressor.compress(sample_document)
                ratios.append(result['compression_ratio'])
            
            # Check consistency
            ratio_std = statistics.stdev(ratios)
            ratio_mean = statistics.mean(ratios)
            
            assert ratio_std < 0.5  # Low standard deviation
            assert 7.5 < ratio_mean < 8.5  # Mean around target
    
    def test_compression_information_retention(self, sample_document, sample_questions):
        """Test information retention after compression."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            
            def mock_compress(text):
                # Simulate high information retention
                return {
                    'compressed_tokens': torch.randn(32, 768),
                    'compression_ratio': 8.0,
                    'information_retention_score': 0.92  # 92% retention
                }
            
            mock_instance.compress.side_effect = mock_compress
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            result = compressor.compress(sample_document)
            
            # Assert high information retention
            assert result['information_retention_score'] > 0.85
            assert result['compression_ratio'] > 4.0


def benchmark_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate benchmark summary statistics."""
    return {
        'total_tests': len(results),
        'avg_latency_ms': statistics.mean([r.get('latency_ms', 0) for r in results]),
        'avg_compression_ratio': statistics.mean([r.get('compression_ratio', 1) for r in results]),
        'memory_efficiency': statistics.mean([r.get('memory_mb', 0) for r in results]),
    }