"""Integration tests for the complete system."""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import tempfile
import os
import json


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all heavy dependencies for integration testing."""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'torch.nn': MagicMock(),
        'transformers': MagicMock(),
        'sklearn': MagicMock(),
        'sklearn.cluster': MagicMock(),
        'sentence_transformers': MagicMock(),
        'psutil': MagicMock(),
        'faiss': MagicMock(),
        'datasets': MagicMock(),
        'einops': MagicMock(),
    }):
        yield


class TestSystemIntegration:
    """Test complete system integration."""
    
    def test_system_import_structure(self):
        """Test that all modules can be imported."""
        # Test main package imports
        from src.retrieval_free import __version__
        assert __version__ == "0.1.0"
        
        # Test core components
        from src.retrieval_free.core.base import CompressorBase, MegaToken, CompressionResult
        assert CompressorBase is not None
        assert MegaToken is not None
        assert CompressionResult is not None
        
        # Test validation components
        from src.retrieval_free.validation import InputValidator, ValidationResult
        from src.retrieval_free.security import ModelSecurityValidator, SecurityScan
        from src.retrieval_free.monitoring import MetricsCollector, HealthChecker
        from src.retrieval_free.exceptions import RetrievalFreeError, CompressionError
        
        # Test optimization components
        from src.retrieval_free.caching import MemoryCache, TieredCache
        from src.retrieval_free.optimization import BatchProcessor, MemoryOptimizer
        
        assert all([
            InputValidator, ValidationResult, ModelSecurityValidator, SecurityScan,
            MetricsCollector, HealthChecker, RetrievalFreeError, CompressionError,
            MemoryCache, TieredCache, BatchProcessor, MemoryOptimizer
        ])
    
    def test_compressor_factory_integration(self):
        """Test AutoCompressor factory integration."""
        from src.retrieval_free.core.auto_compressor import AutoCompressor, ModelRegistry
        
        # Test model registry
        models = ModelRegistry.list_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Test model creation
        with patch('src.retrieval_free.core.auto_compressor.ContextCompressor') as mock_compressor:
            mock_instance = MagicMock()
            mock_compressor.return_value = mock_instance
            
            compressor = AutoCompressor.create_custom_compressor(
                compressor_type="context",
                compression_ratio=8.0
            )
            
            mock_compressor.assert_called_once()
            mock_instance.load_model.assert_called_once()
    
    def test_validation_security_integration(self):
        """Test validation and security integration."""
        from src.retrieval_free.validation import validate_compression_request
        from src.retrieval_free.security import ModelSecurityValidator
        
        # Test validation with malicious input
        malicious_text = "<script>alert('xss')</script>"
        parameters = {'compression_ratio': 8.0}
        
        result = validate_compression_request(malicious_text, parameters)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.risk_score > 0.5
        
        # Test model security validation
        validator = ModelSecurityValidator()
        
        scan_result = validator.validate_model_source("https://evil-site.com/model")
        assert not scan_result.passed
        assert len(scan_result.vulnerabilities) > 0
    
    def test_caching_optimization_integration(self):
        """Test caching and optimization integration."""
        from src.retrieval_free.caching import TieredCache, MemoryCache, create_cache_key
        from src.retrieval_free.optimization import BatchProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tiered cache
            memory_cache = MemoryCache(max_size=10)
            cache = TieredCache(memory_cache=memory_cache, cache_dir=temp_dir)
            
            # Test cache operations
            cache.put("test_key", "test_value")
            assert cache.get("test_key") == "test_value"
            
            # Test cache key generation
            key = create_cache_key("test text", "model", {"ratio": 8.0})
            assert isinstance(key, str)
            assert "compress" in key
            
            # Test batch processing
            processor = BatchProcessor(batch_size=2)
            
            def process_batch(batch):
                return [f"processed_{item}" for item in batch]
            
            items = ["a", "b", "c", "d"]
            results = processor.process_batch(items, process_batch)
            
            assert len(results) == 4
            assert all("processed_" in result for result in results)
            
            processor.shutdown()
    
    def test_monitoring_health_integration(self):
        """Test monitoring and health check integration."""
        from src.retrieval_free.monitoring import MetricsCollector, HealthChecker, HealthStatus
        
        # Test metrics collection
        collector = MetricsCollector()
        
        collector.record_compression(
            input_tokens=1000,
            output_tokens=125,
            processing_time_ms=500.0,
            model_name="test-model"
        )
        
        stats = collector.get_summary_stats()
        assert stats['total_operations'] == 1
        
        # Test health checking
        checker = HealthChecker()
        
        def test_health_check():
            return HealthStatus(
                service="test",
                healthy=True,
                message="OK",
                response_time_ms=10.0,
                timestamp=1234567890
            )
        
        checker.register_check("test", test_health_check)
        result = checker.run_check("test")
        
        assert result.healthy
        assert result.service == "test"
    
    def test_exception_handling_integration(self):
        """Test exception handling across components."""
        from src.retrieval_free.exceptions import (
            RetrievalFreeError, CompressionError, ValidationError,
            handle_exception, create_error_response
        )
        
        # Test exception creation
        error = CompressionError(
            "Test compression error",
            input_length=1000,
            model_name="test-model"
        )
        
        assert error.error_code == "COMPRESSION_FAILED"
        assert error.details['input_length'] == 1000
        
        # Test exception conversion
        original_error = ValueError("Test error")
        converted = handle_exception(original_error, "test_context", reraise=False)
        
        assert isinstance(converted, RetrievalFreeError)
        assert "test_context" in str(converted)
        
        # Test error response
        response = create_error_response(converted)
        assert response['success'] is False
        assert 'error' in response
    
    def test_plugin_integration(self):
        """Test plugin system integration."""
        from src.retrieval_free.plugins import CompressorPlugin
        
        # Mock transformer components
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(100))  # 100 tokens
        
        # Create plugin with mocked compressor
        with patch('src.retrieval_free.plugins.AutoCompressor') as mock_auto_compressor:
            mock_compressor = MagicMock()
            mock_auto_compressor.from_pretrained.return_value = mock_compressor
            
            plugin = CompressorPlugin(
                model=mock_model,
                tokenizer=mock_tokenizer,
                compressor="test-compressor",
                compression_threshold=50
            )
            
            # Mock compression result
            mock_result = MagicMock()
            mock_result.mega_tokens = [MagicMock(), MagicMock()]
            mock_compressor.compress.return_value = mock_result
            
            # Test auto-compression (should trigger due to 100 > 50 threshold)
            with patch.object(plugin, '_mega_tokens_to_input', return_value="compressed"):
                mock_tokenizer.return_value = {"input_ids": [[1, 2, 3]]}
                mock_model.generate.return_value = [[1, 2, 3, 4, 5]]
                mock_tokenizer.decode.return_value = "compressedgenerated text"
                
                result = plugin.generate("long input text", max_new_tokens=10)
                
                # Should have attempted compression
                mock_compressor.compress.assert_called_once()
                assert isinstance(result, str)
    
    def test_streaming_processing_integration(self):
        """Test streaming processing integration."""
        from src.retrieval_free.streaming import StreamingCompressor
        
        with patch('src.retrieval_free.streaming.ContextCompressor') as mock_base:
            mock_instance = MagicMock()
            mock_base.return_value = mock_instance
            
            # Mock compression result
            mock_result = MagicMock()
            mock_result.mega_tokens = [MagicMock(), MagicMock()]
            mock_instance.compress.return_value = mock_result
            
            # Create streaming compressor
            compressor = StreamingCompressor(
                window_size=1000,
                compression_ratio=8.0
            )
            
            # Test streaming processing
            chunk1 = "First chunk of streaming text."
            chunk2 = "Second chunk of streaming text."
            
            tokens1 = compressor.add_chunk(chunk1)
            tokens2 = compressor.add_chunk(chunk2)
            
            # Should accumulate tokens
            assert isinstance(tokens1, list)
            assert isinstance(tokens2, list)
            
            # Test query functionality
            query_result = compressor.query("test query", top_k=5)
            assert isinstance(query_result, list)
    
    def test_selective_compression_integration(self):
        """Test selective compression integration."""
        from src.retrieval_free.selective import SelectiveCompressor
        
        with patch('src.retrieval_free.selective.ContextCompressor') as mock_base:
            mock_instance = MagicMock()
            mock_base.return_value = mock_instance
            
            # Mock compression result
            mock_result = MagicMock()
            mock_result.mega_tokens = [MagicMock()]
            mock_result.original_length = 100
            mock_result.compressed_length = 12
            mock_result.compression_ratio = 8.0
            mock_instance.compress.return_value = mock_result
            
            # Create selective compressor
            compressor = SelectiveCompressor()
            
            # Test content analysis
            test_text = "This is a legal document with hereby and pursuant clauses."
            analysis = compressor.get_content_analysis(test_text)
            
            assert 'total_segments' in analysis
            assert 'content_distribution' in analysis
            assert 'recommended_ratios' in analysis
            
            # Test selective compression
            result = compressor.compress(test_text)
            
            assert hasattr(result, 'mega_tokens')
            assert hasattr(result, 'metadata')
            assert result.metadata.get('selective_compression') is True
    
    def test_full_pipeline_simulation(self):
        """Test complete pipeline with all components."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        from src.retrieval_free.validation import validate_compression_request
        from src.retrieval_free.monitoring import MetricsCollector
        
        # Mock all dependencies
        with patch('src.retrieval_free.core.context_compressor.validate_compression_request') as mock_validate:
            with patch('src.retrieval_free.core.context_compressor.TieredCache') as mock_cache_class:
                
                # Setup validation mock
                from src.retrieval_free.validation import ValidationResult
                mock_validate.return_value = ValidationResult(
                    is_valid=True,
                    errors=[],
                    warnings=[],
                    sanitized_input={'text': 'test text'},
                    risk_score=0.0
                )
                
                # Setup cache mock
                mock_cache = MagicMock()
                mock_cache.get.return_value = None  # Cache miss
                mock_cache_class.return_value = mock_cache
                
                # Create compressor
                compressor = ContextCompressor()
                
                # Mock model loading and processing
                with patch.object(compressor, 'load_model'):
                    with patch.object(compressor, '_chunk_text', return_value=['chunk1', 'chunk2']):
                        with patch.object(compressor, '_encode_chunks_optimized') as mock_encode:
                            with patch.object(compressor, '_cluster_embeddings') as mock_cluster:
                                
                                # Setup encoding mock
                                mock_embedding = MagicMock()
                                mock_embedding.shape = [2, 384]
                                mock_encode.return_value = mock_embedding
                                
                                # Setup clustering mock
                                mock_centers = [MagicMock(), MagicMock()]
                                mock_labels = [0, 1]
                                mock_cluster.return_value = (mock_centers, mock_labels)
                                
                                # Run compression
                                result = compressor.compress("Test document for compression pipeline")
                                
                                # Verify result
                                assert hasattr(result, 'mega_tokens')
                                assert hasattr(result, 'compression_ratio')
                                assert hasattr(result, 'processing_time')
                                assert len(result.mega_tokens) == 2
                                
                                # Verify validation was called
                                mock_validate.assert_called_once()
                                
                                # Verify caching was attempted
                                mock_cache.get.assert_called_once()
                                mock_cache.put.assert_called_once()


class TestPerformanceIntegration:
    """Test performance aspects of integration."""
    
    def test_benchmark_integration(self):
        """Test benchmark runner integration."""
        # Import benchmark runner
        import sys
        import os
        
        # Add benchmarks to path
        benchmark_path = os.path.join(os.path.dirname(__file__), '..', 'benchmarks')
        if benchmark_path not in sys.path:
            sys.path.insert(0, benchmark_path)
        
        from benchmark_runner import (
            BenchmarkRunner, create_compression_benchmark,
            mock_compression_function
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = BenchmarkRunner(temp_dir)
            config = create_compression_benchmark()
            
            # Create minimal test data
            test_data = ["test"] * 10
            
            # Run benchmark  
            result = runner.run_benchmark(config, mock_compression_function, test_data)
            
            assert result.name == "compression_benchmark"
            assert result.success_rate == 1.0
            assert result.duration_ms > 0
    
    def test_security_scan_integration(self):
        """Test security scanner integration."""
        from src.retrieval_free.security import scan_for_vulnerabilities
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test file with potential issues
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("import os\nprint('hello')")  # Safe code
            
            # Run scan
            result = scan_for_vulnerabilities(temp_dir)
            
            assert hasattr(result, 'passed')
            assert hasattr(result, 'vulnerabilities')
            assert hasattr(result, 'warnings')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])