"""Run integration tests without pytest dependency."""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock dependencies
from unittest.mock import MagicMock, patch

# Setup global mocks
mock_modules = {
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
    'numpy': MagicMock(),
    'tqdm': MagicMock(),
}

for name, mock_module in mock_modules.items():
    sys.modules[name] = mock_module


class TestRunner:
    """Simple test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def run_test(self, test_name, test_func):
        """Run a single test."""
        print(f"Running {test_name}... ", end="")
        try:
            test_func()
            print("✅ PASSED")
            self.passed += 1
        except Exception as e:
            print("❌ FAILED")
            self.failed += 1
            self.errors.append((test_name, str(e), traceback.format_exc()))
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed} passed, {self.failed} failed")
        
        if self.errors:
            print(f"\nFailures:")
            for test_name, error, tb in self.errors:
                print(f"\n{test_name}:")
                print(f"  Error: {error}")
        
        return self.failed == 0


def test_system_imports():
    """Test that all modules can be imported."""
    # Test main package imports
    from retrieval_free import __version__
    assert __version__ == "0.1.0"
    
    # Test core components
    from retrieval_free.core.base import CompressorBase, MegaToken, CompressionResult
    assert CompressorBase is not None
    assert MegaToken is not None
    assert CompressionResult is not None
    
    # Test validation components
    from retrieval_free.validation import InputValidator, ValidationResult
    from retrieval_free.security import ModelSecurityValidator, SecurityScan
    from retrieval_free.monitoring import MetricsCollector, HealthChecker
    from retrieval_free.exceptions import RetrievalFreeError, CompressionError
    
    # Test optimization components
    from retrieval_free.caching import MemoryCache, TieredCache
    from retrieval_free.optimization import BatchProcessor, MemoryOptimizer
    
    assert all([
        InputValidator, ValidationResult, ModelSecurityValidator, SecurityScan,
        MetricsCollector, HealthChecker, RetrievalFreeError, CompressionError,
        MemoryCache, TieredCache, BatchProcessor, MemoryOptimizer
    ])


def test_compressor_factory():
    """Test AutoCompressor factory."""
    from retrieval_free.core.auto_compressor import AutoCompressor, ModelRegistry
    
    # Test model registry
    models = ModelRegistry.list_models()
    assert isinstance(models, dict)
    assert len(models) > 0
    
    # Test model creation
    with patch('retrieval_free.core.auto_compressor.ContextCompressor') as mock_compressor:
        mock_instance = MagicMock()
        mock_compressor.return_value = mock_instance
        
        compressor = AutoCompressor.create_custom_compressor(
            compressor_type="context",
            compression_ratio=8.0
        )
        
        mock_compressor.assert_called_once()
        mock_instance.load_model.assert_called_once()


def test_validation_security():
    """Test validation and security integration."""
    from retrieval_free.validation import validate_compression_request
    from retrieval_free.security import ModelSecurityValidator
    
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


def test_caching_optimization():
    """Test caching and optimization integration."""
    from retrieval_free.caching import TieredCache, MemoryCache, create_cache_key
    from retrieval_free.optimization import BatchProcessor
    import tempfile
    
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


def test_monitoring_health():
    """Test monitoring and health check integration."""
    from retrieval_free.monitoring import MetricsCollector, HealthChecker, HealthStatus
    
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


def test_exception_handling():
    """Test exception handling across components."""
    from retrieval_free.exceptions import (
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


def test_full_pipeline_simulation():
    """Test complete pipeline with all components."""
    from retrieval_free.core.context_compressor import ContextCompressor
    from retrieval_free.validation import ValidationResult
    
    # Mock all dependencies
    with patch('retrieval_free.core.context_compressor.validate_compression_request') as mock_validate:
        with patch('retrieval_free.core.context_compressor.TieredCache') as mock_cache_class:
            
            # Setup validation mock  
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


def main():
    """Run all integration tests."""
    print("Running Integration Tests")
    print("=" * 40)
    
    runner = TestRunner()
    
    # Run all tests
    runner.run_test("test_system_imports", test_system_imports)
    runner.run_test("test_compressor_factory", test_compressor_factory)
    runner.run_test("test_validation_security", test_validation_security)
    runner.run_test("test_caching_optimization", test_caching_optimization)
    runner.run_test("test_monitoring_health", test_monitoring_health)
    runner.run_test("test_exception_handling", test_exception_handling)
    runner.run_test("test_full_pipeline_simulation", test_full_pipeline_simulation)
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("\n✅ All integration tests passed!")
        return 0
    else:
        print("\n❌ Some integration tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())