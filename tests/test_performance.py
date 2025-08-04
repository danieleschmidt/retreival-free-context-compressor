"""Performance and optimization tests."""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import time
import threading
from concurrent.futures import Future


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies for testing."""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'transformers': MagicMock(),
        'sklearn': MagicMock(),
        'psutil': MagicMock(),
        'faiss': MagicMock(),
    }):
        yield


class TestCachingSystem:
    """Test caching system performance."""
    
    def test_memory_cache_performance(self):
        """Test memory cache operations."""
        from src.retrieval_free.caching import MemoryCache
        
        cache = MemoryCache(max_size=100)
        
        # Test basic operations
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("missing_key") is None
        
        # Test statistics
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        assert stats['size'] == 1
    
    def test_memory_cache_eviction(self):
        """Test LRU eviction in memory cache."""
        from src.retrieval_free.caching import MemoryCache
        
        # Small cache for testing eviction
        cache = MemoryCache(max_size=2)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Add third item (should evict first)
        cache.put("key3", "value3")
        
        # First key should be evicted
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        from src.retrieval_free.caching import MemoryCache
        
        cache = MemoryCache()
        
        # Put with short TTL
        cache.put("ttl_key", "ttl_value", ttl=0.1)  # 100ms
        
        # Should be available immediately
        assert cache.get("ttl_key") == "ttl_value"
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        assert cache.get("ttl_key") is None
    
    def test_cache_key_generation(self):
        """Test cache key generation consistency."""
        from src.retrieval_free.caching import create_cache_key
        
        # Same inputs should generate same key
        key1 = create_cache_key("test text", "model", {"ratio": 8.0})
        key2 = create_cache_key("test text", "model", {"ratio": 8.0})
        assert key1 == key2
        
        # Different inputs should generate different keys
        key3 = create_cache_key("different text", "model", {"ratio": 8.0})
        assert key1 != key3
    
    def test_tiered_cache(self):
        """Test tiered cache functionality."""
        from src.retrieval_free.caching import TieredCache, MemoryCache
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tiered cache
            memory_cache = MemoryCache(max_size=2)
            cache = TieredCache(memory_cache=memory_cache, cache_dir=temp_dir)
            
            # Test put/get
            cache.put("test_key", "test_value")
            
            # Should be in memory cache
            assert memory_cache.get("test_key") == "test_value"
            
            # Should also get from tiered cache
            assert cache.get("test_key") == "test_value"


class TestBatchProcessing:
    """Test batch processing performance."""
    
    def test_batch_processor_creation(self):
        """Test batch processor creation."""
        from src.retrieval_free.optimization import BatchProcessor
        
        processor = BatchProcessor(batch_size=4, max_workers=2)
        
        assert processor.batch_size == 4
        assert processor.max_workers == 2
        
        processor.shutdown()
    
    def test_batch_processing_functionality(self):
        """Test batch processing with mock data."""
        from src.retrieval_free.optimization import BatchProcessor
        
        processor = BatchProcessor(batch_size=2)
        
        # Mock processing function
        def process_batch(batch):
            return [f"processed_{item}" for item in batch]
        
        # Test data
        items = ["item1", "item2", "item3", "item4", "item5"]
        
        # Process
        results = processor.process_batch(items, process_batch)
        
        # Verify results
        assert len(results) == 5
        assert all("processed_" in result for result in results)
        
        processor.shutdown()
    
    def test_batch_processing_progress(self):
        """Test batch processing with progress callback."""
        from src.retrieval_free.optimization import BatchProcessor
        
        processor = BatchProcessor(batch_size=2)
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        def process_batch(batch):
            return batch
        
        items = ["a", "b", "c", "d"]
        
        results = processor.process_batch(
            items, 
            process_batch,
            progress_callback=progress_callback
        )
        
        # Should have received progress updates
        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == progress_calls[-1][1]  # Final should be complete
        
        processor.shutdown()


class TestMemoryOptimization:
    """Test memory optimization features."""
    
    def test_memory_optimizer_creation(self):
        """Test memory optimizer creation."""
        from src.retrieval_free.optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        assert optimizer is not None
    
    def test_memory_context_manager(self):
        """Test memory-efficient context manager."""
        from src.retrieval_free.optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Should not raise any errors
        with optimizer.memory_efficient_context():
            pass
        
        # Test with parameters
        with optimizer.memory_efficient_context(clear_cache=True, gradient_checkpointing=True):
            pass
    
    @patch('src.retrieval_free.optimization.psutil')
    def test_memory_usage_tracking(self, mock_psutil):
        """Test memory usage tracking."""
        from src.retrieval_free.optimization import MemoryOptimizer
        
        # Mock memory info
        mock_memory = MagicMock()
        mock_memory.total = 8 * 1024**3  # 8GB
        mock_memory.used = 4 * 1024**3   # 4GB
        mock_memory.available = 4 * 1024**3  # 4GB
        mock_memory.percent = 50.0
        
        mock_psutil.virtual_memory.return_value = mock_memory
        
        optimizer = MemoryOptimizer()
        stats = optimizer.get_memory_usage()
        
        assert 'system_total_mb' in stats
        assert stats['system_total_mb'] == 8192  # 8GB in MB
        assert stats['system_percent'] == 50.0
    
    @patch('src.retrieval_free.optimization.torch')
    def test_tensor_optimization(self, mock_torch):
        """Test tensor memory optimization."""
        from src.retrieval_free.optimization import MemoryOptimizer
        
        # Mock tensor
        mock_tensor = MagicMock()
        mock_tensor.dtype = mock_torch.float64
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.dim.return_value = 4
        mock_tensor.contiguous.return_value = mock_tensor
        mock_tensor.is_contiguous.return_value = False
        
        optimizer = MemoryOptimizer()
        result = optimizer.optimize_tensor_memory(mock_tensor)
        
        # Should have attempted optimization
        mock_tensor.float.assert_called_once()
        mock_tensor.contiguous.assert_called()


class TestConcurrencyManagement:
    """Test concurrency management."""
    
    def test_concurrency_manager_creation(self):
        """Test concurrency manager creation."""
        from src.retrieval_free.optimization import ConcurrencyManager
        
        manager = ConcurrencyManager(max_concurrent_operations=4)
        assert manager.max_concurrent == 4
        
        manager.shutdown()
    
    def test_concurrency_slot_management(self):
        """Test concurrency slot acquisition."""
        from src.retrieval_free.optimization import ConcurrencyManager
        
        manager = ConcurrencyManager(max_concurrent_operations=1)
        
        # Test acquiring slot
        with manager.acquire_slot():
            # Should have slot
            pass
        
        manager.shutdown()
    
    def test_task_submission(self):
        """Test task submission to thread pools."""
        from src.retrieval_free.optimization import ConcurrencyManager
        
        manager = ConcurrencyManager(max_concurrent_operations=2)
        
        def test_task(x):
            return x * 2
        
        # Submit compute task
        future = manager.submit_compute_task(test_task, 5)
        result = future.result()
        assert result == 10
        
        # Submit I/O task
        future = manager.submit_io_task(test_task, 3)
        result = future.result()
        assert result == 6
        
        manager.shutdown()
    
    def test_concurrent_task_processing(self):
        """Test processing multiple tasks concurrently."""
        from src.retrieval_free.optimization import ConcurrencyManager
        
        manager = ConcurrencyManager(max_concurrent_operations=2)
        
        def multiply_task(x):
            return x * 2
        
        # Create tasks
        tasks = [
            (multiply_task, (1,), {}),
            (multiply_task, (2,), {}),
            (multiply_task, (3,), {}),
        ]
        
        results = manager.process_concurrent_tasks(tasks, task_type="compute")
        
        # Should have results for all tasks (order may vary)
        assert len(results) == 3
        assert set(results) == {2, 4, 6}
        
        manager.shutdown()


class TestPerformanceProfiling:
    """Test performance profiling utilities."""
    
    def test_profile_decorator(self):
        """Test function profiling decorator."""
        from src.retrieval_free.optimization import profile_function
        
        call_count = 0
        
        @profile_function
        def test_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Small delay
            return x * 2
        
        result = test_function(5)
        
        assert result == 10
        assert call_count == 1
    
    def test_performance_tuner(self):
        """Test performance tuning functionality."""
        from src.retrieval_free.optimization import PerformanceTuner
        
        tuner = PerformanceTuner()
        
        # Mock processing function
        def mock_process_func(batch):
            time.sleep(0.001)  # Small delay
            return [f"processed_{item}" for item in batch]
        
        # Test data
        test_data = [f"item_{i}" for i in range(20)]
        
        # Test batch size tuning (with limited options for speed)
        optimal_batch_size = tuner.tune_batch_size(
            mock_process_func,
            test_data,
            batch_sizes=[2, 4]  # Limited for testing
        )
        
        assert optimal_batch_size in [2, 4]
    
    def test_worker_count_tuning(self):
        """Test worker count tuning."""
        from src.retrieval_free.optimization import PerformanceTuner
        
        tuner = PerformanceTuner()
        
        def mock_process_func(batch):
            time.sleep(0.001)
            return batch
        
        test_data = ["item1", "item2", "item3", "item4"]
        
        # Test with limited worker options
        optimal_workers = tuner.tune_worker_count(
            mock_process_func,
            test_data,
            max_workers_range=[1, 2]  # Limited for testing
        )
        
        assert optimal_workers in [1, 2]


class TestModelOptimization:
    """Test model optimization features."""
    
    @patch('src.retrieval_free.optimization.torch')
    def test_model_optimizer_creation(self, mock_torch):
        """Test model optimizer creation."""
        from src.retrieval_free.optimization import ModelOptimizer
        
        mock_torch.cuda.is_available.return_value = False
        
        optimizer = ModelOptimizer()
        assert optimizer.device == "cpu"
    
    @patch('src.retrieval_free.optimization.torch')
    def test_model_optimization(self, mock_torch):
        """Test model optimization pipeline."""
        from src.retrieval_free.optimization import ModelOptimizer
        
        # Mock PyTorch components
        mock_torch.cuda.is_available.return_value = False
        mock_torch.jit.trace.return_value = MagicMock()
        
        # Mock model
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.modules.return_value = []
        mock_model.parameters.return_value = [MagicMock()]
        
        optimizer = ModelOptimizer(device="cpu")
        
        optimized_model = optimizer.optimize_model(
            mock_model,
            use_mixed_precision=False,
            use_jit=False,
            optimize_for_inference=True
        )
        
        # Should have called optimization methods
        mock_model.to.assert_called_with("cpu")
        mock_model.eval.assert_called()


class TestMetricsCollection:
    """Test metrics collection performance."""
    
    def test_metrics_collector_creation(self):
        """Test metrics collector creation."""
        from src.retrieval_free.monitoring import MetricsCollector
        
        collector = MetricsCollector(max_history=100)
        assert collector.max_history == 100
    
    def test_compression_metrics_recording(self):
        """Test recording compression metrics."""
        from src.retrieval_free.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record metrics
        collector.record_compression(
            input_tokens=1000,
            output_tokens=125,
            processing_time_ms=500.0,
            memory_usage_mb=100.0,
            model_name="test-model"
        )
        
        # Check counters
        assert collector.counters['total_compressions'] == 1
        assert collector.counters['total_input_tokens'] == 1000
        assert collector.counters['total_output_tokens'] == 125
        
        # Check gauges
        assert collector.gauges['last_compression_ratio'] == 8.0  # 1000/125
        assert collector.gauges['last_processing_time_ms'] == 500.0
    
    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        from src.retrieval_free.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record multiple metrics
        for i in range(5):
            collector.record_compression(
                input_tokens=1000 + i * 100,
                output_tokens=125 + i * 10,
                processing_time_ms=500.0 + i * 50,
                model_name="test-model"
            )
        
        # Get summary
        stats = collector.get_summary_stats(window_minutes=60)
        
        assert stats['total_operations'] == 5
        assert 'compression_ratio' in stats
        assert 'processing_time_ms' in stats
        assert 'mean' in stats['compression_ratio']
        assert 'median' in stats['compression_ratio']
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        from src.retrieval_free.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_compression(1000, 125, 500.0, model_name="test")
        
        # Test JSON export
        json_export = collector.export_metrics("json")
        assert "counters" in json_export
        assert "gauges" in json_export
        
        # Test Prometheus export
        prom_export = collector.export_metrics("prometheus")
        assert "retrieval_free_" in prom_export


class TestHealthChecks:
    """Test health check system."""
    
    def test_health_checker_creation(self):
        """Test health checker creation."""
        from src.retrieval_free.monitoring import HealthChecker
        
        checker = HealthChecker()
        assert checker is not None
    
    def test_health_check_registration(self):
        """Test health check registration."""
        from src.retrieval_free.monitoring import HealthChecker, HealthStatus
        
        checker = HealthChecker()
        
        def dummy_check():
            return HealthStatus(
                service="test",
                healthy=True,
                message="OK",
                response_time_ms=0,
                timestamp=0
            )
        
        checker.register_check("test_check", dummy_check)
        
        # Run the check
        result = checker.run_check("test_check")
        assert result.healthy
        assert result.service == "test"
    
    def test_overall_health_assessment(self):
        """Test overall health assessment."""
        from src.retrieval_free.monitoring import HealthChecker, HealthStatus
        
        checker = HealthChecker()
        
        # Register passing check
        def passing_check():
            return HealthStatus("pass", True, "OK", 0, 0)
        
        # Register failing check
        def failing_check():
            return HealthStatus("fail", False, "Error", 0, 0)
        
        checker.register_check("pass", passing_check)
        checker.register_check("fail", failing_check)
        
        # Get overall health
        overall = checker.get_overall_health()
        
        assert not overall['healthy']  # Should fail due to one failing check
        assert overall['total_checks'] == 2
        assert overall['passing_checks'] == 1
        assert overall['failing_checks'] == 1


class TestAlertSystem:
    """Test alert system performance."""
    
    def test_alert_manager_creation(self):
        """Test alert manager creation."""
        from src.retrieval_free.monitoring import AlertManager
        
        manager = AlertManager()
        assert manager is not None
    
    def test_threshold_checking(self):
        """Test threshold checking and alerting."""
        from src.retrieval_free.monitoring import AlertManager
        
        manager = AlertManager()
        
        # Test metrics within thresholds
        normal_metrics = {
            'processing_time_ms': 1000,  # Below warning threshold
            'memory_usage_percent': 70   # Below warning threshold
        }
        
        alerts = manager.check_thresholds(normal_metrics)
        assert len(alerts) == 0  # No alerts expected
        
        # Test metrics exceeding thresholds
        high_metrics = {
            'processing_time_ms': 10000,  # Above warning threshold
            'memory_usage_percent': 95    # Above critical threshold
        }
        
        alerts = manager.check_thresholds(high_metrics)
        assert len(alerts) > 0  # Should have alerts
    
    def test_alert_handler_registration(self):
        """Test alert handler registration."""
        from src.retrieval_free.monitoring import AlertManager
        
        manager = AlertManager()
        
        handled_alerts = []
        
        def test_handler(metric, level, details):
            handled_alerts.append((metric, level, details))
        
        manager.add_alert_handler(test_handler)
        
        # Trigger alert
        high_metrics = {'processing_time_ms': 20000}
        alerts = manager.check_thresholds(high_metrics)
        
        # Should have called handler
        assert len(handled_alerts) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])