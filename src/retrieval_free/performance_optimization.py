"""
Performance Optimization Framework - Generation 3
Advanced performance optimization and scaling capabilities.
"""

import time
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import logging
from queue import Queue, Empty
from collections import defaultdict
import json
import gc
import resource
import sys
from functools import lru_cache, wraps

from .observability import PerformanceMonitor, MetricsCollector
from .exceptions import ResourceError, ScalingError

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    
    # Threading and concurrency
    max_workers: int = 4
    enable_async: bool = True
    queue_size: int = 1000
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # Memory management
    gc_threshold: int = 100  # Number of operations before GC
    max_memory_mb: int = 2048
    
    # Performance monitoring
    enable_profiling: bool = True
    profile_interval_seconds: int = 60
    
    # Batch processing
    batch_size: int = 10
    batch_timeout_seconds: float = 1.0
    
    # Resource limits
    cpu_limit_percent: float = 80.0
    memory_limit_percent: float = 85.0


class PerformanceProfiler:
    """Profiles and analyzes performance metrics."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.function_times = defaultdict(list)
        self.memory_samples = []
        self.cpu_samples = []
    
    def profile_function(self, func_name: str = None):
        """Decorator to profile function performance."""
        def decorator(func):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Record successful execution
                    duration = time.perf_counter() - start_time
                    end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    memory_delta = end_memory - start_memory
                    
                    self.function_times[name].append(duration)
                    # Store timing directly in timers dict
                    self.metrics.timers[f"function.{name}"] = duration
                    self.metrics.set_gauge(f"memory.{name}", memory_delta)
                    
                    return result
                    
                except Exception as e:
                    # Record failed execution
                    self.metrics.increment(f"errors.{name}")
                    raise
                    
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": time.time(),
            "function_stats": {},
            "system_stats": self.metrics.get_all_metrics(),
        }
        
        for func_name, times in self.function_times.items():
            if times:
                report["function_stats"][func_name] = {
                    "call_count": len(times),
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                }
        
        return report


class CacheManager:
    """Advanced caching with TTL and memory management."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.cache = {}
        self.access_times = {}
        self.creation_times = {}
        self.metrics = MetricsCollector()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.config.enable_caching:
            return None
        
        if key not in self.cache:
            self.metrics.increment("cache.miss")
            return None
        
        # Check TTL
        if time.time() - self.creation_times[key] > self.config.cache_ttl_seconds:
            self._evict(key)
            self.metrics.increment("cache.expired")
            return None
        
        # Update access time
        self.access_times[key] = time.time()
        self.metrics.increment("cache.hit")
        
        return self.cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if not self.config.enable_caching:
            return
        
        # Check if we need to evict items
        if len(self.cache) >= self.config.cache_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.creation_times[key] = time.time()
        
        self.metrics.increment("cache.set")
        self.metrics.set_gauge("cache.size", len(self.cache))
    
    def _evict(self, key: str) -> None:
        """Evict specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
            
            self.metrics.increment("cache.evicted")
            self.metrics.set_gauge("cache.size", len(self.cache))
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._evict(lru_key)
        
        logger.debug(f"Evicted LRU cache item: {lru_key}")
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.access_times.clear()
        self.creation_times.clear()
        self.metrics.increment("cache.cleared")
        self.metrics.set_gauge("cache.size", 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.config.cache_size,
            "ttl_seconds": self.config.cache_ttl_seconds,
            "metrics": self.metrics.get_all_metrics()
        }


class BatchProcessor:
    """Processes items in optimized batches."""
    
    def __init__(self, 
                 process_func: Callable,
                 config: Optional[PerformanceConfig] = None):
        self.process_func = process_func
        self.config = config or PerformanceConfig()
        self.queue = Queue(maxsize=self.config.queue_size)
        self.results = {}
        self.metrics = MetricsCollector()
        self._processing = False
        self._thread = None
    
    def start_processing(self):
        """Start background batch processing."""
        if self._processing:
            return
        
        self._processing = True
        self._thread = threading.Thread(target=self._process_batches, daemon=True)
        self._thread.start()
        
        logger.info("Started batch processing thread")
    
    def stop_processing(self):
        """Stop background batch processing."""
        self._processing = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("Stopped batch processing thread")
    
    def submit(self, item_id: str, item: Any) -> None:
        """Submit item for batch processing."""
        try:
            self.queue.put((item_id, item), timeout=1.0)
            self.metrics.increment("batch.submitted")
        except Exception as e:
            self.metrics.increment("batch.submit_failed")
            raise ResourceError(f"Failed to submit item for processing: {e}")
    
    def get_result(self, item_id: str, timeout: float = 5.0) -> Any:
        """Get processing result for item."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if item_id in self.results:
                result = self.results.pop(item_id)
                self.metrics.increment("batch.result_retrieved")
                return result
            time.sleep(0.1)
        
        self.metrics.increment("batch.timeout")
        raise ResourceError(f"Timeout waiting for result: {item_id}")
    
    def _process_batches(self):
        """Background batch processing loop."""
        batch = []
        batch_start_time = time.time()
        
        while self._processing:
            try:
                # Try to get item from queue
                try:
                    item_id, item = self.queue.get(timeout=0.1)
                    batch.append((item_id, item))
                except Empty:
                    pass
                
                # Process batch if conditions are met
                should_process = (
                    len(batch) >= self.config.batch_size or
                    (batch and time.time() - batch_start_time >= self.config.batch_timeout_seconds)
                )
                
                if should_process and batch:
                    self._process_batch(batch)
                    batch = []
                    batch_start_time = time.time()
                    
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                self.metrics.increment("batch.error")
        
        # Process remaining items
        if batch:
            self._process_batch(batch)
    
    def _process_batch(self, batch: List[Tuple[str, Any]]):
        """Process a batch of items."""
        start_time = time.perf_counter()
        
        try:
            # Extract items for processing
            items = [item for _, item in batch]
            
            # Process batch
            results = self.process_func(items)
            
            # Store results
            for (item_id, _), result in zip(batch, results):
                self.results[item_id] = result
            
            # Record metrics
            duration = time.perf_counter() - start_time
            # Store timing directly in timers dict
            self.metrics.timers["batch.process_time"] = duration
            self.metrics.increment("batch.processed")
            self.metrics.set_gauge("batch.size", len(batch))
            
            logger.debug(f"Processed batch of {len(batch)} items in {duration:.3f}s")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.metrics.increment("batch.failed")
            
            # Store error for all items in batch
            for item_id, _ in batch:
                self.results[item_id] = ResourceError(f"Batch processing failed: {e}")


class AutoScaler:
    """Automatically scales resources based on load."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.monitor = PerformanceMonitor()
        self.metrics = MetricsCollector()
        self.current_workers = self.config.max_workers
        self._executor = None
        self._scaling_enabled = True
    
    def get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor."""
        if self._executor is None or self._executor._threads is None:
            self._executor = ThreadPoolExecutor(max_workers=self.current_workers)
        return self._executor
    
    def submit_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit task with automatic scaling."""
        if self._scaling_enabled:
            self._check_and_scale()
        
        executor = self.get_executor()
        future = executor.submit(func, *args, **kwargs)
        
        self.metrics.increment("autoscaler.task_submitted")
        return future
    
    def _check_and_scale(self):
        """Check system load and scale if necessary."""
        try:
            system_metrics = self.monitor.get_system_metrics()
            cpu_percent = system_metrics.get("cpu_percent", 0)
            memory_mb = system_metrics.get("memory_mb", 0)
            
            # Calculate resource utilization
            memory_percent = (memory_mb / self.config.max_memory_mb) * 100
            
            # Scale up conditions
            should_scale_up = (
                cpu_percent > self.config.cpu_limit_percent * 0.8 and
                self.current_workers < self.config.max_workers
            )
            
            # Scale down conditions  
            should_scale_down = (
                cpu_percent < self.config.cpu_limit_percent * 0.3 and
                self.current_workers > 1
            )
            
            if should_scale_up:
                self._scale_up()
            elif should_scale_down:
                self._scale_down()
            
            # Record metrics
            self.metrics.set_gauge("autoscaler.workers", self.current_workers)
            self.metrics.set_gauge("autoscaler.cpu_percent", cpu_percent)
            self.metrics.set_gauge("autoscaler.memory_percent", memory_percent)
            
        except Exception as e:
            logger.error(f"Error in auto-scaling: {e}")
            self.metrics.increment("autoscaler.error")
    
    def _scale_up(self):
        """Increase number of workers."""
        old_workers = self.current_workers
        self.current_workers = min(self.current_workers + 1, self.config.max_workers)
        
        if self.current_workers != old_workers:
            # Recreate executor with new worker count
            if self._executor:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=self.current_workers)
            
            self.metrics.increment("autoscaler.scaled_up")
            logger.info(f"Scaled up from {old_workers} to {self.current_workers} workers")
    
    def _scale_down(self):
        """Decrease number of workers."""
        old_workers = self.current_workers
        self.current_workers = max(self.current_workers - 1, 1)
        
        if self.current_workers != old_workers:
            # Recreate executor with new worker count
            if self._executor:
                self._executor.shutdown(wait=False)
            self._executor = ThreadPoolExecutor(max_workers=self.current_workers)
            
            self.metrics.increment("autoscaler.scaled_down")
            logger.info(f"Scaled down from {old_workers} to {self.current_workers} workers")


class MemoryOptimizer:
    """Optimizes memory usage and prevents leaks."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        self.operation_count = 0
        self.metrics = MetricsCollector()
    
    def optimize_operation(self, func: Callable):
        """Decorator to optimize memory usage for operations."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Track operation
            self.operation_count += 1
            
            # Run garbage collection periodically
            if self.operation_count % self.config.gc_threshold == 0:
                self._run_garbage_collection()
            
            # Check memory usage
            self._check_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Clean up after operation
                self._cleanup_after_operation()
            
        return wrapper
    
    def _run_garbage_collection(self):
        """Run garbage collection and record metrics."""
        start_time = time.perf_counter()
        collected = gc.collect()
        duration = time.perf_counter() - start_time
        
        self.metrics.increment("gc.runs")
        self.metrics.set_gauge("gc.collected", collected)
        # Store timing directly in timers dict
        self.metrics.timers["gc.duration"] = duration
        
        logger.debug(f"Garbage collection collected {collected} objects in {duration:.3f}s")
    
    def _check_memory_usage(self):
        """Check and warn about high memory usage."""
        try:
            memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            memory_mb = memory_usage / (1024 * 1024)  # Convert to MB on Linux
            
            if memory_mb > self.config.max_memory_mb:
                logger.warning(f"High memory usage: {memory_mb:.1f}MB exceeds limit {self.config.max_memory_mb}MB")
                self.metrics.increment("memory.high_usage_warning")
                
                # Force garbage collection on high usage
                self._run_garbage_collection()
                
            self.metrics.set_gauge("memory.current_mb", memory_mb)
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
    
    def _cleanup_after_operation(self):
        """Cleanup resources after operation."""
        # Clear any temporary variables from local scope
        # This is a placeholder for more sophisticated cleanup
        pass


# Global instances for performance optimization
_performance_profiler = PerformanceProfiler()
_cache_manager = CacheManager()
_auto_scaler = AutoScaler()
_memory_optimizer = MemoryOptimizer()


def optimized_function(cache_key: Optional[str] = None, profile: bool = True):
    """Decorator for comprehensive function optimization."""
    def decorator(func):
        # Apply memory optimization
        func = _memory_optimizer.optimize_operation(func)
        
        # Apply profiling if enabled
        if profile:
            func = _performance_profiler.profile_function()(func)
        
        # Apply caching if cache_key provided
        if cache_key:
            original_func = func
            
            @wraps(original_func)
            def cached_wrapper(*args, **kwargs):
                # Generate cache key
                key = f"{cache_key}:{hash(str(args))}{hash(str(sorted(kwargs.items())))}"
                
                # Try cache first
                cached_result = _cache_manager.get(key)
                if cached_result is not None:
                    return cached_result
                
                # Compute and cache result
                result = original_func(*args, **kwargs)
                _cache_manager.set(key, result)
                
                return result
            
            func = cached_wrapper
        
        return func
    return decorator


def get_performance_status() -> Dict[str, Any]:
    """Get comprehensive performance status."""
    return {
        "profiler": _performance_profiler.get_performance_report(),
        "cache": _cache_manager.get_stats(),
        "autoscaler": {
            "current_workers": _auto_scaler.current_workers,
            "max_workers": _auto_scaler.config.max_workers,
            "metrics": _auto_scaler.metrics.get_all_metrics()
        },
        "memory": {
            "operation_count": _memory_optimizer.operation_count,
            "gc_threshold": _memory_optimizer.config.gc_threshold,
            "metrics": _memory_optimizer.metrics.get_all_metrics()
        }
    }