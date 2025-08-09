"""Performance optimization components for the retrieval-free context compressor."""

import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Optional

import psutil


logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available - some optimizations will be disabled")


class MemoryOptimizer:
    """Memory optimization utilities."""

    def __init__(self, enable_gc: bool = True, gc_threshold: float = 0.8):
        """Initialize memory optimizer.
        
        Args:
            enable_gc: Whether to enable automatic garbage collection
            gc_threshold: Memory usage threshold (0-1) to trigger GC
        """
        self.enable_gc = enable_gc
        self.gc_threshold = gc_threshold
        self._process = psutil.Process()
        self._baseline_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self._process.memory_info().rss / 1024 / 1024

    def _get_memory_percent(self) -> float:
        """Get current memory usage as percentage of total system memory."""
        return self._process.memory_percent()

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.
        
        Returns:
            True if memory usage is above threshold
        """
        return self._get_memory_percent() > self.gc_threshold * 100

    def force_cleanup(self) -> dict[str, float]:
        """Force memory cleanup and return statistics.
        
        Returns:
            Dictionary with memory statistics before and after cleanup
        """
        memory_before = self._get_memory_usage()

        # Python garbage collection
        if self.enable_gc:
            collected = gc.collect()
            logger.debug(f"Python GC collected {collected} objects")

        # PyTorch cleanup if available
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared PyTorch CUDA cache")

        memory_after = self._get_memory_usage()
        freed_mb = memory_before - memory_after

        stats = {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'freed_mb': freed_mb,
            'freed_percent': (freed_mb / memory_before * 100) if memory_before > 0 else 0
        }

        if freed_mb > 1.0:  # Only log if significant cleanup
            logger.info(f"Memory cleanup freed {freed_mb:.1f}MB ({stats['freed_percent']:.1f}%)")

        return stats

    @contextmanager
    def memory_efficient_context(self, clear_cache: bool = True):
        """Context manager for memory-efficient operations.
        
        Args:
            clear_cache: Whether to clear GPU cache after operation
        """
        initial_memory = self._get_memory_usage()

        try:
            yield
        finally:
            if clear_cache and self.check_memory_pressure():
                self.force_cleanup()

            final_memory = self._get_memory_usage()
            memory_delta = final_memory - initial_memory

            if memory_delta > 100:  # Log significant memory increases
                logger.warning(f"Operation increased memory usage by {memory_delta:.1f}MB")

    def optimize_tensor_memory(self, tensor: 'torch.Tensor') -> 'torch.Tensor':
        """Optimize tensor memory usage.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Memory-optimized tensor
        """
        if not HAS_TORCH:
            return tensor

        # Convert to half precision if possible (saves ~50% memory)
        if tensor.dtype == torch.float32 and tensor.numel() > 1000:
            return tensor.half()

        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            return tensor.contiguous()

        return tensor

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        stats = {
            'current_mb': self._get_memory_usage(),
            'baseline_mb': self._baseline_memory,
            'delta_mb': self._get_memory_usage() - self._baseline_memory,
            'percent_used': self._get_memory_percent(),
            'system_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'system_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
        }

        if HAS_TORCH and torch.cuda.is_available():
            stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['cuda_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024

        return stats


class BatchProcessor:
    """Batch processing utilities for efficient GPU utilization."""

    def __init__(
        self,
        batch_size: int = 8,
        device: str = 'auto',
        max_memory_mb: int | None = None
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Default batch size
            device: Target device ('cpu', 'cuda', 'auto')
            max_memory_mb: Maximum memory usage limit in MB
        """
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.max_memory_mb = max_memory_mb
        self._memory_optimizer = MemoryOptimizer()

    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == 'auto':
            if HAS_TORCH and torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device

    def process_in_batches(
        self,
        items: list[Any],
        process_func: Callable[[list[Any]], Any],
        batch_size: int | None = None,
        show_progress: bool = False
    ) -> list[Any]:
        """Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            batch_size: Optional batch size override
            show_progress: Whether to show progress logging
            
        Returns:
            List of processed results
        """
        batch_size = batch_size or self.batch_size
        results = []

        total_batches = (len(items) + batch_size - 1) // batch_size

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_idx = i // batch_size + 1

            if show_progress:
                logger.info(f"Processing batch {batch_idx}/{total_batches} ({len(batch)} items)")

            # Check memory before processing
            if self.max_memory_mb:
                current_memory = self._memory_optimizer._get_memory_usage()
                if current_memory > self.max_memory_mb:
                    logger.warning(f"Memory usage ({current_memory:.1f}MB) exceeds limit ({self.max_memory_mb}MB)")
                    self._memory_optimizer.force_cleanup()

            # Process batch
            with self._memory_optimizer.memory_efficient_context():
                batch_result = process_func(batch)
                results.append(batch_result)

        return results

    def adaptive_batch_size(
        self,
        items: list[Any],
        process_func: Callable[[list[Any]], Any],
        initial_batch_size: int | None = None,
        max_memory_mb: float = 1024
    ) -> tuple[list[Any], int]:
        """Automatically find optimal batch size based on memory usage.
        
        Args:
            items: Items to process
            process_func: Processing function
            initial_batch_size: Starting batch size
            max_memory_mb: Maximum memory usage in MB
            
        Returns:
            Tuple of (results, optimal_batch_size)
        """
        batch_size = initial_batch_size or self.batch_size
        results = []

        # Start with a small test batch to measure memory usage
        if len(items) > 0:
            test_batch = items[:min(2, len(items))]

            memory_before = self._memory_optimizer._get_memory_usage()
            test_result = process_func(test_batch)
            memory_after = self._memory_optimizer._get_memory_usage()

            memory_per_item = (memory_after - memory_before) / len(test_batch)

            # Calculate optimal batch size based on memory usage
            if memory_per_item > 0:
                optimal_batch_size = min(
                    batch_size,
                    max(1, int(max_memory_mb / memory_per_item))
                )
            else:
                optimal_batch_size = batch_size

            logger.info(f"Adaptive batching: {memory_per_item:.1f}MB/item, using batch_size={optimal_batch_size}")

            # Process remaining items with optimal batch size
            remaining_items = items[len(test_batch):]
            remaining_results = self.process_in_batches(
                remaining_items,
                process_func,
                optimal_batch_size,
                show_progress=True
            )

            results = [test_result] + remaining_results

            return results, optimal_batch_size

        return [], batch_size


class ModelOptimizer:
    """Model-specific optimization utilities."""

    def __init__(self):
        """Initialize model optimizer."""
        self._optimization_cache = {}

    def optimize_model_inference(self, model: Any) -> Any:
        """Apply inference optimizations to model.
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        if not HAS_TORCH:
            return model

        model_id = id(model)

        # Check if already optimized
        if model_id in self._optimization_cache:
            return self._optimization_cache[model_id]

        try:
            # Set to eval mode
            if hasattr(model, 'eval'):
                model = model.eval()

            # Disable gradient computation
            if hasattr(model, 'requires_grad_'):
                for param in model.parameters():
                    param.requires_grad_(False)

            # Apply torch.jit.script if possible (for CPU)
            if hasattr(torch.jit, 'script') and model.training == False:
                try:
                    model = torch.jit.script(model)
                    logger.debug("Applied TorchScript optimization")
                except Exception as e:
                    logger.debug(f"TorchScript optimization failed: {e}")

            # Cache optimized model
            self._optimization_cache[model_id] = model

        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")

        return model

    def optimize_embedding_computation(
        self,
        embeddings: 'torch.Tensor',
        target_dtype: Optional['torch.dtype'] = None
    ) -> 'torch.Tensor':
        """Optimize embedding tensor for computation.
        
        Args:
            embeddings: Input embeddings
            target_dtype: Target data type
            
        Returns:
            Optimized embeddings
        """
        if not HAS_TORCH:
            return embeddings

        # Convert to target dtype for memory/speed optimization
        if target_dtype and embeddings.dtype != target_dtype:
            embeddings = embeddings.to(target_dtype)

        # Ensure contiguous memory layout
        if not embeddings.is_contiguous():
            embeddings = embeddings.contiguous()

        return embeddings


class ConcurrencyOptimizer:
    """Concurrency and threading optimization utilities."""

    def __init__(self, max_workers: int | None = None):
        """Initialize concurrency optimizer.
        
        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers or min(8, (os.cpu_count() or 1) + 4)
        self._thread_local = threading.local()

    def parallel_process(
        self,
        items: list[Any],
        process_func: Callable[[Any], Any],
        max_workers: int | None = None
    ) -> list[Any]:
        """Process items in parallel using ThreadPoolExecutor.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            max_workers: Optional max workers override
            
        Returns:
            List of processed results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        max_workers = max_workers or self.max_workers
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_func, item): i
                for i, item in enumerate(items)
            }

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error processing item {index}: {e}")
                    results[index] = None

        return results

    @contextmanager
    def thread_local_context(self, **kwargs):
        """Context manager for thread-local storage.
        
        Args:
            **kwargs: Key-value pairs to store in thread-local storage
        """
        # Store values
        for key, value in kwargs.items():
            setattr(self._thread_local, key, value)

        try:
            yield self._thread_local
        finally:
            # Clean up
            for key in kwargs:
                if hasattr(self._thread_local, key):
                    delattr(self._thread_local, key)


class PerformanceProfiler:
    """Performance profiling utilities."""

    def __init__(self, enabled: bool = True):
        """Initialize performance profiler.
        
        Args:
            enabled: Whether profiling is enabled
        """
        self.enabled = enabled
        self._profiles: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    @contextmanager
    def profile(self, operation_name: str):
        """Profile an operation.
        
        Args:
            operation_name: Name of the operation being profiled
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time

            with self._lock:
                if operation_name not in self._profiles:
                    self._profiles[operation_name] = []
                self._profiles[operation_name].append(duration)

    def get_stats(self, operation_name: str | None = None) -> dict[str, Any]:
        """Get profiling statistics.
        
        Args:
            operation_name: Optional specific operation to get stats for
            
        Returns:
            Dictionary with profiling statistics
        """
        with self._lock:
            if operation_name:
                if operation_name in self._profiles:
                    times = self._profiles[operation_name]
                    return self._calculate_stats(operation_name, times)
                else:
                    return {}

            # Return stats for all operations
            stats = {}
            for name, times in self._profiles.items():
                stats[name] = self._calculate_stats(name, times)

            return stats

    def _calculate_stats(self, name: str, times: list[float]) -> dict[str, Any]:
        """Calculate statistics for a list of times."""
        if not times:
            return {}

        return {
            'count': len(times),
            'total_time': sum(times),
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'max_time': max(times),
            'last_time': times[-1],
        }

    def reset(self, operation_name: str | None = None) -> None:
        """Reset profiling data.
        
        Args:
            operation_name: Optional specific operation to reset
        """
        with self._lock:
            if operation_name:
                self._profiles.pop(operation_name, None)
            else:
                self._profiles.clear()


# Global instances
memory_optimizer = MemoryOptimizer()
batch_processor = BatchProcessor()
model_optimizer = ModelOptimizer()
concurrency_optimizer = ConcurrencyOptimizer()
performance_profiler = PerformanceProfiler()
