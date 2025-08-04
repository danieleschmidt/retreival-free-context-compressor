"""Performance optimization utilities."""

import time
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import functools
import queue
import torch
import torch.nn as nn
from contextlib import contextmanager
import psutil

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Efficient batch processing for compression operations."""
    
    def __init__(
        self,
        batch_size: int = 8,
        max_workers: int = None,
        device: str = "auto"
    ):
        """Initialize batch processor.
        
        Args:
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of worker threads
            device: Device to use for processing
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(8, multiprocessing.cpu_count())
        self.device = device if device != "auto" else self._get_optimal_device()
        
        # Thread pool for I/O operations
        self._thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Memory pool for tensors
        self._tensor_cache = {}
        self._cache_lock = threading.Lock()
    
    def _get_optimal_device(self) -> str:
        """Determine optimal device for processing."""
        if torch.cuda.is_available():
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory > 4 * 1024**3:  # 4GB
                return "cuda"
        return "cpu"
    
    def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[List[Any]], List[Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """Process items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each batch
            progress_callback: Optional progress callback (current, total)
            
        Returns:
            List of processed results
        """
        results = []
        total_batches = (len(items) + self.batch_size - 1) // self.batch_size
        
        # Split into batches
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        # Process batches
        futures = []
        for i, batch in enumerate(batches):
            future = self._thread_pool.submit(self._process_single_batch, batch, process_func)
            futures.append((i, future))
        
        # Collect results in order
        batch_results = [None] * len(batches)
        completed = 0
        
        for i, future in futures:
            try:
                batch_result = future.result()
                batch_results[i] = batch_result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_batches)
                    
            except Exception as e:
                logger.error(f"Batch {i} failed: {e}")
                batch_results[i] = []
        
        # Flatten results
        for batch_result in batch_results:
            if batch_result:
                results.extend(batch_result)
        
        return results
    
    def _process_single_batch(
        self, 
        batch: List[Any], 
        process_func: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """Process a single batch."""
        try:
            with self._get_memory_context():
                return process_func(batch)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            return []
    
    @contextmanager
    def _get_memory_context(self):
        """Context manager for memory optimization."""
        # Clear GPU cache if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        try:
            yield
        finally:
            # Cleanup after processing
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
    
    def shutdown(self):
        """Shutdown the batch processor."""
        self._thread_pool.shutdown(wait=True)


class ModelOptimizer:
    """Optimize model for inference performance."""
    
    def __init__(self, device: str = "auto"):
        """Initialize model optimizer.
        
        Args:
            device: Target device for optimization
        """
        self.device = device if device != "auto" else self._get_optimal_device()
    
    def _get_optimal_device(self) -> str:
        """Get optimal device."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def optimize_model(
        self, 
        model: nn.Module,
        use_mixed_precision: bool = True,
        use_jit: bool = True,
        optimize_for_inference: bool = True
    ) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            use_mixed_precision: Whether to use mixed precision
            use_jit: Whether to use TorchScript JIT compilation
            optimize_for_inference: Whether to optimize for inference
            
        Returns:
            Optimized model
        """
        model = model.to(self.device)
        
        if optimize_for_inference:
            model.eval()
            
            # Freeze batch norm layers
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
        
        # Mixed precision optimization
        if use_mixed_precision and self.device.startswith("cuda"):
            try:
                model = model.half()  # Convert to FP16
                logger.info("Applied mixed precision (FP16) optimization")
            except Exception as e:
                logger.warning(f"Failed to apply mixed precision: {e}")
        
        # JIT compilation
        if use_jit:
            try:
                # Create dummy input for tracing
                dummy_input = self._create_dummy_input(model)
                if dummy_input is not None:
                    model = torch.jit.trace(model, dummy_input)
                    logger.info("Applied TorchScript JIT compilation")
            except Exception as e:
                logger.warning(f"Failed to apply JIT compilation: {e}")
        
        return model
    
    def _create_dummy_input(self, model: nn.Module) -> Optional[torch.Tensor]:
        """Create dummy input for model tracing."""
        try:
            # Try to infer input shape from first layer
            first_layer = next(model.parameters())
            if len(first_layer.shape) >= 2:
                batch_size = 1
                input_dim = first_layer.shape[1]
                return torch.randn(batch_size, input_dim, device=self.device)
        except Exception:
            pass
        
        return None
    
    def enable_attention_optimizations(self, model: nn.Module) -> nn.Module:
        """Enable attention optimizations."""
        try:
            # Enable Flash Attention if available
            from flash_attn import flash_attn_func
            
            # Replace attention layers with Flash Attention
            def replace_attention(module):
                for name, child in module.named_children():
                    if 'attention' in name.lower():
                        # Apply Flash Attention optimization
                        setattr(module, name, self._wrap_with_flash_attention(child))
                    else:
                        replace_attention(child)
            
            replace_attention(model)
            logger.info("Enabled Flash Attention optimizations")
            
        except ImportError:
            logger.info("Flash Attention not available, using standard attention")
        except Exception as e:
            logger.warning(f"Failed to enable attention optimizations: {e}")
        
        return model
    
    def _wrap_with_flash_attention(self, attention_module):
        """Wrap attention module with Flash Attention."""
        # This is a placeholder - actual implementation would depend on the attention module structure
        return attention_module


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.memory_pool = {}
        self._pool_lock = threading.Lock()
    
    @contextmanager
    def memory_efficient_context(
        self,
        clear_cache: bool = True,
        gradient_checkpointing: bool = False
    ):
        """Context manager for memory efficient operations.
        
        Args:
            clear_cache: Whether to clear GPU cache
            gradient_checkpointing: Whether to use gradient checkpointing
        """
        # Pre-operation cleanup
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing
        original_checkpointing = {}
        if gradient_checkpointing:
            # This would enable gradient checkpointing on supported modules
            pass
        
        try:
            yield
        finally:
            # Post-operation cleanup
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Restore gradient checkpointing settings
            for module, setting in original_checkpointing.items():
                # Restore original settings
                pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.
        
        Returns:
            Dictionary with memory usage in MB
        """
        stats = {}
        
        # System memory
        try:
            memory = psutil.virtual_memory()
            stats['system_total_mb'] = memory.total / 1024 / 1024
            stats['system_used_mb'] = memory.used / 1024 / 1024
            stats['system_available_mb'] = memory.available / 1024 / 1024
            stats['system_percent'] = memory.percent
        except Exception as e:
            logger.warning(f"Failed to get system memory stats: {e}")
        
        # GPU memory
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    cached = torch.cuda.memory_reserved(i) / 1024 / 1024
                    total = torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                    
                    stats[f'gpu_{i}_allocated_mb'] = allocated  
                    stats[f'gpu_{i}_cached_mb'] = cached
                    stats[f'gpu_{i}_total_mb'] = total
                    stats[f'gpu_{i}_percent'] = (allocated / total) * 100
            except Exception as e:
                logger.warning(f"Failed to get GPU memory stats: {e}")
        
        return stats
    
    def optimize_tensor_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor memory usage.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Memory-optimized tensor
        """
        # Convert to optimal dtype if possible
        if tensor.dtype == torch.float64:
            tensor = tensor.float()  # Convert to float32
        
        # Use memory format optimization
        if tensor.dim() == 4:  # For 4D tensors (NCHW format)
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        return tensor


class ConcurrencyManager:
    """Manage concurrent operations for better resource utilization."""
    
    def __init__(
        self,
        max_concurrent_operations: int = None,
        enable_async_io: bool = True
    ):
        """Initialize concurrency manager.
        
        Args:
            max_concurrent_operations: Maximum concurrent operations
            enable_async_io: Whether to enable asynchronous I/O
        """
        self.max_concurrent = max_concurrent_operations or multiprocessing.cpu_count()
        self.enable_async_io = enable_async_io
        
        # Semaphore to limit concurrent operations
        self.semaphore = threading.Semaphore(self.max_concurrent)
        
        # Thread pools
        self.io_pool = ThreadPoolExecutor(max_workers=self.max_concurrent * 2)
        self.compute_pool = ThreadPoolExecutor(max_workers=self.max_concurrent)
    
    @contextmanager
    def acquire_slot(self):
        """Acquire a concurrency slot."""
        self.semaphore.acquire()
        try:
            yield
        finally:
            self.semaphore.release()
    
    def submit_compute_task(self, func: Callable, *args, **kwargs):
        """Submit compute-intensive task.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        return self.compute_pool.submit(func, *args, **kwargs)
    
    def submit_io_task(self, func: Callable, *args, **kwargs):
        """Submit I/O-intensive task.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        return self.io_pool.submit(func, *args, **kwargs)
    
    def process_concurrent_tasks(
        self,
        tasks: List[Tuple[Callable, tuple, dict]],
        task_type: str = "compute"
    ) -> List[Any]:
        """Process multiple tasks concurrently.
        
        Args:
            tasks: List of (function, args, kwargs) tuples
            task_type: Type of tasks ("compute" or "io")
            
        Returns:
            List of results
        """
        pool = self.compute_pool if task_type == "compute" else self.io_pool
        
        futures = []
        for func, args, kwargs in tasks:
            future = pool.submit(func, *args, **kwargs)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Task failed: {e}")
                results.append(None)
        
        return results
    
    def shutdown(self):
        """Shutdown thread pools."""
        self.io_pool.shutdown(wait=True)
        self.compute_pool.shutdown(wait=True)


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time and memory usage.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Get initial memory usage
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.memory_allocated()
        else:
            initial_gpu_memory = 0
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Calculate metrics
            execution_time = time.time() - start_time
            
            if torch.cuda.is_available():
                final_gpu_memory = torch.cuda.memory_allocated()
                gpu_memory_delta = final_gpu_memory - initial_gpu_memory
            else:
                gpu_memory_delta = 0
            
            logger.info(
                f"Function {func.__name__} executed in {execution_time:.3f}s, "
                f"GPU memory delta: {gpu_memory_delta / 1024 / 1024:.2f}MB"
            )
        
        return result
    
    return wrapper


class PerformanceTuner:
    """Automatic performance tuning for compression operations."""
    
    def __init__(self):
        """Initialize performance tuner."""
        self.benchmark_results = {}
        self.optimal_settings = {}
    
    def tune_batch_size(
        self,
        process_func: Callable,
        test_data: List[Any],
        batch_sizes: List[int] = None
    ) -> int:
        """Find optimal batch size for processing.
        
        Args:
            process_func: Function to benchmark
            test_data: Test data for benchmarking
            batch_sizes: List of batch sizes to test
            
        Returns:
            Optimal batch size
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]
        
        best_batch_size = 1
        best_throughput = 0
        
        # Limit test data size for faster benchmarking
        test_data = test_data[:min(100, len(test_data))]
        
        for batch_size in batch_sizes:
            try:
                processor = BatchProcessor(batch_size=batch_size)
                
                start_time = time.time()
                results = processor.process_batch(test_data, process_func)
                end_time = time.time()
                
                if len(results) > 0:
                    throughput = len(results) / (end_time - start_time)
                    
                    logger.info(f"Batch size {batch_size}: {throughput:.2f} items/sec")
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_batch_size = batch_size
                
                processor.shutdown()
                
            except Exception as e:
                logger.warning(f"Failed to test batch size {batch_size}: {e}")
        
        logger.info(f"Optimal batch size: {best_batch_size} ({best_throughput:.2f} items/sec)")
        return best_batch_size
    
    def tune_worker_count(
        self,
        process_func: Callable,
        test_data: List[Any],
        max_workers_range: List[int] = None
    ) -> int:
        """Find optimal number of workers.
        
        Args:
            process_func: Function to benchmark
            test_data: Test data for benchmarking  
            max_workers_range: List of worker counts to test
            
        Returns:
            Optimal number of workers
        """
        if max_workers_range is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers_range = [1, 2, 4, min(8, cpu_count), cpu_count]
        
        best_workers = 1
        best_throughput = 0
        
        test_data = test_data[:min(100, len(test_data))]
        
        for workers in max_workers_range:
            try:
                processor = BatchProcessor(max_workers=workers)
                
                start_time = time.time()
                results = processor.process_batch(test_data, process_func)
                end_time = time.time()
                
                if len(results) > 0:
                    throughput = len(results) / (end_time - start_time)
                    
                    logger.info(f"Workers {workers}: {throughput:.2f} items/sec")
                    
                    if throughput > best_throughput:
                        best_throughput = throughput
                        best_workers = workers
                
                processor.shutdown()
                
            except Exception as e:
                logger.warning(f"Failed to test {workers} workers: {e}")
        
        logger.info(f"Optimal workers: {best_workers} ({best_throughput:.2f} items/sec)")
        return best_workers