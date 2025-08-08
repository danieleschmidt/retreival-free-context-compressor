"""High-performance scaling features for the retrieval-free context compressor.

This module implements Generation 3 scaling capabilities including:
- Multi-GPU processing with CUDA optimizations
- Mixed precision training and inference
- Distributed computing support
- Async processing capabilities
- Auto-scaling and load management
"""

import asyncio
import logging
import math
import multiprocessing as mp
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DataParallel

from .core import CompressionResult, CompressorBase, MegaToken
from .exceptions import ResourceError, ScalingError
from .monitoring import MetricsCollector


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about available compute devices."""
    device_id: int
    device_type: str  # 'cuda', 'cpu', 'mps'
    memory_total: int  # in bytes
    memory_available: int  # in bytes
    compute_capability: tuple[int, int] | None = None
    name: str = ""


@dataclass
class ProcessingTask:
    """Represents a compression task with priority and metadata."""
    priority: int
    task_id: str
    text: str
    parameters: dict[str, Any]
    callback: Callable | None = None
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.priority < other.priority


class DeviceManager:
    """Manages GPU devices and memory allocation."""

    def __init__(self):
        self.devices = self._discover_devices()
        self.device_usage = {device.device_id: 0.0 for device in self.devices}
        self._lock = threading.RLock()

    def _discover_devices(self) -> list[DeviceInfo]:
        """Discover available compute devices."""
        devices = []

        # Add CPU
        devices.append(DeviceInfo(
            device_id=-1,
            device_type="cpu",
            memory_total=8 * 1024**3,  # Assume 8GB
            memory_available=4 * 1024**3,
            name="CPU"
        ))

        # Add CUDA devices if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append(DeviceInfo(
                    device_id=i,
                    device_type="cuda",
                    memory_total=props.total_memory,
                    memory_available=props.total_memory - torch.cuda.memory_allocated(i),
                    compute_capability=(props.major, props.minor),
                    name=props.name
                ))

        # Add MPS device if available (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append(DeviceInfo(
                device_id=0,
                device_type="mps",
                memory_total=8 * 1024**3,  # Unified memory
                memory_available=4 * 1024**3,
                name="Apple Silicon MPS"
            ))

        logger.info(f"Discovered {len(devices)} compute devices")
        return devices

    def get_best_device(self, memory_required: int = 0) -> DeviceInfo | None:
        """Get the best available device for processing."""
        with self._lock:
            # Filter devices with enough memory
            available_devices = [
                d for d in self.devices
                if d.memory_available >= memory_required and self.device_usage[d.device_id] < 0.9
            ]

            if not available_devices:
                return None

            # Prefer CUDA, then MPS, then CPU
            priority_order = {'cuda': 3, 'mps': 2, 'cpu': 1}
            available_devices.sort(
                key=lambda d: (priority_order.get(d.device_type, 0), -self.device_usage[d.device_id]),
                reverse=True
            )

            return available_devices[0]

    @contextmanager
    def allocate_device(self, device: DeviceInfo, usage: float = 0.5):
        """Context manager for device allocation."""
        with self._lock:
            old_usage = self.device_usage[device.device_id]
            self.device_usage[device.device_id] += usage

        try:
            yield device
        finally:
            with self._lock:
                self.device_usage[device.device_id] = old_usage


class MultiGPUProcessor:
    """Multi-GPU processing manager with CUDA optimizations."""

    def __init__(self, compressor: CompressorBase, enable_mixed_precision: bool = True):
        self.compressor = compressor
        self.enable_mixed_precision = enable_mixed_precision
        self.device_manager = DeviceManager()
        self.scaler = GradScaler() if enable_mixed_precision else None

        # Setup multi-GPU if available
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_data_parallel = self.gpu_count > 1

        if self.use_data_parallel and hasattr(self.compressor, 'model'):
            self.compressor.model = DataParallel(self.compressor.model)
            logger.info(f"Enabled DataParallel across {self.gpu_count} GPUs")

    def process_batch(self, texts: list[str], **kwargs) -> list[CompressionResult]:
        """Process a batch of texts across multiple GPUs."""
        if not texts:
            return []

        batch_size = len(texts)

        if self.gpu_count <= 1:
            # Single GPU/CPU processing
            return self._process_single_device(texts, **kwargs)

        # Multi-GPU processing
        return self._process_multi_gpu(texts, **kwargs)

    def _process_single_device(self, texts: list[str], **kwargs) -> list[CompressionResult]:
        """Process batch on single device with mixed precision."""
        device = self.device_manager.get_best_device()
        if not device:
            raise ResourceError("No available devices for processing")

        results = []

        with self.device_manager.allocate_device(device):
            for text in texts:
                if self.enable_mixed_precision and device.device_type == "cuda":
                    with autocast():
                        result = self.compressor.compress(text, **kwargs)
                else:
                    result = self.compressor.compress(text, **kwargs)

                results.append(result)

        return results

    def _process_multi_gpu(self, texts: list[str], **kwargs) -> list[CompressionResult]:
        """Process batch across multiple GPUs."""
        # Split batch across available GPUs
        chunks = self._split_batch(texts, self.gpu_count)
        futures = []

        with ThreadPoolExecutor(max_workers=self.gpu_count) as executor:
            for gpu_id, chunk in enumerate(chunks):
                if chunk:  # Skip empty chunks
                    future = executor.submit(
                        self._process_gpu_chunk,
                        chunk,
                        gpu_id,
                        **kwargs
                    )
                    futures.append(future)

            # Collect results
            all_results = []
            for future in futures:
                try:
                    chunk_results = future.result(timeout=300)  # 5 minute timeout
                    all_results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"GPU processing failed: {e}")
                    raise ScalingError(f"Multi-GPU processing failed: {e}")

        return all_results

    def _process_gpu_chunk(self, texts: list[str], gpu_id: int, **kwargs) -> list[CompressionResult]:
        """Process a chunk of texts on specific GPU."""
        device = torch.device(f"cuda:{gpu_id}")

        # Move model to specific GPU for this thread
        if hasattr(self.compressor, 'model') and not isinstance(self.compressor.model, DataParallel):
            original_device = next(self.compressor.model.parameters()).device
            self.compressor.model = self.compressor.model.to(device)

        try:
            results = []
            for text in texts:
                if self.enable_mixed_precision:
                    with autocast():
                        result = self.compressor.compress(text, **kwargs)
                else:
                    result = self.compressor.compress(text, **kwargs)
                results.append(result)

            return results

        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _split_batch(self, items: list[Any], num_chunks: int) -> list[list[Any]]:
        """Split batch into roughly equal chunks."""
        chunk_size = math.ceil(len(items) / num_chunks)
        return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


class AsyncProcessor:
    """Asynchronous processing with queue management."""

    def __init__(
        self,
        compressor: CompressorBase,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        batch_size: int = 8
    ):
        self.compressor = compressor
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size

        # Initialize queues
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self.result_futures: dict[str, Future] = {}

        # Thread pools
        self.io_executor = ThreadPoolExecutor(max_workers=max_workers // 2)
        self.compute_executor = ThreadPoolExecutor(max_workers=max_workers)

        # Processing state
        self._running = False
        self._worker_threads = []

        # Multi-GPU processor
        self.gpu_processor = MultiGPUProcessor(compressor)

    def start(self):
        """Start the async processing workers."""
        if self._running:
            return

        self._running = True

        # Start batch processing workers
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._batch_worker, daemon=True)
            worker.start()
            self._worker_threads.append(worker)

        logger.info(f"Started {self.max_workers} async processing workers")

    def stop(self):
        """Stop the async processing workers."""
        self._running = False

        # Wait for workers to finish
        for worker in self._worker_threads:
            worker.join(timeout=5.0)

        self._worker_threads.clear()

        # Shutdown executors
        self.io_executor.shutdown(wait=True)
        self.compute_executor.shutdown(wait=True)

        logger.info("Stopped async processing workers")

    async def compress_async(
        self,
        text: str,
        priority: int = 5,
        **kwargs
    ) -> CompressionResult:
        """Compress text asynchronously."""
        task_id = f"task_{int(time.time() * 1000000)}"

        # Create task
        task = ProcessingTask(
            priority=priority,
            task_id=task_id,
            text=text,
            parameters=kwargs
        )

        # Create future for result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.result_futures[task_id] = future

        try:
            # Add to queue (this might block if queue is full)
            await loop.run_in_executor(None, self.task_queue.put, task)
        except Exception as e:
            self.result_futures.pop(task_id, None)
            raise ScalingError(f"Failed to queue task: {e}")

        # Wait for result
        try:
            return await future
        finally:
            self.result_futures.pop(task_id, None)

    def compress_batch_async(
        self,
        texts: list[str],
        priority: int = 5,
        **kwargs
    ) -> list[Future]:
        """Submit a batch of texts for async processing."""
        futures = []

        for text in texts:
            try:
                future = asyncio.ensure_future(
                    self.compress_async(text, priority, **kwargs)
                )
                futures.append(future)
            except Exception as e:
                logger.error(f"Failed to submit async task: {e}")
                # Return failed future
                failed_future = asyncio.Future()
                failed_future.set_exception(e)
                futures.append(failed_future)

        return futures

    def _batch_worker(self):
        """Worker thread for batch processing."""
        while self._running:
            try:
                # Collect batch of tasks
                batch = []

                # Get first task (blocking)
                try:
                    task = self.task_queue.get(timeout=1.0)
                    batch.append(task)
                except:
                    continue  # Timeout, check if still running

                # Collect additional tasks up to batch_size
                while len(batch) < self.batch_size and not self.task_queue.empty():
                    try:
                        task = self.task_queue.get_nowait()
                        batch.append(task)
                    except:
                        break

                # Process batch
                self._process_batch(batch)

            except Exception as e:
                logger.error(f"Batch worker error: {e}")

    def _process_batch(self, batch: list[ProcessingTask]):
        """Process a batch of tasks."""
        try:
            # Extract texts and parameters
            texts = [task.text for task in batch]

            # Process with multi-GPU
            start_time = time.time()
            results = self.gpu_processor.process_batch(texts)
            processing_time = time.time() - start_time

            # Set results
            for task, result in zip(batch, results, strict=False):
                future = self.result_futures.get(task.task_id)
                if future and not future.done():
                    # Add processing metadata
                    result.metadata.update({
                        'async_processing_time': processing_time / len(batch),
                        'batch_size': len(batch),
                        'task_id': task.task_id
                    })
                    future.set_result(result)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Set exception for all futures
            for task in batch:
                future = self.result_futures.get(task.task_id)
                if future and not future.done():
                    future.set_exception(ScalingError(f"Batch processing failed: {e}"))

        finally:
            # Mark tasks as done
            for _ in batch:
                self.task_queue.task_done()

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        return {
            'queue_size': self.task_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_futures': len(self.result_futures),
            'workers_running': self._running,
            'worker_count': len(self._worker_threads)
        }


class DistributedProcessor:
    """Distributed computing support using multiprocessing and Ray/Dask."""

    def __init__(self, compressor: CompressorBase, use_ray: bool = True):
        self.compressor = compressor
        self.use_ray = use_ray
        self._ray_available = False
        self._dask_available = False

        # Try to initialize Ray
        if use_ray:
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self._ray_available = True
                logger.info("Ray initialized for distributed processing")
            except ImportError:
                logger.warning("Ray not available, falling back to multiprocessing")

        # Try Dask as fallback
        if not self._ray_available:
            try:
                import dask
                from dask.distributed import Client
                self._dask_client = Client(processes=False, silence_logs=False)
                self._dask_available = True
                logger.info("Dask initialized for distributed processing")
            except ImportError:
                logger.warning("Dask not available, using multiprocessing only")

    def process_distributed(
        self,
        texts: list[str],
        num_workers: int | None = None,
        **kwargs
    ) -> list[CompressionResult]:
        """Process texts using distributed computing."""
        if not texts:
            return []

        num_workers = num_workers or min(mp.cpu_count(), len(texts))

        if self._ray_available:
            return self._process_with_ray(texts, num_workers, **kwargs)
        elif self._dask_available:
            return self._process_with_dask(texts, num_workers, **kwargs)
        else:
            return self._process_with_multiprocessing(texts, num_workers, **kwargs)

    def _process_with_ray(
        self,
        texts: list[str],
        num_workers: int,
        **kwargs
    ) -> list[CompressionResult]:
        """Process using Ray distributed framework."""
        import ray

        # Create remote compression function
        @ray.remote
        def compress_remote(compressor_state, text_chunk, params):
            # Reconstruct compressor from state
            compressor = self._reconstruct_compressor(compressor_state)
            results = []

            for text in text_chunk:
                result = compressor.compress(text, **params)
                results.append(result)

            return results

        # Serialize compressor
        compressor_state = self._serialize_compressor()

        # Split work into chunks
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [
            texts[i:i + chunk_size]
            for i in range(0, len(texts), chunk_size)
        ]

        # Submit remote tasks
        futures = []
        for chunk in text_chunks:
            future = compress_remote.remote(compressor_state, chunk, kwargs)
            futures.append(future)

        # Collect results
        chunk_results = ray.get(futures)

        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

        return all_results

    def _process_with_dask(
        self,
        texts: list[str],
        num_workers: int,
        **kwargs
    ) -> list[CompressionResult]:
        """Process using Dask distributed framework."""
        import dask
        from dask import delayed

        # Create delayed compression functions
        @delayed
        def compress_delayed(text, compressor_state, params):
            compressor = self._reconstruct_compressor(compressor_state)
            return compressor.compress(text, **params)

        compressor_state = self._serialize_compressor()

        # Create delayed tasks
        delayed_results = [
            compress_delayed(text, compressor_state, kwargs)
            for text in texts
        ]

        # Compute results
        results = dask.compute(*delayed_results)
        return list(results)

    def _process_with_multiprocessing(
        self,
        texts: list[str],
        num_workers: int,
        **kwargs
    ) -> list[CompressionResult]:
        """Process using standard multiprocessing."""
        def worker_func(args):
            compressor_state, text_chunk, params = args
            compressor = self._reconstruct_compressor(compressor_state)
            results = []

            for text in text_chunk:
                result = compressor.compress(text, **params)
                results.append(result)

            return results

        compressor_state = self._serialize_compressor()

        # Split work into chunks
        chunk_size = max(1, len(texts) // num_workers)
        text_chunks = [
            texts[i:i + chunk_size]
            for i in range(0, len(texts), chunk_size)
        ]

        # Prepare arguments
        args_list = [
            (compressor_state, chunk, kwargs)
            for chunk in text_chunks
        ]

        # Process with multiprocessing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            chunk_results = list(executor.map(worker_func, args_list))

        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)

        return all_results

    def _serialize_compressor(self) -> dict[str, Any]:
        """Serialize compressor for distributed processing."""
        return {
            'model_name': getattr(self.compressor, 'model_name', ''),
            'class_name': self.compressor.__class__.__name__,
            'init_params': {
                attr: getattr(self.compressor, attr)
                for attr in ['chunk_size', 'compression_ratio', 'overlap_ratio']
                if hasattr(self.compressor, attr)
            }
        }

    def _reconstruct_compressor(self, state: dict[str, Any]) -> CompressorBase:
        """Reconstruct compressor from serialized state."""
        # This is a simplified reconstruction
        # In practice, you'd need more sophisticated serialization
        from .core import ContextCompressor

        class_name = state.get('class_name', 'ContextCompressor')

        if class_name == 'ContextCompressor':
            return ContextCompressor(
                model_name=state.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                **state.get('init_params', {})
            )

        # Fallback to original compressor
        return self.compressor


class AutoScaler:
    """Automatic scaling based on load and resource utilization."""

    def __init__(
        self,
        async_processor: AsyncProcessor,
        min_workers: int = 2,
        max_workers: int = 16,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        scale_interval: float = 30.0
    ):
        self.async_processor = async_processor
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_interval = scale_interval

        self._running = False
        self._scaler_thread = None
        self._metrics = MetricsCollector()

    def start(self):
        """Start auto-scaling."""
        if self._running:
            return

        self._running = True
        self._scaler_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self._scaler_thread.start()

        logger.info("Auto-scaler started")

    def stop(self):
        """Stop auto-scaling."""
        self._running = False
        if self._scaler_thread:
            self._scaler_thread.join(timeout=5.0)

        logger.info("Auto-scaler stopped")

    def _scaling_loop(self):
        """Main scaling loop."""
        while self._running:
            try:
                self._evaluate_scaling()
                time.sleep(self.scale_interval)
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")

    def _evaluate_scaling(self):
        """Evaluate if scaling is needed."""
        # Get current metrics
        queue_stats = self.async_processor.get_queue_stats()

        # Calculate utilization metrics
        queue_utilization = queue_stats['queue_size'] / queue_stats['max_queue_size']
        current_workers = len(self.async_processor._worker_threads)

        # Check if scaling up is needed
        if (queue_utilization > self.scale_up_threshold and
            current_workers < self.max_workers):

            new_worker_count = min(self.max_workers, current_workers + 2)
            self._scale_workers(new_worker_count)

            logger.info(
                f"Scaled up from {current_workers} to {new_worker_count} workers "
                f"(queue utilization: {queue_utilization:.2f})"
            )

        # Check if scaling down is needed
        elif (queue_utilization < self.scale_down_threshold and
              current_workers > self.min_workers):

            new_worker_count = max(self.min_workers, current_workers - 1)
            self._scale_workers(new_worker_count)

            logger.info(
                f"Scaled down from {current_workers} to {new_worker_count} workers "
                f"(queue utilization: {queue_utilization:.2f})"
            )

    def _scale_workers(self, target_count: int):
        """Scale worker threads to target count."""
        current_count = len(self.async_processor._worker_threads)

        if target_count > current_count:
            # Scale up - add workers
            for _ in range(target_count - current_count):
                worker = threading.Thread(
                    target=self.async_processor._batch_worker,
                    daemon=True
                )
                worker.start()
                self.async_processor._worker_threads.append(worker)

        elif target_count < current_count:
            # Scale down - this is more complex as we need to gracefully stop workers
            # For now, we'll rely on the workers to check _running flag
            excess_workers = current_count - target_count

            # Mark excess workers for termination
            # In a production system, you'd implement graceful worker shutdown
            for _ in range(excess_workers):
                if self.async_processor._worker_threads:
                    self.async_processor._worker_threads.pop()


class HighPerformanceCompressor(CompressorBase):
    """High-performance compressor with all scaling features integrated."""

    def __init__(
        self,
        base_compressor: CompressorBase,
        enable_multi_gpu: bool = True,
        enable_async: bool = True,
        enable_distributed: bool = False,
        enable_auto_scaling: bool = True,
        **kwargs
    ):
        # Don't call super().__init__ as we wrap an existing compressor
        self.base_compressor = base_compressor
        self.model_name = getattr(base_compressor, 'model_name', 'unknown')

        # Initialize scaling components
        self.multi_gpu_processor = (
            MultiGPUProcessor(base_compressor) if enable_multi_gpu else None
        )

        self.async_processor = (
            AsyncProcessor(base_compressor, **kwargs) if enable_async else None
        )

        self.distributed_processor = (
            DistributedProcessor(base_compressor) if enable_distributed else None
        )

        self.auto_scaler = None
        if enable_auto_scaling and self.async_processor:
            self.auto_scaler = AutoScaler(self.async_processor)

        # Start services
        if self.async_processor:
            self.async_processor.start()

        if self.auto_scaler:
            self.auto_scaler.start()

    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text with automatic best-path selection."""
        # For single text, use base compressor with optional GPU optimization
        if self.multi_gpu_processor and self.multi_gpu_processor.gpu_count > 0:
            results = self.multi_gpu_processor.process_batch([text], **kwargs)
            return results[0] if results else self.base_compressor.compress(text, **kwargs)

        return self.base_compressor.compress(text, **kwargs)

    def compress_batch(
        self,
        texts: list[str],
        use_distributed: bool = False,
        **kwargs
    ) -> list[CompressionResult]:
        """Compress batch of texts with optimal scaling strategy."""
        if not texts:
            return []

        batch_size = len(texts)

        # Choose processing strategy based on batch size
        if use_distributed and self.distributed_processor and batch_size > 100:
            # Large batch - use distributed processing
            return self.distributed_processor.process_distributed(texts, **kwargs)

        elif self.multi_gpu_processor and batch_size > 10:
            # Medium batch - use multi-GPU
            return self.multi_gpu_processor.process_batch(texts, **kwargs)

        else:
            # Small batch - use base compressor
            return [self.base_compressor.compress(text, **kwargs) for text in texts]

    async def compress_async(
        self,
        text: str,
        priority: int = 5,
        **kwargs
    ) -> CompressionResult:
        """Compress text asynchronously."""
        if self.async_processor:
            return await self.async_processor.compress_async(text, priority, **kwargs)

        # Fallback to synchronous
        return self.compress(text, **kwargs)

    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Decompress mega-tokens."""
        return self.base_compressor.decompress(mega_tokens, **kwargs)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            'devices': [
                {
                    'device_id': d.device_id,
                    'device_type': d.device_type,
                    'name': d.name,
                    'memory_total_mb': d.memory_total / 1024**2,
                    'memory_available_mb': d.memory_available / 1024**2
                }
                for d in (self.multi_gpu_processor.device_manager.devices
                         if self.multi_gpu_processor else [])
            ],
            'multi_gpu_enabled': self.multi_gpu_processor is not None,
            'async_enabled': self.async_processor is not None,
            'distributed_enabled': self.distributed_processor is not None,
            'auto_scaling_enabled': self.auto_scaler is not None
        }

        if self.async_processor:
            stats['async_queue'] = self.async_processor.get_queue_stats()

        return stats

    def shutdown(self):
        """Gracefully shutdown all scaling components."""
        if self.auto_scaler:
            self.auto_scaler.stop()

        if self.async_processor:
            self.async_processor.stop()

        logger.info("High-performance compressor shutdown complete")
