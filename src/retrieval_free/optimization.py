"""Performance optimization utilities and techniques."""

import gc
import math
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np

from .caching import cached, vector_cache, batch_process
from .observability import monitor_performance, metrics_collector


class PerformanceProfiler:
    """Lightweight performance profiler for optimization."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
    
    def profile(self, name: str):
        """Context manager for profiling code blocks."""
        return ProfileContext(self, name)
    
    def record_time(self, name: str, duration: float) -> None:
        """Record execution time."""
        if name not in self.profiles:
            self.profiles[name] = []
        self.profiles[name].append(duration)
    
    def record_memory(self, name: str, memory_mb: float) -> None:
        """Record memory usage."""
        if name not in self.memory_usage:
            self.memory_usage[name] = []
        self.memory_usage[name].append(memory_mb)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get performance statistics for a profiled operation."""
        if name not in self.profiles:
            return {}
        
        times = self.profiles[name]
        memory = self.memory_usage.get(name, [])
        
        stats = {
            "count": len(times),
            "total_time": sum(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
        }
        
        if memory:
            stats.update({
                "avg_memory_mb": sum(memory) / len(memory),
                "peak_memory_mb": max(memory),
            })
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all profiled operations."""
        return {name: self.get_stats(name) for name in self.profiles.keys()}
    
    def clear(self) -> None:
        """Clear all profiling data."""
        self.profiles.clear()
        self.memory_usage.clear()


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        end_memory = self._get_memory_usage()
        
        self.profiler.record_time(self.name, duration)
        if self.start_memory is not None and end_memory is not None:
            self.profiler.record_memory(self.name, end_memory - self.start_memory)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return None


# Global profiler instance
profiler = PerformanceProfiler()


def optimize_embeddings(embeddings: List[np.ndarray], target_dim: Optional[int] = None) -> List[np.ndarray]:
    """Optimize embeddings for better performance and memory usage."""
    if not embeddings:
        return embeddings
    
    with profiler.profile("optimize_embeddings"):
        # Convert to float32 if needed (saves memory)
        optimized = []
        for emb in embeddings:
            if emb.dtype != np.float32:
                emb = emb.astype(np.float32)
            
            # Dimensionality reduction if requested
            if target_dim and len(emb) > target_dim:
                # Simple truncation (could use PCA for better results)
                emb = emb[:target_dim]
            
            # Normalize for better similarity computation
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            
            optimized.append(emb)
        
        return optimized


def optimize_chunks(chunks: List[str], max_chunk_size: int = 1000) -> List[str]:
    """Optimize text chunks for better processing."""
    with profiler.profile("optimize_chunks"):
        optimized = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.strip():
                continue
            
            # Split overly long chunks
            if len(chunk) > max_chunk_size:
                # Split by sentences first
                sentences = chunk.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > max_chunk_size:
                        if current_chunk:
                            optimized.append(current_chunk.strip())
                            current_chunk = sentence
                        else:
                            # Single sentence too long, force split
                            optimized.append(sentence[:max_chunk_size])
                            current_chunk = sentence[max_chunk_size:]
                    else:
                        current_chunk += ". " + sentence if current_chunk else sentence
                
                if current_chunk:
                    optimized.append(current_chunk.strip())
            else:
                optimized.append(chunk)
        
        return optimized


@cached(ttl=7200, key_func=lambda text, model: f"embedding_{hash(text)}_{model}")
def cached_embedding(text: str, model: Any) -> np.ndarray:
    """Cache embeddings to avoid recomputation."""
    # This would normally call the actual embedding model
    # For now, return a placeholder
    return np.random.rand(384).astype(np.float32)


class BatchProcessor:
    """Optimized batch processing for embeddings and compression."""
    
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    @monitor_performance
    def process_embeddings(
        self, 
        texts: List[str], 
        embedding_func: callable,
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """Process embeddings in optimized batches."""
        if not texts:
            return []
        
        with profiler.profile("batch_embeddings"):
            results = []
            
            # Check cache first if enabled
            if use_cache:
                cached_results = []
                uncached_texts = []
                uncached_indices = []
                
                for i, text in enumerate(texts):
                    # Try vector similarity cache
                    text_hash = hash(text)
                    dummy_vector = np.array([float(text_hash % 1000) / 1000] * 384)
                    
                    cached = vector_cache.get_similar(dummy_vector, f"embed_{text_hash}")
                    if cached is not None:
                        cached_results.append((i, cached))
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                
                # Process uncached texts
                if uncached_texts:
                    new_embeddings = batch_process(
                        uncached_texts,
                        embedding_func,
                        batch_size=self.batch_size,
                        num_workers=self.max_workers
                    )
                    
                    # Cache new embeddings
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        text_hash = hash(text)
                        dummy_vector = np.array([float(text_hash % 1000) / 1000] * 384)
                        vector_cache.put(f"embed_{text_hash}", dummy_vector, embedding)
                
                # Combine cached and new results
                results = [None] * len(texts)
                for i, embedding in cached_results:
                    results[i] = embedding
                for i, embedding in zip(uncached_indices, new_embeddings if uncached_texts else []):
                    results[i] = embedding
            
            else:
                # Process all without caching
                results = batch_process(
                    texts,
                    embedding_func,
                    batch_size=self.batch_size,
                    num_workers=self.max_workers
                )
            
            return results
    
    def process_compression(
        self,
        embeddings: List[np.ndarray],
        compression_func: callable,
        target_ratio: float = 8.0
    ) -> List[np.ndarray]:
        """Process compression in optimized batches."""
        with profiler.profile("batch_compression"):
            # Optimize embeddings first
            optimized_embeddings = optimize_embeddings(embeddings)
            
            # Group similar embeddings for better compression
            groups = self._group_similar_embeddings(optimized_embeddings)
            
            compressed_results = []
            for group in groups:
                group_compressed = compression_func(group, target_ratio)
                compressed_results.extend(group_compressed)
            
            return compressed_results
    
    def _group_similar_embeddings(
        self, 
        embeddings: List[np.ndarray], 
        similarity_threshold: float = 0.8
    ) -> List[List[np.ndarray]]:
        """Group similar embeddings for batch processing."""
        if not embeddings:
            return []
        
        groups = []
        used = set()
        
        for i, emb1 in enumerate(embeddings):
            if i in used:
                continue
            
            group = [emb1]
            used.add(i)
            
            # Find similar embeddings
            for j, emb2 in enumerate(embeddings[i+1:], i+1):
                if j in used:
                    continue
                
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if similarity >= similarity_threshold:
                    group.append(emb2)
                    used.add(j)
            
            groups.append(group)
        
        return groups


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    @staticmethod
    def optimize_arrays(arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory efficiency."""
        optimized = []
        
        for arr in arrays:
            # Use smallest dtype that preserves precision
            if arr.dtype == np.float64:
                # Check if values can fit in float32
                if np.allclose(arr, arr.astype(np.float32)):
                    arr = arr.astype(np.float32)
            
            # Ensure C-contiguous for better performance
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)
            
            optimized.append(arr)
        
        return optimized
    
    @staticmethod
    def cleanup_memory():
        """Force garbage collection and memory cleanup."""
        gc.collect()
        
        # Clear caches if memory pressure is high
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 85:  # High memory usage
                from .caching import cache_manager
                cache_manager.memory_cache.clear()
                vector_cache.clear()
                warnings.warn("High memory usage detected. Cache cleared.")
        except ImportError:
            pass


def auto_optimize(func):
    """Decorator that applies automatic optimizations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Memory cleanup before processing
        if hasattr(func, '_call_count'):
            func._call_count += 1
        else:
            func._call_count = 1
        
        # Periodic memory cleanup
        if func._call_count % 100 == 0:
            MemoryOptimizer.cleanup_memory()
        
        # Profile the function
        with profiler.profile(func.__name__):
            result = func(*args, **kwargs)
        
        # Record performance metrics
        stats = profiler.get_stats(func.__name__)
        if stats:
            metrics_collector.set_gauge(f"{func.__name__}_avg_time", stats["avg_time"])
            if "avg_memory_mb" in stats:
                metrics_collector.set_gauge(f"{func.__name__}_avg_memory", stats["avg_memory_mb"])
        
        return result
    
    return wrapper


class CompressionOptimizer:
    """Optimization strategies for compression operations."""
    
    @staticmethod
    def adaptive_chunk_size(text_length: int, target_chunks: int = 50) -> int:
        """Calculate optimal chunk size based on text length."""
        if text_length < 1000:
            return max(100, text_length // 2)
        
        base_chunk_size = text_length // target_chunks
        
        # Ensure chunk size is within reasonable bounds
        chunk_size = max(256, min(2048, base_chunk_size))
        
        return chunk_size
    
    @staticmethod
    def adaptive_compression_ratio(
        text_length: int, 
        target_tokens: int = 1000
    ) -> float:
        """Calculate optimal compression ratio based on text length."""
        if text_length < target_tokens:
            return 2.0  # Minimal compression
        
        # Estimate tokens (rough approximation)
        estimated_tokens = text_length / 4  # ~4 chars per token
        
        if estimated_tokens <= target_tokens:
            return 2.0
        
        ratio = estimated_tokens / target_tokens
        
        # Cap ratio to reasonable bounds
        return max(2.0, min(50.0, ratio))
    
    @staticmethod
    def optimize_compression_params(
        text: str,
        performance_target: str = "balanced"  # "speed", "quality", "balanced"
    ) -> Dict[str, Any]:
        """Get optimized compression parameters."""
        text_length = len(text)
        
        if performance_target == "speed":
            return {
                "chunk_size": CompressionOptimizer.adaptive_chunk_size(text_length, 20),
                "compression_ratio": CompressionOptimizer.adaptive_compression_ratio(text_length, 2000),
                "overlap_ratio": 0.05,
                "batch_size": 64
            }
        elif performance_target == "quality":
            return {
                "chunk_size": CompressionOptimizer.adaptive_chunk_size(text_length, 100),
                "compression_ratio": CompressionOptimizer.adaptive_compression_ratio(text_length, 500),
                "overlap_ratio": 0.15,
                "batch_size": 16
            }
        else:  # balanced
            return {
                "chunk_size": CompressionOptimizer.adaptive_chunk_size(text_length, 50),
                "compression_ratio": CompressionOptimizer.adaptive_compression_ratio(text_length, 1000),
                "overlap_ratio": 0.1,
                "batch_size": 32
            }


# Global instances
batch_processor = BatchProcessor()
memory_optimizer = MemoryOptimizer()
compression_optimizer = CompressionOptimizer()