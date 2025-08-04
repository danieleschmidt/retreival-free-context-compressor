"""Caching and performance optimization utilities."""

import hashlib
import json
import os
import pickle
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from threading import Lock
import weakref

import numpy as np

from .observability import monitor_performance


class MemoryCache:
    """Thread-safe in-memory cache with LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: List[str] = []  # LRU tracking
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value into cache."""
        with self.lock:
            current_time = time.time()
            
            # If key exists, update it
            if key in self.cache:
                self.cache[key] = (value, current_time)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                return
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = (value, current_time)
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache (assumes lock is held)."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item (assumes lock is held)."""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_key(lru_key)


class DiskCache:
    """Persistent disk-based cache for large objects."""
    
    def __init__(self, cache_dir: str = ".rfcc_cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "index.json"
        self.index = self._load_index()
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self.lock:
            if key not in self.index:
                return None
            
            entry = self.index[key]
            file_path = self.cache_dir / entry["filename"]
            
            # Check if file exists and TTL
            if not file_path.exists():
                del self.index[key]
                self._save_index()
                return None
            
            if time.time() - entry["timestamp"] > entry.get("ttl", 3600):
                self._remove_entry(key)
                return None
            
            # Load and return data
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                entry["last_access"] = time.time()
                self._save_index()
                
                return data
            except Exception:
                # Corrupted file, remove entry
                self._remove_entry(key)
                return None
    
    def put(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Put value into disk cache."""
        with self.lock:
            # Generate filename
            filename = f"{hashlib.md5(key.encode()).hexdigest()}.pkl"
            file_path = self.cache_dir / filename
            
            try:
                # Save data to disk
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Update index
                self.index[key] = {
                    "filename": filename,
                    "timestamp": time.time(),
                    "last_access": time.time(),
                    "ttl": ttl,
                    "size": file_path.stat().st_size
                }
                
                self._save_index()
                self._cleanup_if_needed()
                
            except Exception as e:
                # Clean up if save failed
                if file_path.exists():
                    file_path.unlink()
                raise e
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            for entry in self.index.values():
                file_path = self.cache_dir / entry["filename"]
                if file_path.exists():
                    file_path.unlink()
            
            self.index.clear()
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_size = sum(entry.get("size", 0) for entry in self.index.values())
            return {
                "entries": len(self.index),
                "total_size_mb": total_size / (1024 * 1024),
                "max_size_mb": self.max_size_mb,
                "utilization": (total_size / (1024 * 1024)) / self.max_size_mb
            }
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2)
        except Exception:
            pass  # Ignore save errors
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry from cache (assumes lock is held)."""
        if key in self.index:
            entry = self.index[key]
            file_path = self.cache_dir / entry["filename"]
            if file_path.exists():
                file_path.unlink()
            del self.index[key]
            self._save_index()
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit (assumes lock is held)."""
        total_size = sum(entry.get("size", 0) for entry in self.index.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
        
        # Sort by last access time (oldest first)
        entries_by_access = sorted(
            self.index.items(),
            key=lambda x: x[1].get("last_access", 0)
        )
        
        # Remove oldest entries until under limit
        for key, entry in entries_by_access:
            self._remove_entry(key)
            total_size -= entry.get("size", 0)
            
            if total_size <= max_size_bytes * 0.8:  # Leave some headroom
                break


class CacheManager:
    """Unified cache manager with memory and disk tiers."""
    
    def __init__(
        self,
        memory_size: int = 500,
        disk_size_mb: int = 1024,
        cache_dir: str = ".rfcc_cache"
    ):
        self.memory_cache = MemoryCache(max_size=memory_size)
        self.disk_cache = DiskCache(cache_dir=cache_dir, max_size_mb=disk_size_mb)
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, disk_only: bool = False) -> None:
        """Put value into cache."""
        if not disk_only:
            self.memory_cache.put(key, value)
        
        # Also save to disk for large objects or if requested
        try:
            # Estimate size (rough)
            size_estimate = len(pickle.dumps(value))
            if size_estimate > 1024 * 100 or disk_only:  # >100KB or explicitly requested
                self.disk_cache.put(key, value)
        except Exception:
            pass  # Ignore disk cache errors
    
    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            "memory": {
                "entries": self.memory_cache.size(),
                "max_size": self.memory_cache.max_size
            },
            "disk": self.disk_cache.get_stats()
        }


# Global cache manager
cache_manager = CacheManager()


def cached(
    ttl: int = 3600,
    key_func: Optional[callable] = None,
    disk_only: bool = False,
    use_cache: bool = True
):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result, disk_only=disk_only)
            
            return result
        
        # Add cache control methods
        wrapper.cache_clear = lambda: cache_manager.clear()
        wrapper.cache_stats = lambda: cache_manager.get_stats()
        
        return wrapper
    return decorator


def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function name and arguments."""
    key_data = {
        "func": func_name,
        "args": _serialize_args(args),
        "kwargs": _serialize_args(kwargs)
    }
    
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _serialize_args(args) -> Any:
    """Serialize arguments for cache key generation."""
    if isinstance(args, (list, tuple)):
        return [_serialize_args(arg) for arg in args]
    elif isinstance(args, dict):
        return {k: _serialize_args(v) for k, v in args.items()}
    elif isinstance(args, np.ndarray):
        return f"np_array_{args.shape}_{hashlib.md5(args.tobytes()).hexdigest()[:16]}"
    elif hasattr(args, '__dict__'):
        # For objects, use class name and a hash of their dict
        obj_dict = getattr(args, '__dict__', {})
        obj_hash = hashlib.md5(str(sorted(obj_dict.items())).encode()).hexdigest()[:16]
        return f"{args.__class__.__name__}_{obj_hash}"
    else:
        return str(args)


class VectorCache:
    """Specialized cache for embedding vectors with similarity search."""
    
    def __init__(self, max_vectors: int = 10000, similarity_threshold: float = 0.95):
        self.max_vectors = max_vectors
        self.similarity_threshold = similarity_threshold
        self.vectors: List[np.ndarray] = []
        self.keys: List[str] = []
        self.values: List[Any] = []
        self.lock = Lock()
    
    def get_similar(self, query_vector: np.ndarray, key_prefix: str = "") -> Optional[Any]:
        """Get cached value for similar vector."""
        with self.lock:
            if not self.vectors:
                return None
            
            # Calculate similarities
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return None
            
            similarities = []
            for i, vector in enumerate(self.vectors):
                if key_prefix and not self.keys[i].startswith(key_prefix):
                    continue
                
                vector_norm = np.linalg.norm(vector)
                if vector_norm == 0:
                    continue
                
                similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
                similarities.append((i, similarity))
            
            # Find best match
            if similarities:
                best_idx, best_sim = max(similarities, key=lambda x: x[1])
                if best_sim >= self.similarity_threshold:
                    return self.values[best_idx]
            
            return None
    
    def put(self, key: str, vector: np.ndarray, value: Any) -> None:
        """Store vector and associated value."""
        with self.lock:
            # Check if we need to evict
            if len(self.vectors) >= self.max_vectors:
                # Remove oldest entry
                self.vectors.pop(0)
                self.keys.pop(0)
                self.values.pop(0)
            
            self.vectors.append(vector.copy())
            self.keys.append(key)
            self.values.append(value)
    
    def clear(self) -> None:
        """Clear all cached vectors."""
        with self.lock:
            self.vectors.clear()
            self.keys.clear()
            self.values.clear()


# Global vector cache for embeddings
vector_cache = VectorCache()


@monitor_performance
def batch_process(
    items: List[Any],
    process_func: callable,
    batch_size: int = 32,
    num_workers: int = 1
) -> List[Any]:
    """Process items in batches for better performance."""
    if num_workers == 1:
        # Single-threaded processing
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
        return results
    
    # Multi-threaded processing
    import concurrent.futures
    
    results = [None] * len(items)
    
    def process_batch(start_idx: int, batch: List[Any]) -> None:
        batch_results = [process_func(item) for item in batch]
        for i, result in enumerate(batch_results):
            results[start_idx + i] = result
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            future = executor.submit(process_batch, i, batch)
            futures.append(future)
        
        # Wait for all batches to complete
        concurrent.futures.wait(futures)
    
    return results