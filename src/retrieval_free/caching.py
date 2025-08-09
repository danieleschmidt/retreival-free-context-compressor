"""Caching system for compression results."""

import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .exceptions import CacheError


logger = logging.getLogger(__name__)


def create_cache_key(
    text: str,
    model_name: str,
    parameters: dict[str, Any]
) -> str:
    """Create a unique cache key for compression request.
    
    Args:
        text: Input text
        model_name: Model name used
        parameters: Compression parameters
        
    Returns:
        Unique cache key string
    """
    # Create a deterministic representation
    key_data = {
        'text_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'model': model_name,
        'params': parameters
    }

    # Sort parameters for consistency
    sorted_key = json.dumps(key_data, sort_keys=True, default=str)

    # Create final hash
    return hashlib.md5(sorted_key.encode('utf-8')).hexdigest()


class MemoryCache:
    """In-memory LRU cache for compression results."""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                return None

            # Check if expired
            if self._is_expired(key):
                self._remove(key)
                return None

            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value

            logger.debug(f"Cache hit for key: {key[:8]}...")
            return value

    def put(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override
        """
        with self._lock:
            # Remove if already exists
            if key in self._cache:
                self._remove(key)

            # Check capacity
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)
                logger.debug(f"Evicted cache item: {oldest_key[:8]}...")

            # Add new item
            self._cache[key] = value
            self._timestamps[key] = time.time() + (ttl or self.ttl)

            logger.debug(f"Cached item: {key[:8]}...")

    def remove(self, key: str) -> bool:
        """Remove item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            return self._remove(key)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            logger.debug("Cache cleared")

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def cleanup_expired(self) -> int:
        """Remove expired items.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            expired_keys = [
                key for key in self._cache.keys()
                if self._is_expired(key)
            ]

            for key in expired_keys:
                self._remove(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")

            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            current_time = time.time()
            expired_count = sum(
                1 for timestamp in self._timestamps.values()
                if timestamp <= current_time
            )

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'expired_items': expired_count,
                'utilization': len(self._cache) / self.max_size if self.max_size > 0 else 0
            }

    def _remove(self, key: str) -> bool:
        """Internal method to remove item."""
        if key in self._cache:
            self._cache.pop(key)
            self._timestamps.pop(key, None)
            return True
        return False

    def _is_expired(self, key: str) -> bool:
        """Check if item is expired."""
        timestamp = self._timestamps.get(key, 0)
        return time.time() > timestamp


class DiskCache:
    """Persistent disk cache for compression results."""

    def __init__(self, cache_dir: str | Path, max_size_mb: int = 1024):
        """Initialize disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

        # Create metadata file if it doesn't exist
        self.metadata_file = self.cache_dir / "metadata.json"
        if not self.metadata_file.exists():
            self._save_metadata({})

    def get(self, key: str) -> Any | None:
        """Get item from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            cache_file = self.cache_dir / f"{key}.pkl"

            if not cache_file.exists():
                return None

            try:
                # Load metadata to check expiration
                metadata = self._load_metadata()
                if key in metadata:
                    ttl = metadata[key].get('ttl', 0)
                    if time.time() > ttl:
                        # Expired
                        self.remove(key)
                        return None

                # Load from disk
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)

                # Update access time
                metadata[key]['last_accessed'] = time.time()
                self._save_metadata(metadata)

                logger.debug(f"Disk cache hit for key: {key[:8]}...")
                return value

            except Exception as e:
                logger.warning(f"Failed to load from disk cache: {e}")
                # Clean up corrupted file
                cache_file.unlink(missing_ok=True)
                return None

    def put(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Put item in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        with self._lock:
            try:
                # Check disk space before writing
                self._cleanup_if_needed()

                cache_file = self.cache_dir / f"{key}.pkl"

                # Save to disk
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)

                # Update metadata
                metadata = self._load_metadata()
                metadata[key] = {
                    'ttl': time.time() + ttl,
                    'last_accessed': time.time(),
                    'size_bytes': cache_file.stat().st_size
                }
                self._save_metadata(metadata)

                logger.debug(f"Saved to disk cache: {key[:8]}...")

            except Exception as e:
                logger.error(f"Failed to save to disk cache: {e}")
                raise CacheError(f"Disk cache write failed: {e}", cache_key=key)

    def remove(self, key: str) -> bool:
        """Remove item from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed, False if not found
        """
        with self._lock:
            cache_file = self.cache_dir / f"{key}.pkl"

            if cache_file.exists():
                try:
                    cache_file.unlink()

                    # Update metadata
                    metadata = self._load_metadata()
                    metadata.pop(key, None)
                    self._save_metadata(metadata)

                    return True
                except Exception as e:
                    logger.error(f"Failed to remove from disk cache: {e}")

            return False

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            try:
                # Remove all .pkl files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

                # Clear metadata
                self._save_metadata({})

                logger.debug("Disk cache cleared")

            except Exception as e:
                logger.error(f"Failed to clear disk cache: {e}")

    def size(self) -> int:
        """Get current cache size in items."""
        with self._lock:
            metadata = self._load_metadata()
            return len(metadata)

    def size_bytes(self) -> int:
        """Get current cache size in bytes."""
        with self._lock:
            metadata = self._load_metadata()
            return sum(
                item.get('size_bytes', 0)
                for item in metadata.values()
            )

    def cleanup_expired(self) -> int:
        """Remove expired items.
        
        Returns:
            Number of items removed
        """
        with self._lock:
            metadata = self._load_metadata()
            current_time = time.time()

            expired_keys = [
                key for key, meta in metadata.items()
                if meta.get('ttl', 0) <= current_time
            ]

            for key in expired_keys:
                self.remove(key)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired disk cache items")

            return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            size_bytes = self.size_bytes()
            return {
                'size': self.size(),
                'size_bytes': size_bytes,
                'size_mb': size_bytes / 1024 / 1024,
                'max_size_mb': self.max_size_mb,
                'utilization': (size_bytes / 1024 / 1024) / self.max_size_mb if self.max_size_mb > 0 else 0
            }

    def _load_metadata(self) -> dict[str, Any]:
        """Load metadata from disk."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")

        return {}

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")

    def _cleanup_if_needed(self) -> None:
        """Clean up cache if size limit exceeded."""
        current_size_mb = self.size_bytes() / 1024 / 1024

        if current_size_mb > self.max_size_mb:
            # Remove least recently accessed items
            metadata = self._load_metadata()

            # Sort by last accessed time
            items_by_access = sorted(
                metadata.items(),
                key=lambda x: x[1].get('last_accessed', 0)
            )

            # Remove items until under limit
            removed_count = 0
            for key, _ in items_by_access:
                if current_size_mb <= self.max_size_mb * 0.8:  # Leave some headroom
                    break

                if self.remove(key):
                    removed_count += 1
                    current_size_mb = self.size_bytes() / 1024 / 1024

            if removed_count > 0:
                logger.debug(f"Cleaned up {removed_count} items to free disk cache space")


class TieredCache:
    """Two-tier cache with memory and disk layers."""

    def __init__(
        self,
        memory_cache: MemoryCache | None = None,
        disk_cache: DiskCache | None = None,
        cache_dir: str | Path | None = None
    ):
        """Initialize tiered cache.
        
        Args:
            memory_cache: Optional memory cache instance
            disk_cache: Optional disk cache instance
            cache_dir: Cache directory for default disk cache
        """
        self.memory_cache = memory_cache or MemoryCache()

        if disk_cache:
            self.disk_cache = disk_cache
        elif cache_dir:
            self.disk_cache = DiskCache(cache_dir)
        else:
            # Default cache directory
            import tempfile
            cache_dir = Path(tempfile.gettempdir()) / "retrieval_free_cache"
            self.disk_cache = DiskCache(cache_dir)

    def get(self, key: str) -> Any | None:
        """Get item from cache, checking memory first then disk.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value

        # Check disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.put(key, value)
            return value

        return None

    def put(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Put item in both cache layers.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
        """
        # Store in both layers
        self.memory_cache.put(key, value, ttl)
        self.disk_cache.put(key, value, ttl)

    def remove(self, key: str) -> bool:
        """Remove item from both cache layers.
        
        Args:
            key: Cache key
            
        Returns:
            True if item was removed from at least one layer
        """
        memory_removed = self.memory_cache.remove(key)
        disk_removed = self.disk_cache.remove(key)
        return memory_removed or disk_removed

    def clear(self) -> None:
        """Clear both cache layers."""
        self.memory_cache.clear()
        self.disk_cache.clear()

    def cleanup_expired(self) -> tuple[int, int]:
        """Clean up expired items in both layers.
        
        Returns:
            Tuple of (memory_cleaned, disk_cleaned) counts
        """
        memory_cleaned = self.memory_cache.cleanup_expired()
        disk_cleaned = self.disk_cache.cleanup_expired()
        return memory_cleaned, disk_cleaned

    def get_stats(self) -> dict[str, Any]:
        """Get statistics for both cache layers."""
        return {
            'memory': self.memory_cache.get_stats(),
            'disk': self.disk_cache.get_stats()
        }


# Global cache instance
_default_cache: TieredCache | None = None


def get_default_cache() -> TieredCache:
    """Get or create the default cache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = TieredCache()
    return _default_cache


def clear_default_cache() -> None:
    """Clear the default cache."""
    cache = get_default_cache()
    cache.clear()
