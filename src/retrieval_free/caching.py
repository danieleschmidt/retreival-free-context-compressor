"""Caching system for performance optimization."""

import os
import pickle
import hashlib
import logging
import threading
import time
from typing import Any, Dict, Optional, Tuple, List, Union
from pathlib import Path
from dataclasses import dataclass
import json
import sqlite3
from collections import OrderedDict
import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.timestamp > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp


class MemoryCache:
    """High-performance in-memory cache with LRU eviction."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 512,
        default_ttl: Optional[float] = None
    ):
        """Initialize memory cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_size_bytes = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            entry = self._cache[key]
            
            # Check expiration
            if entry.is_expired:
                self._remove_entry(key)
                self._misses += 1
                return None
            
            # Update access statistics
            entry.access_count += 1
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._hits += 1
            return entry.value
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            # Calculate size
            try:
                size_bytes = self._estimate_size(value)
            except Exception as e:
                logger.warning(f"Failed to estimate size for cache key {key}: {e}")
                return False
            
            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Evict entries to make space
            self._evict_to_fit(size_bytes)
            
            # Create and store entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.default_ttl
            )
            
            self._cache[key] = entry
            self._total_size_bytes += size_bytes
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._total_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_used_mb': self._total_size_bytes / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'avg_entry_size_kb': (self._total_size_bytes / len(self._cache) / 1024) if self._cache else 0
            }
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update size tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._total_size_bytes -= entry.size_bytes
            return True
        return False
    
    def _evict_to_fit(self, required_bytes: int) -> None:
        """Evict entries to fit required bytes."""
        # First evict expired entries
        expired_keys = [k for k, e in self._cache.items() if e.is_expired]
        for key in expired_keys:
            self._remove_entry(key)
            self._evictions += 1
        
        # Then evict LRU entries if needed
        while (self._total_size_bytes + required_bytes > self.max_memory_bytes or 
               len(self._cache) >= self.max_size):
            if not self._cache:
                break
            
            # Remove least recently used (first item)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self._evictions += 1
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            # For torch tensors
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            
            # For other objects, use pickle serialization size
            return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        
        except Exception:
            # Fallback estimation
            if hasattr(value, '__len__'):
                return len(value) * 8  # Rough estimate
            return 64  # Default estimate


class DiskCache:
    """Persistent disk-based cache with SQLite backend."""
    
    def __init__(
        self,
        cache_dir: str,
        max_size_gb: float = 5.0,
        default_ttl: Optional[float] = None
    ):
        """Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size_gb: Maximum cache size in GB
            default_ttl: Default TTL in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.default_ttl = default_ttl
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
        
        self._lock = threading.Lock()
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    size_bytes INTEGER NOT NULL,
                    ttl_seconds REAL,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON cache_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_count 
                ON cache_entries(access_count)
            """)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT filename, timestamp, ttl_seconds, access_count
                        FROM cache_entries 
                        WHERE key = ?
                    """, (key,))
                    
                    row = cursor.fetchone()
                    if not row:
                        return None
                    
                    filename, timestamp, ttl_seconds, access_count = row
                    
                    # Check expiration
                    if ttl_seconds and time.time() - timestamp > ttl_seconds:
                        self._remove_entry(key)
                        return None
                    
                    # Load from file
                    file_path = self.cache_dir / filename
                    if not file_path.exists():
                        self._remove_entry(key)
                        return None
                    
                    with open(file_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Update access count
                    conn.execute("""
                        UPDATE cache_entries 
                        SET access_count = access_count + 1
                        WHERE key = ?
                    """, (key,))
                    
                    return value
            
            except Exception as e:
                logger.error(f"Error reading from disk cache: {e}")
                return None
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                # Generate unique filename
                key_hash = hashlib.md5(key.encode()).hexdigest()
                filename = f"{key_hash}.pkl"
                file_path = self.cache_dir / filename
                
                # Serialize to file
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                size_bytes = file_path.stat().st_size
                
                # Store metadata in database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache_entries 
                        (key, filename, timestamp, size_bytes, ttl_seconds)
                        VALUES (?, ?, ?, ?, ?)
                    """, (key, filename, time.time(), size_bytes, ttl or self.default_ttl))
                
                # Cleanup if needed
                self._cleanup_if_needed()
                
                return True
            
            except Exception as e:
                logger.error(f"Error writing to disk cache: {e}")
                return False
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            try:
                # Remove all files
                for file_path in self.cache_dir.glob("*.pkl"):
                    file_path.unlink()
                
                # Clear database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM cache_entries")
            
            except Exception as e:
                logger.error(f"Error clearing disk cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as entry_count,
                            SUM(size_bytes) as total_size,
                            AVG(size_bytes) as avg_size,
                            SUM(access_count) as total_accesses
                        FROM cache_entries
                    """)
                    
                    row = cursor.fetchone()
                    entry_count, total_size, avg_size, total_accesses = row
                    
                    return {
                        'entry_count': entry_count or 0,
                        'total_size_mb': (total_size or 0) / 1024 / 1024,
                        'max_size_gb': self.max_size_bytes / 1024 / 1024 / 1024,
                        'avg_entry_size_kb': (avg_size or 0) / 1024,
                        'total_accesses': total_accesses or 0,
                        'cache_dir': str(self.cache_dir)
                    }
            
            except Exception as e:
                logger.error(f"Error getting disk cache stats: {e}")
                return {}
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry and associated file."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get filename
                cursor = conn.execute(
                    "SELECT filename FROM cache_entries WHERE key = ?", 
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    filename = row[0]
                    file_path = self.cache_dir / filename
                    
                    # Remove file
                    if file_path.exists():
                        file_path.unlink()
                    
                    # Remove from database
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    
                    return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error removing cache entry {key}: {e}")
            return False
    
    def _cleanup_if_needed(self) -> None:
        """Cleanup old entries if cache is too large."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check total size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size_bytes:
                    # Remove expired entries first
                    current_time = time.time()
                    expired_keys = []
                    
                    cursor = conn.execute("""
                        SELECT key FROM cache_entries 
                        WHERE ttl_seconds IS NOT NULL 
                        AND ? - timestamp > ttl_seconds
                    """, (current_time,))
                    
                    for row in cursor:
                        expired_keys.append(row[0])
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    # If still too large, remove least recently used
                    cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_entries")
                    total_size = cursor.fetchone()[0] or 0
                    
                    if total_size > self.max_size_bytes:
                        cursor = conn.execute("""
                            SELECT key FROM cache_entries 
                            ORDER BY access_count ASC, timestamp ASC
                        """)
                        
                        for row in cursor:
                            if total_size <= self.max_size_bytes * 0.9:  # Leave some headroom
                                break
                            
                            key = row[0]
                            # Get size before removing
                            size_cursor = conn.execute(
                                "SELECT size_bytes FROM cache_entries WHERE key = ?", 
                                (key,)
                            )
                            size_row = size_cursor.fetchone()
                            
                            if size_row and self._remove_entry(key):
                                total_size -= size_row[0]
        
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


class TieredCache:
    """Multi-tier cache with memory and disk layers."""
    
    def __init__(
        self,
        memory_cache: Optional[MemoryCache] = None,
        disk_cache: Optional[DiskCache] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize tiered cache.
        
        Args:
            memory_cache: Memory cache instance
            disk_cache: Disk cache instance
            cache_dir: Directory for disk cache (if disk_cache not provided)
        """
        self.memory_cache = memory_cache or MemoryCache()
        
        if disk_cache:
            self.disk_cache = disk_cache
        elif cache_dir:
            self.disk_cache = DiskCache(cache_dir)
        else:
            # Use default cache directory
            default_cache_dir = Path.home() / '.retrieval_free' / 'cache'
            self.disk_cache = DiskCache(str(default_cache_dir))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from tiered cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
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
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None
    ) -> bool:
        """Put value in tiered cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successfully cached in at least one tier
        """
        memory_success = self.memory_cache.put(key, value, ttl)
        disk_success = self.disk_cache.put(key, value, ttl)
        
        return memory_success or disk_success
    
    def remove(self, key: str) -> bool:
        """Remove entry from all cache tiers.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if entry was removed from any tier
        """
        memory_removed = self.memory_cache.remove(key)
        disk_removed = self.disk_cache.remove(key)
        
        return memory_removed or disk_removed
    
    def clear(self) -> None:
        """Clear all cache tiers."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            'memory_cache': self.memory_cache.get_stats(),
            'disk_cache': self.disk_cache.get_stats()
        }


def create_cache_key(
    text: str, 
    model_name: str, 
    parameters: Dict[str, Any]
) -> str:
    """Create cache key for compression operation.
    
    Args:
        text: Input text
        model_name: Model name
        parameters: Compression parameters
        
    Returns:
        Cache key string
    """
    # Create hash of input text
    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    # Create hash of parameters
    param_str = json.dumps(parameters, sort_keys=True)
    param_hash = hashlib.sha256(param_str.encode('utf-8')).hexdigest()[:8]
    
    # Combine into cache key
    return f"compress_{model_name}_{text_hash}_{param_hash}"