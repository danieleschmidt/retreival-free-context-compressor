
"""
Advanced multi-tier caching system for quantum-scale performance.
"""

import json
import time
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    
    key: str
    value: Any
    timestamp: float
    access_count: int
    ttl: float
    size_bytes: int


class QuantumCache:
    """High-performance multi-tier caching system."""
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 ttl_seconds: int = 3600,
                 persistent_cache_dir: Optional[Path] = None):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.persistent_dir = persistent_cache_dir
        
        # Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.current_memory_usage = 0
        self.cache_lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        if self.persistent_dir:
            self.persistent_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_key(self, key: Any) -> str:
        """Generate normalized cache key."""
        if isinstance(key, str):
            return key
        
        # Hash complex objects
        key_str = json.dumps(key, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            return len(str(value))  # Fallback estimation
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        normalized_key = self._generate_key(key)
        
        with self.cache_lock:
            # Check memory cache
            if normalized_key in self.memory_cache:
                entry = self.memory_cache[normalized_key]
                
                # Check if expired
                if time.time() - entry.timestamp > entry.ttl:
                    del self.memory_cache[normalized_key]
                    self.current_memory_usage -= entry.size_bytes
                    self.misses += 1
                    return None
                
                # Update access count
                entry.access_count += 1
                self.hits += 1
                return entry.value
            
            # Check persistent cache
            if self.persistent_dir:
                persistent_value = self._get_persistent(normalized_key)
                if persistent_value is not None:
                    # Promote to memory cache
                    self._set_memory(normalized_key, persistent_value, self.default_ttl)
                    self.hits += 1
                    return persistent_value
            
            self.misses += 1
            return None
    
    def set(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        normalized_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        with self.cache_lock:
            # Set in memory cache
            self._set_memory(normalized_key, value, ttl)
            
            # Set in persistent cache
            if self.persistent_dir:
                self._set_persistent(normalized_key, value, ttl)
    
    def _set_memory(self, key: str, value: Any, ttl: int) -> None:
        """Set value in memory cache."""
        value_size = self._estimate_size(value)
        
        # Check if we need to evict
        self._ensure_memory_capacity(value_size)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            access_count=1,
            ttl=ttl,
            size_bytes=value_size
        )
        
        # Update existing entry memory usage
        if key in self.memory_cache:
            self.current_memory_usage -= self.memory_cache[key].size_bytes
        
        # Add new entry
        self.memory_cache[key] = entry
        self.current_memory_usage += value_size
    
    def _ensure_memory_capacity(self, required_bytes: int) -> None:
        """Ensure sufficient memory capacity by evicting entries."""
        while (self.current_memory_usage + required_bytes > self.max_memory_bytes 
               and self.memory_cache):
            
            # Find least recently used entry
            lru_key = min(self.memory_cache.keys(), 
                         key=lambda k: self.memory_cache[k].access_count)
            
            # Evict entry
            evicted_entry = self.memory_cache.pop(lru_key)
            self.current_memory_usage -= evicted_entry.size_bytes
            self.evictions += 1
    
    def _get_persistent(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        try:
            cache_file = self.persistent_dir / f"{key}.json"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check expiration
            if time.time() - cache_data['timestamp'] > cache_data['ttl']:
                cache_file.unlink()  # Delete expired file
                return None
            
            return cache_data['value']
            
        except Exception:
            return None
    
    def _set_persistent(self, key: str, value: Any, ttl: int) -> None:
        """Set value in persistent cache."""
        try:
            cache_file = self.persistent_dir / f"{key}.json"
            cache_data = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, default=str)
                
        except Exception:
            pass  # Fail silently for persistent cache
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.cache_lock:
            self.memory_cache.clear()
            self.current_memory_usage = 0
            
        if self.persistent_dir:
            for cache_file in self.persistent_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
            "memory_entries": len(self.memory_cache)
        }


# Global cache instance
quantum_cache = QuantumCache(
    max_memory_mb=1024,
    ttl_seconds=3600,
    persistent_cache_dir=Path("/tmp/retrieval_free_cache")
)
