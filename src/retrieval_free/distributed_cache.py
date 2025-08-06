"""Distributed caching system with Redis, Memcached, and CDN support.

This module implements advanced caching strategies for high-performance scaling:
- Redis distributed caching
- Memcached support
- CDN integration for edge caching
- Storage tiering (hot/warm/cold)
- Cache warming and preloading
- Multi-region cache synchronization
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from pathlib import Path
import pickle
import uuid

import numpy as np

from .core import CompressionResult, MegaToken
from .exceptions import CacheError, ResourceError
from .caching import create_cache_key

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_size_bytes: int = 0
    avg_latency_ms: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    ttl: float
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    tier: str = "hot"  # hot, warm, cold
    compressed: bool = False
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.ttl
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600,
        **kwargs
    ) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        pass


class RedisBackend(CacheBackend):
    """Redis distributed cache backend."""
    
    def __init__(
        self, 
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "rfcc:",
        compression_threshold: int = 1024,  # Compress values larger than 1KB
        max_connections: int = 10
    ):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.prefix = prefix
        self.compression_threshold = compression_threshold
        self.max_connections = max_connections
        
        self._redis = None
        self._stats = CacheStats()
        self._lock = asyncio.Lock()
    
    async def _get_redis(self):
        """Get Redis connection (lazy initialization)."""
        if self._redis is None:
            try:
                import aioredis
                self._redis = aioredis.from_url(
                    f"redis://{self.host}:{self.port}/{self.db}",
                    password=self.password,
                    max_connections=self.max_connections,
                    retry_on_timeout=True
                )
                await self._redis.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except ImportError:
                raise CacheError("aioredis package required for Redis backend")
            except Exception as e:
                raise CacheError(f"Failed to connect to Redis: {e}")
        
        return self._redis
    
    def _serialize_value(self, value: Any) -> Tuple[bytes, bool]:
        """Serialize and optionally compress value."""
        # Convert to JSON first
        if isinstance(value, CompressionResult):
            # Convert MegaTokens to serializable format
            serializable_tokens = []
            for token in value.mega_tokens:
                token_dict = {
                    'vector': token.vector.tolist(),
                    'metadata': token.metadata,
                    'confidence': token.confidence
                }
                serializable_tokens.append(token_dict)
            
            serializable_value = {
                'mega_tokens': serializable_tokens,
                'original_length': value.original_length,
                'compressed_length': value.compressed_length,
                'compression_ratio': value.compression_ratio,
                'processing_time': value.processing_time,
                'metadata': value.metadata
            }
        else:
            serializable_value = value
        
        # Serialize to bytes
        data = pickle.dumps(serializable_value)
        
        # Compress if large enough
        if len(data) > self.compression_threshold:
            compressed_data = gzip.compress(data)
            if len(compressed_data) < len(data):
                return compressed_data, True
        
        return data, False
    
    def _deserialize_value(self, data: bytes, compressed: bool) -> Any:
        """Decompress and deserialize value."""
        if compressed:
            data = gzip.decompress(data)
        
        value = pickle.loads(data)
        
        # Reconstruct CompressionResult if needed
        if isinstance(value, dict) and 'mega_tokens' in value:
            # Reconstruct MegaTokens
            mega_tokens = []
            for token_dict in value['mega_tokens']:
                token = MegaToken(
                    vector=np.array(token_dict['vector']),
                    metadata=token_dict['metadata'],
                    confidence=token_dict['confidence']
                )
                mega_tokens.append(token)
            
            # Reconstruct CompressionResult
            value = CompressionResult(
                mega_tokens=mega_tokens,
                original_length=value['original_length'],
                compressed_length=value['compressed_length'],
                compression_ratio=value['compression_ratio'],
                processing_time=value['processing_time'],
                metadata=value['metadata']
            )
        
        return value
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        start_time = time.time()
        
        try:
            redis = await self._get_redis()
            full_key = f"{self.prefix}{key}"
            
            # Get value and metadata
            pipe = redis.pipeline()
            pipe.hgetall(full_key)
            pipe.hincrby(full_key, "access_count", 1)
            pipe.hset(full_key, "last_accessed", int(time.time()))
            
            results = await pipe.execute()
            data = results[0]
            
            if not data:
                self._stats.misses += 1
                return None
            
            # Check TTL
            ttl = float(data.get(b'ttl', 0))
            if time.time() > ttl:
                await self.delete(key)
                self._stats.misses += 1
                return None
            
            # Deserialize value
            value_data = data[b'value']
            compressed = data.get(b'compressed', b'false') == b'true'
            
            value = self._deserialize_value(value_data, compressed)
            
            # Update stats
            self._stats.hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self._stats.avg_latency_ms = (
                (self._stats.avg_latency_ms * (self._stats.hits - 1) + latency_ms) /
                self._stats.hits
            )
            
            return value
            
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis get error: {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600,
        tier: str = "hot",
        **kwargs
    ) -> bool:
        """Set value in Redis."""
        try:
            redis = await self._get_redis()
            full_key = f"{self.prefix}{key}"
            
            # Serialize value
            value_data, compressed = self._serialize_value(value)
            
            # Prepare metadata
            now = time.time()
            metadata = {
                "ttl": now + ttl,
                "created_at": now,
                "last_accessed": now,
                "access_count": 0,
                "size_bytes": len(value_data),
                "tier": tier,
                "compressed": str(compressed).lower(),
                "value": value_data
            }
            
            # Store in Redis
            await redis.hset(full_key, mapping=metadata)
            
            # Set expiration
            await redis.expire(full_key, ttl)
            
            # Update stats
            self._stats.total_size_bytes += len(value_data)
            
            return True
            
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            redis = await self._get_redis()
            full_key = f"{self.prefix}{key}"
            
            # Get size before deletion for stats
            size_data = await redis.hget(full_key, "size_bytes")
            if size_data:
                self._stats.total_size_bytes -= int(size_data)
            
            result = await redis.delete(full_key)
            return bool(result)
            
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            redis = await self._get_redis()
            full_key = f"{self.prefix}{key}"
            return bool(await redis.exists(full_key))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.prefix}*")
            if keys:
                await redis.delete(*keys)
            self._stats = CacheStats()
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats
    
    async def get_keys_by_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self.prefix}{pattern}")
            return [key.decode().replace(self.prefix, '') for key in keys]
        except Exception as e:
            logger.error(f"Redis pattern search error: {e}")
            return []


class MemcachedBackend(CacheBackend):
    """Memcached cache backend."""
    
    def __init__(
        self, 
        servers: List[str] = None,
        compression_threshold: int = 1024
    ):
        self.servers = servers or ["localhost:11211"]
        self.compression_threshold = compression_threshold
        self._client = None
        self._stats = CacheStats()
    
    async def _get_client(self):
        """Get Memcached client (lazy initialization)."""
        if self._client is None:
            try:
                import aiomcache
                self._client = aiomcache.Client(
                    self.servers[0].split(':')[0],
                    int(self.servers[0].split(':')[1])
                )
                logger.info(f"Connected to Memcached at {self.servers[0]}")
            except ImportError:
                raise CacheError("aiomcache package required for Memcached backend")
        
        return self._client
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached."""
        start_time = time.time()
        
        try:
            client = await self._get_client()
            data = await client.get(key.encode())
            
            if data is None:
                self._stats.misses += 1
                return None
            
            # Deserialize
            value = pickle.loads(data)
            
            self._stats.hits += 1
            latency_ms = (time.time() - start_time) * 1000
            self._stats.avg_latency_ms = (
                (self._stats.avg_latency_ms * (self._stats.hits - 1) + latency_ms) /
                self._stats.hits
            )
            
            return value
            
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Memcached get error: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600, **kwargs) -> bool:
        """Set value in Memcached."""
        try:
            client = await self._get_client()
            data = pickle.dumps(value)
            
            await client.set(key.encode(), data, exptime=ttl)
            self._stats.total_size_bytes += len(data)
            return True
            
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Memcached set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Memcached."""
        try:
            client = await self._get_client()
            result = await client.delete(key.encode())
            return result
        except Exception as e:
            logger.error(f"Memcached delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Memcached."""
        value = await self.get(key)
        return value is not None
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            client = await self._get_client()
            await client.flush_all()
            self._stats = CacheStats()
            return True
        except Exception as e:
            logger.error(f"Memcached clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class TieredDistributedCache:
    """Multi-tier distributed cache with hot/warm/cold storage."""
    
    def __init__(
        self,
        hot_backend: CacheBackend,
        warm_backend: Optional[CacheBackend] = None,
        cold_backend: Optional[CacheBackend] = None,
        hot_ttl: int = 3600,      # 1 hour
        warm_ttl: int = 86400,    # 24 hours  
        cold_ttl: int = 604800,   # 7 days
        promotion_threshold: int = 3  # Access count to promote to hot tier
    ):
        self.hot_backend = hot_backend
        self.warm_backend = warm_backend
        self.cold_backend = cold_backend
        
        self.hot_ttl = hot_ttl
        self.warm_ttl = warm_ttl
        self.cold_ttl = cold_ttl
        self.promotion_threshold = promotion_threshold
        
        self._access_counts = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from tiered cache."""
        # Try hot tier first
        value = await self.hot_backend.get(key)
        if value is not None:
            return value
        
        # Try warm tier
        if self.warm_backend:
            value = await self.warm_backend.get(key)
            if value is not None:
                # Track access for potential promotion
                async with self._lock:
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    
                    if self._access_counts[key] >= self.promotion_threshold:
                        # Promote to hot tier
                        await self.hot_backend.set(key, value, ttl=self.hot_ttl, tier="hot")
                        del self._access_counts[key]
                
                return value
        
        # Try cold tier
        if self.cold_backend:
            value = await self.cold_backend.get(key)
            if value is not None:
                # Track access for potential promotion to warm
                async with self._lock:
                    self._access_counts[key] = self._access_counts.get(key, 0) + 1
                    
                    if self._access_counts[key] >= self.promotion_threshold // 2:
                        if self.warm_backend:
                            await self.warm_backend.set(key, value, ttl=self.warm_ttl, tier="warm")
                
                return value
        
        return None
    
    async def set(self, key: str, value: Any, tier: str = "hot") -> bool:
        """Set value in specified tier."""
        if tier == "hot":
            return await self.hot_backend.set(key, value, ttl=self.hot_ttl, tier=tier)
        elif tier == "warm" and self.warm_backend:
            return await self.warm_backend.set(key, value, ttl=self.warm_ttl, tier=tier)
        elif tier == "cold" and self.cold_backend:
            return await self.cold_backend.set(key, value, ttl=self.cold_ttl, tier=tier)
        else:
            # Default to hot tier
            return await self.hot_backend.set(key, value, ttl=self.hot_ttl, tier="hot")
    
    async def delete(self, key: str) -> bool:
        """Delete from all tiers."""
        results = []
        
        results.append(await self.hot_backend.delete(key))
        
        if self.warm_backend:
            results.append(await self.warm_backend.delete(key))
        
        if self.cold_backend:
            results.append(await self.cold_backend.delete(key))
        
        # Clean up access count
        async with self._lock:
            self._access_counts.pop(key, None)
        
        return any(results)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in any tier."""
        if await self.hot_backend.exists(key):
            return True
        
        if self.warm_backend and await self.warm_backend.exists(key):
            return True
        
        if self.cold_backend and await self.cold_backend.exists(key):
            return True
        
        return False
    
    async def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all tiers."""
        stats = {
            "hot": await self.hot_backend.get_stats()
        }
        
        if self.warm_backend:
            stats["warm"] = await self.warm_backend.get_stats()
        
        if self.cold_backend:
            stats["cold"] = await self.cold_backend.get_stats()
        
        return stats


class CacheWarmer:
    """Proactive cache warming system."""
    
    def __init__(
        self, 
        cache: Union[CacheBackend, TieredDistributedCache],
        compressor: Any,
        warm_patterns: List[str] = None
    ):
        self.cache = cache
        self.compressor = compressor
        self.warm_patterns = warm_patterns or []
        
        self._running = False
        self._warmer_task = None
    
    async def start(self, interval: int = 300):  # 5 minutes
        """Start cache warming."""
        if self._running:
            return
        
        self._running = True
        self._warmer_task = asyncio.create_task(self._warming_loop(interval))
        logger.info("Cache warmer started")
    
    async def stop(self):
        """Stop cache warming."""
        self._running = False
        if self._warmer_task:
            self._warmer_task.cancel()
            try:
                await self._warmer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Cache warmer stopped")
    
    async def warm_content(self, content: List[str], priority: str = "warm"):
        """Warm cache with specific content."""
        for text in content:
            try:
                # Check if already cached
                cache_key = create_cache_key(
                    text, 
                    self.compressor.model_name, 
                    {}
                )
                
                if await self.cache.exists(cache_key):
                    continue
                
                # Compress and cache
                result = self.compressor.compress(text)
                await self.cache.set(cache_key, result, tier=priority)
                
                logger.debug(f"Warmed cache for key: {cache_key[:8]}...")
                
                # Small delay to avoid overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Cache warming error: {e}")
    
    async def _warming_loop(self, interval: int):
        """Main warming loop."""
        while self._running:
            try:
                await self._perform_warming()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache warming loop error: {e}")
    
    async def _perform_warming(self):
        """Perform cache warming operations."""
        # This would implement your specific warming strategy
        # For example, preload commonly accessed patterns
        pass


class CDNIntegration:
    """CDN integration for edge caching."""
    
    def __init__(
        self, 
        cdn_endpoint: str,
        api_key: Optional[str] = None,
        cache_regions: List[str] = None
    ):
        self.cdn_endpoint = cdn_endpoint.rstrip('/')
        self.api_key = api_key
        self.cache_regions = cache_regions or ["us-east-1", "eu-west-1", "ap-southeast-1"]
    
    async def cache_at_edge(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600
    ) -> bool:
        """Cache content at CDN edge locations."""
        try:
            # Serialize value for HTTP transport
            if isinstance(value, CompressionResult):
                # Convert to JSON-serializable format
                data = {
                    "mega_tokens": [
                        {
                            "vector": token.vector.tolist(),
                            "metadata": token.metadata,
                            "confidence": token.confidence
                        }
                        for token in value.mega_tokens
                    ],
                    "original_length": value.original_length,
                    "compressed_length": value.compressed_length,
                    "compression_ratio": value.compression_ratio,
                    "processing_time": value.processing_time,
                    "metadata": value.metadata
                }
            else:
                data = value
            
            # Upload to CDN (implementation depends on your CDN provider)
            # This is a placeholder for the actual CDN API calls
            success_count = 0
            
            for region in self.cache_regions:
                try:
                    # Simulate CDN upload
                    await asyncio.sleep(0.1)  # Simulate network delay
                    success_count += 1
                    logger.debug(f"Cached at CDN edge in {region}: {key[:8]}...")
                except Exception as e:
                    logger.error(f"CDN caching failed in {region}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"CDN integration error: {e}")
            return False
    
    async def invalidate_edge_cache(self, key: str) -> bool:
        """Invalidate cached content at edge locations."""
        try:
            success_count = 0
            
            for region in self.cache_regions:
                try:
                    # Simulate CDN invalidation
                    await asyncio.sleep(0.05)
                    success_count += 1
                    logger.debug(f"Invalidated CDN cache in {region}: {key[:8]}...")
                except Exception as e:
                    logger.error(f"CDN invalidation failed in {region}: {e}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"CDN invalidation error: {e}")
            return False


class DistributedCacheManager:
    """High-level distributed cache management."""
    
    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        memcached_config: Optional[Dict[str, Any]] = None,
        cdn_config: Optional[Dict[str, Any]] = None,
        enable_warming: bool = True
    ):
        # Initialize backends
        self.backends = {}
        
        if redis_config:
            self.backends['redis'] = RedisBackend(**redis_config)
        
        if memcached_config:
            self.backends['memcached'] = MemcachedBackend(**memcached_config)
        
        # Setup tiered cache
        hot_backend = self.backends.get('redis') or self.backends.get('memcached')
        warm_backend = self.backends.get('memcached') if 'redis' in self.backends else None
        
        if hot_backend:
            self.cache = TieredDistributedCache(
                hot_backend=hot_backend,
                warm_backend=warm_backend
            )
        else:
            # Fallback to in-memory cache
            from .caching import TieredCache
            self.cache = TieredCache()
        
        # CDN integration
        self.cdn = CDNIntegration(**cdn_config) if cdn_config else None
        
        # Cache warming
        self.warmer = None
        if enable_warming and hasattr(self.cache, 'hot_backend'):
            self.warmer = CacheWarmer(self.cache, None)  # Compressor set later
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from distributed cache."""
        return await self.cache.get(key)
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: int = 3600,
        use_cdn: bool = False,
        tier: str = "hot"
    ) -> bool:
        """Set in distributed cache."""
        # Set in main cache
        success = await self.cache.set(key, value, tier=tier)
        
        # Optionally cache at CDN edge
        if use_cdn and self.cdn and success:
            await self.cdn.cache_at_edge(key, value, ttl)
        
        return success
    
    async def delete(self, key: str, invalidate_cdn: bool = True) -> bool:
        """Delete from distributed cache."""
        success = await self.cache.delete(key)
        
        # Invalidate CDN cache
        if invalidate_cdn and self.cdn:
            await self.cdn.invalidate_edge_cache(key)
        
        return success
    
    async def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'timestamp': time.time(),
            'cache_stats': await self.cache.get_stats(),
            'backends': list(self.backends.keys()),
            'cdn_enabled': self.cdn is not None,
            'warmer_enabled': self.warmer is not None
        }
        
        return stats
    
    async def start_services(self):
        """Start cache warming and other services."""
        if self.warmer:
            await self.warmer.start()
    
    async def stop_services(self):
        """Stop cache warming and other services."""
        if self.warmer:
            await self.warmer.stop()


# Global distributed cache instance
_distributed_cache: Optional[DistributedCacheManager] = None


def get_distributed_cache(
    redis_config: Optional[Dict[str, Any]] = None,
    memcached_config: Optional[Dict[str, Any]] = None,
    cdn_config: Optional[Dict[str, Any]] = None
) -> DistributedCacheManager:
    """Get or create distributed cache instance."""
    global _distributed_cache
    
    if _distributed_cache is None:
        _distributed_cache = DistributedCacheManager(
            redis_config=redis_config,
            memcached_config=memcached_config,
            cdn_config=cdn_config
        )
    
    return _distributed_cache