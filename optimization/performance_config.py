
"""
Performance optimization configuration for quantum-scale processing.
"""

import os
import multiprocessing
from typing import Dict, Any


class PerformanceConfig:
    """Centralized performance configuration management."""
    
    # CPU and memory optimization
    MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) + 4)
    MAX_MEMORY_GB = int(os.getenv("MAX_MEMORY_GB", "16"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    
    # Caching configuration  
    CACHE_SIZE_MB = int(os.getenv("CACHE_SIZE_MB", "1024"))
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    ENABLE_PERSISTENT_CACHE = os.getenv("ENABLE_PERSISTENT_CACHE", "true").lower() == "true"
    
    # Compression optimization
    COMPRESSION_BATCH_SIZE = int(os.getenv("COMPRESSION_BATCH_SIZE", "100"))
    COMPRESSION_QUALITY = float(os.getenv("COMPRESSION_QUALITY", "0.95"))
    ENABLE_GPU_ACCELERATION = os.getenv("ENABLE_GPU_ACCELERATION", "false").lower() == "true"
    
    # Network optimization
    CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "20"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    
    # Monitoring and metrics
    ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    METRICS_INTERVAL = int(os.getenv("METRICS_INTERVAL", "60"))
    ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    
    @classmethod
    def get_optimization_params(cls) -> Dict[str, Any]:
        """Get all optimization parameters."""
        return {
            "max_workers": cls.MAX_WORKERS,
            "max_memory_gb": cls.MAX_MEMORY_GB,
            "chunk_size": cls.CHUNK_SIZE,
            "cache_size_mb": cls.CACHE_SIZE_MB,
            "cache_ttl": cls.CACHE_TTL_SECONDS,
            "compression_batch_size": cls.COMPRESSION_BATCH_SIZE,
            "compression_quality": cls.COMPRESSION_QUALITY,
            "connection_pool_size": cls.CONNECTION_POOL_SIZE,
            "enable_metrics": cls.ENABLE_METRICS,
            "enable_profiling": cls.ENABLE_PROFILING
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration parameters."""
        if cls.MAX_WORKERS <= 0:
            raise ValueError("MAX_WORKERS must be positive")
        if cls.MAX_MEMORY_GB <= 0:
            raise ValueError("MAX_MEMORY_GB must be positive") 
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if cls.COMPRESSION_QUALITY <= 0 or cls.COMPRESSION_QUALITY > 1:
            raise ValueError("COMPRESSION_QUALITY must be between 0 and 1")
        return True


# Global performance configuration
perf_config = PerformanceConfig()
