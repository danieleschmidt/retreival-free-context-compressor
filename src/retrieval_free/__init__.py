"""Retrieval-Free Context Compressor - Generation 3: Scale-Ready

A high-performance transformer plug-in that compresses long documents into dense
"mega-tokens," enabling 256k-token context processing with enterprise-grade scaling.

Generation 3 Features:
- Multi-GPU processing with CUDA optimizations and mixed precision
- Distributed computing with Ray/Dask for multi-node parallel processing
- Async API endpoints with advanced queue management
- Auto-scaling with dynamic resource management
- Distributed caching with Redis/Memcached and CDN integration
- Multi-region deployment with global load balancing
- Real-time performance monitoring and bottleneck analysis
- Adaptive compression algorithms with streaming capabilities
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"


# Lazy imports to avoid dependency issues
def __getattr__(name):
    # Core compressors
    if name == "ContextCompressor":
        from .core import ContextCompressor

        return ContextCompressor
    elif name == "AutoCompressor":
        from .core import AutoCompressor

        return AutoCompressor
    elif name == "StreamingCompressor":
        from .streaming import StreamingCompressor

        return StreamingCompressor
    elif name == "SelectiveCompressor":
        from .selective import SelectiveCompressor

        return SelectiveCompressor
    elif name == "MultiDocCompressor":
        from .multi_doc import MultiDocCompressor

        return MultiDocCompressor
    elif name == "CompressorPlugin":
        from .plugins import CompressorPlugin

        return CompressorPlugin

    # Generation 3 Scaling Features
    elif name == "HighPerformanceCompressor":
        from .scaling import HighPerformanceCompressor

        return HighPerformanceCompressor
    elif name == "MultiGPUProcessor":
        from .scaling import MultiGPUProcessor

        return MultiGPUProcessor
    elif name == "AsyncProcessor":
        from .scaling import AsyncProcessor

        return AsyncProcessor
    elif name == "DistributedProcessor":
        from .scaling import DistributedProcessor

        return DistributedProcessor
    elif name == "AutoScaler":
        from .scaling import AutoScaler

        return AutoScaler

    # Distributed caching
    elif name == "DistributedCacheManager":
        from .distributed_cache import DistributedCacheManager

        return DistributedCacheManager
    elif name == "RedisBackend":
        from .distributed_cache import RedisBackend

        return RedisBackend
    elif name == "TieredDistributedCache":
        from .distributed_cache import TieredDistributedCache

        return TieredDistributedCache

    # Async API
    elif name == "AsyncCompressionAPI":
        from .async_api import AsyncCompressionAPI

        return AsyncCompressionAPI
    elif name == "create_api_server":
        from .async_api import create_api_server

        return create_api_server

    # Performance monitoring
    elif name == "PerformanceMonitor":
        from .performance_monitor import PerformanceMonitor

        return PerformanceMonitor
    elif name == "get_performance_monitor":
        from .performance_monitor import get_performance_monitor

        return get_performance_monitor
    elif name == "performance_profile":
        from .performance_monitor import performance_profile

        return performance_profile

    # Multi-region deployment
    elif name == "MultiRegionManager":
        from .multi_region import MultiRegionManager

        return MultiRegionManager
    elif name == "LoadBalancer":
        from .multi_region import LoadBalancer

        return LoadBalancer
    elif name == "setup_multi_region_deployment":
        from .multi_region import setup_multi_region_deployment

        return setup_multi_region_deployment

    # Adaptive compression
    elif name == "AdaptiveCompressor":
        from .adaptive_compression import AdaptiveCompressor

        return AdaptiveCompressor
    elif name == "ContentAnalyzer":
        from .adaptive_compression import ContentAnalyzer

        return ContentAnalyzer
    elif name == "StreamingCompressor":
        from .adaptive_compression import StreamingCompressor

        return StreamingCompressor

    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Core compressors
    "ContextCompressor",
    "AutoCompressor",
    "StreamingCompressor",
    "SelectiveCompressor",
    "MultiDocCompressor",
    "CompressorPlugin",
    # Generation 3 Scaling Features
    "HighPerformanceCompressor",
    "MultiGPUProcessor",
    "AsyncProcessor",
    "DistributedProcessor",
    "AutoScaler",
    # Distributed caching
    "DistributedCacheManager",
    "RedisBackend",
    "TieredDistributedCache",
    # Async API
    "AsyncCompressionAPI",
    "create_api_server",
    # Performance monitoring
    "PerformanceMonitor",
    "get_performance_monitor",
    "performance_profile",
    # Multi-region deployment
    "MultiRegionManager",
    "LoadBalancer",
    "setup_multi_region_deployment",
    # Adaptive compression
    "AdaptiveCompressor",
    "ContentAnalyzer",
    "StreamingCompressor",
]
