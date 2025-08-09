"""Advanced performance monitoring with real-time metrics and bottleneck analysis.

This module provides comprehensive performance monitoring capabilities:
- Real-time metrics collection and analysis
- Bottleneck detection and recommendations
- Performance profiling with flame graphs
- Resource usage monitoring
- A/B testing framework for compression strategies
- Capacity planning and predictive analytics
"""

import asyncio
import logging
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import psutil

from .core import CompressionResult


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class Severity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class PerformanceAlert:
    """Performance alert."""
    id: str
    name: str
    severity: Severity
    message: str
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: float | None = None


@dataclass
class BottleneckAnalysis:
    """Analysis of performance bottlenecks."""
    component: str
    severity: Severity
    description: str
    impact_score: float  # 0-100
    recommendations: list[str]
    metrics: dict[str, float]
    timestamp: float


@dataclass
class ResourceUsage:
    """System resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_bytes_sent: float
    network_bytes_recv: float
    gpu_usage: dict[str, float] | None = None


class MetricsCollector:
    """Collects and stores performance metrics."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(float)
        self.histograms = defaultdict(list)
        self._lock = threading.RLock()

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: dict[str, str] | None = None,
        timestamp: float | None = None
    ):
        """Record a metric value."""
        timestamp = timestamp or time.time()
        labels = labels or {}

        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=timestamp,
            labels=labels,
            metric_type=metric_type
        )

        with self._lock:
            # Store in time series
            self.metrics[name].append(metric_point)

            # Update aggregates based on type
            if metric_type == MetricType.COUNTER:
                self.counters[name] += value
            elif metric_type == MetricType.HISTOGRAM:
                self.histograms[name].append(value)
                # Keep only recent values for histograms
                if len(self.histograms[name]) > 1000:
                    self.histograms[name] = self.histograms[name][-1000:]

    def get_metric_history(
        self,
        name: str,
        time_window: float | None = None
    ) -> list[MetricPoint]:
        """Get metric history, optionally filtered by time window."""
        with self._lock:
            if name not in self.metrics:
                return []

            points = list(self.metrics[name])

            if time_window:
                cutoff_time = time.time() - time_window
                points = [p for p in points if p.timestamp >= cutoff_time]

            return points

    def get_metric_stats(self, name: str, time_window: float | None = None) -> dict[str, float]:
        """Get statistical summary of metric."""
        points = self.get_metric_history(name, time_window)

        if not points:
            return {}

        values = [p.value for p in points]

        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }

    def get_counter_value(self, name: str) -> float:
        """Get current counter value."""
        with self._lock:
            return self.counters.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            values = self.histograms.get(name, [])

            if not values:
                return {}

            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'p50': np.percentile(values, 50),
                'p90': np.percentile(values, 90),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }


class ResourceMonitor:
    """Monitors system resource usage."""

    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self._running = False
        self._monitor_task = None

        # Initialize baseline measurements
        self._last_disk_io = None
        self._last_network_io = None

    async def start(self):
        """Start resource monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")

    async def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                usage = self._collect_resource_usage()
                self._record_usage_metrics(usage)
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0.0
        disk_write_mb = 0.0

        if disk_io and self._last_disk_io:
            read_bytes = disk_io.read_bytes - self._last_disk_io.read_bytes
            write_bytes = disk_io.write_bytes - self._last_disk_io.write_bytes
            disk_read_mb = read_bytes / 1024 / 1024
            disk_write_mb = write_bytes / 1024 / 1024

        if disk_io:
            self._last_disk_io = disk_io

        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent = 0.0
        network_recv = 0.0

        if network_io and self._last_network_io:
            sent_bytes = network_io.bytes_sent - self._last_network_io.bytes_sent
            recv_bytes = network_io.bytes_recv - self._last_network_io.bytes_recv
            network_sent = sent_bytes / 1024 / 1024
            network_recv = recv_bytes / 1024 / 1024

        if network_io:
            self._last_network_io = network_io

        # GPU usage (if available)
        gpu_usage = self._collect_gpu_usage()

        return ResourceUsage(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_mb=memory.available / 1024 / 1024,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_bytes_sent=network_sent,
            network_bytes_recv=network_recv,
            gpu_usage=gpu_usage
        )

    def _collect_gpu_usage(self) -> dict[str, float] | None:
        """Collect GPU usage statistics."""
        try:
            import torch
            if not torch.cuda.is_available():
                return None

            gpu_usage = {}

            for i in range(torch.cuda.device_count()):
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                gpu_usage[f"gpu_{i}_memory_allocated_gb"] = memory_allocated
                gpu_usage[f"gpu_{i}_memory_reserved_gb"] = memory_reserved
                gpu_usage[f"gpu_{i}_memory_total_gb"] = memory_total
                gpu_usage[f"gpu_{i}_memory_utilization"] = (memory_allocated / memory_total) * 100

                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage[f"gpu_{i}_utilization"] = utilization.gpu
                    gpu_usage[f"gpu_{i}_memory_utilization_nvml"] = utilization.memory
                except ImportError:
                    pass  # nvidia-ml-py not available
                except Exception:
                    pass  # Other NVML errors

            return gpu_usage if gpu_usage else None

        except ImportError:
            return None  # PyTorch not available
        except Exception as e:
            logger.debug(f"GPU monitoring error: {e}")
            return None

    def _record_usage_metrics(self, usage: ResourceUsage):
        """Record usage metrics."""
        self.metrics_collector.record_metric("cpu_percent", usage.cpu_percent)
        self.metrics_collector.record_metric("memory_percent", usage.memory_percent)
        self.metrics_collector.record_metric("memory_available_mb", usage.memory_available_mb)
        self.metrics_collector.record_metric("disk_io_read_mb_per_sec", usage.disk_io_read_mb)
        self.metrics_collector.record_metric("disk_io_write_mb_per_sec", usage.disk_io_write_mb)
        self.metrics_collector.record_metric("network_sent_mb_per_sec", usage.network_bytes_sent)
        self.metrics_collector.record_metric("network_recv_mb_per_sec", usage.network_bytes_recv)

        if usage.gpu_usage:
            for metric_name, value in usage.gpu_usage.items():
                self.metrics_collector.record_metric(metric_name, value)


class PerformanceProfiler:
    """Performance profiler with flame graph generation."""

    def __init__(self):
        self.active_profiles = {}
        self.profile_results = {}
        self._lock = threading.RLock()

    @contextmanager
    def profile_operation(self, operation_name: str, labels: dict[str, str] | None = None):
        """Context manager for profiling operations."""
        profile_id = str(uuid.uuid4())
        labels = labels or {}

        start_time = time.time()

        profile_data = {
            'operation': operation_name,
            'start_time': start_time,
            'labels': labels,
            'call_stack': [],
            'resource_usage_start': self._get_current_resource_usage()
        }

        with self._lock:
            self.active_profiles[profile_id] = profile_data

        try:
            yield profile_id
        finally:
            end_time = time.time()

            with self._lock:
                if profile_id in self.active_profiles:
                    profile_data = self.active_profiles.pop(profile_id)

                    profile_data.update({
                        'end_time': end_time,
                        'duration': end_time - start_time,
                        'resource_usage_end': self._get_current_resource_usage()
                    })

                    self.profile_results[profile_id] = profile_data

    def _get_current_resource_usage(self) -> dict[str, float]:
        """Get current resource usage snapshot."""
        try:
            process = psutil.Process()
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()

            usage = {
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
            }

            # Add GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                        reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
                        usage[f'gpu_{i}_memory_allocated_mb'] = allocated
                        usage[f'gpu_{i}_memory_reserved_mb'] = reserved
            except ImportError:
                pass

            return usage

        except Exception as e:
            logger.debug(f"Resource usage collection error: {e}")
            return {}

    def get_profile_results(self, operation_name: str | None = None) -> list[dict[str, Any]]:
        """Get profile results, optionally filtered by operation name."""
        with self._lock:
            results = list(self.profile_results.values())

            if operation_name:
                results = [r for r in results if r['operation'] == operation_name]

            return results

    def generate_flame_graph_data(self, operation_name: str) -> dict[str, Any]:
        """Generate flame graph data for operation."""
        results = self.get_profile_results(operation_name)

        if not results:
            return {}

        # Aggregate data for flame graph
        flame_data = {
            'name': operation_name,
            'value': sum(r['duration'] for r in results),
            'children': [],
            'samples': len(results),
            'avg_duration': sum(r['duration'] for r in results) / len(results),
            'min_duration': min(r['duration'] for r in results),
            'max_duration': max(r['duration'] for r in results)
        }

        return flame_data


class BottleneckDetector:
    """Detects and analyzes performance bottlenecks."""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0,
            'gpu_memory_high': 90.0,
            'latency_high': 5.0,  # seconds
            'error_rate_high': 0.05,  # 5%
            'queue_depth_high': 100
        }

    def analyze_bottlenecks(self, time_window: float = 300) -> list[BottleneckAnalysis]:
        """Analyze system for performance bottlenecks."""
        analyses = []

        # CPU bottleneck analysis
        cpu_analysis = self._analyze_cpu_bottleneck(time_window)
        if cpu_analysis:
            analyses.append(cpu_analysis)

        # Memory bottleneck analysis
        memory_analysis = self._analyze_memory_bottleneck(time_window)
        if memory_analysis:
            analyses.append(memory_analysis)

        # GPU bottleneck analysis
        gpu_analysis = self._analyze_gpu_bottleneck(time_window)
        if gpu_analysis:
            analyses.append(gpu_analysis)

        # Latency bottleneck analysis
        latency_analysis = self._analyze_latency_bottleneck(time_window)
        if latency_analysis:
            analyses.append(latency_analysis)

        # Queue depth analysis
        queue_analysis = self._analyze_queue_bottleneck(time_window)
        if queue_analysis:
            analyses.append(queue_analysis)

        return analyses

    def _analyze_cpu_bottleneck(self, time_window: float) -> BottleneckAnalysis | None:
        """Analyze CPU usage bottlenecks."""
        stats = self.metrics_collector.get_metric_stats('cpu_percent', time_window)

        if not stats or stats.get('mean', 0) < self.thresholds['cpu_high']:
            return None

        severity = Severity.WARNING
        if stats['mean'] > 95:
            severity = Severity.CRITICAL
        elif stats['mean'] > 90:
            severity = Severity.ERROR

        recommendations = [
            "Consider scaling horizontally by adding more worker processes",
            "Optimize CPU-intensive operations",
            "Use multi-threading for I/O bound operations",
            "Consider using faster CPU or upgrading to more cores"
        ]

        if stats['p99'] > 98:
            recommendations.append("CPU usage spikes detected - investigate periodic high-load operations")

        return BottleneckAnalysis(
            component="CPU",
            severity=severity,
            description=f"High CPU usage detected: {stats['mean']:.1f}% average",
            impact_score=min(100, stats['mean']),
            recommendations=recommendations,
            metrics=stats,
            timestamp=time.time()
        )

    def _analyze_memory_bottleneck(self, time_window: float) -> BottleneckAnalysis | None:
        """Analyze memory usage bottlenecks."""
        stats = self.metrics_collector.get_metric_stats('memory_percent', time_window)

        if not stats or stats.get('mean', 0) < self.thresholds['memory_high']:
            return None

        severity = Severity.WARNING
        if stats['mean'] > 95:
            severity = Severity.CRITICAL
        elif stats['mean'] > 90:
            severity = Severity.ERROR

        recommendations = [
            "Increase system memory or optimize memory usage",
            "Implement more aggressive caching eviction policies",
            "Use memory-mapped files for large datasets",
            "Profile memory usage to identify memory leaks"
        ]

        return BottleneckAnalysis(
            component="Memory",
            severity=severity,
            description=f"High memory usage detected: {stats['mean']:.1f}% average",
            impact_score=min(100, stats['mean']),
            recommendations=recommendations,
            metrics=stats,
            timestamp=time.time()
        )

    def _analyze_gpu_bottleneck(self, time_window: float) -> BottleneckAnalysis | None:
        """Analyze GPU usage bottlenecks."""
        gpu_metrics = []

        # Check all GPU metrics
        for i in range(8):  # Check up to 8 GPUs
            metric_name = f"gpu_{i}_memory_utilization"
            stats = self.metrics_collector.get_metric_stats(metric_name, time_window)
            if stats:
                gpu_metrics.append((i, stats))

        if not gpu_metrics:
            return None

        # Find GPUs with high utilization
        high_utilization_gpus = [
            (gpu_id, stats) for gpu_id, stats in gpu_metrics
            if stats.get('mean', 0) > self.thresholds['gpu_memory_high']
        ]

        if not high_utilization_gpus:
            return None

        avg_utilization = sum(stats['mean'] for _, stats in high_utilization_gpus) / len(high_utilization_gpus)

        severity = Severity.WARNING
        if avg_utilization > 98:
            severity = Severity.CRITICAL
        elif avg_utilization > 95:
            severity = Severity.ERROR

        recommendations = [
            "Consider using multiple GPUs for parallel processing",
            "Implement gradient checkpointing to reduce memory usage",
            "Use mixed precision training to reduce memory footprint",
            "Optimize batch sizes for better GPU utilization"
        ]

        if len(high_utilization_gpus) > 1:
            recommendations.append("Load balance work across available GPUs")

        return BottleneckAnalysis(
            component="GPU",
            severity=severity,
            description=f"High GPU memory usage detected on {len(high_utilization_gpus)} GPUs",
            impact_score=min(100, avg_utilization),
            recommendations=recommendations,
            metrics={'avg_gpu_memory_utilization': avg_utilization},
            timestamp=time.time()
        )

    def _analyze_latency_bottleneck(self, time_window: float) -> BottleneckAnalysis | None:
        """Analyze request latency bottlenecks."""
        stats = self.metrics_collector.get_metric_stats('request_latency', time_window)

        if not stats or stats.get('p95', 0) < self.thresholds['latency_high']:
            return None

        severity = Severity.WARNING
        if stats['p99'] > 10:
            severity = Severity.CRITICAL
        elif stats['p95'] > 8:
            severity = Severity.ERROR

        recommendations = [
            "Optimize compression algorithms for speed",
            "Implement request batching to improve throughput",
            "Use caching to reduce computation time",
            "Consider using faster hardware or GPU acceleration"
        ]

        if stats['p99'] > stats['p95'] * 2:
            recommendations.append("High latency variance detected - investigate tail latencies")

        return BottleneckAnalysis(
            component="Latency",
            severity=severity,
            description=f"High request latency: p95={stats['p95']:.2f}s, p99={stats['p99']:.2f}s",
            impact_score=min(100, stats['p95'] * 20),  # Scale to 0-100
            recommendations=recommendations,
            metrics=stats,
            timestamp=time.time()
        )

    def _analyze_queue_bottleneck(self, time_window: float) -> BottleneckAnalysis | None:
        """Analyze request queue bottlenecks."""
        stats = self.metrics_collector.get_metric_stats('queue_depth', time_window)

        if not stats or stats.get('mean', 0) < self.thresholds['queue_depth_high']:
            return None

        severity = Severity.WARNING
        if stats['mean'] > 500:
            severity = Severity.CRITICAL
        elif stats['mean'] > 200:
            severity = Severity.ERROR

        recommendations = [
            "Increase worker count to process requests faster",
            "Implement request prioritization",
            "Add horizontal scaling to handle load",
            "Consider implementing backpressure mechanisms"
        ]

        return BottleneckAnalysis(
            component="Queue",
            severity=severity,
            description=f"High queue depth: {stats['mean']:.0f} average requests queued",
            impact_score=min(100, stats['mean'] / 10),
            recommendations=recommendations,
            metrics=stats,
            timestamp=time.time()
        )


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(
        self,
        collection_interval: float = 5.0,
        analysis_interval: float = 60.0,
        alert_callback: Callable[[PerformanceAlert], None] | None = None
    ):
        self.collection_interval = collection_interval
        self.analysis_interval = analysis_interval
        self.alert_callback = alert_callback

        # Components
        self.metrics_collector = MetricsCollector()
        self.resource_monitor = ResourceMonitor(collection_interval)
        self.profiler = PerformanceProfiler()
        self.bottleneck_detector = BottleneckDetector(self.metrics_collector)

        # State
        self.alerts = {}
        self._running = False
        self._analysis_task = None

    async def start(self):
        """Start performance monitoring."""
        if self._running:
            return

        self._running = True

        # Start resource monitoring
        await self.resource_monitor.start()

        # Start analysis loop
        self._analysis_task = asyncio.create_task(self._analysis_loop())

        logger.info("Performance monitoring started")

    async def stop(self):
        """Stop performance monitoring."""
        self._running = False

        # Stop resource monitoring
        await self.resource_monitor.stop()

        # Stop analysis task
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _analysis_loop(self):
        """Main analysis loop."""
        while self._running:
            try:
                await self._perform_analysis()
                await asyncio.sleep(self.analysis_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance analysis error: {e}")

    async def _perform_analysis(self):
        """Perform comprehensive performance analysis."""
        # Detect bottlenecks
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks()

        # Generate alerts for critical bottlenecks
        for bottleneck in bottlenecks:
            if bottleneck.severity in [Severity.ERROR, Severity.CRITICAL]:
                alert = PerformanceAlert(
                    id=str(uuid.uuid4()),
                    name=f"{bottleneck.component}_bottleneck",
                    severity=bottleneck.severity,
                    message=bottleneck.description,
                    timestamp=time.time(),
                    labels={'component': bottleneck.component}
                )

                self.alerts[alert.id] = alert

                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")

    def record_compression_metrics(self, result: CompressionResult, operation_type: str = "compress"):
        """Record metrics from compression operation."""
        self.metrics_collector.record_metric(
            f"{operation_type}_latency",
            result.processing_time,
            MetricType.HISTOGRAM
        )

        self.metrics_collector.record_metric(
            f"{operation_type}_compression_ratio",
            result.compression_ratio,
            MetricType.HISTOGRAM
        )

        self.metrics_collector.record_metric(
            f"{operation_type}_original_length",
            result.original_length,
            MetricType.HISTOGRAM
        )

        self.metrics_collector.record_metric(
            f"{operation_type}_requests_total",
            1,
            MetricType.COUNTER
        )

    def get_comprehensive_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        # Get recent bottlenecks
        bottlenecks = self.bottleneck_detector.analyze_bottlenecks()

        # Get active alerts
        active_alerts = [
            alert for alert in self.alerts.values()
            if not alert.resolved
        ]

        # Get key metrics
        key_metrics = {}
        for metric_name in ['cpu_percent', 'memory_percent', 'request_latency', 'queue_depth']:
            stats = self.metrics_collector.get_metric_stats(metric_name, time_window=3600)
            if stats:
                key_metrics[metric_name] = stats

        # Get resource usage trends
        cpu_history = self.resource_monitor.metrics_collector.get_metric_history('cpu_percent', 3600)
        memory_history = self.resource_monitor.metrics_collector.get_metric_history('memory_percent', 3600)

        return {
            'timestamp': time.time(),
            'summary': {
                'active_alerts': len(active_alerts),
                'critical_alerts': sum(1 for a in active_alerts if a.severity == Severity.CRITICAL),
                'bottlenecks_detected': len(bottlenecks),
                'critical_bottlenecks': sum(1 for b in bottlenecks if b.severity == Severity.CRITICAL)
            },
            'bottlenecks': [
                {
                    'component': b.component,
                    'severity': b.severity.value,
                    'description': b.description,
                    'impact_score': b.impact_score,
                    'recommendations': b.recommendations
                }
                for b in bottlenecks
            ],
            'alerts': [
                {
                    'id': a.id,
                    'name': a.name,
                    'severity': a.severity.value,
                    'message': a.message,
                    'timestamp': a.timestamp
                }
                for a in active_alerts
            ],
            'metrics': key_metrics,
            'resource_trends': {
                'cpu_samples': len(cpu_history),
                'memory_samples': len(memory_history),
                'time_range_hours': 1.0
            }
        }


# Global performance monitor instance
_performance_monitor: PerformanceMonitor | None = None


def get_performance_monitor(
    collection_interval: float = 5.0,
    analysis_interval: float = 60.0,
    alert_callback: Callable[[PerformanceAlert], None] | None = None
) -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _performance_monitor

    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(
            collection_interval=collection_interval,
            analysis_interval=analysis_interval,
            alert_callback=alert_callback
        )

    return _performance_monitor


@contextmanager
def performance_profile(operation_name: str, labels: dict[str, str] | None = None):
    """Context manager for performance profiling."""
    monitor = get_performance_monitor()
    with monitor.profiler.profile_operation(operation_name, labels) as profile_id:
        yield profile_id
