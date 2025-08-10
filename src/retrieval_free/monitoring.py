"""Enhanced monitoring, distributed tracing, and alerting system."""

import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import psutil
import torch


logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status."""

    service: str
    healthy: bool
    message: str
    response_time_ms: float
    timestamp: float
    details: dict[str, Any] = None


@dataclass
class Span:
    """Distributed tracing span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout

    def finish(self) -> None:
        """Finish the span."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000

    def log(self, message: str, level: str = "info", **fields) -> None:
        """Add log entry to span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **fields,
        }
        self.logs.append(log_entry)

    def set_tag(self, key: str, value: Any) -> None:
        """Set tag on span."""
        self.tags[key] = value

    def set_error(self, error: Exception) -> None:
        """Mark span as error."""
        self.status = "error"
        self.set_tag("error", True)
        self.set_tag("error.type", type(error).__name__)
        self.set_tag("error.message", str(error))


@dataclass
class Trace:
    """Complete trace with multiple spans."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    root_span: Span | None = None

    def add_span(self, span: Span) -> None:
        """Add span to trace."""
        self.spans.append(span)
        if self.root_span is None or span.parent_span_id is None:
            self.root_span = span

    def finish(self) -> None:
        """Finish the trace."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    name: str
    metric_name: str
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    window_minutes: int = 5
    severity: str = "warning"  # info, warning, critical
    description: str = ""
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def evaluate(self, value: float) -> bool:
        """Evaluate if alert should fire."""
        if not self.enabled:
            return False

        if self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return value == self.threshold
        elif self.operator == "!=":
            return value != self.threshold

        return False


@dataclass
class AlertInstance:
    """Active alert instance."""

    alert_id: str
    alert_name: str
    metric_name: str
    value: float
    threshold: float
    severity: str
    fired_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None
    is_resolved: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    timestamp: float
    compression_ratio: float
    processing_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: float
    input_tokens: int
    output_tokens: int
    throughput_tps: float  # tokens per second
    error_count: int = 0
    success_count: int = 0
    trace_id: str | None = None
    span_id: str | None = None
    user_id: str | None = None
    model_name: str | None = None
    operation: str | None = None


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_history: int = 1000):
        """Initialize metrics collector.

        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.counters: dict[str, int] = defaultdict(int)
        self.gauges: dict[str, float] = defaultdict(float)
        self.timers: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_compression(
        self,
        input_tokens: int,
        output_tokens: int,
        processing_time_ms: float,
        memory_usage_mb: float = 0,
        model_name: str = "unknown",
    ) -> None:
        """Record compression operation metrics.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in MB
            model_name: Name of model used
        """
        with self._lock:
            # Calculate derived metrics
            compression_ratio = input_tokens / output_tokens if output_tokens > 0 else 0
            throughput_tps = (
                input_tokens / (processing_time_ms / 1000)
                if processing_time_ms > 0
                else 0
            )

            # Get GPU memory if available
            gpu_memory_mb = 0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

            # Create metrics record
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                compression_ratio=compression_ratio,
                processing_time_ms=processing_time_ms,
                memory_usage_mb=memory_usage_mb,
                gpu_memory_mb=gpu_memory_mb,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                throughput_tps=throughput_tps,
            )

            self.metrics_history.append(metrics)

            # Update counters
            self.counters["total_compressions"] += 1
            self.counters[f"compressions_{model_name}"] += 1
            self.counters["total_input_tokens"] += input_tokens
            self.counters["total_output_tokens"] += output_tokens

            # Update gauges
            self.gauges["last_compression_ratio"] = compression_ratio
            self.gauges["last_processing_time_ms"] = processing_time_ms
            self.gauges["last_throughput_tps"] = throughput_tps

            # Update timers
            self.timers["processing_time_ms"].append(processing_time_ms)
            self.timers["compression_ratio"].append(compression_ratio)

            # Limit timer history
            max_timer_history = 100
            for timer_name in self.timers:
                if len(self.timers[timer_name]) > max_timer_history:
                    self.timers[timer_name] = self.timers[timer_name][
                        -max_timer_history:
                    ]

    def get_summary_stats(self, window_minutes: int = 60) -> dict[str, Any]:
        """Get summary statistics for recent operations.

        Args:
            window_minutes: Time window for statistics in minutes

        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            cutoff_time = time.time() - (window_minutes * 60)
            recent_metrics = [
                m for m in self.metrics_history if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {"message": "No recent metrics available"}

            # Calculate statistics
            compression_ratios = [m.compression_ratio for m in recent_metrics]
            processing_times = [m.processing_time_ms for m in recent_metrics]
            throughputs = [m.throughput_tps for m in recent_metrics]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]

            return {
                "window_minutes": window_minutes,
                "total_operations": len(recent_metrics),
                "compression_ratio": {
                    "mean": statistics.mean(compression_ratios),
                    "median": statistics.median(compression_ratios),
                    "min": min(compression_ratios),
                    "max": max(compression_ratios),
                    "stddev": (
                        statistics.stdev(compression_ratios)
                        if len(compression_ratios) > 1
                        else 0
                    ),
                },
                "processing_time_ms": {
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "p95": self._percentile(processing_times, 95),
                    "p99": self._percentile(processing_times, 99),
                },
                "throughput_tps": {
                    "mean": statistics.mean(throughputs),
                    "median": statistics.median(throughputs),
                    "min": min(throughputs),
                    "max": max(throughputs),
                },
                "memory_usage_mb": {
                    "mean": statistics.mean(memory_usage),
                    "max": max(memory_usage),
                },
            }

    def _percentile(self, data: list[float], percentile: int) -> float:
        """Calculate percentile of data.

        Args:
            data: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not data:
            return 0.0

        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))

    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.

        Args:
            format: Export format ('json', 'csv', 'prometheus')

        Returns:
            Formatted metrics string
        """
        with self._lock:
            if format.lower() == "json":
                return json.dumps(
                    {
                        "counters": dict(self.counters),
                        "gauges": dict(self.gauges),
                        "recent_metrics": [
                            asdict(m) for m in list(self.metrics_history)[-10:]
                        ],
                    },
                    indent=2,
                )

            elif format.lower() == "prometheus":
                lines = []

                # Counters
                for name, value in self.counters.items():
                    lines.append(f"retrieval_free_{name}_total {value}")

                # Gauges
                for name, value in self.gauges.items():
                    lines.append(f"retrieval_free_{name} {value}")

                return "\n".join(lines)

            else:
                return f"Unsupported format: {format}"


class HealthChecker:
    """Health check system for monitoring service status."""

    def __init__(self):
        """Initialize health checker."""
        self.checks: dict[str, Callable] = {}
        self.last_results: dict[str, HealthStatus] = {}
        self._lock = threading.Lock()

    def register_check(self, name: str, check_func: Callable[[], HealthStatus]) -> None:
        """Register a health check function.

        Args:
            name: Name of the health check
            check_func: Function that returns HealthStatus
        """
        with self._lock:
            self.checks[name] = check_func
            logger.info(f"Registered health check: {name}")

    def run_check(self, name: str) -> HealthStatus:
        """Run a specific health check.

        Args:
            name: Name of health check to run

        Returns:
            HealthStatus result
        """
        if name not in self.checks:
            return HealthStatus(
                service=name,
                healthy=False,
                message=f"Health check '{name}' not found",
                response_time_ms=0,
                timestamp=time.time(),
            )

        start_time = time.time()

        try:
            result = self.checks[name]()
            result.response_time_ms = (time.time() - start_time) * 1000
            result.timestamp = time.time()

            with self._lock:
                self.last_results[name] = result

            return result

        except Exception as e:
            result = HealthStatus(
                service=name,
                healthy=False,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                timestamp=time.time(),
            )

            with self._lock:
                self.last_results[name] = result

            return result

    def run_all_checks(self) -> dict[str, HealthStatus]:
        """Run all registered health checks.

        Returns:
            Dictionary mapping check names to results
        """
        results = {}

        for name in self.checks:
            results[name] = self.run_check(name)

        return results

    def get_overall_health(self) -> dict[str, Any]:
        """Get overall system health status.

        Returns:
            Overall health summary
        """
        results = self.run_all_checks()

        total_checks = len(results)
        passing_checks = sum(1 for r in results.values() if r.healthy)

        overall_healthy = passing_checks == total_checks
        health_percentage = (
            (passing_checks / total_checks * 100) if total_checks > 0 else 0
        )

        return {
            "healthy": overall_healthy,
            "health_percentage": health_percentage,
            "total_checks": total_checks,
            "passing_checks": passing_checks,
            "failing_checks": total_checks - passing_checks,
            "checks": {name: asdict(status) for name, status in results.items()},
            "timestamp": time.time(),
        }


def create_default_health_checks(compressor) -> HealthChecker:
    """Create default health checks for the compressor.

    Args:
        compressor: Compressor instance to monitor

    Returns:
        Configured HealthChecker
    """
    health_checker = HealthChecker()

    # Model availability check
    def check_model_availability() -> HealthStatus:
        """Check if compression model is available."""
        try:
            if (
                hasattr(compressor, "_encoder_model")
                and compressor._encoder_model is not None
            ):
                # Try a simple inference
                test_result = compressor.compress("Test input for health check")

                return HealthStatus(
                    service="model_availability",
                    healthy=True,
                    message=f"Model available, compressed to {len(test_result.mega_tokens)} tokens",
                    response_time_ms=0,  # Will be set by health checker
                    timestamp=0,
                )
            else:
                return HealthStatus(
                    service="model_availability",
                    healthy=False,
                    message="Model not loaded",
                    response_time_ms=0,
                    timestamp=0,
                )
        except Exception as e:
            return HealthStatus(
                service="model_availability",
                healthy=False,
                message=f"Model check failed: {str(e)}",
                response_time_ms=0,
                timestamp=0,
            )

    # Memory check
    def check_memory_usage() -> HealthStatus:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                healthy = False
                message = f"High memory usage: {memory_percent:.1f}%"
            elif memory_percent > 80:
                healthy = True
                message = f"Moderate memory usage: {memory_percent:.1f}%"
            else:
                healthy = True
                message = f"Normal memory usage: {memory_percent:.1f}%"

            return HealthStatus(
                service="memory_usage",
                healthy=healthy,
                message=message,
                response_time_ms=0,
                timestamp=0,
                details={
                    "memory_percent": memory_percent,
                    "available_gb": memory.available / 1024 / 1024 / 1024,
                    "total_gb": memory.total / 1024 / 1024 / 1024,
                },
            )
        except Exception as e:
            return HealthStatus(
                service="memory_usage",
                healthy=False,
                message=f"Memory check failed: {str(e)}",
                response_time_ms=0,
                timestamp=0,
            )

    # GPU check (if available)
    def check_gpu_availability() -> HealthStatus:
        """Check GPU availability and memory."""
        try:
            if not torch.cuda.is_available():
                return HealthStatus(
                    service="gpu_availability",
                    healthy=True,
                    message="GPU not available (CPU mode)",
                    response_time_ms=0,
                    timestamp=0,
                )

            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_memory_used = (
                torch.cuda.memory_allocated(current_device) / 1024 / 1024 / 1024
            )  # GB
            gpu_memory_total = (
                torch.cuda.get_device_properties(current_device).total_memory
                / 1024
                / 1024
                / 1024
            )  # GB
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100

            if gpu_memory_percent > 90:
                healthy = False
                message = f"High GPU memory usage: {gpu_memory_percent:.1f}%"
            else:
                healthy = True
                message = f"GPU available, memory usage: {gpu_memory_percent:.1f}%"

            return HealthStatus(
                service="gpu_availability",
                healthy=healthy,
                message=message,
                response_time_ms=0,
                timestamp=0,
                details={
                    "gpu_count": gpu_count,
                    "current_device": current_device,
                    "memory_used_gb": gpu_memory_used,
                    "memory_total_gb": gpu_memory_total,
                    "memory_percent": gpu_memory_percent,
                },
            )
        except Exception as e:
            return HealthStatus(
                service="gpu_availability",
                healthy=False,
                message=f"GPU check failed: {str(e)}",
                response_time_ms=0,
                timestamp=0,
            )

    # Register all checks
    health_checker.register_check("model_availability", check_model_availability)
    health_checker.register_check("memory_usage", check_memory_usage)
    health_checker.register_check("gpu_availability", check_gpu_availability)

    return health_checker


class AlertManager:
    """Alert manager for monitoring thresholds."""

    def __init__(self):
        """Initialize alert manager."""
        self.thresholds: dict[str, dict[str, float]] = {
            "processing_time_ms": {"warning": 5000, "critical": 15000},
            "memory_usage_percent": {"warning": 80, "critical": 90},
            "error_rate_percent": {"warning": 5, "critical": 10},
            "compression_ratio": {"warning": 2, "critical": 1},  # Too low compression
        }

        self.alert_handlers: list[Callable] = []
        self.active_alerts: dict[str, dict[str, Any]] = {}

    def add_alert_handler(
        self, handler: Callable[[str, str, dict[str, Any]], None]
    ) -> None:
        """Add alert handler function.

        Args:
            handler: Function that handles alerts (metric, level, details)
        """
        self.alert_handlers.append(handler)

    def check_thresholds(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        """Check metrics against thresholds and trigger alerts.

        Args:
            metrics: Dictionary of metric values

        Returns:
            List of triggered alerts
        """
        alerts = []

        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue

            thresholds = self.thresholds[metric_name]
            alert_level = None

            if value >= thresholds.get("critical", float("inf")):
                alert_level = "critical"
            elif value >= thresholds.get("warning", float("inf")):
                alert_level = "warning"

            if alert_level:
                alert_key = f"{metric_name}_{alert_level}"

                # Check if this alert is already active
                if alert_key not in self.active_alerts:
                    alert = {
                        "metric": metric_name,
                        "level": alert_level,
                        "value": value,
                        "threshold": thresholds[alert_level],
                        "timestamp": time.time(),
                        "message": f"{metric_name} is {value} (threshold: {thresholds[alert_level]})",
                    }

                    alerts.append(alert)
                    self.active_alerts[alert_key] = alert

                    # Trigger handlers
                    for handler in self.alert_handlers:
                        try:
                            handler(metric_name, alert_level, alert)
                        except Exception as e:
                            logger.error(f"Alert handler failed: {e}")
            else:
                # Clear any existing alerts for this metric
                keys_to_remove = [
                    k
                    for k in self.active_alerts.keys()
                    if k.startswith(f"{metric_name}_")
                ]
                for key in keys_to_remove:
                    del self.active_alerts[key]

        return alerts


def log_alert_handler(metric: str, level: str, details: dict[str, Any]) -> None:
    """Default alert handler that logs alerts.

    Args:
        metric: Metric name that triggered alert
        level: Alert level (warning, critical)
        details: Alert details
    """
    logger_level = logging.WARNING if level == "warning" else logging.ERROR
    logger.log(logger_level, f"ALERT [{level.upper()}] {metric}: {details['message']}")
