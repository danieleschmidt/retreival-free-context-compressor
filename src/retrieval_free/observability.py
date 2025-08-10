"""Observability and monitoring configuration for retrieval-free context compressor."""

import json
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import psutil


class MetricsCollector:
    """Collects and reports performance metrics."""

    def __init__(self):
        self.metrics: dict[str, Any] = {}
        self.counters: dict[str, int] = {}
        self.timers: dict[str, float] = {}

    def increment(self, metric: str, value: int = 1) -> None:
        """Increment a counter metric."""
        self.counters[metric] = self.counters.get(metric, 0) + value

    def set_gauge(self, metric: str, value: float) -> None:
        """Set a gauge metric."""
        self.metrics[metric] = value

    @contextmanager
    def timer(self, metric: str):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timers[metric] = elapsed

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": self.counters,
            "gauges": self.metrics,
            "timers": self.timers,
            "timestamp": time.time(),
        }


class PerformanceMonitor:
    """Monitors system performance during compression operations."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self._get_memory_usage()

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def get_system_metrics(self) -> dict[str, float]:
        """Get current system performance metrics."""
        return {
            "memory_mb": self._get_memory_usage(),
            "memory_delta_mb": self._get_memory_usage() - self.baseline_memory,
            "cpu_percent": self._get_cpu_usage(),
            "num_threads": self.process.num_threads(),
        }


class StructuredLogger:
    """Structured logging for compression operations."""

    def __init__(self, name: str = "retrieval_free", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Create structured formatter
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def log_compression_start(self, input_size: int, model: str) -> None:
        """Log compression operation start."""
        self.logger.info(
            json.dumps(
                {
                    "event": "compression_start",
                    "input_size": input_size,
                    "model": model,
                }
            )
        )

    def log_compression_complete(
        self,
        input_size: int,
        output_size: int,
        compression_ratio: float,
        duration: float,
    ) -> None:
        """Log compression operation completion."""
        self.logger.info(
            json.dumps(
                {
                    "event": "compression_complete",
                    "input_size": input_size,
                    "output_size": output_size,
                    "compression_ratio": compression_ratio,
                    "duration_seconds": duration,
                    "throughput_tokens_per_second": (
                        input_size / duration if duration > 0 else 0
                    ),
                }
            )
        )

    def log_error(self, error: Exception, context: dict[str, Any]) -> None:
        """Log error with context."""
        self.logger.error(
            json.dumps(
                {
                    "event": "error",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": context,
                }
            )
        )


class HealthChecker:
    """Health check functionality for the compression service."""

    def __init__(self):
        self.checks: dict[str, Callable[[], bool]] = {}
        self.last_check_results: dict[str, bool] = {}

    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function."""
        self.checks[name] = check_func

    def run_checks(self) -> dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "success": result,
                }
                self.last_check_results[name] = result
                if not result:
                    all_healthy = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "success": False,
                    "error": str(e),
                }
                all_healthy = False

        return {
            "overall_status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": time.time(),
        }


# Global instances
metrics_collector = MetricsCollector()
performance_monitor = PerformanceMonitor()
logger = StructuredLogger()
health_checker = HealthChecker()


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"

        # Start monitoring
        metrics_collector.increment(f"{func_name}.calls")
        start_memory = performance_monitor._get_memory_usage()

        with metrics_collector.timer(f"{func_name}.duration"):
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment(f"{func_name}.success")
                return result
            except Exception as e:
                metrics_collector.increment(f"{func_name}.errors")
                logger.log_error(e, {"function": func_name, "args": str(args)[:100]})
                raise
            finally:
                # Record memory usage
                end_memory = performance_monitor._get_memory_usage()
                memory_delta = end_memory - start_memory
                metrics_collector.set_gauge(
                    f"{func_name}.memory_delta_mb", memory_delta
                )

    return wrapper


def log_compression_operation(func: Callable) -> Callable:
    """Decorator to log compression operations."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract input size from args (assumes first arg is text/document)
        input_size = len(str(args[0])) if args else 0
        model = getattr(args[0] if args else None, "model_name", "unknown")

        logger.log_compression_start(input_size, model)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)

            # Extract compression metrics from result
            output_size = len(result) if isinstance(result, list) else 1
            compression_ratio = input_size / output_size if output_size > 0 else 1.0
            duration = time.time() - start_time

            logger.log_compression_complete(
                input_size, output_size, compression_ratio, duration
            )

            return result
        except Exception as e:
            logger.log_error(
                e,
                {
                    "function": func.__name__,
                    "input_size": input_size,
                    "model": model,
                },
            )
            raise

    return wrapper


# Default health checks
def _check_memory_usage() -> bool:
    """Check if memory usage is within acceptable limits."""
    current_memory = performance_monitor._get_memory_usage()
    return current_memory < 8192  # Less than 8GB


def _check_disk_space() -> bool:
    """Check if disk space is sufficient."""
    disk_usage = psutil.disk_usage("/")
    free_percentage = (disk_usage.free / disk_usage.total) * 100
    return free_percentage > 10  # More than 10% free


# Register default health checks
health_checker.register_check("memory_usage", _check_memory_usage)
health_checker.register_check("disk_space", _check_disk_space)


def get_observability_status() -> dict[str, Any]:
    """Get comprehensive observability status."""
    return {
        "metrics": metrics_collector.get_all_metrics(),
        "system": performance_monitor.get_system_metrics(),
        "health": health_checker.run_checks(),
    }
