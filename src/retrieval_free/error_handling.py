"""Advanced error handling, resilience patterns, and recovery mechanisms."""

import gc
import logging
import signal
import threading
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any

import psutil
import torch

from .exceptions import CompressionError, ResourceError


logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures to open circuit
    recovery_timeout: int = 60  # Seconds before attempting recovery
    expected_exception: type = Exception
    success_threshold: int = 3  # Successes needed in half-open to close


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    max_retries: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to delays
    retryable_exceptions: tuple[type, ...] = (
        ConnectionError,
        TimeoutError,
        ResourceError,
    )


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascade failures."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker
        """
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self._lock = threading.RLock()

        logger.info(f"Circuit breaker '{name}' initialized")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CompressionError: If circuit is open or function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._move_to_half_open()
                else:
                    raise CompressionError(
                        f"Circuit breaker '{self.name}' is open",
                        details={
                            "circuit_breaker": self.name,
                            "state": self.state.value,
                        },
                    )

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result

            except self.config.expected_exception as e:
                self._on_failure()
                raise CompressionError(
                    f"Circuit breaker '{self.name}' caught exception: {e}",
                    details={
                        "circuit_breaker": self.name,
                        "original_exception": str(e),
                    },
                )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False

        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout

    def _move_to_half_open(self) -> None:
        """Move circuit to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' moved to half-open")

    def _on_success(self) -> None:
        """Handle successful function execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._move_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    def _move_to_closed(self) -> None:
        """Move circuit to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker '{self.name}' moved to closed")

    def _on_failure(self) -> None:
        """Handle failed function execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if (
            self.state == CircuitBreakerState.CLOSED
            and self.failure_count >= self.config.failure_threshold
        ):
            self._move_to_open()
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._move_to_open()

    def _move_to_open(self) -> None:
        """Move circuit to open state."""
        self.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": (
                    self.last_failure_time.isoformat()
                    if self.last_failure_time
                    else None
                ),
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                },
            }


class RetryMechanism:
    """Retry mechanism with exponential backoff and jitter."""

    def __init__(self, name: str, config: RetryConfig):
        """Initialize retry mechanism.

        Args:
            name: Name of the retry mechanism
            config: Retry configuration
        """
        self.name = name
        self.config = config
        self.attempt_count = 0

        logger.info(f"Retry mechanism '{name}' initialized")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry mechanism to a function."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)

        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with retry mechanism.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(
                        f"Retry mechanism '{self.name}' succeeded on attempt {attempt + 1}"
                    )
                return result

            except Exception as e:
                last_exception = e

                if not isinstance(e, self.config.retryable_exceptions):
                    logger.warning(f"Non-retryable exception in '{self.name}': {e}")
                    raise

                if attempt == self.config.max_retries:
                    logger.error(f"Retry mechanism '{self.name}' exhausted all retries")
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Retry mechanism '{self.name}' attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)

        # This should never be reached, but included for completeness
        if last_exception:
            raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base**attempt)
        delay = min(delay, self.config.max_delay)

        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random

            delay = delay * (0.5 + random.random() * 0.5)

        return delay


class TimeoutHandler:
    """Timeout handling for operations."""

    def __init__(self, timeout: float, name: str = "operation"):
        """Initialize timeout handler.

        Args:
            timeout: Timeout in seconds
            name: Name of the operation
        """
        self.timeout = timeout
        self.name = name
        self._start_time: float | None = None

    def __enter__(self):
        """Enter timeout context."""
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit timeout context."""
        if self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed > self.timeout:
                logger.warning(
                    f"Operation '{self.name}' exceeded timeout: {elapsed:.2f}s > {self.timeout}s"
                )

    def check_timeout(self) -> None:
        """Check if timeout has been exceeded.

        Raises:
            TimeoutError: If timeout exceeded
        """
        if self._start_time is None:
            return

        elapsed = time.time() - self._start_time
        if elapsed > self.timeout:
            raise TimeoutError(
                f"Operation '{self.name}' timed out after {elapsed:.2f}s"
            )

    def remaining_time(self) -> float:
        """Get remaining time before timeout.

        Returns:
            Remaining seconds (can be negative if exceeded)
        """
        if self._start_time is None:
            return self.timeout

        elapsed = time.time() - self._start_time
        return self.timeout - elapsed


def timeout_after(seconds: float, name: str = "operation"):
    """Decorator to add timeout to a function.

    Args:
        seconds: Timeout in seconds
        name: Name of the operation
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with TimeoutHandler(seconds, name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ResourceManager:
    """Manages system resources and cleanup."""

    def __init__(self):
        """Initialize resource manager."""
        self._tracked_resources: list[weakref.ref] = []
        self._memory_threshold_mb = 8192  # 8GB
        self._gpu_memory_threshold_mb = 6144  # 6GB
        self._cleanup_callbacks: list[Callable] = []
        self._lock = threading.RLock()

        # Register cleanup on process exit
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def track_resource(self, resource: Any) -> None:
        """Track a resource for cleanup.

        Args:
            resource: Resource to track
        """
        with self._lock:
            # Clean up dead references first
            self._cleanup_dead_references()

            # Add new resource
            weak_ref = weakref.ref(resource, self._resource_finalizer)
            self._tracked_resources.append(weak_ref)

            logger.debug(f"Tracking resource: {type(resource).__name__}")

    def register_cleanup_callback(self, callback: Callable) -> None:
        """Register a cleanup callback.

        Args:
            callback: Function to call during cleanup
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def check_memory_usage(self) -> dict[str, Any]:
        """Check current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # Check GPU memory if available
            gpu_memory_mb = 0
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024

            usage_info = {
                "memory_mb": memory_mb,
                "memory_threshold_mb": self._memory_threshold_mb,
                "memory_usage_percent": (memory_mb / self._memory_threshold_mb) * 100,
                "gpu_memory_mb": gpu_memory_mb,
                "gpu_memory_threshold_mb": self._gpu_memory_threshold_mb,
                "gpu_memory_usage_percent": (
                    (gpu_memory_mb / self._gpu_memory_threshold_mb) * 100
                    if self._gpu_memory_threshold_mb > 0
                    else 0
                ),
                "tracked_resources": len(self._tracked_resources),
            }

            # Check if cleanup is needed
            if (
                memory_mb > self._memory_threshold_mb
                or gpu_memory_mb > self._gpu_memory_threshold_mb
            ):
                logger.warning(
                    f"High memory usage detected: RAM {memory_mb:.1f}MB, GPU {gpu_memory_mb:.1f}MB"
                )
                usage_info["cleanup_recommended"] = True
            else:
                usage_info["cleanup_recommended"] = False

            return usage_info

        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"error": str(e)}

    def cleanup_resources(self, force: bool = False) -> dict[str, Any]:
        """Clean up tracked resources.

        Args:
            force: Force cleanup even if memory usage is low

        Returns:
            Cleanup summary
        """
        with self._lock:
            cleanup_summary = {
                "resources_before": len(self._tracked_resources),
                "memory_before_mb": 0,
                "memory_after_mb": 0,
                "gpu_memory_before_mb": 0,
                "gpu_memory_after_mb": 0,
                "cleanup_forced": force,
            }

            try:
                # Get initial memory usage
                initial_usage = self.check_memory_usage()
                cleanup_summary["memory_before_mb"] = initial_usage.get("memory_mb", 0)
                cleanup_summary["gpu_memory_before_mb"] = initial_usage.get(
                    "gpu_memory_mb", 0
                )

                # Check if cleanup is needed
                if not force and not initial_usage.get("cleanup_recommended", False):
                    logger.debug("Cleanup not needed - memory usage within limits")
                    return cleanup_summary

                # Run cleanup callbacks
                for callback in self._cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Cleanup callback failed: {e}")

                # Clean up dead references
                self._cleanup_dead_references()

                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gc.collect()

                # Get final memory usage
                final_usage = self.check_memory_usage()
                cleanup_summary["memory_after_mb"] = final_usage.get("memory_mb", 0)
                cleanup_summary["gpu_memory_after_mb"] = final_usage.get(
                    "gpu_memory_mb", 0
                )
                cleanup_summary["resources_after"] = len(self._tracked_resources)

                # Calculate savings
                memory_saved = (
                    cleanup_summary["memory_before_mb"]
                    - cleanup_summary["memory_after_mb"]
                )
                gpu_memory_saved = (
                    cleanup_summary["gpu_memory_before_mb"]
                    - cleanup_summary["gpu_memory_after_mb"]
                )

                cleanup_summary["memory_saved_mb"] = memory_saved
                cleanup_summary["gpu_memory_saved_mb"] = gpu_memory_saved

                logger.info(
                    f"Resource cleanup completed: "
                    f"RAM saved {memory_saved:.1f}MB, GPU saved {gpu_memory_saved:.1f}MB"
                )

            except Exception as e:
                logger.error(f"Error during resource cleanup: {e}")
                cleanup_summary["error"] = str(e)

            return cleanup_summary

    def _cleanup_dead_references(self) -> None:
        """Clean up dead weak references."""
        self._tracked_resources = [
            ref for ref in self._tracked_resources if ref() is not None
        ]

    def _resource_finalizer(self, weak_ref: weakref.ref) -> None:
        """Finalizer for tracked resources."""
        logger.debug("Resource finalized")

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle process signals for cleanup."""
        logger.info(f"Received signal {signum}, cleaning up resources")
        self.cleanup_resources(force=True)


class GracefulDegradation:
    """Provides graceful degradation and fallback mechanisms."""

    def __init__(self, name: str):
        """Initialize graceful degradation handler.

        Args:
            name: Name of the degradation handler
        """
        self.name = name
        self.fallback_stack: list[Callable] = []
        self.primary_function: Callable | None = None

    def primary(self, func: Callable) -> Callable:
        """Register primary function.

        Args:
            func: Primary function to execute

        Returns:
            The original function
        """
        self.primary_function = func
        return func

    def fallback(self, func: Callable) -> Callable:
        """Register fallback function.

        Args:
            func: Fallback function to execute if primary fails

        Returns:
            The original function
        """
        self.fallback_stack.append(func)
        return func

    def execute(self, *args, **kwargs) -> Any:
        """Execute with graceful degradation.

        Args:
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Result from primary or fallback function

        Raises:
            Exception: If all functions fail
        """
        functions_to_try = []
        if self.primary_function:
            functions_to_try.append(("primary", self.primary_function))

        for i, fallback_func in enumerate(self.fallback_stack):
            functions_to_try.append((f"fallback_{i}", fallback_func))

        last_exception = None

        for func_type, func in functions_to_try:
            try:
                result = func(*args, **kwargs)
                if func_type != "primary":
                    logger.warning(
                        f"Graceful degradation '{self.name}' used {func_type}"
                    )
                return result

            except Exception as e:
                logger.warning(f"Function {func_type} failed in '{self.name}': {e}")
                last_exception = e
                continue

        # All functions failed
        logger.error(f"All functions failed in graceful degradation '{self.name}'")
        if last_exception:
            raise last_exception
        else:
            raise CompressionError(f"All fallback mechanisms failed for '{self.name}'")


# Global instances
_resource_manager: ResourceManager | None = None
_circuit_breakers: dict[str, CircuitBreaker] = {}
_retry_mechanisms: dict[str, RetryMechanism] = {}
_degradation_handlers: dict[str, GracefulDegradation] = {}


def get_resource_manager() -> ResourceManager:
    """Get global resource manager.

    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager


def get_circuit_breaker(
    name: str, config: CircuitBreakerConfig | None = None
) -> CircuitBreaker:
    """Get or create circuit breaker.

    Args:
        name: Circuit breaker name
        config: Configuration (uses default if not provided)

    Returns:
        CircuitBreaker instance
    """
    global _circuit_breakers

    if name not in _circuit_breakers:
        if config is None:
            config = CircuitBreakerConfig()
        _circuit_breakers[name] = CircuitBreaker(name, config)

    return _circuit_breakers[name]


def get_retry_mechanism(name: str, config: RetryConfig | None = None) -> RetryMechanism:
    """Get or create retry mechanism.

    Args:
        name: Retry mechanism name
        config: Configuration (uses default if not provided)

    Returns:
        RetryMechanism instance
    """
    global _retry_mechanisms

    if name not in _retry_mechanisms:
        if config is None:
            config = RetryConfig()
        _retry_mechanisms[name] = RetryMechanism(name, config)

    return _retry_mechanisms[name]


def get_graceful_degradation(name: str) -> GracefulDegradation:
    """Get or create graceful degradation handler.

    Args:
        name: Handler name

    Returns:
        GracefulDegradation instance
    """
    global _degradation_handlers

    if name not in _degradation_handlers:
        _degradation_handlers[name] = GracefulDegradation(name)

    return _degradation_handlers[name]


def with_resilience(
    name: str,
    circuit_breaker_config: CircuitBreakerConfig | None = None,
    retry_config: RetryConfig | None = None,
    timeout_seconds: float | None = None,
    enable_graceful_degradation: bool = False,
):
    """Decorator to add comprehensive resilience to a function.

    Args:
        name: Name for the resilience mechanisms
        circuit_breaker_config: Circuit breaker configuration
        retry_config: Retry configuration
        timeout_seconds: Timeout in seconds
        enable_graceful_degradation: Enable graceful degradation
    """

    def decorator(func: Callable) -> Callable:
        # Get resilience components
        circuit_breaker = get_circuit_breaker(f"{name}_circuit", circuit_breaker_config)
        retry_mechanism = get_retry_mechanism(f"{name}_retry", retry_config)
        resource_manager = get_resource_manager()

        if enable_graceful_degradation:
            degradation = get_graceful_degradation(f"{name}_degradation")
            degradation.primary(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Track resource usage
            resource_manager.track_resource(wrapper)

            # Apply timeout if specified
            if timeout_seconds:
                with TimeoutHandler(timeout_seconds, name):
                    # Apply retry and circuit breaker
                    return circuit_breaker.call(
                        retry_mechanism.call, func, *args, **kwargs
                    )
            else:
                # Apply retry and circuit breaker
                return circuit_breaker.call(retry_mechanism.call, func, *args, **kwargs)

        return wrapper

    return decorator


def get_resilience_status() -> dict[str, Any]:
    """Get status of all resilience mechanisms.

    Returns:
        Status dictionary
    """
    return {
        "circuit_breakers": {
            name: cb.get_status() for name, cb in _circuit_breakers.items()
        },
        "resource_manager": get_resource_manager().check_memory_usage(),
        "tracked_resources": len(get_resource_manager()._tracked_resources),
        "retry_mechanisms": list(_retry_mechanisms.keys()),
        "degradation_handlers": list(_degradation_handlers.keys()),
    }


class ErrorHandler:
    """Central error handling and recovery coordination."""
    
    def __init__(self):
        """Initialize error handler."""
        self.circuit_breakers = _circuit_breakers
        self.resource_manager = get_resource_manager()
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = "unknown") -> None:
        """Handle and log error with appropriate recovery actions.
        
        Args:
            error: The exception that occurred
            context: Context where error occurred
        """
        self.logger.error(f"Error in {context}: {error}", exc_info=True)
        
        # Trigger appropriate recovery mechanisms
        if isinstance(error, MemoryError):
            self.resource_manager.cleanup_resources()
        elif isinstance(error, ConnectionError):
            # Circuit breaker will handle connection issues
            pass
    
    def get_status(self) -> dict:
        """Get comprehensive error handling status."""
        return get_resilience_status()
