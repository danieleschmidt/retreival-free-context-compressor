"""Enhanced monitoring, distributed tracing, and alerting system."""

import time
import logging
import json
import threading
import uuid
import queue
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
import statistics
import psutil
import torch
from datetime import datetime, timedelta
from contextlib import contextmanager
from functools import wraps
import socket
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health check status."""
    
    service: str
    healthy: bool
    message: str
    response_time_ms: float
    timestamp: float
    details: Dict[str, Any] = None


@dataclass
class Span:
    """Distributed tracing span."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
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
            **fields
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
    spans: List[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    root_span: Optional[Span] = None
    
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
    resolved_at: Optional[datetime] = None
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
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    model_name: Optional[str] = None
    operation: Optional[str] = None


class DistributedTracer:
    """Distributed tracing system."""
    
    def __init__(self, service_name: str = "retrieval-free-compressor"):
        """Initialize distributed tracer.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}  # thread_id -> span
        self._lock = threading.RLock()
        self.enabled = True
        self.max_traces = 1000
        
        logger.info(f"Distributed tracer initialized for service: {service_name}")
    
    def create_trace(self, operation_name: str) -> str:
        """Create a new trace.
        
        Args:
            operation_name: Name of the root operation
            
        Returns:
            Trace ID
        """
        trace_id = str(uuid.uuid4())
        
        with self._lock:
            # Clean up old traces if we have too many
            if len(self.traces) >= self.max_traces:
                oldest_trace_id = min(self.traces.keys(), 
                                    key=lambda tid: self.traces[tid].start_time)
                del self.traces[oldest_trace_id]
            
            trace = Trace(trace_id=trace_id)
            self.traces[trace_id] = trace
            
            # Create root span
            root_span = self.create_span(operation_name, trace_id=trace_id)
            trace.add_span(root_span)
            
            return trace_id
    
    def create_span(
        self, 
        operation_name: str, 
        trace_id: Optional[str] = None, 
        parent_span_id: Optional[str] = None
    ) -> Span:
        """Create a new span.
        
        Args:
            operation_name: Name of the operation
            trace_id: Trace ID (creates new trace if not provided)
            parent_span_id: Parent span ID
            
        Returns:
            Created span
        """
        if not self.enabled:
            return Span(
                trace_id=trace_id or "disabled",
                span_id="disabled",
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time()
            )
        
        if trace_id is None:
            trace_id = self.create_trace(operation_name)
        
        span_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        # Add service tags
        span.set_tag("service.name", self.service_name)
        span.set_tag("host.name", socket.gethostname())
        span.set_tag("process.pid", os.getpid())
        
        with self._lock:
            if trace_id in self.traces:
                self.traces[trace_id].add_span(span)
            
            # Set as active span for current thread
            thread_id = threading.current_thread().ident
            self.active_spans[thread_id] = span
        
        return span
    
    def get_active_span(self) -> Optional[Span]:
        """Get active span for current thread."""
        thread_id = threading.current_thread().ident
        return self.active_spans.get(thread_id)
    
    def finish_span(self, span: Span) -> None:
        """Finish a span."""
        span.finish()
        
        with self._lock:
            # Remove from active spans if it's the current one
            thread_id = threading.current_thread().ident
            if (thread_id in self.active_spans and 
                self.active_spans[thread_id].span_id == span.span_id):
                del self.active_spans[thread_id]
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self.traces.get(trace_id)
    
    def export_traces(self, format: str = "json") -> str:
        """Export traces in specified format."""
        with self._lock:
            if format.lower() == "json":
                traces_data = []
                for trace in self.traces.values():
                    trace_data = {
                        "trace_id": trace.trace_id,
                        "start_time": trace.start_time,
                        "end_time": trace.end_time,
                        "duration_ms": trace.duration_ms,
                        "spans": [asdict(span) for span in trace.spans]
                    }
                    traces_data.append(trace_data)
                
                return json.dumps(traces_data, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    @contextmanager
    def span(self, operation_name: str, **tags):
        """Context manager for creating spans."""
        active_span = self.get_active_span()
        parent_span_id = active_span.span_id if active_span else None
        trace_id = active_span.trace_id if active_span else None
        
        span = self.create_span(operation_name, trace_id, parent_span_id)
        
        # Set provided tags
        for key, value in tags.items():
            span.set_tag(key, value)
        
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.finish_span(span)


class EnhancedMetricsCollector:
    """Enhanced metrics collector with distributed tracing support."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize enhanced metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
        
        # Error tracking
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.error_rates: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Performance tracking
        self.sla_violations: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # SLA thresholds
        self.sla_thresholds = {
            'compression_time_ms': 5000,  # 5 second max
            'memory_usage_mb': 4096,      # 4GB max
            'error_rate_percent': 1.0     # 1% max error rate
        }
        
        logger.info("Enhanced metrics collector initialized")
    
    def record_compression(
        self,
        input_tokens: int,
        output_tokens: int,
        processing_time_ms: float,
        memory_usage_mb: float = 0,
        model_name: str = "unknown",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        operation: str = "compress",
        success: bool = True
    ) -> None:
        """Record compression operation metrics.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens  
            processing_time_ms: Processing time in milliseconds
            memory_usage_mb: Memory usage in MB
            model_name: Name of model used
            trace_id: Distributed tracing trace ID
            span_id: Distributed tracing span ID
            user_id: User identifier
            operation: Operation type
            success: Whether operation was successful
        """
        with self._lock:
            # Calculate derived metrics
            compression_ratio = input_tokens / output_tokens if output_tokens > 0 else 0
            throughput_tps = input_tokens / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            
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
                error_count=0 if success else 1,
                success_count=1 if success else 0,
                trace_id=trace_id,
                span_id=span_id,
                user_id=user_id,
                model_name=model_name,
                operation=operation
            )
            
            self.metrics_history.append(metrics)
            
            # Update counters
            self.counters['total_compressions'] += 1
            self.counters[f'compressions_{model_name}'] += 1
            self.counters['total_input_tokens'] += input_tokens
            self.counters['total_output_tokens'] += output_tokens
            
            if success:
                self.counters['successful_compressions'] += 1
            else:
                self.counters['failed_compressions'] += 1
                self.error_counts[operation] += 1
                
                # Track error rate
                self.error_rates[operation].append(time.time())
            
            # Update gauges
            self.gauges['last_compression_ratio'] = compression_ratio
            self.gauges['last_processing_time_ms'] = processing_time_ms
            self.gauges['last_throughput_tps'] = throughput_tps
            self.gauges['last_memory_usage_mb'] = memory_usage_mb
            self.gauges['last_gpu_memory_mb'] = gpu_memory_mb
            
            # Update timers and histograms
            self.timers['processing_time_ms'].append(processing_time_ms)
            self.timers['compression_ratio'].append(compression_ratio)
            self.histograms[f'{operation}_response_time'].append(processing_time_ms)
            self.response_times[operation].append(processing_time_ms)
            
            # Check SLA violations
            self._check_sla_violations(operation, processing_time_ms, memory_usage_mb)
            
            # Limit timer history
            max_timer_history = 1000
            for timer_name in self.timers:
                if len(self.timers[timer_name]) > max_timer_history:
                    self.timers[timer_name] = self.timers[timer_name][-max_timer_history:]
    
    def _check_sla_violations(self, operation: str, processing_time_ms: float, memory_usage_mb: float) -> None:
        """Check for SLA violations."""
        if processing_time_ms > self.sla_thresholds['compression_time_ms']:
            self.sla_violations[f'{operation}_time'] += 1
            logger.warning(f"SLA violation: {operation} took {processing_time_ms}ms")
        
        if memory_usage_mb > self.sla_thresholds['memory_usage_mb']:
            self.sla_violations[f'{operation}_memory'] += 1
            logger.warning(f"SLA violation: {operation} used {memory_usage_mb}MB")
    
    def get_error_rate(self, operation: str, window_minutes: int = 5) -> float:
        """Get error rate for an operation.
        
        Args:
            operation: Operation name
            window_minutes: Time window in minutes
            
        Returns:
            Error rate as percentage
        """
        with self._lock:
            if operation not in self.error_rates:
                return 0.0
            
            cutoff_time = time.time() - (window_minutes * 60)
            recent_errors = sum(1 for error_time in self.error_rates[operation] 
                              if error_time >= cutoff_time)
            
            # Get total operations in the same window
            recent_metrics = [m for m in self.metrics_history 
                            if m.timestamp >= cutoff_time and m.operation == operation]
            total_operations = len(recent_metrics)
            
            if total_operations == 0:
                return 0.0
            
            return (recent_errors / total_operations) * 100
    
    def get_percentile(self, metric_name: str, percentile: int, window_minutes: int = 60) -> float:
        """Get percentile for a metric.
        
        Args:
            metric_name: Name of the metric
            percentile: Percentile to calculate (0-100)
            window_minutes: Time window in minutes
            
        Returns:
            Percentile value
        """
        with self._lock:
            if metric_name in self.histograms:
                values = self.histograms[metric_name]
            elif metric_name in self.response_times:
                values = list(self.response_times[metric_name])
            else:
                return 0.0
            
            if not values:
                return 0.0
            
            # Filter by time window if needed
            if window_minutes and hasattr(self, 'metrics_history'):
                cutoff_time = time.time() - (window_minutes * 60)
                # This is a simplified approach - in production, you'd want 
                # to track timestamps for each value
                recent_count = int(len(values) * 0.5)  # Approximate recent values
                values = values[-recent_count:] if recent_count > 0 else values
            
            sorted_values = sorted(values)
            index = (percentile / 100) * (len(sorted_values) - 1)
            
            if index.is_integer():
                return sorted_values[int(index)]
            else:
                lower = sorted_values[int(index)]
                upper = sorted_values[min(int(index) + 1, len(sorted_values) - 1)]
                return lower + (upper - lower) * (index - int(index))
    
    def get_summary_stats(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive summary statistics.
        
        Args:
            window_minutes: Time window for statistics in minutes
            
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            cutoff_time = time.time() - (window_minutes * 60)
            recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {'message': 'No recent metrics available'}
            
            # Calculate statistics
            compression_ratios = [m.compression_ratio for m in recent_metrics if m.compression_ratio > 0]
            processing_times = [m.processing_time_ms for m in recent_metrics]
            throughputs = [m.throughput_tps for m in recent_metrics if m.throughput_tps > 0]
            memory_usage = [m.memory_usage_mb for m in recent_metrics]
            
            success_count = sum(m.success_count for m in recent_metrics)
            error_count = sum(m.error_count for m in recent_metrics)
            total_operations = len(recent_metrics)
            
            return {
                'window_minutes': window_minutes,
                'total_operations': total_operations,
                'success_rate': (success_count / total_operations) * 100 if total_operations > 0 else 0,
                'error_rate': (error_count / total_operations) * 100 if total_operations > 0 else 0,
                'compression_ratio': {
                    'mean': statistics.mean(compression_ratios) if compression_ratios else 0,
                    'median': statistics.median(compression_ratios) if compression_ratios else 0,
                    'p95': self.get_percentile('compression_ratio', 95),
                    'p99': self.get_percentile('compression_ratio', 99),
                },
                'processing_time_ms': {
                    'mean': statistics.mean(processing_times) if processing_times else 0,
                    'median': statistics.median(processing_times) if processing_times else 0,
                    'p95': self.get_percentile('processing_time_ms', 95),
                    'p99': self.get_percentile('processing_time_ms', 99),
                },
                'throughput_tps': {
                    'mean': statistics.mean(throughputs) if throughputs else 0,
                    'median': statistics.median(throughputs) if throughputs else 0,
                },
                'memory_usage_mb': {
                    'mean': statistics.mean(memory_usage) if memory_usage else 0,
                    'max': max(memory_usage) if memory_usage else 0,
                },
                'sla_violations': dict(self.sla_violations),
                'error_counts_by_operation': dict(self.error_counts)
            }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'prometheus')
            
        Returns:
            Formatted metrics string
        """
        with self._lock:
            if format.lower() == "json":
                return json.dumps({
                    'counters': dict(self.counters),
                    'gauges': dict(self.gauges),
                    'summary_stats': self.get_summary_stats(),
                    'recent_metrics': [asdict(m) for m in list(self.metrics_history)[-100:]]
                }, indent=2, default=str)
            
            elif format.lower() == "prometheus":
                lines = []
                
                # Counters
                for name, value in self.counters.items():
                    lines.append(f"retrieval_free_{name}_total {value}")
                
                # Gauges  
                for name, value in self.gauges.items():
                    lines.append(f"retrieval_free_{name} {value}")
                
                # Histograms - basic percentiles
                for operation in ['compress', 'decompress']:
                    p95 = self.get_percentile(f'{operation}_response_time', 95)
                    p99 = self.get_percentile(f'{operation}_response_time', 99)
                    lines.append(f"retrieval_free_{operation}_response_time_p95 {p95}")
                    lines.append(f"retrieval_free_{operation}_response_time_p99 {p99}")
                
                return "\n".join(lines)
            
            else:
                return f"Unsupported format: {format}"


class AlertManager:
    """Enhanced alert manager with multiple notification channels."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        self._lock = threading.RLock()
        
        # Default thresholds
        self._setup_default_alerts()
        
        logger.info("Alert manager initialized")
    
    def _setup_default_alerts(self) -> None:
        """Set up default alerts."""
        default_alerts = [
            Alert(
                alert_id="high_processing_time",
                name="High Processing Time",
                metric_name="processing_time_ms",
                operator=">",
                threshold=10000,  # 10 seconds
                severity="warning",
                description="Processing time exceeded normal threshold"
            ),
            Alert(
                alert_id="high_error_rate",
                name="High Error Rate",
                metric_name="error_rate_percent",
                operator=">",
                threshold=5.0,
                severity="critical",
                description="Error rate exceeded acceptable threshold"
            ),
            Alert(
                alert_id="high_memory_usage",
                name="High Memory Usage",
                metric_name="memory_usage_mb",
                operator=">",
                threshold=6144,  # 6GB
                severity="warning",
                description="Memory usage exceeded normal threshold"
            ),
            Alert(
                alert_id="low_compression_ratio",
                name="Low Compression Ratio",
                metric_name="compression_ratio",
                operator="<",
                threshold=2.0,
                severity="warning",
                description="Compression ratio below expected threshold"
            )
        ]
        
        for alert in default_alerts:
            self.alerts[alert.alert_id] = alert
    
    def register_alert(self, alert: Alert) -> None:
        """Register a new alert.
        
        Args:
            alert: Alert definition
        """
        with self._lock:
            self.alerts[alert.alert_id] = alert
            logger.info(f"Registered alert: {alert.name}")
    
    def add_notification_handler(self, handler: Callable[[AlertInstance], None]) -> None:
        """Add notification handler.
        
        Args:
            handler: Function that handles alert notifications
        """
        self.notification_handlers.append(handler)
    
    def check_metrics(self, metrics: Dict[str, float]) -> List[AlertInstance]:
        """Check metrics against all alerts.
        
        Args:
            metrics: Dictionary of metric values
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        
        with self._lock:
            current_time = datetime.now()
            
            for alert in self.alerts.values():
                if alert.metric_name in metrics:
                    value = metrics[alert.metric_name]
                    
                    if alert.evaluate(value):
                        # Alert should fire
                        if alert.alert_id not in self.active_alerts:
                            # New alert
                            alert_instance = AlertInstance(
                                alert_id=alert.alert_id,
                                alert_name=alert.name,
                                metric_name=alert.metric_name,
                                value=value,
                                threshold=alert.threshold,
                                severity=alert.severity,
                                fired_at=current_time
                            )
                            
                            self.active_alerts[alert.alert_id] = alert_instance
                            self.alert_history.append(alert_instance)
                            triggered_alerts.append(alert_instance)
                            
                            # Send notifications
                            for handler in self.notification_handlers:
                                try:
                                    handler(alert_instance)
                                except Exception as e:
                                    logger.error(f"Notification handler failed: {e}")
                    
                    else:
                        # Alert should not fire, resolve if active
                        if alert.alert_id in self.active_alerts:
                            alert_instance = self.active_alerts[alert.alert_id]
                            alert_instance.resolved_at = current_time
                            alert_instance.is_resolved = True
                            
                            del self.active_alerts[alert.alert_id]
                            
                            logger.info(f"Alert resolved: {alert.name}")
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[AlertInstance]:
        """Get all active alerts."""
        with self._lock:
            return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[AlertInstance]:
        """Get alert history.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of alert instances
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [alert for alert in self.alert_history 
                   if alert.fired_at >= cutoff_time]


def log_alert_handler(alert_instance: AlertInstance) -> None:
    """Default alert handler that logs alerts.
    
    Args:
        alert_instance: Alert instance to log
    """
    logger_level = logging.WARNING if alert_instance.severity == 'warning' else logging.ERROR
    logger.log(
        logger_level, 
        f"ALERT [{alert_instance.severity.upper()}] {alert_instance.alert_name}: "
        f"{alert_instance.metric_name} = {alert_instance.value} (threshold: {alert_instance.threshold})"
    )


def trace_operation(operation_name: str):
    """Decorator to trace function execution.
    
    Args:
        operation_name: Name of the operation to trace
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_distributed_tracer()
            
            with tracer.span(operation_name) as span:
                # Add function metadata
                span.set_tag("function.name", func.__name__)
                span.set_tag("function.module", func.__module__)
                
                # Add arguments if available
                if args:
                    span.set_tag("args.count", len(args))
                if kwargs:
                    span.set_tag("kwargs.count", len(kwargs))
                    # Add specific known kwargs
                    for key in ["model_name", "user_id", "compression_ratio"]:
                        if key in kwargs:
                            span.set_tag(f"arg.{key}", kwargs[key])
                
                try:
                    result = func(*args, **kwargs)
                    span.set_tag("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    span.set_tag("success", False)
                    raise
        
        return wrapper
    return decorator


# Global instances
_distributed_tracer: Optional[DistributedTracer] = None
_enhanced_metrics_collector: Optional[EnhancedMetricsCollector] = None
_alert_manager: Optional[AlertManager] = None


def get_distributed_tracer() -> DistributedTracer:
    """Get global distributed tracer.
    
    Returns:
        DistributedTracer instance
    """
    global _distributed_tracer
    if _distributed_tracer is None:
        _distributed_tracer = DistributedTracer()
    return _distributed_tracer


def get_enhanced_metrics_collector() -> EnhancedMetricsCollector:
    """Get global enhanced metrics collector.
    
    Returns:
        EnhancedMetricsCollector instance
    """
    global _enhanced_metrics_collector
    if _enhanced_metrics_collector is None:
        _enhanced_metrics_collector = EnhancedMetricsCollector()
    return _enhanced_metrics_collector


def get_alert_manager() -> AlertManager:
    """Get global alert manager.
    
    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        # Add default log handler
        _alert_manager.add_notification_handler(log_alert_handler)
    return _alert_manager


def get_monitoring_status() -> Dict[str, Any]:
    """Get comprehensive monitoring status.
    
    Returns:
        Status dictionary with all monitoring components
    """
    tracer = get_distributed_tracer()
    metrics_collector = get_enhanced_metrics_collector()
    alert_manager = get_alert_manager()
    
    return {
        "distributed_tracing": {
            "enabled": tracer.enabled,
            "service_name": tracer.service_name,
            "active_traces": len(tracer.traces),
            "active_spans": len(tracer.active_spans)
        },
        "metrics": metrics_collector.get_summary_stats(),
        "alerts": {
            "active_alerts": len(alert_manager.get_active_alerts()),
            "total_alerts_defined": len(alert_manager.alerts),
            "alert_history_24h": len(alert_manager.get_alert_history(24))
        },
        "system_health": {
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.Process().cpu_percent(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
    }