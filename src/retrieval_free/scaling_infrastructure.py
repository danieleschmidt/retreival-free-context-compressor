"""
Scaling Infrastructure - Generation 3
Advanced scaling and distributed processing capabilities.
"""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
import logging
import json
import multiprocessing as mp
from queue import Queue, Empty
import threading
import socket
import subprocess
import os
from pathlib import Path

from .observability import MetricsCollector, PerformanceMonitor
from .exceptions import ScalingError, ResourceError

logger = logging.getLogger(__name__)


@dataclass
class ScalingConfig:
    """Configuration for scaling infrastructure."""
    
    # Processing configuration
    max_processes: int = mp.cpu_count()
    max_threads_per_process: int = 4
    enable_multiprocessing: bool = True
    enable_async: bool = True
    
    # Load balancing
    load_balance_strategy: str = "round_robin"  # round_robin, least_connections, weighted
    health_check_interval: int = 30  # seconds
    
    # Horizontal scaling
    enable_horizontal_scaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    scale_up_threshold: float = 0.70  # CPU utilization
    scale_down_threshold: float = 0.30
    
    # Resource limits
    memory_limit_mb: int = 4096
    cpu_limit_cores: int = 4
    disk_limit_gb: int = 100
    
    # Monitoring and alerting
    enable_monitoring: bool = True
    alert_on_failures: bool = True
    max_failure_rate: float = 0.10  # 10% failure rate


class WorkerNode:
    """Represents a worker node in the scaling infrastructure."""
    
    def __init__(self, node_id: str, host: str = "localhost", port: int = 8080):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.status = "initializing"
        self.last_heartbeat = time.time()
        self.active_connections = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.metrics = MetricsCollector()
        self.performance_monitor = PerformanceMonitor()
    
    def start(self):
        """Start the worker node."""
        try:
            self.status = "starting"
            self._initialize_worker()
            self.status = "healthy"
            self.last_heartbeat = time.time()
            
            logger.info(f"Worker node {self.node_id} started successfully")
            self.metrics.increment("worker.started")
            
        except Exception as e:
            self.status = "failed"
            logger.error(f"Failed to start worker node {self.node_id}: {e}")
            self.metrics.increment("worker.start_failed")
            raise ScalingError(f"Worker node start failed: {e}", node_id=self.node_id)
    
    def stop(self):
        """Stop the worker node."""
        try:
            self.status = "stopping"
            self._cleanup_worker()
            self.status = "stopped"
            
            logger.info(f"Worker node {self.node_id} stopped")
            self.metrics.increment("worker.stopped")
            
        except Exception as e:
            self.status = "failed"
            logger.error(f"Failed to stop worker node {self.node_id}: {e}")
            self.metrics.increment("worker.stop_failed")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request on this worker node."""
        start_time = time.perf_counter()
        self.active_connections += 1
        self.total_requests += 1
        
        try:
            # Simulate request processing
            result = self._process_request_impl(request)
            
            # Record success metrics
            duration = time.perf_counter() - start_time
            # Store timing directly in timers dict
            self.metrics.timers["request.duration"] = duration
            self.metrics.increment("request.success")
            
            return {
                "status": "success",
                "result": result,
                "node_id": self.node_id,
                "duration": duration
            }
            
        except Exception as e:
            self.failed_requests += 1
            self.metrics.increment("request.failed")
            
            logger.error(f"Request failed on node {self.node_id}: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "node_id": self.node_id,
                "duration": time.perf_counter() - start_time
            }
        
        finally:
            self.active_connections -= 1
            self.last_heartbeat = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of this worker node."""
        system_metrics = self.performance_monitor.get_system_metrics()
        
        failure_rate = (
            self.failed_requests / max(self.total_requests, 1) 
            if self.total_requests > 0 else 0
        )
        
        return {
            "node_id": self.node_id,
            "status": self.status,
            "last_heartbeat": self.last_heartbeat,
            "active_connections": self.active_connections,
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "failure_rate": failure_rate,
            "system_metrics": system_metrics,
            "is_healthy": self._is_healthy()
        }
    
    def _initialize_worker(self):
        """Initialize worker resources."""
        # Check if port is available
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((self.host, self.port))
            sock.close()
        except OSError:
            # Port in use, find next available
            self.port = self._find_available_port(self.port)
        
        logger.debug(f"Worker {self.node_id} initialized on {self.host}:{self.port}")
    
    def _cleanup_worker(self):
        """Cleanup worker resources."""
        # Close any open connections, cleanup resources
        pass
    
    def _process_request_impl(self, request: Dict[str, Any]) -> Any:
        """Actual request processing implementation."""
        # Simulate compression processing
        text = request.get("text", "")
        compression_ratio = request.get("compression_ratio", 8.0)
        
        # Simulate processing time based on text length
        processing_time = len(text) / 10000  # 10ms per 1k characters
        time.sleep(min(processing_time, 2.0))  # Cap at 2 seconds
        
        return {
            "compressed_text": f"[COMPRESSED:{len(text)} chars @ {compression_ratio}x]",
            "original_length": len(text),
            "compressed_length": max(1, int(len(text) / compression_ratio)),
            "compression_ratio": compression_ratio
        }
    
    def _find_available_port(self, start_port: int) -> int:
        """Find next available port starting from given port."""
        for port in range(start_port, start_port + 100):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('localhost', port))
                sock.close()
                return port
            except OSError:
                continue
        
        raise ScalingError(f"No available ports found starting from {start_port}")
    
    def _is_healthy(self) -> bool:
        """Check if node is healthy."""
        if self.status not in ["healthy", "busy"]:
            return False
        
        # Check heartbeat
        if time.time() - self.last_heartbeat > 60:  # 1 minute timeout
            return False
        
        # Check failure rate
        failure_rate = self.failed_requests / max(self.total_requests, 1)
        if failure_rate > 0.5:  # 50% failure rate
            return False
        
        return True


class LoadBalancer:
    """Distributes load across multiple worker nodes."""
    
    def __init__(self, config: Optional[ScalingConfig] = None):
        self.config = config or ScalingConfig()
        self.workers: Dict[str, WorkerNode] = {}
        self.current_worker_index = 0
        self.metrics = MetricsCollector()
        self._health_check_thread = None
        self._health_check_running = False
    
    def add_worker(self, node_id: str, host: str = "localhost", port: int = 8080) -> WorkerNode:
        """Add a worker node to the load balancer."""
        worker = WorkerNode(node_id, host, port)
        worker.start()
        
        self.workers[node_id] = worker
        self.metrics.set_gauge("workers.count", len(self.workers))
        
        logger.info(f"Added worker node: {node_id}")
        return worker
    
    def remove_worker(self, node_id: str):
        """Remove a worker node from the load balancer."""
        if node_id in self.workers:
            worker = self.workers[node_id]
            worker.stop()
            del self.workers[node_id]
            
            self.metrics.set_gauge("workers.count", len(self.workers))
            logger.info(f"Removed worker node: {node_id}")
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request using load balancing."""
        if not self.workers:
            raise ScalingError("No worker nodes available")
        
        # Select worker based on strategy
        worker = self._select_worker()
        if not worker:
            raise ScalingError("No healthy worker nodes available")
        
        try:
            result = worker.process_request(request)
            self.metrics.increment("loadbalancer.request_success")
            return result
            
        except Exception as e:
            self.metrics.increment("loadbalancer.request_failed")
            logger.error(f"Request failed on load balancer: {e}")
            
            # Try backup worker
            backup_worker = self._select_backup_worker(worker.node_id)
            if backup_worker:
                try:
                    result = backup_worker.process_request(request)
                    self.metrics.increment("loadbalancer.failover_success")
                    return result
                except Exception as backup_e:
                    logger.error(f"Backup worker also failed: {backup_e}")
            
            raise ScalingError(f"All workers failed to process request: {e}")
    
    def start_health_checking(self):
        """Start background health checking."""
        if self._health_check_running:
            return
        
        self._health_check_running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        
        logger.info("Started health checking")
    
    def stop_health_checking(self):
        """Stop background health checking."""
        self._health_check_running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        
        logger.info("Stopped health checking")
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        healthy_workers = sum(1 for w in self.workers.values() if w._is_healthy())
        total_requests = sum(w.total_requests for w in self.workers.values())
        total_failures = sum(w.failed_requests for w in self.workers.values())
        
        return {
            "total_workers": len(self.workers),
            "healthy_workers": healthy_workers,
            "unhealthy_workers": len(self.workers) - healthy_workers,
            "total_requests": total_requests,
            "total_failures": total_failures,
            "failure_rate": total_failures / max(total_requests, 1),
            "workers": {
                node_id: worker.get_health_status()
                for node_id, worker in self.workers.items()
            },
            "metrics": self.metrics.get_all_metrics()
        }
    
    def _select_worker(self) -> Optional[WorkerNode]:
        """Select worker based on load balancing strategy."""
        healthy_workers = [w for w in self.workers.values() if w._is_healthy()]
        
        if not healthy_workers:
            return None
        
        if self.config.load_balance_strategy == "round_robin":
            worker = healthy_workers[self.current_worker_index % len(healthy_workers)]
            self.current_worker_index += 1
            return worker
        
        elif self.config.load_balance_strategy == "least_connections":
            return min(healthy_workers, key=lambda w: w.active_connections)
        
        elif self.config.load_balance_strategy == "weighted":
            # Simple weight based on failure rate (lower is better)
            weights = [1.0 / (1.0 + w.failed_requests) for w in healthy_workers]
            total_weight = sum(weights)
            
            if total_weight > 0:
                import random
                rand_val = random.random() * total_weight
                weight_sum = 0
                
                for i, weight in enumerate(weights):
                    weight_sum += weight
                    if rand_val <= weight_sum:
                        return healthy_workers[i]
        
        # Fallback to first healthy worker
        return healthy_workers[0]
    
    def _select_backup_worker(self, exclude_node_id: str) -> Optional[WorkerNode]:
        """Select a backup worker excluding the failed one."""
        healthy_workers = [
            w for w in self.workers.values() 
            if w._is_healthy() and w.node_id != exclude_node_id
        ]
        
        return healthy_workers[0] if healthy_workers else None
    
    def _health_check_loop(self):
        """Background health checking loop."""
        while self._health_check_running:
            try:
                unhealthy_workers = []
                
                for node_id, worker in self.workers.items():
                    if not worker._is_healthy():
                        unhealthy_workers.append(node_id)
                        logger.warning(f"Worker {node_id} is unhealthy: {worker.get_health_status()}")
                
                # Update metrics
                healthy_count = len(self.workers) - len(unhealthy_workers)
                self.metrics.set_gauge("workers.healthy", healthy_count)
                self.metrics.set_gauge("workers.unhealthy", len(unhealthy_workers))
                
                # Alert on high failure rate
                if len(unhealthy_workers) > len(self.workers) * 0.5:
                    self.metrics.increment("loadbalancer.high_failure_alert")
                    logger.critical(f"High failure rate: {len(unhealthy_workers)}/{len(self.workers)} workers unhealthy")
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
            
            time.sleep(self.config.health_check_interval)


class HorizontalScaler:
    """Manages horizontal scaling of worker nodes."""
    
    def __init__(self, 
                 load_balancer: LoadBalancer,
                 config: Optional[ScalingConfig] = None):
        self.load_balancer = load_balancer
        self.config = config or ScalingConfig()
        self.metrics = MetricsCollector()
        self._scaling_enabled = True
        self._last_scale_time = 0
        self._scale_cooldown = 60  # seconds
    
    def enable_scaling(self):
        """Enable automatic scaling."""
        self._scaling_enabled = True
        logger.info("Horizontal scaling enabled")
    
    def disable_scaling(self):
        """Disable automatic scaling."""
        self._scaling_enabled = False
        logger.info("Horizontal scaling disabled")
    
    def check_and_scale(self):
        """Check metrics and scale if necessary."""
        if not self._scaling_enabled:
            return
        
        # Respect cooldown period
        if time.time() - self._last_scale_time < self._scale_cooldown:
            return
        
        try:
            status = self.load_balancer.get_status()
            
            # Calculate average CPU usage across healthy workers
            healthy_workers = [
                w for w in status["workers"].values() 
                if w["is_healthy"]
            ]
            
            if not healthy_workers:
                logger.warning("No healthy workers for scaling decisions")
                return
            
            avg_cpu = sum(
                w["system_metrics"].get("cpu_percent", 0) 
                for w in healthy_workers
            ) / len(healthy_workers)
            
            current_replicas = status["healthy_workers"]
            
            # Scale up decision
            if (avg_cpu > self.config.scale_up_threshold * 100 and 
                current_replicas < self.config.max_replicas):
                self._scale_up()
            
            # Scale down decision
            elif (avg_cpu < self.config.scale_down_threshold * 100 and 
                  current_replicas > self.config.min_replicas):
                self._scale_down()
            
            # Update metrics
            self.metrics.set_gauge("scaler.current_replicas", current_replicas)
            self.metrics.set_gauge("scaler.avg_cpu", avg_cpu)
            
        except Exception as e:
            logger.error(f"Error in horizontal scaling check: {e}")
            self.metrics.increment("scaler.error")
    
    def _scale_up(self):
        """Add new worker node."""
        try:
            # Generate new node ID
            node_id = f"worker-{len(self.load_balancer.workers) + 1}-{int(time.time())}"
            
            # Find available port
            base_port = 8080 + len(self.load_balancer.workers)
            
            # Add new worker
            self.load_balancer.add_worker(node_id, port=base_port)
            
            self._last_scale_time = time.time()
            self.metrics.increment("scaler.scaled_up")
            
            logger.info(f"Scaled up: Added worker {node_id}")
            
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
            self.metrics.increment("scaler.scale_up_failed")
    
    def _scale_down(self):
        """Remove a worker node."""
        try:
            # Find worker with least connections to remove
            workers_by_load = sorted(
                self.load_balancer.workers.values(),
                key=lambda w: w.active_connections
            )
            
            if workers_by_load:
                worker_to_remove = workers_by_load[0]
                self.load_balancer.remove_worker(worker_to_remove.node_id)
                
                self._last_scale_time = time.time()
                self.metrics.increment("scaler.scaled_down")
                
                logger.info(f"Scaled down: Removed worker {worker_to_remove.node_id}")
        
        except Exception as e:
            logger.error(f"Scale down failed: {e}")
            self.metrics.increment("scaler.scale_down_failed")


# Global scaling infrastructure instance
_load_balancer = None
_horizontal_scaler = None

def get_scaling_infrastructure(config: Optional[ScalingConfig] = None) -> Tuple[LoadBalancer, HorizontalScaler]:
    """Get or create global scaling infrastructure."""
    global _load_balancer, _horizontal_scaler
    
    if _load_balancer is None:
        _load_balancer = LoadBalancer(config)
        _horizontal_scaler = HorizontalScaler(_load_balancer, config)
        
        # Initialize with minimum workers
        min_replicas = config.min_replicas if config else 2
        for i in range(min_replicas):
            _load_balancer.add_worker(f"worker-{i+1}", port=8080+i)
        
        # Start health checking
        _load_balancer.start_health_checking()
    
    return _load_balancer, _horizontal_scaler


def scale_compression_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Process compression request with scaling infrastructure."""
    load_balancer, scaler = get_scaling_infrastructure()
    
    # Check and scale based on current load
    scaler.check_and_scale()
    
    # Process request
    return load_balancer.process_request(request)


def get_scaling_status() -> Dict[str, Any]:
    """Get comprehensive scaling infrastructure status."""
    if _load_balancer is None:
        return {"status": "not_initialized"}
    
    return {
        "load_balancer": _load_balancer.get_status(),
        "horizontal_scaler": {
            "enabled": _horizontal_scaler._scaling_enabled,
            "last_scale_time": _horizontal_scaler._last_scale_time,
            "metrics": _horizontal_scaler.metrics.get_all_metrics()
        }
    }