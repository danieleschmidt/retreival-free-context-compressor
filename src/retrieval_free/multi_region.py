"""Multi-region deployment support with global load balancing.

This module provides capabilities for deploying the compression service across
multiple regions with intelligent load balancing and failover:
- Multi-cloud deployment (AWS, GCP, Azure)
- Global load balancing with health checks
- Data replication and synchronization
- Latency-based routing
- Disaster recovery and failover
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from .exceptions import DeploymentError


logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


class RegionStatus(str, Enum):
    """Region health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class RoutingStrategy(str, Enum):
    """Load balancing routing strategies."""

    ROUND_ROBIN = "round_robin"
    LATENCY_BASED = "latency_based"
    CAPACITY_BASED = "capacity_based"
    GEOGRAPHIC = "geographic"
    RANDOM = "random"


@dataclass
class RegionInfo:
    """Information about a deployment region."""

    region_id: str
    cloud_provider: CloudProvider
    endpoint: str
    location: str  # e.g., "us-east-1", "europe-west1"
    capacity: int  # Max requests per second
    current_load: float = 0.0
    status: RegionStatus = RegionStatus.HEALTHY
    last_health_check: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancingDecision:
    """Result of load balancing decision."""

    selected_region: RegionInfo
    reason: str
    alternatives: list[RegionInfo]
    routing_strategy: RoutingStrategy


@dataclass
class ReplicationTask:
    """Data replication task."""

    task_id: str
    source_region: str
    target_regions: list[str]
    data_type: str  # "cache", "model", "config"
    data_key: str
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    status: str = "pending"


class HealthChecker:
    """Health checking for deployed regions."""

    def __init__(self, check_interval: float = 30.0, timeout: float = 10.0):
        self.check_interval = check_interval
        self.timeout = timeout
        self._running = False
        self._check_task = None

    async def start(self, regions: list[RegionInfo]):
        """Start health checking."""
        if self._running:
            return

        self._running = True
        self.regions = {r.region_id: r for r in regions}
        self._check_task = asyncio.create_task(self._health_check_loop())

        logger.info(f"Health checker started for {len(regions)} regions")

    async def stop(self):
        """Stop health checking."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_regions()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _check_all_regions(self):
        """Check health of all regions."""
        tasks = []
        for region in self.regions.values():
            task = asyncio.create_task(self._check_region_health(region))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_region_health(self, region: RegionInfo):
        """Check health of a specific region."""
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # Health check endpoint
                health_url = f"{region.endpoint.rstrip('/')}/health"

                response = await client.get(health_url)
                latency = time.time() - start_time

                if response.status_code == 200:
                    health_data = response.json()

                    # Update region status
                    region.status = RegionStatus.HEALTHY
                    region.average_latency = latency
                    region.last_health_check = time.time()
                    region.error_rate = 0.0

                    # Update capacity metrics if available
                    if "active_tasks" in health_data and "queue_size" in health_data:
                        active_tasks = health_data["active_tasks"]
                        queue_size = health_data["queue_size"]
                        region.current_load = (
                            active_tasks + queue_size
                        ) / region.capacity

                    logger.debug(
                        f"Region {region.region_id} health: OK (latency: {latency:.2f}s)"
                    )

                else:
                    self._mark_region_unhealthy(region, f"HTTP {response.status_code}")

        except asyncio.TimeoutError:
            self._mark_region_unhealthy(region, "Timeout")
        except Exception as e:
            self._mark_region_unhealthy(region, str(e))

    def _mark_region_unhealthy(self, region: RegionInfo, reason: str):
        """Mark region as unhealthy."""
        region.status = RegionStatus.UNHEALTHY
        region.last_health_check = time.time()
        region.error_rate = min(1.0, region.error_rate + 0.1)

        logger.warning(f"Region {region.region_id} unhealthy: {reason}")

    def get_healthy_regions(self) -> list[RegionInfo]:
        """Get list of healthy regions."""
        return [
            region
            for region in self.regions.values()
            if region.status == RegionStatus.HEALTHY
        ]


class LoadBalancer:
    """Global load balancer with multiple routing strategies."""

    def __init__(
        self,
        regions: list[RegionInfo],
        default_strategy: RoutingStrategy = RoutingStrategy.LATENCY_BASED,
    ):
        self.regions = {r.region_id: r for r in regions}
        self.default_strategy = default_strategy

        # Strategy state
        self._round_robin_index = 0
        self._latency_cache = {}

        # Client location mapping (simplified)
        self.geographic_mapping = {
            "us": ["us-east-1", "us-west-2"],
            "eu": ["europe-west1", "europe-north1"],
            "asia": ["asia-southeast1", "asia-northeast1"],
        }

    async def select_region(
        self,
        strategy: RoutingStrategy | None = None,
        client_location: str | None = None,
        exclude_regions: set[str] | None = None,
    ) -> LoadBalancingDecision:
        """Select optimal region based on strategy."""
        strategy = strategy or self.default_strategy
        exclude_regions = exclude_regions or set()

        # Get healthy regions
        available_regions = [
            region
            for region in self.regions.values()
            if (
                region.status == RegionStatus.HEALTHY
                and region.region_id not in exclude_regions
            )
        ]

        if not available_regions:
            # Fallback to degraded regions if no healthy ones
            available_regions = [
                region
                for region in self.regions.values()
                if (
                    region.status == RegionStatus.DEGRADED
                    and region.region_id not in exclude_regions
                )
            ]

        if not available_regions:
            raise DeploymentError("No available regions for routing")

        # Apply routing strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            selected = await self._round_robin_selection(available_regions)
        elif strategy == RoutingStrategy.LATENCY_BASED:
            selected = await self._latency_based_selection(available_regions)
        elif strategy == RoutingStrategy.CAPACITY_BASED:
            selected = await self._capacity_based_selection(available_regions)
        elif strategy == RoutingStrategy.GEOGRAPHIC:
            selected = await self._geographic_selection(
                available_regions, client_location
            )
        else:  # RANDOM
            selected = random.choice(available_regions)

        return LoadBalancingDecision(
            selected_region=selected,
            reason=f"Selected by {strategy.value} strategy",
            alternatives=[r for r in available_regions if r != selected],
            routing_strategy=strategy,
        )

    async def _round_robin_selection(self, regions: list[RegionInfo]) -> RegionInfo:
        """Round-robin region selection."""
        if not regions:
            raise DeploymentError("No regions available for round-robin")

        selected = regions[self._round_robin_index % len(regions)]
        self._round_robin_index += 1
        return selected

    async def _latency_based_selection(self, regions: list[RegionInfo]) -> RegionInfo:
        """Latency-based region selection."""
        # Sort by average latency (lower is better)
        sorted_regions = sorted(regions, key=lambda r: r.average_latency)

        # Use weighted selection favoring lower latency
        if len(sorted_regions) == 1:
            return sorted_regions[0]

        # Weight calculation: inverse of latency
        weights = []
        for region in sorted_regions:
            # Avoid division by zero
            weight = 1.0 / max(region.average_latency, 0.001)
            weights.append(weight)

        # Weighted random selection
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)

        cumulative = 0.0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand_val <= cumulative:
                return sorted_regions[i]

        return sorted_regions[0]  # Fallback

    async def _capacity_based_selection(self, regions: list[RegionInfo]) -> RegionInfo:
        """Capacity-based region selection."""
        # Select region with lowest current load
        available_capacity_regions = [
            r for r in regions if r.current_load < 0.9  # Under 90% capacity
        ]

        if not available_capacity_regions:
            # All regions are highly loaded, select least loaded
            available_capacity_regions = regions

        return min(available_capacity_regions, key=lambda r: r.current_load)

    async def _geographic_selection(
        self, regions: list[RegionInfo], client_location: str | None
    ) -> RegionInfo:
        """Geographic proximity-based selection."""
        if not client_location:
            # Fallback to latency-based if no location info
            return await self._latency_based_selection(regions)

        # Find preferred regions for client location
        preferred_region_ids = self.geographic_mapping.get(client_location, [])

        # Filter available regions by preference
        preferred_regions = [r for r in regions if r.location in preferred_region_ids]

        if preferred_regions:
            # Select best from preferred regions based on load
            return min(preferred_regions, key=lambda r: r.current_load)
        else:
            # No preferred regions available, fallback to latency
            return await self._latency_based_selection(regions)


class DataReplicator:
    """Handles data replication across regions."""

    def __init__(self):
        self.replication_tasks = {}
        self.replication_queue = asyncio.Queue()
        self._workers = []
        self._running = False

    async def start(self, num_workers: int = 3):
        """Start replication workers."""
        if self._running:
            return

        self._running = True

        for i in range(num_workers):
            worker = asyncio.create_task(self._replication_worker(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"Data replicator started with {num_workers} workers")

    async def stop(self):
        """Stop replication workers."""
        self._running = False

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def replicate_data(
        self,
        data_type: str,
        data_key: str,
        source_region: str,
        target_regions: list[str],
        priority: int = 5,
    ) -> str:
        """Queue data for replication."""
        task_id = str(uuid.uuid4())

        task = ReplicationTask(
            task_id=task_id,
            source_region=source_region,
            target_regions=target_regions,
            data_type=data_type,
            data_key=data_key,
            priority=priority,
        )

        self.replication_tasks[task_id] = task
        await self.replication_queue.put(task)

        logger.info(f"Queued replication task {task_id}: {data_type}:{data_key}")
        return task_id

    async def _replication_worker(self, worker_name: str):
        """Replication worker process."""
        logger.info(f"Replication worker {worker_name} started")

        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self.replication_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process replication task
                await self._process_replication_task(task)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Replication worker {worker_name} error: {e}")

        logger.info(f"Replication worker {worker_name} stopped")

    async def _process_replication_task(self, task: ReplicationTask):
        """Process a single replication task."""
        try:
            task.status = "processing"

            logger.debug(f"Processing replication task {task.task_id}")

            if task.data_type == "cache":
                await self._replicate_cache_data(task)
            elif task.data_type == "model":
                await self._replicate_model_data(task)
            elif task.data_type == "config":
                await self._replicate_config_data(task)
            else:
                raise ValueError(f"Unknown data type: {task.data_type}")

            task.status = "completed"
            task.completed_at = time.time()

            logger.info(f"Replication task {task.task_id} completed")

        except Exception as e:
            task.status = "failed"
            logger.error(f"Replication task {task.task_id} failed: {e}")

    async def _replicate_cache_data(self, task: ReplicationTask):
        """Replicate cache data between regions."""
        # This would implement cache data synchronization
        # For now, simulate the operation
        await asyncio.sleep(0.1)  # Simulate network latency

    async def _replicate_model_data(self, task: ReplicationTask):
        """Replicate model data between regions."""
        # This would implement model synchronization
        await asyncio.sleep(0.5)  # Simulate larger data transfer

    async def _replicate_config_data(self, task: ReplicationTask):
        """Replicate configuration data between regions."""
        # This would implement config synchronization
        await asyncio.sleep(0.05)  # Simulate small config transfer


class FailoverManager:
    """Handles failover scenarios and disaster recovery."""

    def __init__(self, regions: list[RegionInfo]):
        self.regions = {r.region_id: r for r in regions}
        self.failover_rules = {}
        self.active_failovers = {}

    def configure_failover(
        self,
        primary_region: str,
        backup_regions: list[str],
        auto_failover: bool = True,
        failover_threshold: float = 0.5,  # Error rate threshold
    ):
        """Configure failover rules for a region."""
        self.failover_rules[primary_region] = {
            "backup_regions": backup_regions,
            "auto_failover": auto_failover,
            "threshold": failover_threshold,
        }

    async def check_failover_conditions(self) -> list[str]:
        """Check if any regions need failover."""
        regions_to_failover = []

        for region_id, region in self.regions.items():
            if region_id not in self.failover_rules:
                continue

            rules = self.failover_rules[region_id]

            # Check if region is unhealthy and exceeds threshold
            if (
                region.status == RegionStatus.UNHEALTHY
                and region.error_rate > rules["threshold"]
                and rules["auto_failover"]
            ):

                regions_to_failover.append(region_id)

        return regions_to_failover

    async def execute_failover(self, primary_region: str) -> bool:
        """Execute failover for a region."""
        if primary_region not in self.failover_rules:
            logger.error(f"No failover rules configured for {primary_region}")
            return False

        rules = self.failover_rules[primary_region]
        backup_regions = rules["backup_regions"]

        # Find healthy backup region
        healthy_backups = [
            backup
            for backup in backup_regions
            if (
                backup in self.regions
                and self.regions[backup].status == RegionStatus.HEALTHY
            )
        ]

        if not healthy_backups:
            logger.error(f"No healthy backup regions for {primary_region}")
            return False

        # Select best backup region (lowest load)
        selected_backup = min(
            healthy_backups, key=lambda r: self.regions[r].current_load
        )

        # Record failover
        failover_id = str(uuid.uuid4())
        self.active_failovers[failover_id] = {
            "primary_region": primary_region,
            "backup_region": selected_backup,
            "started_at": time.time(),
            "status": "active",
        }

        logger.warning(f"Executed failover from {primary_region} to {selected_backup}")

        # Here you would implement actual traffic redirection
        # This might involve updating DNS records, load balancer config, etc.

        return True

    async def check_failback_conditions(self) -> list[str]:
        """Check if any failed regions can be restored."""
        regions_to_failback = []

        for failover in self.active_failovers.values():
            if failover["status"] != "active":
                continue

            primary_region = failover["primary_region"]

            # Check if primary region is healthy again
            if (
                primary_region in self.regions
                and self.regions[primary_region].status == RegionStatus.HEALTHY
                and self.regions[primary_region].error_rate < 0.1
            ):

                regions_to_failback.append(primary_region)

        return regions_to_failback


class MultiRegionManager:
    """Comprehensive multi-region deployment manager."""

    def __init__(
        self,
        regions: list[RegionInfo],
        default_routing: RoutingStrategy = RoutingStrategy.LATENCY_BASED,
    ):
        self.regions = {r.region_id: r for r in regions}

        # Components
        self.health_checker = HealthChecker()
        self.load_balancer = LoadBalancer(regions, default_routing)
        self.data_replicator = DataReplicator()
        self.failover_manager = FailoverManager(regions)

        # State
        self._running = False
        self._management_task = None

    async def start(self):
        """Start multi-region management."""
        if self._running:
            return

        self._running = True

        # Start components
        await self.health_checker.start(list(self.regions.values()))
        await self.data_replicator.start()

        # Start management loop
        self._management_task = asyncio.create_task(self._management_loop())

        logger.info(f"Multi-region manager started for {len(self.regions)} regions")

    async def stop(self):
        """Stop multi-region management."""
        self._running = False

        # Stop management task
        if self._management_task:
            self._management_task.cancel()
            try:
                await self._management_task
            except asyncio.CancelledError:
                pass

        # Stop components
        await self.health_checker.stop()
        await self.data_replicator.stop()

        logger.info("Multi-region manager stopped")

    async def _management_loop(self):
        """Main management loop."""
        while self._running:
            try:
                # Check for failover conditions
                regions_to_failover = (
                    await self.failover_manager.check_failover_conditions()
                )

                for region_id in regions_to_failover:
                    await self.failover_manager.execute_failover(region_id)

                # Check for failback conditions
                regions_to_failback = (
                    await self.failover_manager.check_failback_conditions()
                )

                for region_id in regions_to_failback:
                    # Implement failback logic here
                    logger.info(f"Region {region_id} ready for failback")

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Multi-region management error: {e}")

    async def route_request(
        self,
        strategy: RoutingStrategy | None = None,
        client_location: str | None = None,
    ) -> LoadBalancingDecision:
        """Route request to optimal region."""
        return await self.load_balancer.select_region(
            strategy=strategy, client_location=client_location
        )

    async def replicate_across_regions(
        self, data_type: str, data_key: str, source_region: str
    ) -> str:
        """Replicate data across all other regions."""
        target_regions = [
            region_id for region_id in self.regions.keys() if region_id != source_region
        ]

        return await self.data_replicator.replicate_data(
            data_type=data_type,
            data_key=data_key,
            source_region=source_region,
            target_regions=target_regions,
        )

    def get_region_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all regions."""
        status = {}

        for region_id, region in self.regions.items():
            status[region_id] = {
                "status": region.status.value,
                "location": region.location,
                "cloud_provider": region.cloud_provider.value,
                "current_load": region.current_load,
                "average_latency": region.average_latency,
                "error_rate": region.error_rate,
                "last_health_check": region.last_health_check,
                "capacity": region.capacity,
            }

        return status

    def get_deployment_summary(self) -> dict[str, Any]:
        """Get comprehensive deployment summary."""
        healthy_regions = len(
            [r for r in self.regions.values() if r.status == RegionStatus.HEALTHY]
        )
        total_capacity = sum(r.capacity for r in self.regions.values())
        total_load = sum(r.current_load * r.capacity for r in self.regions.values())

        return {
            "total_regions": len(self.regions),
            "healthy_regions": healthy_regions,
            "total_capacity": total_capacity,
            "current_load": total_load,
            "utilization": (total_load / total_capacity) if total_capacity > 0 else 0,
            "active_failovers": len(
                [
                    f
                    for f in self.failover_manager.active_failovers.values()
                    if f["status"] == "active"
                ]
            ),
            "replication_tasks": len(self.data_replicator.replication_tasks),
            "cloud_providers": list(
                set(r.cloud_provider for r in self.regions.values())
            ),
            "regions_by_status": {
                status.value: len(
                    [r for r in self.regions.values() if r.status == status]
                )
                for status in RegionStatus
            },
        }


# Example region configurations
def create_example_regions() -> list[RegionInfo]:
    """Create example multi-region configuration."""
    return [
        RegionInfo(
            region_id="us-east-1",
            cloud_provider=CloudProvider.AWS,
            endpoint="https://api-us-east-1.example.com",
            location="us-east-1",
            capacity=1000,
        ),
        RegionInfo(
            region_id="us-west-2",
            cloud_provider=CloudProvider.AWS,
            endpoint="https://api-us-west-2.example.com",
            location="us-west-2",
            capacity=800,
        ),
        RegionInfo(
            region_id="europe-west1",
            cloud_provider=CloudProvider.GCP,
            endpoint="https://api-eu-west1.example.com",
            location="europe-west1",
            capacity=600,
        ),
        RegionInfo(
            region_id="asia-southeast1",
            cloud_provider=CloudProvider.GCP,
            endpoint="https://api-asia-se1.example.com",
            location="asia-southeast1",
            capacity=400,
        ),
    ]


async def setup_multi_region_deployment(
    regions: list[RegionInfo] | None = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.LATENCY_BASED,
) -> MultiRegionManager:
    """Setup and start multi-region deployment."""
    if regions is None:
        regions = create_example_regions()

    manager = MultiRegionManager(regions, routing_strategy)

    # Configure failover rules
    manager.failover_manager.configure_failover(
        "us-east-1", ["us-west-2", "europe-west1"]
    )
    manager.failover_manager.configure_failover(
        "europe-west1", ["us-east-1", "asia-southeast1"]
    )

    await manager.start()
    return manager
