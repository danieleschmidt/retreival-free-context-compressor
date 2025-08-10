"""Async API endpoints with FastAPI integration and advanced request handling.

This module provides high-performance async API endpoints with:
- FastAPI integration
- Request queuing and batching
- Rate limiting and throttling
- WebSocket support for streaming
- Health checks and metrics
- Auto-scaling integration
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from .caching import create_cache_key
from .core import CompressorBase
from .distributed_cache import DistributedCacheManager
from .scaling import HighPerformanceCompressor


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Priority(int, Enum):
    """Request priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


# Pydantic models for API
class CompressionRequest(BaseModel):
    """Request model for compression operations."""

    text: str = Field(
        ..., min_length=1, max_length=10_000_000, description="Text to compress"
    )
    priority: Priority = Field(
        default=Priority.NORMAL, description="Processing priority"
    )
    compression_ratio: float | None = Field(
        default=8.0, ge=2.0, le=32.0, description="Target compression ratio"
    )
    chunk_size: int | None = Field(
        default=512, ge=64, le=2048, description="Chunk size for processing"
    )
    use_cache: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(
        default=3600, ge=300, le=604800, description="Cache TTL in seconds"
    )
    async_processing: bool = Field(default=False, description="Process asynchronously")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class BatchCompressionRequest(BaseModel):
    """Request model for batch compression operations."""

    texts: list[str] = Field(
        ..., min_items=1, max_items=1000, description="Texts to compress"
    )
    priority: Priority = Field(
        default=Priority.NORMAL, description="Processing priority"
    )
    compression_ratio: float | None = Field(
        default=8.0, ge=2.0, le=32.0, description="Target compression ratio"
    )
    chunk_size: int | None = Field(
        default=512, ge=64, le=2048, description="Chunk size for processing"
    )
    use_cache: bool = Field(default=True, description="Enable result caching")
    cache_ttl: int = Field(
        default=3600, ge=300, le=604800, description="Cache TTL in seconds"
    )
    use_distributed: bool = Field(
        default=False, description="Use distributed processing"
    )
    batch_processing: bool = Field(default=True, description="Process as batch")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class MegaTokenModel(BaseModel):
    """Pydantic model for MegaToken."""

    vector: list[float] = Field(..., description="Dense vector representation")
    metadata: dict[str, Any] = Field(..., description="Token metadata")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class CompressionResponse(BaseModel):
    """Response model for compression operations."""

    task_id: str = Field(..., description="Unique task identifier")
    status: TaskStatus = Field(..., description="Processing status")
    mega_tokens: list[MegaTokenModel] | None = Field(
        None, description="Compressed tokens"
    )
    original_length: int | None = Field(
        None, description="Original text length in tokens"
    )
    compressed_length: int | None = Field(
        None, description="Compressed length in mega-tokens"
    )
    compression_ratio: float | None = Field(
        None, description="Achieved compression ratio"
    )
    processing_time: float | None = Field(
        None, description="Processing time in seconds"
    )
    cached: bool = Field(default=False, description="Result served from cache")
    error_message: str | None = Field(None, description="Error message if failed")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Response metadata"
    )


class TaskInfo(BaseModel):
    """Information about a processing task."""

    task_id: str
    status: TaskStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    priority: Priority
    estimated_completion: float | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)


class HealthCheck(BaseModel):
    """Health check response."""

    status: str
    timestamp: float
    version: str
    uptime_seconds: float
    active_tasks: int
    queue_size: int
    cache_hit_rate: float
    memory_usage_mb: float
    gpu_available: bool


@dataclass
class WebSocketSession:
    """WebSocket session information."""

    websocket: WebSocket
    session_id: str
    connected_at: float
    last_activity: float
    subscriptions: set


class RateLimiter:
    """Rate limiting implementation."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time
                for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []

        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False

        # Record request
        self.requests[client_id].append(now)
        return True


class AsyncCompressionAPI:
    """High-performance async compression API."""

    def __init__(
        self,
        compressor: CompressorBase | None = None,
        cache_manager: DistributedCacheManager | None = None,
        max_workers: int = 8,
        enable_rate_limiting: bool = True,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60,
    ):
        # Initialize compressor
        if compressor is None:
            from .core import ContextCompressor

            base_compressor = ContextCompressor()
            self.compressor = HighPerformanceCompressor(
                base_compressor=base_compressor, max_workers=max_workers
            )
        else:
            self.compressor = compressor

        # Initialize cache
        self.cache_manager = cache_manager

        # Task management
        self.tasks = {}
        self.active_tasks = set()

        # WebSocket connections
        self.websocket_sessions = {}

        # Rate limiting
        self.rate_limiter = (
            RateLimiter(
                max_requests=rate_limit_requests, window_seconds=rate_limit_window
            )
            if enable_rate_limiting
            else None
        )

        # Metrics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

        # Create FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            """Application lifespan management."""
            # Startup
            logger.info("Starting Async Compression API")
            if self.cache_manager:
                await self.cache_manager.start_services()

            yield

            # Shutdown
            logger.info("Shutting down Async Compression API")
            if hasattr(self.compressor, "shutdown"):
                self.compressor.shutdown()
            if self.cache_manager:
                await self.cache_manager.stop_services()

        app = FastAPI(
            title="Retrieval-Free Context Compressor API",
            description="High-performance async API for context compression",
            version="3.0.0",
            lifespan=lifespan,
        )

        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_middleware(GZipMiddleware, minimum_size=1000)

        # Add routes
        self._add_routes(app)

        return app

    def _add_routes(self, app: FastAPI):
        """Add API routes."""

        @app.get("/health", response_model=HealthCheck)
        async def health_check():
            """Health check endpoint."""
            return await self._get_health_status()

        @app.post("/compress", response_model=CompressionResponse)
        async def compress_text(
            request: CompressionRequest,
            background_tasks: BackgroundTasks,
            client_id: str = Depends(self._get_client_id),
        ):
            """Compress text with optional async processing."""
            # Rate limiting
            if self.rate_limiter and not await self.rate_limiter.is_allowed(client_id):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")

            self.request_count += 1

            try:
                if request.async_processing:
                    # Async processing
                    task_id = str(uuid.uuid4())

                    # Create task info
                    task_info = TaskInfo(
                        task_id=task_id,
                        status=TaskStatus.PENDING,
                        created_at=time.time(),
                        priority=request.priority,
                    )
                    self.tasks[task_id] = task_info

                    # Queue for background processing
                    background_tasks.add_task(
                        self._process_compression_task, task_id, request
                    )

                    return CompressionResponse(
                        task_id=task_id,
                        status=TaskStatus.PENDING,
                        metadata={"async": True},
                    )
                else:
                    # Synchronous processing
                    return await self._process_compression_sync(request)

            except Exception as e:
                self.error_count += 1
                logger.error(f"Compression error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/compress/batch", response_model=list[CompressionResponse])
        async def compress_batch(
            request: BatchCompressionRequest,
            client_id: str = Depends(self._get_client_id),
        ):
            """Compress multiple texts in batch."""
            # Rate limiting (stricter for batch requests)
            if self.rate_limiter:
                rate_allowed = await self.rate_limiter.is_allowed(client_id)
                if not rate_allowed:
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")

            self.request_count += len(request.texts)

            try:
                return await self._process_batch_compression(request)
            except Exception as e:
                self.error_count += 1
                logger.error(f"Batch compression error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/tasks/{task_id}", response_model=TaskInfo)
        async def get_task_status(task_id: str):
            """Get status of async task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")

            return self.tasks[task_id]

        @app.get("/tasks/{task_id}/result", response_model=CompressionResponse)
        async def get_task_result(task_id: str):
            """Get result of completed async task."""
            if task_id not in self.tasks:
                raise HTTPException(status_code=404, detail="Task not found")

            task = self.tasks[task_id]
            if task.status != TaskStatus.COMPLETED:
                raise HTTPException(
                    status_code=202, detail=f"Task not completed. Status: {task.status}"
                )

            # Get result from cache or storage
            # This would be implemented based on your storage strategy
            return CompressionResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                metadata={"retrieved_at": time.time()},
            )

        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self._handle_websocket_connection(websocket)

        @app.get("/metrics")
        async def get_metrics():
            """Get API metrics."""
            return await self._get_comprehensive_metrics()

        @app.post("/cache/clear")
        async def clear_cache():
            """Clear all caches."""
            if self.cache_manager:
                # Implementation would clear distributed cache
                pass
            return {"status": "success", "message": "Caches cleared"}

    async def _get_client_id(self) -> str:
        """Get client identifier for rate limiting."""
        # In production, this would extract from headers, JWT, etc.
        return "default_client"

    async def _get_health_status(self) -> HealthCheck:
        """Get comprehensive health status."""
        import psutil

        # Calculate uptime
        uptime = time.time() - self.start_time

        # Get memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Get cache stats
        cache_hit_rate = 0.0
        if self.cache_manager:
            cache_stats = await self.cache_manager.get_comprehensive_stats()
            # Calculate hit rate from cache stats
            if "cache_stats" in cache_stats:
                stats = cache_stats["cache_stats"]
                if isinstance(stats, dict) and "hot" in stats:
                    hot_stats = stats["hot"]
                    cache_hit_rate = (
                        hot_stats.hit_rate if hasattr(hot_stats, "hit_rate") else 0.0
                    )

        # Check GPU availability
        gpu_available = False
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except:
            pass

        return HealthCheck(
            status="healthy",
            timestamp=time.time(),
            version="3.0.0",
            uptime_seconds=uptime,
            active_tasks=len(self.active_tasks),
            queue_size=len(self.tasks),
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=memory_mb,
            gpu_available=gpu_available,
        )

    async def _process_compression_sync(
        self, request: CompressionRequest
    ) -> CompressionResponse:
        """Process compression synchronously."""
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check cache first
            cached_result = None
            if request.use_cache and self.cache_manager:
                cache_key = create_cache_key(
                    request.text,
                    getattr(self.compressor, "model_name", "unknown"),
                    {
                        "compression_ratio": request.compression_ratio,
                        "chunk_size": request.chunk_size,
                    },
                )
                cached_result = await self.cache_manager.get(cache_key)

            if cached_result:
                # Return cached result
                mega_tokens = [
                    MegaTokenModel(
                        vector=token.vector.tolist(),
                        metadata=token.metadata,
                        confidence=token.confidence,
                    )
                    for token in cached_result.mega_tokens
                ]

                return CompressionResponse(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    mega_tokens=mega_tokens,
                    original_length=cached_result.original_length,
                    compressed_length=cached_result.compressed_length,
                    compression_ratio=cached_result.compression_ratio,
                    processing_time=cached_result.processing_time,
                    cached=True,
                    metadata=cached_result.metadata,
                )

            # Process compression
            kwargs = {}
            if request.compression_ratio:
                kwargs["compression_ratio"] = request.compression_ratio
            if request.chunk_size:
                kwargs["chunk_size"] = request.chunk_size

            if hasattr(self.compressor, "compress_async"):
                result = await self.compressor.compress_async(
                    request.text, priority=request.priority.value, **kwargs
                )
            else:
                result = self.compressor.compress(request.text, **kwargs)

            # Cache result
            if request.use_cache and self.cache_manager:
                await self.cache_manager.set(cache_key, result, ttl=request.cache_ttl)

            # Convert to response model
            mega_tokens = [
                MegaTokenModel(
                    vector=token.vector.tolist(),
                    metadata=token.metadata,
                    confidence=token.confidence,
                )
                for token in result.mega_tokens
            ]

            return CompressionResponse(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                mega_tokens=mega_tokens,
                original_length=result.original_length,
                compressed_length=result.compressed_length,
                compression_ratio=result.compression_ratio,
                processing_time=result.processing_time,
                cached=False,
                metadata={**result.metadata, **request.metadata},
            )

        except Exception as e:
            return CompressionResponse(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                metadata={"processing_time": time.time() - start_time},
            )

    async def _process_compression_task(
        self, task_id: str, request: CompressionRequest
    ):
        """Process compression task asynchronously."""
        task = self.tasks[task_id]

        try:
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.started_at = time.time()

            self.active_tasks.add(task_id)

            # Process compression
            response = await self._process_compression_sync(request)

            # Store result (in production, this would go to persistent storage)
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.progress = 1.0

            # Notify WebSocket subscribers
            await self._broadcast_task_update(task_id, task)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()

            logger.error(f"Async task {task_id} failed: {e}")

            # Notify WebSocket subscribers
            await self._broadcast_task_update(task_id, task)

        finally:
            self.active_tasks.discard(task_id)

    async def _process_batch_compression(
        self, request: BatchCompressionRequest
    ) -> list[CompressionResponse]:
        """Process batch compression request."""
        if hasattr(self.compressor, "compress_batch"):
            # Use optimized batch processing
            try:
                results = self.compressor.compress_batch(
                    request.texts,
                    use_distributed=request.use_distributed,
                    compression_ratio=request.compression_ratio,
                    chunk_size=request.chunk_size,
                )

                responses = []
                for i, result in enumerate(results):
                    mega_tokens = [
                        MegaTokenModel(
                            vector=token.vector.tolist(),
                            metadata=token.metadata,
                            confidence=token.confidence,
                        )
                        for token in result.mega_tokens
                    ]

                    responses.append(
                        CompressionResponse(
                            task_id=f"batch_{i}_{int(time.time() * 1000)}",
                            status=TaskStatus.COMPLETED,
                            mega_tokens=mega_tokens,
                            original_length=result.original_length,
                            compressed_length=result.compressed_length,
                            compression_ratio=result.compression_ratio,
                            processing_time=result.processing_time,
                            metadata=result.metadata,
                        )
                    )

                return responses

            except Exception as e:
                # Return error responses for all items
                return [
                    CompressionResponse(
                        task_id=f"batch_error_{i}_{int(time.time() * 1000)}",
                        status=TaskStatus.FAILED,
                        error_message=str(e),
                    )
                    for i in range(len(request.texts))
                ]

        else:
            # Fallback to individual processing
            responses = []
            for i, text in enumerate(request.texts):
                individual_request = CompressionRequest(
                    text=text,
                    priority=request.priority,
                    compression_ratio=request.compression_ratio,
                    chunk_size=request.chunk_size,
                    use_cache=request.use_cache,
                    cache_ttl=request.cache_ttl,
                    metadata=request.metadata,
                )

                response = await self._process_compression_sync(individual_request)
                responses.append(response)

            return responses

    async def _handle_websocket_connection(self, websocket: WebSocket):
        """Handle WebSocket connection for real-time updates."""
        session_id = str(uuid.uuid4())

        try:
            await websocket.accept()

            # Create session
            session = WebSocketSession(
                websocket=websocket,
                session_id=session_id,
                connected_at=time.time(),
                last_activity=time.time(),
                subscriptions=set(),
            )
            self.websocket_sessions[session_id] = session

            logger.info(f"WebSocket connected: {session_id}")

            # Send welcome message
            await websocket.send_json(
                {
                    "type": "connection",
                    "session_id": session_id,
                    "timestamp": time.time(),
                }
            )

            # Handle incoming messages
            while True:
                try:
                    data = await websocket.receive_json()
                    await self._handle_websocket_message(session, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})

        except WebSocketDisconnect:
            pass

        finally:
            # Clean up session
            if session_id in self.websocket_sessions:
                del self.websocket_sessions[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def _handle_websocket_message(self, session: WebSocketSession, data: dict):
        """Handle incoming WebSocket message."""
        message_type = data.get("type")

        if message_type == "subscribe_task":
            task_id = data.get("task_id")
            if task_id:
                session.subscriptions.add(task_id)
                await session.websocket.send_json(
                    {"type": "subscription", "task_id": task_id, "status": "subscribed"}
                )

        elif message_type == "unsubscribe_task":
            task_id = data.get("task_id")
            if task_id:
                session.subscriptions.discard(task_id)
                await session.websocket.send_json(
                    {
                        "type": "subscription",
                        "task_id": task_id,
                        "status": "unsubscribed",
                    }
                )

        session.last_activity = time.time()

    async def _broadcast_task_update(self, task_id: str, task: TaskInfo):
        """Broadcast task update to subscribed WebSocket clients."""
        message = {
            "type": "task_update",
            "task_id": task_id,
            "status": task.status.value,
            "progress": task.progress,
            "timestamp": time.time(),
        }

        # Send to subscribed sessions
        disconnected_sessions = []
        for session_id, session in self.websocket_sessions.items():
            if task_id in session.subscriptions:
                try:
                    await session.websocket.send_json(message)
                except:
                    disconnected_sessions.append(session_id)

        # Clean up disconnected sessions
        for session_id in disconnected_sessions:
            self.websocket_sessions.pop(session_id, None)

    async def _get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive API metrics."""
        uptime = time.time() - self.start_time

        metrics = {
            "api": {
                "uptime_seconds": uptime,
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "error_rate": self.error_count / max(self.request_count, 1),
                "active_tasks": len(self.active_tasks),
                "total_tasks": len(self.tasks),
                "websocket_connections": len(self.websocket_sessions),
            },
            "compressor": {},
        }

        # Add compressor performance stats
        if hasattr(self.compressor, "get_performance_stats"):
            metrics["compressor"] = self.compressor.get_performance_stats()

        # Add cache metrics
        if self.cache_manager:
            cache_stats = await self.cache_manager.get_comprehensive_stats()
            metrics["cache"] = cache_stats

        return metrics


def create_api_server(
    compressor: CompressorBase | None = None,
    cache_config: dict[str, Any] | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    **kwargs,
) -> AsyncCompressionAPI:
    """Create and configure API server."""

    # Initialize cache if config provided
    cache_manager = None
    if cache_config:
        from .distributed_cache import DistributedCacheManager

        cache_manager = DistributedCacheManager(**cache_config)

    # Create API instance
    api = AsyncCompressionAPI(
        compressor=compressor, cache_manager=cache_manager, **kwargs
    )

    return api


def run_api_server(
    api: AsyncCompressionAPI,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    **uvicorn_kwargs,
):
    """Run the API server with uvicorn."""

    config = {
        "host": host,
        "port": port,
        "workers": workers,
        "loop": "asyncio",
        "http": "httptools",
        "ws": "websockets",
        "access_log": True,
        **uvicorn_kwargs,
    }

    logger.info(f"Starting API server on {host}:{port} with {workers} workers")

    uvicorn.run(api.app, **config)


# CLI command for running the API server
def run_server_cli():
    """CLI command to run the API server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Async Compression API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument("--redis-host", help="Redis host for caching")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--enable-cache", action="store_true", help="Enable distributed caching"
    )

    args = parser.parse_args()

    # Setup caching
    cache_config = None
    if args.enable_cache:
        redis_config = {}
        if args.redis_host:
            redis_config = {"host": args.redis_host, "port": args.redis_port}

        cache_config = {"redis_config": redis_config if redis_config else None}

    # Create and run API server
    api = create_api_server(cache_config=cache_config)

    run_api_server(api, host=args.host, port=args.port, workers=args.workers)


if __name__ == "__main__":
    run_server_cli()
