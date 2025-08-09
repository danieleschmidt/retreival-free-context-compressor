"""Streaming compressor for handling infinite context lengths."""

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .core.base import CompressionResult, CompressorBase, MegaToken
from .exceptions import CompressionError
from .optimization import BatchProcessor, MemoryOptimizer
from .validation import InputValidator


logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Represents a chunk of streaming data."""
    data: str
    chunk_id: int
    timestamp: float
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        """Get chunk size in characters."""
        return len(self.data)


@dataclass
class StreamState:
    """Maintains state for streaming compression."""
    active_mega_tokens: list[MegaToken]
    buffer: deque
    processed_chunks: int
    total_characters: int
    compression_stats: dict[str, Any]

    def __post_init__(self):
        """Initialize default values."""
        if not hasattr(self, 'compression_stats') or self.compression_stats is None:
            self.compression_stats = {
                'total_compression_ratio': 1.0,
                'avg_processing_time': 0.0,
                'memory_usage_mb': 0.0
            }


class StreamingCompressor(CompressorBase):
    """Compressor optimized for streaming/infinite context scenarios."""

    def __init__(
        self,
        model_name: str = "context-compressor-base",
        device: str | None = None,
        max_length: int = 1_000_000,  # Much higher for streaming
        compression_ratio: float = 8.0,
        buffer_size: int = 10000,
        overlap_ratio: float = 0.1,
        sliding_window_size: int = 5,
        auto_flush_threshold: int = 50000
    ):
        """Initialize streaming compressor.
        
        Args:
            model_name: Name of compression model
            device: Computing device  
            max_length: Maximum total context length to maintain
            compression_ratio: Target compression ratio
            buffer_size: Size of streaming buffer in characters
            overlap_ratio: Ratio of overlap between chunks (0.0-0.5)
            sliding_window_size: Number of recent mega-tokens to keep active
            auto_flush_threshold: Automatically compress when buffer exceeds this size
        """
        super().__init__(model_name, device, max_length, compression_ratio)

        self.buffer_size = buffer_size
        self.overlap_ratio = min(0.5, max(0.0, overlap_ratio))
        self.sliding_window_size = sliding_window_size
        self.auto_flush_threshold = auto_flush_threshold

        # Streaming state
        self.stream_state = StreamState(
            active_mega_tokens=[],
            buffer=deque(),
            processed_chunks=0,
            total_characters=0,
            compression_stats={}
        )

        # Thread safety
        self._lock = threading.RLock()
        self._processing_queue = queue.Queue()
        self._is_streaming = False

        # Optimization components
        self._memory_optimizer = MemoryOptimizer()
        self._batch_processor = BatchProcessor(batch_size=4)  # Smaller batches for streaming
        self._validator = InputValidator(max_text_length=buffer_size * 2)

        # Background processing
        self._background_thread = None
        self._stop_processing = threading.Event()

    def start_streaming(self) -> None:
        """Start background streaming processing."""
        with self._lock:
            if self._is_streaming:
                logger.warning("Streaming already active")
                return

            self._is_streaming = True
            self._stop_processing.clear()

            # Start background processing thread
            self._background_thread = threading.Thread(
                target=self._background_processor,
                name="StreamingCompressor-Background",
                daemon=True
            )
            self._background_thread.start()

            logger.info("Started streaming compression")

    def stop_streaming(self) -> None:
        """Stop background streaming processing."""
        with self._lock:
            if not self._is_streaming:
                return

            self._is_streaming = False
            self._stop_processing.set()

            # Flush remaining data
            self._flush_buffer()

            # Wait for background thread to finish
            if self._background_thread and self._background_thread.is_alive():
                self._background_thread.join(timeout=5.0)

            logger.info("Stopped streaming compression")

    def feed_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Feed text into the streaming buffer.
        
        Args:
            text: Text to add to stream
            metadata: Optional metadata for this text chunk
        """
        if not self._is_streaming:
            raise CompressionError("Streaming not active. Call start_streaming() first.")

        # Validate input
        validation_result = self._validator.validate_text_input(text)
        if not validation_result.is_valid:
            raise CompressionError(
                f"Invalid streaming input: {validation_result.errors}",
                details={'validation_errors': validation_result.errors}
            )

        # Create stream chunk
        chunk = StreamChunk(
            data=validation_result.sanitized_input['text'],
            chunk_id=self.stream_state.processed_chunks,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        # Add to processing queue
        try:
            self._processing_queue.put_nowait(chunk)
        except queue.Full:
            logger.warning("Processing queue full, dropping oldest chunk")
            try:
                self._processing_queue.get_nowait()  # Remove oldest
                self._processing_queue.put_nowait(chunk)  # Add new
            except queue.Empty:
                pass

    def get_compressed_context(self) -> list[MegaToken]:
        """Get current compressed context.
        
        Returns:
            List of active mega-tokens representing compressed context
        """
        with self._lock:
            return self.stream_state.active_mega_tokens.copy()

    def get_streaming_stats(self) -> dict[str, Any]:
        """Get streaming compression statistics.
        
        Returns:
            Dictionary with streaming statistics
        """
        with self._lock:
            return {
                'active_mega_tokens': len(self.stream_state.active_mega_tokens),
                'buffer_size': len(self.stream_state.buffer),
                'processed_chunks': self.stream_state.processed_chunks,
                'total_characters': self.stream_state.total_characters,
                'is_streaming': self._is_streaming,
                'queue_size': self._processing_queue.qsize(),
                **self.stream_state.compression_stats
            }

    def _background_processor(self) -> None:
        """Background thread for processing streaming chunks."""
        logger.debug("Background streaming processor started")

        while not self._stop_processing.is_set():
            try:
                # Get chunk from queue with timeout
                chunk = self._processing_queue.get(timeout=1.0)
                self._process_streaming_chunk(chunk)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in background processing: {e}")
                continue

        logger.debug("Background streaming processor stopped")

    def _process_streaming_chunk(self, chunk: StreamChunk) -> None:
        """Process a single streaming chunk.
        
        Args:
            chunk: Stream chunk to process
        """
        with self._lock:
            # Add to buffer
            self.stream_state.buffer.append(chunk)
            self.stream_state.total_characters += chunk.size

            # Check if we should compress
            buffer_text_size = sum(c.size for c in self.stream_state.buffer)

            if (buffer_text_size >= self.auto_flush_threshold or
                len(self.stream_state.buffer) >= self.buffer_size):

                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush current buffer and compress accumulated text."""
        if not self.stream_state.buffer:
            return

        start_time = time.time()

        # Combine buffer chunks into text
        combined_text = ""
        chunk_boundaries = []
        current_pos = 0

        for chunk in self.stream_state.buffer:
            chunk_boundaries.append((current_pos, current_pos + chunk.size, chunk.chunk_id))
            combined_text += chunk.data
            current_pos += chunk.size

        # Compress combined text
        try:
            with self._memory_optimizer.memory_efficient_context():
                # Use parent class compression method
                result = super().compress(combined_text)

                # Convert to streaming mega-tokens with chunk tracking
                streaming_tokens = []
                for i, mega_token in enumerate(result.mega_tokens):
                    # Find which chunks this mega-token spans
                    start_pos, end_pos = mega_token.source_range
                    spanned_chunks = [
                        chunk_id for chunk_start, chunk_end, chunk_id in chunk_boundaries
                        if not (chunk_end <= start_pos or chunk_start >= end_pos)
                    ]

                    # Create enhanced metadata
                    enhanced_metadata = {
                        **mega_token.metadata,
                        'chunk_ids': spanned_chunks,
                        'stream_timestamp': time.time(),
                        'buffer_position': i
                    }

                    streaming_token = MegaToken(
                        embedding=mega_token.embedding,
                        metadata=enhanced_metadata,
                        source_range=mega_token.source_range,
                        compression_ratio=mega_token.compression_ratio
                    )

                    streaming_tokens.append(streaming_token)

                # Update sliding window of active tokens
                self._update_sliding_window(streaming_tokens)

                # Update statistics
                processing_time = time.time() - start_time
                self._update_streaming_stats(result, processing_time)

                # Clear buffer
                self.stream_state.processed_chunks += len(self.stream_state.buffer)
                self.stream_state.buffer.clear()

                logger.debug(
                    f"Flushed buffer: {result.original_length} chars -> "
                    f"{len(streaming_tokens)} mega-tokens in {processing_time:.2f}s"
                )

        except Exception as e:
            logger.error(f"Buffer flush failed: {e}")
            # Clear buffer anyway to prevent infinite retries
            self.stream_state.buffer.clear()
            raise CompressionError(f"Streaming compression failed: {e}")

    def _update_sliding_window(self, new_tokens: list[MegaToken]) -> None:
        """Update the sliding window of active mega-tokens.
        
        Args:
            new_tokens: New mega-tokens to add to the window
        """
        # Add new tokens to active set
        self.stream_state.active_mega_tokens.extend(new_tokens)

        # Keep only the most recent tokens within sliding window
        if len(self.stream_state.active_mega_tokens) > self.sliding_window_size:
            # Remove oldest tokens
            excess = len(self.stream_state.active_mega_tokens) - self.sliding_window_size
            removed_tokens = self.stream_state.active_mega_tokens[:excess]
            self.stream_state.active_mega_tokens = self.stream_state.active_mega_tokens[excess:]

            logger.debug(f"Sliding window: removed {len(removed_tokens)} old tokens")

    def _update_streaming_stats(self, result: CompressionResult, processing_time: float) -> None:
        """Update streaming compression statistics.
        
        Args:
            result: Compression result
            processing_time: Processing time for this flush
        """
        current_stats = self.stream_state.compression_stats

        # Update running averages
        if 'total_compression_ratio' not in current_stats:
            current_stats['total_compression_ratio'] = result.compression_ratio
            current_stats['avg_processing_time'] = processing_time
        else:
            # Exponential moving average
            alpha = 0.1
            current_stats['total_compression_ratio'] = (
                alpha * result.compression_ratio +
                (1 - alpha) * current_stats['total_compression_ratio']
            )
            current_stats['avg_processing_time'] = (
                alpha * processing_time +
                (1 - alpha) * current_stats['avg_processing_time']
            )

        # Memory usage
        memory_stats = self._memory_optimizer.get_memory_stats()
        current_stats['memory_usage_mb'] = memory_stats['current_mb']

    # Implement abstract methods from CompressorBase
    def load_model(self) -> None:
        """Load compression model for streaming."""
        # Delegate to parent implementation but with streaming optimizations
        try:
            super().load_model()

            # Apply streaming-specific optimizations
            if self._encoder_model is not None:
                from .optimization import model_optimizer
                self._encoder_model = model_optimizer.optimize_model_inference(self._encoder_model)

        except Exception as e:
            logger.error(f"Failed to load streaming model: {e}")
            raise CompressionError(f"Streaming model load failed: {e}")

    def compress(
        self,
        text: str | list[str],
        **kwargs
    ) -> CompressionResult:
        """Compress text using streaming approach.
        
        This method processes text in streaming fashion if streaming is active,
        otherwise falls back to batch compression.
        
        Args:
            text: Input text to compress
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult
        """
        if self._is_streaming:
            # For streaming mode, add to stream and return current state
            if isinstance(text, list):
                for t in text:
                    self.feed_text(t)
            else:
                self.feed_text(text)

            # Return current compressed state
            mega_tokens = self.get_compressed_context()

            return CompressionResult(
                mega_tokens=mega_tokens,
                original_length=self.stream_state.total_characters,
                compressed_length=len(mega_tokens),
                compression_ratio=self.stream_state.compression_stats.get('total_compression_ratio', 1.0),
                processing_time=self.stream_state.compression_stats.get('avg_processing_time', 0.0),
                metadata={
                    'streaming_mode': True,
                    'active_chunks': self.stream_state.processed_chunks,
                    **self.get_streaming_stats()
                }
            )
        else:
            # Use parent class implementation for non-streaming compression
            return super().compress(text, **kwargs)

    def decompress(
        self,
        mega_tokens: list[MegaToken],
        **kwargs
    ) -> str:
        """Decompress mega-tokens to text (approximate reconstruction).
        
        Args:
            mega_tokens: Mega-tokens to decompress
            **kwargs: Additional parameters
            
        Returns:
            Reconstructed text
        """
        # Simple reconstruction for streaming context
        parts = []

        for i, token in enumerate(mega_tokens):
            if 'chunk_ids' in token.metadata:
                chunk_info = f"chunks {token.metadata['chunk_ids']}"
            else:
                chunk_info = f"chunk range {token.source_range}"

            timestamp = token.metadata.get('stream_timestamp', 0)

            part = (
                f"[StreamSegment {i+1}: {chunk_info}, "
                f"ratio={token.compression_ratio:.1f}x, "
                f"ts={timestamp:.0f}]"
            )
            parts.append(part)

        return " ".join(parts)

    def reset_stream(self) -> None:
        """Reset streaming state while keeping model loaded."""
        with self._lock:
            was_streaming = self._is_streaming

            if was_streaming:
                self.stop_streaming()

            # Clear state
            self.stream_state = StreamState(
                active_mega_tokens=[],
                buffer=deque(),
                processed_chunks=0,
                total_characters=0,
                compression_stats={}
            )

            # Clear queue
            while not self._processing_queue.empty():
                try:
                    self._processing_queue.get_nowait()
                except queue.Empty:
                    break

            if was_streaming:
                self.start_streaming()

            logger.info("Stream state reset")

    def __enter__(self):
        """Context manager entry."""
        self.start_streaming()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_streaming()
