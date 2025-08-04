"""Streaming compression for infinite contexts."""

import time
import logging
from typing import List, Dict, Any, Optional, Deque
from collections import deque
import torch

from .core.base import CompressorBase, MegaToken, CompressionResult
from .core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


class StreamingCompressor(CompressorBase):
    """Compressor for continuous/infinite contexts with automatic pruning."""
    
    def __init__(
        self,
        model_name: str = "rfcc-streaming",
        device: Optional[str] = None,
        window_size: int = 32000,
        compression_ratio: float = 8.0,
        prune_threshold: float = 0.1,
        max_history: int = 1000
    ):
        """Initialize streaming compressor.
        
        Args:
            model_name: Base compression model
            device: Computing device
            window_size: Size of compression window
            compression_ratio: Target compression ratio
            prune_threshold: Threshold for pruning obsolete tokens
            max_history: Maximum history to maintain
        """
        super().__init__(model_name, device, window_size, compression_ratio)
        self.window_size = window_size
        self.prune_threshold = prune_threshold
        self.max_history = max_history
        
        # Internal state
        self._base_compressor = ContextCompressor(
            model_name=model_name,
            device=device,
            compression_ratio=compression_ratio,
            chunk_size=512,
            overlap=64
        )
        self._history: Deque[MegaToken] = deque(maxlen=max_history)
        self._current_window = ""
        self._total_processed = 0
        self._last_access_times: Dict[int, float] = {}
        
    def load_model(self) -> None:
        """Load base compression model."""
        self._base_compressor.load_model()
        
    def add_chunk(self, text_chunk: str) -> List[MegaToken]:
        """Add new text chunk to streaming context.
        
        Args:
            text_chunk: New text to add
            
        Returns:
            Current mega-tokens including new content
        """
        # Add to current window
        self._current_window += " " + text_chunk
        self._total_processed += self.count_tokens(text_chunk)
        
        # Check if window exceeds size limit
        if self.count_tokens(self._current_window) > self.window_size:
            # Compress current window
            result = self._base_compressor.compress(self._current_window)
            
            # Add new mega-tokens to history
            current_time = time.time()
            for token in result.mega_tokens:
                token.metadata["timestamp"] = current_time
                token.metadata["access_count"] = 1
                self._history.append(token)
                self._last_access_times[id(token)] = current_time
            
            # Reset window for next batch
            self._current_window = ""
            
            logger.info(f"Compressed window: {len(result.mega_tokens)} mega-tokens added")
        
        return list(self._history)
    
    def compress(
        self, 
        text: str, 
        **kwargs
    ) -> CompressionResult:
        """Compress text in streaming fashion.
        
        Args:
            text: Input text to compress
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult with streaming state
        """
        start_time = time.time() 
        
        # Process text in chunks
        chunks = self._split_into_chunks(text)
        
        for chunk in chunks:
            self.add_chunk(chunk)
        
        # Create result from current state
        all_tokens = list(self._history)
        processing_time = time.time() - start_time
        
        result = CompressionResult(
            mega_tokens=all_tokens,
            original_length=self.count_tokens(text),
            compressed_length=len(all_tokens),
            compression_ratio=self._total_processed / len(all_tokens) if all_tokens else 1.0,
            processing_time=processing_time,
            metadata={
                "streaming": True,
                "total_processed": self._total_processed,
                "window_size": self.window_size,
                "history_length": len(self._history)
            }
        )
        
        return result
    
    def decompress(
        self, 
        mega_tokens: List[MegaToken],
        **kwargs
    ) -> str:
        """Reconstruct text from streaming mega-tokens."""
        return self._base_compressor.decompress(mega_tokens, **kwargs)
    
    def should_prune(self) -> bool:
        """Check if pruning is needed.
        
        Returns:
            True if pruning should be performed
        """
        if len(self._history) < self.max_history // 2:
            return False
        
        # Check for old, unused tokens
        current_time = time.time()
        old_tokens = 0
        
        for token in self._history:
            last_access = self._last_access_times.get(id(token), current_time)
            if current_time - last_access > 3600:  # 1 hour threshold
                old_tokens += 1
        
        return old_tokens > len(self._history) * self.prune_threshold
    
    def prune_obsolete(self) -> int:
        """Remove obsolete mega-tokens from history.
        
        Returns:
            Number of tokens pruned
        """
        if not self.should_prune():
            return 0
        
        current_time = time.time()
        original_length = len(self._history)
        
        # Create new deque with non-obsolete tokens
        pruned_history = deque(maxlen=self.max_history)
        
        for token in self._history:
            last_access = self._last_access_times.get(id(token), current_time)
            access_count = token.metadata.get("access_count", 1)
            
            # Keep token if recently accessed or frequently used
            if (current_time - last_access < 3600 or  # Less than 1 hour old
                access_count > 5 or  # Frequently accessed
                token.metadata.get("importance", 0) > 0.8):  # High importance
                pruned_history.append(token)
            else:
                # Remove from access tracking
                self._last_access_times.pop(id(token), None)
        
        self._history = pruned_history
        pruned_count = original_length - len(self._history)
        
        logger.info(f"Pruned {pruned_count} obsolete mega-tokens")
        return pruned_count
    
    def query(
        self, 
        question: str, 
        top_k: int = 10
    ) -> List[MegaToken]:
        """Query streaming context for relevant mega-tokens.
        
        Args:
            question: Query text
            top_k: Number of top tokens to return
            
        Returns:
            Most relevant mega-tokens
        """
        if not self._history:
            return []
        
        # Calculate attention weights
        attention_weights = self._base_compressor.get_attention_weights(
            question, list(self._history)
        )
        
        # Update access times and counts for queried tokens
        current_time = time.time()
        for i, token in enumerate(self._history):
            self._last_access_times[id(token)] = current_time
            token.metadata["access_count"] = token.metadata.get("access_count", 0) + 1
        
        # Get top-k tokens
        if len(attention_weights) > 0:
            top_indices = torch.topk(attention_weights, min(top_k, len(attention_weights))).indices
            top_tokens = [list(self._history)[i] for i in top_indices]
        else:
            # Fallback: return most recent tokens
            top_tokens = list(self._history)[-top_k:]
        
        return top_tokens
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming compression statistics.
        
        Returns:
            Dictionary with streaming stats
        """
        current_time = time.time()
        
        # Calculate age statistics
        ages = []
        access_counts = []
        
        for token in self._history:
            timestamp = token.metadata.get("timestamp", current_time)
            ages.append(current_time - timestamp)
            access_counts.append(token.metadata.get("access_count", 1))
        
        return {
            "total_tokens": len(self._history),
            "total_processed": self._total_processed,
            "compression_ratio": self._total_processed / len(self._history) if self._history else 0,
            "avg_token_age": sum(ages) / len(ages) if ages else 0,
            "avg_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
            "window_utilization": self.count_tokens(self._current_window) / self.window_size,
            "memory_usage_mb": len(self._history) * 0.016  # Rough estimate
        }
    
    def reset(self) -> None:
        """Reset streaming state."""
        self._history.clear()
        self._current_window = ""
        self._total_processed = 0
        self._last_access_times.clear()
        logger.info("Reset streaming compressor state")
    
    def _split_into_chunks(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into processing chunks.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks