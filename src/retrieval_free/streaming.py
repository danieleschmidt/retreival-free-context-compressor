"""Streaming compression for infinite contexts."""

import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .core import CompressorBase, CompressionResult, MegaToken
from .observability import monitor_performance


class StreamingCompressor(CompressorBase):
    """Streaming compressor for continuous/infinite contexts."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        window_size: int = 32000,
        compression_ratio: float = 8.0,
        prune_threshold: float = 0.1,
        max_windows: int = 10
    ):
        super().__init__(model_name)
        self.window_size = window_size
        self.compression_ratio = compression_ratio
        self.prune_threshold = prune_threshold
        self.max_windows = max_windows
        
        # Sliding window state
        self.windows: Deque[Tuple[str, List[MegaToken], float]] = deque(maxlen=max_windows)
        self.global_mega_tokens: List[MegaToken] = []
        self.total_processed = 0
        
    @monitor_performance
    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text and update streaming state."""
        # For streaming, this adds to the current context
        return self.add_chunk(text)
    
    def decompress(self, mega_tokens: List[MegaToken], **kwargs) -> str:
        """Decompress mega-tokens to approximate text."""
        if not mega_tokens:
            return ""
        
        parts = []
        for token in mega_tokens:
            if "source_text" in token.metadata:
                parts.append(token.metadata["source_text"])
        
        return " ".join(parts)
    
    def add_chunk(self, chunk: str) -> CompressionResult:
        """Add a new chunk to the streaming context."""
        start_time = time.time()
        
        # Check if we need to prune old windows
        if len(self.windows) >= self.max_windows:
            self._prune_oldest_window()
        
        # Compress the new chunk
        from .core import ContextCompressor
        temp_compressor = ContextCompressor(
            model_name=self.model_name,
            compression_ratio=self.compression_ratio
        )
        
        chunk_result = temp_compressor.compress(chunk)
        chunk_timestamp = time.time()
        
        # Add to sliding window
        self.windows.append((chunk, chunk_result.mega_tokens, chunk_timestamp))
        self.total_processed += len(chunk)
        
        # Update global compressed representation
        self._update_global_representation()
        
        # Check if we should prune obsolete information
        if self.should_prune():
            self.prune_obsolete()
        
        processing_time = time.time() - start_time
        
        return CompressionResult(
            mega_tokens=self.global_mega_tokens.copy(),
            original_length=self.total_processed,
            compressed_length=len(self.global_mega_tokens),
            compression_ratio=self.total_processed / max(len(self.global_mega_tokens), 1),
            processing_time=processing_time,
            metadata={
                "streaming": True,
                "active_windows": len(self.windows),
                "total_processed": self.total_processed,
                "pruning_enabled": True
            }
        )
    
    def should_prune(self) -> bool:
        """Check if obsolete information should be pruned."""
        if len(self.global_mega_tokens) < 100:  # Don't prune if we have few tokens
            return False
        
        # Prune based on time and relevance
        current_time = time.time()
        oldest_window_time = min(
            timestamp for _, _, timestamp in self.windows
        ) if self.windows else current_time
        
        # Prune if oldest content is more than 1 hour old
        return (current_time - oldest_window_time) > 3600
    
    def prune_obsolete(self) -> int:
        """Remove obsolete information from the compressed representation."""
        if not self.global_mega_tokens:
            return 0
        
        initial_count = len(self.global_mega_tokens)
        current_time = time.time()
        
        # Remove tokens with low confidence or old timestamps
        filtered_tokens = []
        for token in self.global_mega_tokens:
            # Check confidence threshold
            if token.confidence < self.prune_threshold:
                continue
            
            # Check age (if timestamp available)
            token_age = current_time - token.metadata.get("timestamp", current_time)
            if token_age > 7200:  # 2 hours
                continue
            
            filtered_tokens.append(token)
        
        self.global_mega_tokens = filtered_tokens
        pruned_count = initial_count - len(filtered_tokens)
        
        return pruned_count
    
    def get_current_context(self) -> List[MegaToken]:
        """Get the current compressed context."""
        return self.global_mega_tokens.copy()
    
    def get_context_summary(self) -> Dict:
        """Get summary of current streaming context."""
        return {
            "total_mega_tokens": len(self.global_mega_tokens),
            "active_windows": len(self.windows),
            "total_processed_chars": self.total_processed,
            "effective_compression_ratio": self.total_processed / max(len(self.global_mega_tokens), 1),
            "oldest_window_age": self._get_oldest_window_age(),
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def reset(self):
        """Reset the streaming state."""
        self.windows.clear()
        self.global_mega_tokens.clear()
        self.total_processed = 0
    
    def _prune_oldest_window(self):
        """Remove the oldest window from state."""
        if self.windows:
            oldest_chunk, oldest_tokens, _ = self.windows.popleft()
            # Remove corresponding tokens from global representation
            self._remove_tokens_from_global(oldest_tokens)
    
    def _update_global_representation(self):
        """Update the global compressed representation."""
        # Combine all mega-tokens from all windows
        all_tokens = []
        for _, tokens, timestamp in self.windows:
            for token in tokens:
                # Add timestamp to metadata
                token_copy = MegaToken(
                    vector=token.vector.copy(),
                    metadata={**token.metadata, "timestamp": timestamp},
                    confidence=token.confidence
                )
                all_tokens.append(token_copy)
        
        # Apply global compression if we have too many tokens
        target_global_size = max(50, int(self.total_processed / (self.compression_ratio * 100)))
        
        if len(all_tokens) > target_global_size:
            # Use similarity-based clustering to reduce global token count
            all_tokens = self._cluster_similar_tokens(all_tokens, target_global_size)
        
        self.global_mega_tokens = all_tokens
    
    def _remove_tokens_from_global(self, tokens_to_remove: List[MegaToken]):
        """Remove specific tokens from global representation."""
        # Simple implementation: remove tokens with similar vectors
        removal_vectors = [token.vector for token in tokens_to_remove]
        
        filtered_global = []
        for global_token in self.global_mega_tokens:
            should_keep = True
            for removal_vector in removal_vectors:
                # Check cosine similarity
                similarity = np.dot(global_token.vector, removal_vector) / (
                    np.linalg.norm(global_token.vector) * np.linalg.norm(removal_vector)
                )
                if similarity > 0.95:  # Very similar, likely the same token
                    should_keep = False
                    break
            
            if should_keep:
                filtered_global.append(global_token)
        
        self.global_mega_tokens = filtered_global
    
    def _cluster_similar_tokens(
        self, 
        tokens: List[MegaToken], 
        target_size: int
    ) -> List[MegaToken]:
        """Cluster similar tokens to reduce count."""
        if len(tokens) <= target_size:
            return tokens
        
        try:
            from sklearn.cluster import KMeans
            
            # Extract vectors for clustering
            vectors = np.array([token.vector for token in tokens])
            
            # Cluster tokens
            kmeans = KMeans(n_clusters=target_size, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vectors)
            
            # Create representative tokens for each cluster
            clustered_tokens = []
            for cluster_id in range(target_size):
                cluster_mask = cluster_labels == cluster_id
                cluster_tokens = [tokens[i] for i in range(len(tokens)) if cluster_mask[i]]
                
                if cluster_tokens:
                    # Use the token with highest confidence as representative
                    representative = max(cluster_tokens, key=lambda t: t.confidence)
                    
                    # Average the vectors for better representation
                    avg_vector = np.mean([t.vector for t in cluster_tokens], axis=0)
                    
                    clustered_token = MegaToken(
                        vector=avg_vector,
                        metadata={
                            **representative.metadata,
                            "cluster_size": len(cluster_tokens),
                            "cluster_id": cluster_id
                        },
                        confidence=np.mean([t.confidence for t in cluster_tokens])
                    )
                    clustered_tokens.append(clustered_token)
            
            return clustered_tokens
            
        except ImportError:
            # Fallback: simple sampling
            step = len(tokens) // target_size
            return [tokens[i] for i in range(0, len(tokens), max(step, 1))]
    
    def _get_oldest_window_age(self) -> float:
        """Get age of oldest window in seconds."""
        if not self.windows:
            return 0.0
        
        current_time = time.time()
        oldest_time = min(timestamp for _, _, timestamp in self.windows)
        return current_time - oldest_time
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.global_mega_tokens:
            return 0.0
        
        # Rough estimate: vector size * number of tokens * 4 bytes per float
        vector_size = len(self.global_mega_tokens[0].vector)
        total_floats = len(self.global_mega_tokens) * vector_size
        bytes_used = total_floats * 4  # 4 bytes per float32
        
        return bytes_used / (1024 * 1024)  # Convert to MB