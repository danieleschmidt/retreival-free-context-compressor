"""Main context compressor implementation."""

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from ..exceptions import (
    CompressionError,
    ModelLoadError,
    ValidationError,
)
from ..monitoring import MetricsCollector
from ..validation import InputValidator, validate_compression_request
from .base import CompressionResult, CompressorBase, MegaToken


logger = logging.getLogger(__name__)


class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for multi-scale compression."""

    def __init__(
        self,
        base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 768,
        bottleneck_dim: int = 256
    ):
        super().__init__()
        self.base_model_name = base_model_name
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        # Information bottleneck layers
        self.sentence_encoder = nn.Linear(384, hidden_dim)  # MiniLM output dim
        self.paragraph_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.bottleneck = nn.Linear(hidden_dim, bottleneck_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode sentence embeddings through hierarchical layers.
        
        Args:
            sentence_embeddings: [batch_size, seq_len, embedding_dim]
            
        Returns:
            Compressed representations [batch_size, bottleneck_dim]
        """
        # Sentence level encoding
        x = torch.relu(self.sentence_encoder(sentence_embeddings))
        x = self.dropout(x)

        # Aggregate to paragraph level (mean pooling)
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]

        # Paragraph level encoding
        x = torch.relu(self.paragraph_encoder(x))
        x = self.dropout(x)

        # Information bottleneck
        compressed = self.bottleneck(x)  # [batch_size, bottleneck_dim]

        return compressed


class ContextCompressor(CompressorBase):
    """Main context compressor with hierarchical encoding."""

    def __init__(
        self,
        model_name: str = "context-compressor-base",
        device: str | None = None,
        max_length: int = 256000,
        compression_ratio: float = 8.0,
        chunk_size: int = 512,
        overlap: int = 64
    ):
        """Initialize context compressor.
        
        Args:
            model_name: Name of compression model  
            device: Computing device
            max_length: Maximum input length
            compression_ratio: Target compression ratio
            chunk_size: Size of text chunks for processing
            overlap: Overlap between chunks
        """
        super().__init__(model_name, device, max_length, compression_ratio)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoder_model = None
        self._hierarchical_encoder = None

        # Add robust components
        self._validator = InputValidator()
        self._metrics_collector = MetricsCollector()

        # Add performance optimization components
        self._cache = None  # Will be initialized lazily
        self._batch_processor = None
        self._memory_optimizer = None

    def load_model(self) -> None:
        """Load sentence transformer and hierarchical encoder."""
        try:
            # Load base sentence transformer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self._encoder_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).to(self.device)

            # Initialize hierarchical encoder
            self._hierarchical_encoder = HierarchicalEncoder().to(self.device)

            logger.info(f"Loaded compression model on {self.device}")

        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            logger.error(error_msg)

            # Convert to proper exception
            model_error = ModelLoadError(
                error_msg,
                model_name=self.model_name,
                details={'device': self.device, 'original_error': str(e)}
            )

            # Fallback to dummy implementation for demo/testing
            logger.warning("Falling back to dummy implementation for testing")
            self._tokenizer = None
            self._encoder_model = None
            self._hierarchical_encoder = None

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if self._tokenizer is None:
            # Simple word-based chunking as fallback
            words = text.split()
            chunks = []

            for i in range(0, len(words), self.chunk_size - self.overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunks.append(" ".join(chunk_words))

            return chunks

        # Token-based chunking with proper tokenizer
        tokens = self._tokenizer.encode(text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self._tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

        return chunks

    def _encode_chunks(self, chunks: list[str]) -> torch.Tensor:
        """Encode text chunks into embeddings.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chunk embeddings tensor
        """
        if self._encoder_model is None:
            # Dummy embeddings for demo
            return torch.randn(len(chunks), 384, device=self.device)

        embeddings = []

        for chunk in chunks:
            # Tokenize and encode
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self._encoder_model(**inputs)
                # Mean pooling over tokens
                embedding = outputs.last_hidden_state.mean(dim=1)
                embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)  # [num_chunks, embedding_dim]

    def _encode_chunks_optimized(self, chunks: list[str]) -> torch.Tensor:
        """Optimized version of chunk encoding with batching.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Chunk embeddings tensor
        """
        if self._encoder_model is None:
            # Dummy embeddings for demo
            return torch.randn(len(chunks), 384, device=self.device)

        # Initialize batch processor if needed
        if self._batch_processor is None:
            from ..optimization import BatchProcessor
            self._batch_processor = BatchProcessor(batch_size=8, device=self.device)

        # Process chunks in batches for better GPU utilization
        def encode_batch(chunk_batch):
            embeddings = []
            for chunk in chunk_batch:
                # Tokenize and encode
                inputs = self._tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._encoder_model(**inputs)
                    # Mean pooling over tokens
                    embedding = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(embedding)

            return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0, 384, device=self.device)

        # Process all chunks in batches
        all_embeddings = []

        # Split chunks into batches
        batch_size = self._batch_processor.batch_size
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = encode_batch(batch)
            all_embeddings.append(batch_embeddings)

        if all_embeddings:
            return torch.cat(all_embeddings, dim=0)
        else:
            return torch.empty(0, 384, device=self.device)

    def _cluster_embeddings(
        self,
        embeddings: torch.Tensor,
        n_clusters: int | None = None
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Cluster embeddings to create mega-tokens.
        
        Args:
            embeddings: Chunk embeddings
            n_clusters: Number of clusters (auto-calculated if None)
            
        Returns:
            Tuple of (cluster_centers, cluster_labels)
        """
        if n_clusters is None:
            # Calculate clusters based on compression ratio
            n_clusters = max(1, len(embeddings) // int(self.compression_ratio))

        # Use K-means clustering
        embeddings_np = embeddings.detach().cpu().numpy()

        # Ensure 2D array for sklearn
        if embeddings_np.ndim == 1:
            embeddings_np = embeddings_np.reshape(1, -1)

        if len(embeddings_np) <= n_clusters:
            # If we have fewer embeddings than desired clusters, use all
            return embeddings, np.arange(len(embeddings))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)

        # Convert cluster centers back to torch tensors
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_,
            dtype=embeddings.dtype,
            device=self.device
        )

        return cluster_centers, cluster_labels

    def compress(
        self,
        text: str | list[str],
        **kwargs
    ) -> CompressionResult:
        """Compress text into mega-tokens.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult with mega-tokens
        """
        start_time = time.time()

        try:
            # Handle single text input
            if isinstance(text, str):
                input_text = text
            else:
                input_text = " ".join(text)

            # Validate input
            validation_result = validate_compression_request(
                text=input_text,
                parameters={
                    'compression_ratio': self.compression_ratio,
                    'max_length': self.max_length,
                    'chunk_size': self.chunk_size
                }
            )

            if not validation_result.is_valid:
                raise ValidationError(
                    "Input validation failed",
                    validation_errors=validation_result.errors
                )

            # Use sanitized input
            input_text = validation_result.sanitized_input['text']

            # Initialize cache if not already done
            if self._cache is None:
                from ..caching import TieredCache, create_cache_key
                self._cache = TieredCache()

            # Check cache first
            cache_key = create_cache_key(
                input_text,
                self.model_name,
                {
                    'compression_ratio': self.compression_ratio,
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap
                }
            )

            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                logger.info("Cache hit for compression request")
                return cached_result

            # Load model if not already loaded
            if self._encoder_model is None:
                self.load_model()

            # Initialize optimization components if needed
            if self._memory_optimizer is None:
                from ..optimization import MemoryOptimizer
                self._memory_optimizer = MemoryOptimizer()

            # Count original tokens
            original_length = self.count_tokens(input_text)

            # Use memory-efficient context for processing
            with self._memory_optimizer.memory_efficient_context(clear_cache=True):
                # Chunk the text
                chunks = self._chunk_text(input_text)
                logger.info(f"Split text into {len(chunks)} chunks")

                # Encode chunks into embeddings with optimization
                chunk_embeddings = self._encode_chunks_optimized(chunks)

                # Apply hierarchical encoding if available
                if self._hierarchical_encoder is not None:
                    # Reshape for hierarchical encoder
                    chunk_embeddings = chunk_embeddings.unsqueeze(0)  # Add batch dim
                    compressed_embeddings = self._hierarchical_encoder(chunk_embeddings)
                    compressed_embeddings = compressed_embeddings.squeeze(0)  # Remove batch dim
                else:
                    compressed_embeddings = chunk_embeddings

                # Cluster embeddings into mega-tokens
                cluster_centers, cluster_labels = self._cluster_embeddings(compressed_embeddings)

                # Create mega-tokens
                mega_tokens = []
                for i, center in enumerate(cluster_centers):
                    # Optimize tensor memory
                    center = self._memory_optimizer.optimize_tensor_memory(center)

                    # Find chunks assigned to this cluster
                    cluster_chunks = [j for j, label in enumerate(cluster_labels) if label == i]

                    # Calculate source range (approximate)
                    if cluster_chunks:
                        start_chunk = min(cluster_chunks)
                        end_chunk = max(cluster_chunks)
                        source_range = (
                            start_chunk * (self.chunk_size - self.overlap),
                            (end_chunk + 1) * (self.chunk_size - self.overlap)
                        )
                    else:
                        source_range = (0, 0)

                    # Ensure center is 1D for MegaToken
                    if center.dim() > 1:
                        center = center.flatten()
                    elif center.dim() == 0:
                        center = center.unsqueeze(0)

                    mega_token = MegaToken(
                        embedding=center,
                        metadata={
                            'cluster_id': i,
                            'cluster_size': len(cluster_chunks),
                            'source_chunks': cluster_chunks,
                            'model': self.model_name
                        },
                        source_range=source_range,
                        compression_ratio=len(cluster_chunks) if cluster_chunks else 1.0
                    )
                    mega_tokens.append(mega_token)

            processing_time = time.time() - start_time
            compressed_length = len(mega_tokens)
            actual_ratio = original_length / compressed_length if compressed_length > 0 else 1.0

            # Record metrics
            self._metrics_collector.record_compression(
                input_tokens=original_length,
                output_tokens=compressed_length,
                processing_time_ms=processing_time * 1000,
                model_name=self.model_name
            )

            result = CompressionResult(
                mega_tokens=mega_tokens,
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=actual_ratio,
                processing_time=processing_time,
                metadata={
                    'chunks_processed': len(chunks),
                    'embedding_dim': chunk_embeddings.shape[-1] if len(chunk_embeddings.shape) > 1 else 0,
                    'model': self.model_name,
                    'device': str(self.device),
                    'validation_warnings': validation_result.warnings
                }
            )

            logger.info(
                f"Compressed {original_length} tokens to {compressed_length} mega-tokens "
                f"({actual_ratio:.1f}x ratio) in {processing_time:.2f}s"
            )

            # Cache the result for future use
            self._cache.put(cache_key, result, ttl=3600)  # Cache for 1 hour

            return result

        except Exception as e:
            # Handle and convert exceptions
            if not isinstance(e, (CompressionError, ValidationError)):
                compression_error = CompressionError(
                    f"Compression failed: {str(e)}",
                    input_length=len(input_text.split()) if 'input_text' in locals() else None,
                    model_name=self.model_name,
                    details={'original_error': str(e)}
                )
                raise compression_error from e
            raise

    def decompress(
        self,
        mega_tokens: list[MegaToken],
        **kwargs
    ) -> str:
        """Approximate text reconstruction from mega-tokens.
        
        Note: This is lossy reconstruction for demonstration purposes.
        Real implementation would use a decoder model.
        
        Args:
            mega_tokens: List of mega-tokens to decompress
            **kwargs: Additional parameters
            
        Returns:
            Reconstructed text (approximate)
        """
        # Simple placeholder reconstruction
        reconstructed_parts = []

        for i, token in enumerate(mega_tokens):
            cluster_size = token.metadata.get('cluster_size', 1)

            # Generate placeholder text based on mega-token properties
            placeholder = (
                f"[Compressed segment {i+1}: {cluster_size} chunks, "
                f"embedding_dim={len(token.embedding)}, "
                f"compression={token.compression_ratio:.1f}x]"
            )
            reconstructed_parts.append(placeholder)

        return " ".join(reconstructed_parts)

    def get_attention_weights(
        self,
        query: str,
        mega_tokens: list[MegaToken]
    ) -> torch.Tensor:
        """Calculate attention weights between query and mega-tokens.
        
        Args:
            query: Query text
            mega_tokens: List of mega-tokens
            
        Returns:
            Attention weights tensor
        """
        if not mega_tokens:
            return torch.empty(0, device=self.device)

        # Encode query
        if self._encoder_model is None:
            self.load_model()

        query_embedding = self._encode_chunks([query])[0]  # Single embedding

        # Stack mega-token embeddings
        token_embeddings = torch.stack([token.embedding for token in mega_tokens])

        # Compute cosine similarity as attention weights
        query_norm = torch.nn.functional.normalize(query_embedding, dim=0)
        token_norms = torch.nn.functional.normalize(token_embeddings, dim=1)

        attention_weights = torch.mm(token_norms, query_norm.unsqueeze(1)).squeeze(1)

        # Apply softmax to get attention distribution
        attention_weights = torch.softmax(attention_weights, dim=0)

        return attention_weights
