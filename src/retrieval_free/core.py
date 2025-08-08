"""Core compression interfaces and implementations."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .exceptions import CompressionError, ModelError, ValidationError
from .observability import log_compression_operation, monitor_performance
from .validation import ParameterValidator, validate_input, validate_parameters


@dataclass
class MegaToken:
    """Represents a compressed token containing dense semantic information."""

    vector: np.ndarray  # Dense representation
    metadata: dict[str, Any]  # Source information, compression ratio, etc.
    confidence: float  # Quality/confidence score

    def __post_init__(self):
        # Validate and convert vector
        if not isinstance(self.vector, np.ndarray):
            try:
                self.vector = np.array(self.vector)
            except (ValueError, TypeError) as e:
                raise ValidationError(f"Invalid vector data: {e}", field="vector")

        # Validate vector properties
        if self.vector.size == 0:
            raise ValidationError("Vector cannot be empty", field="vector")

        if not np.isfinite(self.vector).all():
            raise ValidationError("Vector contains non-finite values", field="vector")

        # Validate confidence
        validation_result = ParameterValidator.validate_confidence(self.confidence)
        if not validation_result.is_valid:
            raise ValidationError(
                f"Invalid confidence value: {'; '.join(validation_result.errors)}",
                field="confidence",
            )

        # Validate metadata
        if not isinstance(self.metadata, dict):
            raise ValidationError("Metadata must be a dictionary", field="metadata")


@dataclass
class CompressionResult:
    """Result of a compression operation."""

    mega_tokens: list[MegaToken]
    original_length: int
    compressed_length: int
    compression_ratio: float
    processing_time: float
    metadata: dict[str, Any]

    @property
    def effective_compression(self) -> float:
        """Calculate effective compression considering vector dimensions."""
        if not self.mega_tokens:
            return 1.0

        vector_size = len(self.mega_tokens[0].vector) if self.mega_tokens else 0
        total_compressed_size = len(self.mega_tokens) * vector_size
        return self.original_length / max(total_compressed_size, 1)


class CompressorBase(ABC):
    """Abstract base class for all compressors with robust error handling."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError(
                "Model name must be a non-empty string", field="model_name"
            )

        self.model_name = model_name.strip()
        self.device = self._get_device()
        self._load_model()

    def _get_device(self) -> torch.device:
        """Safely determine the best available device."""
        try:
            if torch.cuda.is_available():
                # Check if CUDA device is actually accessible
                torch.cuda.current_device()
                return torch.device("cuda")
        except Exception as e:
            # CUDA might be available but not working properly
            import warnings

            warnings.warn(
                f"CUDA available but not accessible: {e}. Falling back to CPU."
            )

        return torch.device("cpu")

    def _load_model(self):
        """Load the underlying model with comprehensive error handling."""
        try:
            # Try sentence-transformers first
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(
                    self.model_name, device=str(self.device)
                )
                self.model_type = "sentence_transformer"
                return
            except ImportError:
                pass  # Fall through to transformers
            except Exception as e:
                raise ModelError(
                    f"Failed to load SentenceTransformer model '{self.model_name}': {e}",
                    model_name=self.model_name,
                    model_type="sentence_transformer",
                )

            # Fallback to transformers
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model_type = "transformer"
            except Exception as e:
                raise ModelError(
                    f"Failed to load transformer model '{self.model_name}': {e}",
                    model_name=self.model_name,
                    model_type="transformer",
                )

        except Exception as e:
            if isinstance(e, ModelError):
                raise
            raise ModelError(
                f"Unexpected error loading model '{self.model_name}': {e}",
                model_name=self.model_name,
            )

    @abstractmethod
    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text into mega-tokens."""
        pass

    @abstractmethod
    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Decompress mega-tokens back to text (approximate)."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if hasattr(self, "tokenizer"):
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation
            return len(text.split()) * 1.3

    def get_compression_ratio(
        self, original_length: int, compressed_length: int
    ) -> float:
        """Calculate compression ratio."""
        return original_length / max(compressed_length, 1)


class ContextCompressor(CompressorBase):
    """Basic context compressor using hierarchical encoding with robust validation."""

    @validate_parameters(
        chunk_size=ParameterValidator.validate_chunk_size,
        compression_ratio=ParameterValidator.validate_compression_ratio,
    )
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        compression_ratio: float = 8.0,
        overlap_ratio: float = 0.1,
    ):
        # Validate overlap ratio
        if (
            not isinstance(overlap_ratio, (int, float))
            or not 0.0 <= overlap_ratio <= 0.5
        ):
            raise ValidationError(
                "Overlap ratio must be between 0.0 and 0.5", field="overlap_ratio"
            )

        super().__init__(model_name)
        self.chunk_size = chunk_size
        self.compression_ratio = compression_ratio
        self.overlap_ratio = overlap_ratio

    @monitor_performance
    @log_compression_operation
    @validate_input(max_size=50_000_000)  # 50MB max
    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Compress text using hierarchical chunking and encoding with error handling."""
        start_time = time.time()

        try:
            # Validate input
            if not text or not text.strip():
                raise ValidationError("Input text cannot be empty", field="text")

            # Step 1: Chunk the text hierarchically
            try:
                chunks = self._chunk_text(text)
                if not chunks:
                    raise CompressionError(
                        "Text chunking produced no chunks", stage="chunking"
                    )
            except Exception as e:
                raise CompressionError(f"Text chunking failed: {e}", stage="chunking")

            # Step 2: Encode chunks into embeddings
            try:
                embeddings = self._encode_chunks(chunks)
                if not embeddings:
                    raise CompressionError(
                        "Embedding generation produced no embeddings", stage="encoding"
                    )
            except Exception as e:
                raise CompressionError(
                    f"Embedding generation failed: {e}", stage="encoding"
                )

            # Step 3: Apply information bottleneck compression
            try:
                compressed_embeddings = self._apply_compression(embeddings)
                if not compressed_embeddings:
                    raise CompressionError(
                        "Compression produced no results", stage="compression"
                    )
            except Exception as e:
                raise CompressionError(
                    f"Information bottleneck compression failed: {e}",
                    stage="compression",
                )

            # Step 4: Create mega-tokens
            try:
                mega_tokens = self._create_mega_tokens(compressed_embeddings, chunks)
                if not mega_tokens:
                    raise CompressionError(
                        "Token creation produced no mega-tokens", stage="tokenization"
                    )
            except Exception as e:
                raise CompressionError(
                    f"Mega-token creation failed: {e}", stage="tokenization"
                )

            # Calculate metrics and create result
            processing_time = time.time() - start_time
            original_length = self.count_tokens(text)
            compressed_length = len(mega_tokens)

            return CompressionResult(
                mega_tokens=mega_tokens,
                original_length=int(original_length),
                compressed_length=compressed_length,
                compression_ratio=self.get_compression_ratio(
                    original_length, compressed_length
                ),
                processing_time=processing_time,
                metadata={
                    "model": self.model_name,
                    "chunk_size": self.chunk_size,
                    "target_compression_ratio": self.compression_ratio,
                    "actual_chunks": len(chunks),
                    "success": True,
                },
            )

        except (ValidationError, CompressionError, ModelError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise CompressionError(
                f"Unexpected compression error: {e}",
                original_length=len(text) if text else 0,
            )

    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Approximate decompression (semantic reconstruction)."""
        if not mega_tokens:
            return ""

        # Simple reconstruction by concatenating metadata text
        reconstructed_parts = []
        for token in mega_tokens:
            if "source_text" in token.metadata:
                reconstructed_parts.append(token.metadata["source_text"])
            elif "summary" in token.metadata:
                reconstructed_parts.append(token.metadata["summary"])

        return " ".join(reconstructed_parts)

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text into overlapping segments."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        overlap_size = int(self.chunk_size * self.overlap_ratio)
        step_size = self.chunk_size - overlap_size

        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + self.chunk_size]
            if chunk_words:  # Avoid empty chunks
                chunks.append(" ".join(chunk_words))

        return chunks

    def _encode_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Encode text chunks into embeddings."""
        if hasattr(self.model, "encode"):
            # SentenceTransformer
            embeddings = self.model.encode(chunks, convert_to_numpy=True)
            return [emb for emb in embeddings]
        else:
            # Transformers model
            embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling of last hidden state
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])

            return embeddings

    def _apply_compression(self, embeddings: list[np.ndarray]) -> list[np.ndarray]:
        """Apply information bottleneck compression."""
        if not embeddings:
            return []

        embeddings_array = np.array(embeddings)
        target_size = max(1, int(len(embeddings) / self.compression_ratio))

        if target_size >= len(embeddings):
            return embeddings

        # Use k-means clustering for compression
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=target_size, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_array)

            # Use cluster centers as compressed representations
            compressed = []
            for i in range(target_size):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    # Average embeddings in each cluster
                    cluster_embedding = embeddings_array[cluster_mask].mean(axis=0)
                    compressed.append(cluster_embedding)

            return compressed

        except ImportError:
            # Fallback: simple uniform sampling
            step = len(embeddings) // target_size
            return [embeddings[i] for i in range(0, len(embeddings), max(step, 1))]

    def _create_mega_tokens(
        self, embeddings: list[np.ndarray], original_chunks: list[str]
    ) -> list[MegaToken]:
        """Create mega-tokens from compressed embeddings."""
        mega_tokens = []

        for i, embedding in enumerate(embeddings):
            # Calculate confidence based on embedding norm and consistency
            confidence = min(1.0, np.linalg.norm(embedding) / 10.0)

            # Create metadata
            chunk_indices = self._find_representative_chunks(
                embedding, original_chunks, i
            )
            source_text = " ".join([original_chunks[idx] for idx in chunk_indices[:2]])

            metadata = {
                "index": i,
                "source_text": (
                    source_text[:200] + "..." if len(source_text) > 200 else source_text
                ),
                "chunk_indices": chunk_indices,
                "embedding_dim": len(embedding),
                "compression_method": "hierarchical_clustering",
            }

            mega_tokens.append(
                MegaToken(vector=embedding, metadata=metadata, confidence=confidence)
            )

        return mega_tokens

    def _find_representative_chunks(
        self, target_embedding: np.ndarray, chunks: list[str], mega_token_index: int
    ) -> list[int]:
        """Find chunks that best represent this mega-token."""
        # Simple heuristic: return chunks near this mega-token's position
        chunks_per_token = len(chunks) // max(1, len([target_embedding]))
        start_idx = mega_token_index * chunks_per_token
        end_idx = min(len(chunks), start_idx + chunks_per_token + 1)

        return list(range(start_idx, end_idx))


class AutoCompressor:
    """Factory for loading pretrained compressors."""

    _MODELS = {
        "rfcc-base-8x": {
            "class": ContextCompressor,
            "params": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "compression_ratio": 8.0,
                "chunk_size": 512,
            },
        },
        "rfcc-aggressive-16x": {
            "class": ContextCompressor,
            "params": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "compression_ratio": 16.0,
                "chunk_size": 256,
            },
        },
        "rfcc-conservative-4x": {
            "class": ContextCompressor,
            "params": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "compression_ratio": 4.0,
                "chunk_size": 1024,
            },
        },
    }

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> CompressorBase:
        """Load a pretrained compressor."""
        if model_name not in cls._MODELS:
            available = ", ".join(cls._MODELS.keys())
            raise ValueError(
                f"Unknown model '{model_name}'. Available models: {available}"
            )

        model_config = cls._MODELS[model_name]
        params = {**model_config["params"], **kwargs}

        return model_config["class"](**params)

    @classmethod
    def list_models(cls) -> list[str]:
        """List available pretrained models."""
        return list(cls._MODELS.keys())
