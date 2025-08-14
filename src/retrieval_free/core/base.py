"""Base compressor interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np

# Try to import torch, fall back to mock if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    from ..mock_torch import tensor, cuda, device, MockTensor
    # Create a minimal torch-like module
    class MockTorch:
        Tensor = MockTensor
        tensor = tensor
        cuda = cuda  
        device = device
    torch = MockTorch()
    HAS_TORCH = False


@dataclass
class MegaToken:
    """Dense representation of compressed text segments."""

    vector: np.ndarray  # Dense vector representation (numpy array for compatibility)
    metadata: dict[str, Any]  # Source info, attention weights, etc.
    confidence: float = 1.0  # Confidence score for this mega-token

    def __post_init__(self):
        """Validate mega-token structure."""
        if not isinstance(self.vector, np.ndarray):
            # Convert from torch tensor if needed
            if hasattr(self.vector, 'numpy'):
                self.vector = self.vector.numpy()
            elif hasattr(self.vector, 'data'):
                self.vector = self.vector.data
            else:
                self.vector = np.array(self.vector)
        
        if self.vector.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")
    
    @property
    def embedding(self):
        """Legacy property for backward compatibility."""
        return self.vector
    
    @property
    def source_range(self):
        """Source range from metadata."""
        return self.metadata.get('source_range', (0, 0))
    
    @property 
    def compression_ratio(self):
        """Compression ratio from metadata."""
        return self.metadata.get('compression_ratio', 1.0)
        if HAS_TORCH:
            return torch.tensor(self.vector)
        else:
            return torch.tensor(self.vector)  # Will use mock
    
    @property 
    def source_range(self):
        """Source range from metadata if available."""
        return self.metadata.get('source_range', (0, len(self.vector)))
    
    @property
    def compression_ratio(self):
        """Compression ratio from metadata if available.""" 
        return self.metadata.get('compression_ratio', 1.0)


@dataclass
class CompressionResult:
    """Result of document compression."""

    mega_tokens: list[MegaToken]
    original_length: int
    compressed_length: int
    compression_ratio: float
    processing_time: float
    metadata: dict[str, Any]

    @property
    def total_tokens(self) -> int:
        """Total number of mega-tokens."""
        return len(self.mega_tokens)

    @property
    def memory_savings(self) -> float:
        """Estimated memory savings percentage."""
        return 1.0 - (self.compressed_length / self.original_length)


class CompressorBase(ABC):
    """Abstract base class for all document compressors."""

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 256000,
        compression_ratio: float = 8.0,
    ):
        """Initialize base compressor.

        Args:
            model_name: Name/path of the compression model
            device: Device to run on ('cpu', 'cuda', etc.)
            max_length: Maximum input length in tokens
            compression_ratio: Target compression ratio
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def compress(self, text: str | list[str], **kwargs) -> CompressionResult:
        """Compress input text into mega-tokens.

        Args:
            text: Input text or list of texts to compress
            **kwargs: Additional compression parameters

        Returns:
            CompressionResult with mega-tokens and metadata
        """
        pass

    @abstractmethod
    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Reconstruct text from mega-tokens (approximate).

        Args:
            mega_tokens: List of mega-tokens to decompress
            **kwargs: Additional decompression parameters

        Returns:
            Reconstructed text (may be lossy)
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in input text.

        Args:
            text: Input text to count

        Returns:
            Number of tokens
        """
        if self._tokenizer is None:
            # Rough approximation: ~4 chars per token
            return len(text) // 4
        return len(self._tokenizer.encode(text))

    def get_compression_ratio(self) -> float:
        """Get current compression ratio."""
        return self.compression_ratio

    def estimate_memory_usage(self, text_length: int) -> dict[str, float]:
        """Estimate memory usage for compression.

        Args:
            text_length: Input text length in tokens

        Returns:
            Dictionary with memory estimates in MB
        """
        # Rough estimates based on transformer memory patterns
        input_memory = text_length * 0.004  # ~4KB per token
        compressed_memory = (text_length / self.compression_ratio) * 0.016  # Denser

        return {
            "input_mb": input_memory,
            "compressed_mb": compressed_memory,
            "peak_mb": input_memory * 1.5,  # Peak during processing
            "savings_mb": input_memory - compressed_memory,
        }

    @abstractmethod
    def load_model(self) -> None:
        """Load compression model and tokenizer."""
        pass

    def to(self, device: str) -> "CompressorBase":
        """Move compressor to specified device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Self for chaining
        """
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', "
            f"device='{self.device}', "
            f"ratio={self.compression_ratio}x"
            f")"
        )
