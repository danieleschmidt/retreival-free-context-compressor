"""Base compressor interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
from dataclasses import dataclass


@dataclass
class MegaToken:
    """Dense representation of compressed text segments."""
    
    embedding: torch.Tensor  # Dense vector representation
    metadata: Dict[str, Any]  # Source info, attention weights, etc.
    source_range: tuple[int, int]  # Original token positions
    compression_ratio: float  # How much this represents
    
    def __post_init__(self):
        """Validate mega-token structure."""
        if self.embedding.dim() != 1:
            raise ValueError("Embedding must be 1-dimensional tensor")
        if self.compression_ratio <= 0:
            raise ValueError("Compression ratio must be positive")


@dataclass 
class CompressionResult:
    """Result of document compression."""
    
    mega_tokens: List[MegaToken]
    original_length: int
    compressed_length: int
    compression_ratio: float
    processing_time: float
    metadata: Dict[str, Any]
    
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
        device: Optional[str] = None,
        max_length: int = 256000,
        compression_ratio: float = 8.0
    ):
        """Initialize base compressor.
        
        Args:
            model_name: Name/path of the compression model
            device: Device to run on ('cpu', 'cuda', etc.)
            max_length: Maximum input length in tokens
            compression_ratio: Target compression ratio
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    def compress(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> CompressionResult:
        """Compress input text into mega-tokens.
        
        Args:
            text: Input text or list of texts to compress
            **kwargs: Additional compression parameters
            
        Returns:
            CompressionResult with mega-tokens and metadata
        """
        pass
    
    @abstractmethod
    def decompress(
        self, 
        mega_tokens: List[MegaToken],
        **kwargs
    ) -> str:
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
    
    def estimate_memory_usage(self, text_length: int) -> Dict[str, float]:
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
            'input_mb': input_memory,
            'compressed_mb': compressed_memory,
            'peak_mb': input_memory * 1.5,  # Peak during processing
            'savings_mb': input_memory - compressed_memory
        }
    
    @abstractmethod
    def load_model(self) -> None:
        """Load compression model and tokenizer."""
        pass
    
    def to(self, device: str) -> 'CompressorBase':
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