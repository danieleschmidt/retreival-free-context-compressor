"""Auto-compressor factory for loading pretrained models."""

import json
import logging
import os
from pathlib import Path
from typing import Any

from .base import CompressorBase
from .context_compressor import ContextCompressor


# Import other compressor types
try:
    from ..streaming import StreamingCompressor
    HAS_STREAMING = True
except ImportError:
    HAS_STREAMING = False

try:
    from ..selective import SelectiveCompressor
    HAS_SELECTIVE = True
except ImportError:
    HAS_SELECTIVE = False

try:
    from ..multi_doc import MultiDocCompressor
    HAS_MULTI_DOC = True
except ImportError:
    HAS_MULTI_DOC = False

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry of available compression models."""

    # Pretrained model configurations
    MODELS = {
        "rfcc-base-8x": {
            "class": "ContextCompressor",
            "compression_ratio": 8.0,
            "chunk_size": 512,
            "overlap": 64,
            "description": "Base 8x compression model for general text"
        },
        "rfcc-streaming": {
            "class": "StreamingCompressor",
            "compression_ratio": 8.0,
            "window_size": 32000,
            "description": "Streaming compression for infinite contexts"
        },
        "rfcc-selective": {
            "class": "SelectiveCompressor",
            "compression_ratios": {"legal": 4.0, "general": 8.0, "repetitive": 16.0},
            "description": "Content-aware selective compression"
        },
        "context-compressor-base": {
            "class": "ContextCompressor",
            "compression_ratio": 8.0,
            "chunk_size": 512,
            "overlap": 64,
            "description": "Default context compressor"
        }
    }

    @classmethod
    def list_models(cls) -> dict[str, dict[str, Any]]:
        """List all available models."""
        return cls.MODELS.copy()

    @classmethod
    def get_model_info(cls, model_name: str) -> dict[str, Any] | None:
        """Get information about a specific model."""
        return cls.MODELS.get(model_name)

    @classmethod
    def register_model(
        cls,
        name: str,
        config: dict[str, Any]
    ) -> None:
        """Register a new model configuration."""
        cls.MODELS[name] = config
        logger.info(f"Registered model: {name}")


class AutoCompressor:
    """Factory class for creating and loading compressors."""

    @staticmethod
    def from_pretrained(
        model_name: str,
        device: str | None = None,
        cache_dir: str | None = None,
        **kwargs
    ) -> CompressorBase:
        """Load a pretrained compressor model.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on
            cache_dir: Directory to cache models
            **kwargs: Additional model parameters
            
        Returns:
            Initialized compressor instance
            
        Raises:
            ValueError: If model not found
            RuntimeError: If model loading fails
        """
        # Check if model is registered
        model_info = ModelRegistry.get_model_info(model_name)
        if model_info is None:
            # Try loading from local path
            if os.path.exists(model_name):
                return AutoCompressor._load_from_path(model_name, device, **kwargs)

            # Model not found
            available = list(ModelRegistry.MODELS.keys())
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {available}"
            )

        # Get compressor class
        compressor_class = AutoCompressor._get_compressor_class(model_info["class"])

        # Merge model config with kwargs
        config = model_info.copy()
        config.update(kwargs)
        config.pop("class", None)  # Remove class key
        config.pop("description", None)  # Remove description

        # Create compressor instance
        try:
            compressor = compressor_class(
                model_name=model_name,
                device=device,
                **config
            )

            # Load the model
            compressor.load_model()

            logger.info(f"Loaded pretrained model: {model_name}")
            return compressor

        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {e}")

    @staticmethod
    def _get_compressor_class(class_name: str) -> type:
        """Get compressor class by name.
        
        Args:
            class_name: Name of compressor class
            
        Returns:
            Compressor class
            
        Raises:
            ValueError: If class not found
        """
        # Map class names to actual classes
        class_map = {
            "ContextCompressor": ContextCompressor,
        }

        # Add other compressor classes if available
        if HAS_STREAMING:
            class_map["StreamingCompressor"] = StreamingCompressor
        else:
            class_map["StreamingCompressor"] = ContextCompressor  # Fallback

        if HAS_SELECTIVE:
            class_map["SelectiveCompressor"] = SelectiveCompressor
        else:
            class_map["SelectiveCompressor"] = ContextCompressor  # Fallback

        if HAS_MULTI_DOC:
            class_map["MultiDocCompressor"] = MultiDocCompressor
        else:
            class_map["MultiDocCompressor"] = ContextCompressor  # Fallback

        if class_name not in class_map:
            raise ValueError(f"Unknown compressor class: {class_name}")

        return class_map[class_name]

    @staticmethod
    def _load_from_path(
        model_path: str,
        device: str | None = None,
        **kwargs
    ) -> CompressorBase:
        """Load compressor from local path.
        
        Args:
            model_path: Path to model directory
            device: Device to load on
            **kwargs: Additional parameters
            
        Returns:
            Loaded compressor
            
        Raises:
            FileNotFoundError: If config file not found
            ValueError: If invalid configuration
        """
        model_path = Path(model_path)
        config_path = model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        # Load configuration
        with open(config_path) as f:
            config = json.load(f)

        # Get compressor class
        class_name = config.get("compressor_class", "ContextCompressor")
        compressor_class = AutoCompressor._get_compressor_class(class_name)

        # Create compressor
        compressor = compressor_class(
            model_name=str(model_path),
            device=device,
            **{**config, **kwargs}
        )

        compressor.load_model()
        return compressor

    @staticmethod
    def save_pretrained(
        compressor: CompressorBase,
        save_path: str,
        push_to_hub: bool = False
    ) -> None:
        """Save compressor to disk.
        
        Args:
            compressor: Compressor to save
            save_path: Directory to save to
            push_to_hub: Whether to push to HuggingFace Hub
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "compressor_class": compressor.__class__.__name__,
            "model_name": compressor.model_name,
            "compression_ratio": compressor.compression_ratio,
            "max_length": compressor.max_length,
        }

        # Add compressor-specific config
        if hasattr(compressor, 'chunk_size'):
            config["chunk_size"] = compressor.chunk_size
        if hasattr(compressor, 'overlap'):
            config["overlap"] = compressor.overlap

        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save model weights if available
        if hasattr(compressor, '_hierarchical_encoder') and compressor._hierarchical_encoder:
            import torch
            model_path = save_path / "pytorch_model.bin"
            torch.save(compressor._hierarchical_encoder.state_dict(), model_path)

        logger.info(f"Saved compressor to: {save_path}")

        if push_to_hub:
            logger.warning("HuggingFace Hub upload not implemented yet")

    @staticmethod
    def list_available_models() -> dict[str, str]:
        """List all available pretrained models.
        
        Returns:
            Dictionary mapping model names to descriptions
        """
        models = {}
        for name, info in ModelRegistry.list_models().items():
            models[name] = info.get("description", "No description available")

        return models

    @staticmethod
    def create_custom_compressor(
        compressor_type: str = "context",
        compression_ratio: float = 8.0,
        device: str | None = None,
        **kwargs
    ) -> CompressorBase:
        """Create a custom compressor with specific parameters.
        
        Args:
            compressor_type: Type of compressor ('context', 'streaming', etc.)
            compression_ratio: Target compression ratio
            device: Device to use
            **kwargs: Additional parameters
            
        Returns:
            Configured compressor instance
        """
        type_map = {
            "context": ContextCompressor,
        }

        # Add other types if available
        if HAS_STREAMING:
            type_map["streaming"] = StreamingCompressor
        else:
            type_map["streaming"] = ContextCompressor  # Fallback

        if HAS_SELECTIVE:
            type_map["selective"] = SelectiveCompressor
        else:
            type_map["selective"] = ContextCompressor  # Fallback

        if HAS_MULTI_DOC:
            type_map["multi_doc"] = MultiDocCompressor
        else:
            type_map["multi_doc"] = ContextCompressor  # Fallback

        if compressor_type not in type_map:
            available = list(type_map.keys())
            raise ValueError(f"Unknown type '{compressor_type}'. Available: {available}")

        compressor_class = type_map[compressor_type]

        compressor = compressor_class(
            model_name=f"custom-{compressor_type}",
            device=device,
            compression_ratio=compression_ratio,
            **kwargs
        )

        compressor.load_model()
        return compressor
