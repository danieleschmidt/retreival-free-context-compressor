"""Selective compressor with adaptive compression ratios based on content importance."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .core.base import CompressionResult, CompressorBase, MegaToken
from .exceptions import CompressionError, ValidationError
from .optimization import MemoryOptimizer
from .validation import InputValidator


logger = logging.getLogger(__name__)


class ImportanceLevel(Enum):
    """Importance levels for selective compression."""

    CRITICAL = "critical"  # 2x compression (preserve detail)
    HIGH = "high"  # 4x compression
    MEDIUM = "medium"  # 8x compression (default)
    LOW = "low"  # 16x compression
    MINIMAL = "minimal"  # 32x compression (heavy compression)


@dataclass
class ContentSegment:
    """Represents a content segment with importance scoring."""

    text: str
    start_pos: int
    end_pos: int
    importance_score: float
    importance_level: ImportanceLevel
    features: dict[str, Any]

    @property
    def length(self) -> int:
        """Get segment length."""
        return len(self.text)

    def get_target_compression_ratio(self) -> float:
        """Get target compression ratio for this segment."""
        ratio_map = {
            ImportanceLevel.CRITICAL: 2.0,
            ImportanceLevel.HIGH: 4.0,
            ImportanceLevel.MEDIUM: 8.0,
            ImportanceLevel.LOW: 16.0,
            ImportanceLevel.MINIMAL: 32.0,
        }
        return ratio_map[self.importance_level]


class ImportanceAnalyzer:
    """Analyzes content to determine importance scores."""

    def __init__(self):
        """Initialize importance analyzer."""
        self._keyword_weights = {
            "summary": 2.0,
            "conclusion": 2.0,
            "introduction": 1.5,
            "abstract": 2.0,
            "key": 1.5,
            "important": 1.5,
            "critical": 2.0,
            "main": 1.3,
            "primary": 1.3,
            "essential": 1.8,
            "fundamental": 1.6,
        }

        self._section_patterns = [
            (r"# (.*)", 2.0),  # Main headings
            (r"## (.*)", 1.5),  # Subheadings
            (r"### (.*)", 1.2),  # Sub-subheadings
            (r"\*\*(.*?)\*\*", 1.3),  # Bold text
            (r"__(.*?)__", 1.3),  # Underlined text
            (r"`(.*?)`", 1.1),  # Code snippets
        ]

    def analyze_segments(
        self,
        segments: list[str],
        segment_positions: list[tuple[int, int]] | None = None,
    ) -> list[ContentSegment]:
        """Analyze text segments for importance.

        Args:
            segments: List of text segments to analyze
            segment_positions: Optional position information for each segment

        Returns:
            List of ContentSegment objects with importance scores
        """
        content_segments = []

        for i, segment in enumerate(segments):
            if segment_positions:
                start_pos, end_pos = segment_positions[i]
            else:
                start_pos, end_pos = i * 1000, (i + 1) * 1000  # Approximate

            # Calculate importance features
            features = self._extract_features(segment)

            # Calculate importance score
            importance_score = self._calculate_importance_score(segment, features)

            # Determine importance level
            importance_level = self._score_to_level(importance_score)

            content_segment = ContentSegment(
                text=segment,
                start_pos=start_pos,
                end_pos=end_pos,
                importance_score=importance_score,
                importance_level=importance_level,
                features=features,
            )

            content_segments.append(content_segment)

        return content_segments

    def _extract_features(self, text: str) -> dict[str, Any]:
        """Extract features from text segment.

        Args:
            text: Text segment to analyze

        Returns:
            Dictionary of extracted features
        """
        import re

        features = {
            "length": len(text),
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r"[.\!?]+", text)),
            "has_numbers": bool(re.search(r"\d", text)),
            "has_punctuation": bool(re.search(r"[\!?;:]", text)),
            "has_formatting": bool(re.search(r"[*_`#]", text)),
            "avg_word_length": (
                np.mean([len(word) for word in text.split()]) if text.split() else 0
            ),
            "keyword_density": 0.0,
            "structural_importance": 0.0,
        }

        # Calculate keyword density
        text_lower = text.lower()
        keyword_matches = sum(
            text_lower.count(keyword) * weight
            for keyword, weight in self._keyword_weights.items()
        )
        features["keyword_density"] = keyword_matches / max(features["word_count"], 1)

        # Calculate structural importance
        structural_score = 0.0
        for pattern, weight in self._section_patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            structural_score += matches * weight

        features["structural_importance"] = structural_score

        return features

    def _calculate_importance_score(self, text: str, features: dict[str, Any]) -> float:
        """Calculate importance score for text segment.

        Args:
            text: Text segment
            features: Extracted features

        Returns:
            Importance score (0.0 to 1.0)
        """
        # Base score from length (longer segments tend to be more important)
        length_score = min(1.0, features["length"] / 1000)

        # Keyword density contribution
        keyword_score = min(1.0, features["keyword_density"] * 0.5)

        # Structural importance
        structural_score = min(1.0, features["structural_importance"] * 0.3)

        # Information density (words per sentence)
        if features["sentence_count"] > 0:
            density_score = min(
                1.0, (features["word_count"] / features["sentence_count"]) / 20
            )
        else:
            density_score = 0.0

        # Formatting indicates importance
        formatting_score = 0.2 if features["has_formatting"] else 0.0

        # Combine scores
        total_score = (
            length_score * 0.2
            + keyword_score * 0.3
            + structural_score * 0.3
            + density_score * 0.1
            + formatting_score * 0.1
        )

        return min(1.0, total_score)

    def _score_to_level(self, score: float) -> ImportanceLevel:
        """Convert importance score to level.

        Args:
            score: Importance score (0.0 to 1.0)

        Returns:
            ImportanceLevel enum
        """
        if score >= 0.8:
            return ImportanceLevel.CRITICAL
        elif score >= 0.6:
            return ImportanceLevel.HIGH
        elif score >= 0.4:
            return ImportanceLevel.MEDIUM
        elif score >= 0.2:
            return ImportanceLevel.LOW
        else:
            return ImportanceLevel.MINIMAL


class SelectiveCompressor(CompressorBase):
    """Compressor that applies different compression ratios based on content importance."""

    def __init__(
        self,
        model_name: str = "context-compressor-base",
        device: str | None = None,
        max_length: int = 256000,
        compression_ratio: float = 8.0,  # Default ratio for medium importance content
        segment_size: int = 1000,
        overlap: int = 100,
        importance_threshold: float = 0.5,
        adaptive_ratios: bool = True,
    ):
        """Initialize selective compressor.

        Args:
            model_name: Name of compression model
            device: Computing device
            max_length: Maximum input length
            compression_ratio: Base compression ratio
            segment_size: Size of content segments for analysis
            overlap: Overlap between segments
            importance_threshold: Threshold for high-importance content
            adaptive_ratios: Whether to use adaptive compression ratios
        """
        super().__init__(model_name, device, max_length, compression_ratio)

        self.segment_size = segment_size
        self.overlap = overlap
        self.importance_threshold = importance_threshold
        self.adaptive_ratios = adaptive_ratios

        # Components
        self._importance_analyzer = ImportanceAnalyzer()
        self._validator = InputValidator()
        self._memory_optimizer = MemoryOptimizer()

        # Statistics tracking
        self._compression_stats = {
            "segments_processed": 0,
            "importance_distribution": {level.value: 0 for level in ImportanceLevel},
            "compression_ratios_used": [],
            "processing_times": [],
        }

    def load_model(self) -> None:
        """Load compression model."""
        try:
            # Use parent implementation
            super().load_model()
            logger.info("Selective compressor model loaded")

        except Exception as e:
            logger.error(f"Failed to load selective compression model: {e}")
            raise CompressionError(f"Model loading failed: {e}")

    def compress(
        self,
        text: str | list[str],
        custom_importance_fn: Callable[[str], float] | None = None,
        **kwargs,
    ) -> CompressionResult:
        """Compress text with selective compression ratios.

        Args:
            text: Input text or list of texts
            custom_importance_fn: Optional custom importance scoring function
            **kwargs: Additional parameters

        Returns:
            CompressionResult with selectively compressed mega-tokens
        """
        start_time = time.time()

        try:
            # Handle input format
            if isinstance(text, list):
                input_text = " ".join(text)
            else:
                input_text = text

            # Validate input
            validation_result = self._validator.validate_text_input(input_text)
            if not validation_result.is_valid:
                raise ValidationError(
                    "Input validation failed",
                    validation_errors=validation_result.errors,
                )

            input_text = validation_result.sanitized_input["text"]
            original_length = len(input_text.split())

            # Load model if needed
            if self._encoder_model is None:
                self.load_model()

            with self._memory_optimizer.memory_efficient_context():
                # Segment text
                segments, positions = self._segment_text(input_text)

                # Analyze importance
                content_segments = self._importance_analyzer.analyze_segments(
                    segments, positions
                )

                # Apply custom importance function if provided
                if custom_importance_fn:
                    for segment in content_segments:
                        custom_score = custom_importance_fn(segment.text)
                        segment.importance_score = max(
                            segment.importance_score, custom_score
                        )
                        segment.importance_level = (
                            self._importance_analyzer._score_to_level(
                                segment.importance_score
                            )
                        )

                # Compress segments with different ratios
                all_mega_tokens = []
                compression_ratios_used = []

                for segment in content_segments:
                    if self.adaptive_ratios:
                        target_ratio = segment.get_target_compression_ratio()
                    else:
                        target_ratio = self.compression_ratio

                    # Compress individual segment
                    segment_tokens = self._compress_segment(segment, target_ratio)
                    all_mega_tokens.extend(segment_tokens)
                    compression_ratios_used.append(target_ratio)

                    # Update statistics
                    self._compression_stats["importance_distribution"][
                        segment.importance_level.value
                    ] += 1

                # Update statistics
                self._compression_stats["segments_processed"] += len(content_segments)
                self._compression_stats["compression_ratios_used"].extend(
                    compression_ratios_used
                )

            processing_time = time.time() - start_time
            self._compression_stats["processing_times"].append(processing_time)

            # Calculate overall compression ratio
            compressed_length = len(all_mega_tokens)
            overall_ratio = (
                original_length / compressed_length if compressed_length > 0 else 1.0
            )

            # Create result
            result = CompressionResult(
                mega_tokens=all_mega_tokens,
                original_length=original_length,
                compressed_length=compressed_length,
                compression_ratio=overall_ratio,
                processing_time=processing_time,
                metadata={
                    "selective_compression": True,
                    "segments_analyzed": len(content_segments),
                    "importance_distribution": {
                        level.value: sum(
                            1 for s in content_segments if s.importance_level == level
                        )
                        for level in ImportanceLevel
                    },
                    "adaptive_ratios_used": self.adaptive_ratios,
                    "compression_ratios_range": {
                        "min": (
                            min(compression_ratios_used)
                            if compression_ratios_used
                            else self.compression_ratio
                        ),
                        "max": (
                            max(compression_ratios_used)
                            if compression_ratios_used
                            else self.compression_ratio
                        ),
                        "avg": (
                            np.mean(compression_ratios_used)
                            if compression_ratios_used
                            else self.compression_ratio
                        ),
                    },
                },
            )

            logger.info(
                f"Selective compression: {original_length} -> {compressed_length} tokens "
                f"({overall_ratio:.1f}x) across {len(content_segments)} segments in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            if not isinstance(e, (CompressionError, ValidationError)):
                raise CompressionError(f"Selective compression failed: {e}") from e
            raise

    def _segment_text(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        """Segment text for importance analysis.

        Args:
            text: Input text to segment

        Returns:
            Tuple of (segments, positions)
        """
        segments = []
        positions = []

        words = text.split()

        for i in range(0, len(words), self.segment_size - self.overlap):
            segment_words = words[i : i + self.segment_size]
            segment_text = " ".join(segment_words)

            # Calculate character positions (approximate)
            start_pos = sum(len(w) + 1 for w in words[:i])
            end_pos = start_pos + len(segment_text)

            segments.append(segment_text)
            positions.append((start_pos, end_pos))

        return segments, positions

    def _compress_segment(
        self, segment: ContentSegment, target_ratio: float
    ) -> list[MegaToken]:
        """Compress individual content segment.

        Args:
            segment: Content segment to compress
            target_ratio: Target compression ratio

        Returns:
            List of mega-tokens for the segment
        """
        # Temporarily adjust compression ratio
        original_ratio = self.compression_ratio
        self.compression_ratio = target_ratio

        try:
            # Use parent compression method
            result = super().compress(segment.text)

            # Enhance mega-tokens with selective compression metadata
            enhanced_tokens = []
            for i, token in enumerate(result.mega_tokens):
                enhanced_metadata = {
                    **token.metadata,
                    "importance_score": segment.importance_score,
                    "importance_level": segment.importance_level.value,
                    "target_compression_ratio": target_ratio,
                    "segment_features": segment.features,
                    "selective_compression": True,
                }

                enhanced_token = MegaToken(
                    embedding=token.embedding,
                    metadata=enhanced_metadata,
                    source_range=(
                        segment.start_pos + token.source_range[0],
                        segment.start_pos + token.source_range[1],
                    ),
                    compression_ratio=target_ratio,
                )

                enhanced_tokens.append(enhanced_token)

            return enhanced_tokens

        finally:
            # Restore original compression ratio
            self.compression_ratio = original_ratio

    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Decompress mega-tokens with selective information.

        Args:
            mega_tokens: Mega-tokens to decompress
            **kwargs: Additional parameters

        Returns:
            Reconstructed text with importance annotations
        """
        parts = []

        for i, token in enumerate(mega_tokens):
            importance_level = token.metadata.get("importance_level", "unknown")
            importance_score = token.metadata.get("importance_score", 0.0)
            compression_ratio = token.metadata.get(
                "target_compression_ratio", token.compression_ratio
            )

            part = (
                f"[SelectiveSegment {i+1}: importance={importance_level} "
                f"(score={importance_score:.2f}), ratio={compression_ratio:.1f}x]"
            )

            # Add segment features if available
            if "segment_features" in token.metadata:
                features = token.metadata["segment_features"]
                part += f" [words={features.get('word_count', '?')}, "
                part += f"density={features.get('keyword_density', 0):.2f}]"

            parts.append(part)

        return " ".join(parts)

    def get_compression_strategy(self, text: str) -> dict[str, Any]:
        """Analyze text and return compression strategy without compressing.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with compression strategy details
        """
        # Validate input
        validation_result = self._validator.validate_text_input(text)
        if not validation_result.is_valid:
            raise ValidationError(
                "Input validation failed", validation_errors=validation_result.errors
            )

        text = validation_result.sanitized_input["text"]

        # Segment and analyze
        segments, positions = self._segment_text(text)
        content_segments = self._importance_analyzer.analyze_segments(
            segments, positions
        )

        # Calculate strategy
        strategy = {
            "total_segments": len(content_segments),
            "total_length": len(text),
            "segments": [],
            "compression_plan": {
                level.value: {
                    "count": 0,
                    "total_length": 0,
                    "compression_ratio": ImportanceLevel(level).name.lower(),
                }
                for level in ImportanceLevel
            },
            "expected_output_size": 0,
        }

        for segment in content_segments:
            segment_info = {
                "text_preview": (
                    segment.text[:100] + "..."
                    if len(segment.text) > 100
                    else segment.text
                ),
                "length": segment.length,
                "importance_score": segment.importance_score,
                "importance_level": segment.importance_level.value,
                "target_ratio": (
                    segment.get_target_compression_ratio()
                    if self.adaptive_ratios
                    else self.compression_ratio
                ),
                "features": segment.features,
            }

            strategy["segments"].append(segment_info)

            # Update compression plan
            level_stats = strategy["compression_plan"][segment.importance_level.value]
            level_stats["count"] += 1
            level_stats["total_length"] += segment.length

            # Estimate compressed size
            target_ratio = (
                segment.get_target_compression_ratio()
                if self.adaptive_ratios
                else self.compression_ratio
            )
            estimated_compressed_size = max(1, segment.length // target_ratio)
            strategy["expected_output_size"] += estimated_compressed_size

        # Calculate overall metrics
        strategy["expected_compression_ratio"] = len(text) / max(
            1, strategy["expected_output_size"]
        )
        strategy["importance_distribution"] = {
            level: level_stats["count"] / len(content_segments)
            for level, level_stats in strategy["compression_plan"].items()
        }

        return strategy

    def get_selective_stats(self) -> dict[str, Any]:
        """Get selective compression statistics.

        Returns:
            Dictionary with selective compression statistics
        """
        stats = self._compression_stats.copy()

        if stats["processing_times"]:
            stats["avg_processing_time"] = np.mean(stats["processing_times"])
            stats["total_processing_time"] = sum(stats["processing_times"])

        if stats["compression_ratios_used"]:
            stats["compression_ratio_stats"] = {
                "min": min(stats["compression_ratios_used"]),
                "max": max(stats["compression_ratios_used"]),
                "avg": np.mean(stats["compression_ratios_used"]),
                "std": np.std(stats["compression_ratios_used"]),
            }

        return stats

    def reset_stats(self) -> None:
        """Reset compression statistics."""
        self._compression_stats = {
            "segments_processed": 0,
            "importance_distribution": {level.value: 0 for level in ImportanceLevel},
            "compression_ratios_used": [],
            "processing_times": [],
        }

        logger.debug("Selective compression statistics reset")
