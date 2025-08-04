"""Selective compression with content-aware ratios."""

import re
import logging
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass

from .core.base import CompressorBase, CompressionResult
from .core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


@dataclass
class ContentClassification:
    """Classification result for text content."""
    
    content_type: str
    confidence: float
    features: Dict[str, float]
    recommended_ratio: float


class ContentClassifier:
    """Classifier for determining content type and optimal compression."""
    
    def __init__(self):
        """Initialize content classifier."""
        # Patterns for different content types
        self.patterns = {
            "legal": [
                r"\b(whereas|hereby|pursuant|notwithstanding|heretofore)\b",
                r"\b(contract|agreement|terms|conditions|liability)\b",
                r"\b(shall|will|must|may|should)\b",
                r"\b(party|parties|entity|entities)\b"
            ],
            "technical": [
                r"\b(function|class|method|variable|parameter)\b",
                r"\b(algorithm|implementation|optimization|performance)\b", 
                r"\b(system|framework|architecture|component)\b",
                r"\b(api|interface|protocol|specification)\b"
            ],
            "academic": [
                r"\b(research|study|analysis|experiment|hypothesis)\b",
                r"\b(results|findings|conclusion|methodology)\b",
                r"\b(significant|correlation|variable|dataset)\b",
                r"\b(furthermore|moreover|however|therefore)\b"
            ],
            "financial": [
                r"\$\d+(?:,\d{3})*(?:\.\d{2})?",
                r"\b(revenue|profit|loss|investment|portfolio)\b",
                r"\b(quarter|annual|fiscal|earnings|dividend)\b",
                r"\b(stock|equity|bond|asset|liability)\b"
            ],
            "repetitive": [
                r"\b(\w+)\s+\1\b",  # Repeated words
                r"(.{10,})\1{2,}",  # Repeated phrases
                r"\n\s*\n\s*\n",   # Multiple empty lines
            ]
        }
        
        # Compression ratios for each content type
        self.compression_ratios = {
            "legal": 4.0,      # Conservative - legal text needs precision
            "technical": 6.0,   # Moderate - technical terms important
            "academic": 8.0,    # Standard - balanced approach
            "financial": 5.0,   # Conservative - numbers important
            "general": 8.0,     # Standard compression
            "repetitive": 16.0, # Aggressive - lots of redundancy
            "narrative": 10.0   # Aggressive - stories can be summarized
        }
    
    def classify(self, text: str) -> ContentClassification:
        """Classify text content and recommend compression ratio.
        
        Args:
            text: Input text to classify
            
        Returns:
            ContentClassification with type and recommended ratio
        """
        scores = {}
        features = {}
        
        # Calculate pattern-based scores
        for content_type, patterns in self.patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            
            # Normalize by text length
            normalized_score = score / (len(text.split()) + 1)
            scores[content_type] = normalized_score
            features[f"{content_type}_score"] = normalized_score
        
        # Additional feature extraction
        features.update(self._extract_features(text))
        
        # Determine primary content type
        if not scores or max(scores.values()) < 0.01:
            content_type = "general"
            confidence = 0.5
        else:
            content_type = max(scores, key=scores.get)
            confidence = min(scores[content_type] * 10, 1.0)  # Scale to 0-1
        
        # Special handling for repetitive content
        if features.get("repetition_ratio", 0) > 0.3:
            content_type = "repetitive"
            confidence = features["repetition_ratio"]
        
        # Get recommended compression ratio
        recommended_ratio = self.compression_ratios.get(content_type, 8.0)
        
        return ContentClassification(
            content_type=content_type,
            confidence=confidence,
            features=features,
            recommended_ratio=recommended_ratio
        )
    
    def _extract_features(self, text: str) -> Dict[str, float]:  
        """Extract additional text features.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted features
        """
        words = text.split()
        sentences = text.split('.')
        
        features = {}
        
        # Basic statistics
        features["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
        features["avg_sentence_length"] = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary diversity (unique words / total words)
        unique_words = set(word.lower() for word in words)
        features["vocabulary_diversity"] = len(unique_words) / len(words) if words else 0
        
        # Repetition analysis
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        if word_counts:
            max_count = max(word_counts.values())
            features["repetition_ratio"] = max_count / len(words)
        else:
            features["repetition_ratio"] = 0
        
        # Punctuation density
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        features["punctuation_density"] = punctuation_count / len(text) if text else 0
        
        # Number density
        number_matches = re.findall(r'\d+', text)
        features["number_density"] = len(number_matches) / len(words) if words else 0
        
        return features


class SelectiveCompressor(CompressorBase):
    """Content-aware compressor with adaptive compression ratios."""
    
    def __init__(
        self,
        model_name: str = "rfcc-selective",
        device: Optional[str] = None,
        max_length: int = 256000,
        compression_ratios: Optional[Dict[str, float]] = None,
        segment_size: int = 1000
    ):
        """Initialize selective compressor.
        
        Args:
            model_name: Base model name
            device: Computing device
            max_length: Maximum input length
            compression_ratios: Custom ratios for content types
            segment_size: Size of segments for classification
        """
        # Use default compression ratio (will be overridden per segment)
        super().__init__(model_name, device, max_length, 8.0)
        
        self.segment_size = segment_size
        self._classifier = ContentClassifier()
        self._base_compressor = ContextCompressor(
            model_name=model_name,
            device=device
        )
        
        # Override default compression ratios if provided
        if compression_ratios:
            self._classifier.compression_ratios.update(compression_ratios)
    
    def load_model(self) -> None:
        """Load base compression model."""
        self._base_compressor.load_model()
    
    def compress(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> CompressionResult:
        """Compress text with content-aware ratios.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult with selective compression applied
        """
        if isinstance(text, list):
            text = " ".join(text)
        
        # Segment text for classification
        segments = self._segment_text(text)
        
        all_mega_tokens = []
        segment_info = []
        total_original = 0
        total_compressed = 0
        
        for i, segment in enumerate(segments):
            # Classify segment
            classification = self._classifier.classify(segment)
            
            # Set compression ratio for this segment
            self._base_compressor.compression_ratio = classification.recommended_ratio
            
            # Compress segment
            result = self._base_compressor.compress(segment)
            
            # Add classification metadata to mega-tokens
            for token in result.mega_tokens:
                token.metadata.update({
                    "segment_id": i,
                    "content_type": classification.content_type,
                    "classification_confidence": classification.confidence,
                    "adaptive_ratio": classification.recommended_ratio
                })
            
            all_mega_tokens.extend(result.mega_tokens)
            total_original += result.original_length
            total_compressed += result.compressed_length
            
            segment_info.append({
                "segment_id": i,
                "content_type": classification.content_type,
                "confidence": classification.confidence,
                "compression_ratio": classification.recommended_ratio,
                "original_tokens": result.original_length,
                "compressed_tokens": result.compressed_length,
                "features": classification.features
            })
            
            logger.info(
                f"Segment {i}: {classification.content_type} "
                f"({classification.confidence:.2f} confidence) -> "
                f"{classification.recommended_ratio:.1f}x compression"
            )
        
        # Calculate overall metrics
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
        
        result = CompressionResult(
            mega_tokens=all_mega_tokens,
            original_length=total_original,
            compressed_length=total_compressed,
            compression_ratio=overall_ratio,
            processing_time=sum(info.get("processing_time", 0) for info in segment_info),
            metadata={
                "selective_compression": True,
                "segments": segment_info,
                "content_types": list(set(info["content_type"] for info in segment_info)),
                "avg_confidence": sum(info["confidence"] for info in segment_info) / len(segment_info)
            }
        )
        
        logger.info(
            f"Selective compression complete: {len(segments)} segments, "
            f"{overall_ratio:.1f}x overall ratio"
        )
        
        return result
    
    def decompress(
        self, 
        mega_tokens: List[MegaToken],
        **kwargs
    ) -> str:
        """Decompress selectively compressed tokens."""
        return self._base_compressor.decompress(mega_tokens, **kwargs)
    
    def compress_smart(self, text: str) -> CompressionResult:
        """Smart compression with automatic content detection.
        
        This is an alias for the standard compress method with
        automatic content-aware compression.
        
        Args:
            text: Input text to compress
            
        Returns:
            CompressionResult with smart compression applied
        """
        return self.compress(text)
    
    def get_content_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze text content without compression.
        
        Args:
            text: Text to analyze
            
        Returns:
            Content analysis results
        """
        segments = self._segment_text(text)
        analysis = {
            "total_segments": len(segments),
            "segment_analyses": [],
            "content_distribution": {},
            "recommended_ratios": {}
        }
        
        for i, segment in enumerate(segments):
            classification = self._classifier.classify(segment)
            
            analysis["segment_analyses"].append({
                "segment_id": i,
                "length": len(segment.split()),
                "content_type": classification.content_type,
                "confidence": classification.confidence,
                "recommended_ratio": classification.recommended_ratio,
                "features": classification.features
            })
            
            # Update distribution
            content_type = classification.content_type
            analysis["content_distribution"][content_type] = (
                analysis["content_distribution"].get(content_type, 0) + 1
            )
            
            # Track recommended ratios
            if content_type not in analysis["recommended_ratios"]:
                analysis["recommended_ratios"][content_type] = classification.recommended_ratio
        
        return analysis
    
    def _segment_text(self, text: str) -> List[str]:
        """Segment text for content-aware processing.
        
        Args:
            text: Input text to segment
            
        Returns:
            List of text segments
        """
        words = text.split()
        segments = []
        
        for i in range(0, len(words), self.segment_size):
            segment_words = words[i:i + self.segment_size]
            segments.append(" ".join(segment_words))
        
        return segments
    
    def update_compression_ratios(self, ratios: Dict[str, float]) -> None:
        """Update compression ratios for content types.
        
        Args:
            ratios: Dictionary mapping content types to compression ratios
        """
        self._classifier.compression_ratios.update(ratios)
        logger.info(f"Updated compression ratios: {ratios}")