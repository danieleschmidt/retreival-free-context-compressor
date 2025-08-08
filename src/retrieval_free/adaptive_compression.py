"""Advanced compression algorithms with adaptive and streaming capabilities.

This module implements sophisticated compression strategies:
- Adaptive compression based on content type and characteristics
- Incremental compression for changed content
- Streaming compression for real-time data
- Domain-specific optimizations
- Multi-algorithm selection and ensembles
"""

import hashlib
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .core import CompressionResult, CompressorBase, MegaToken
from .exceptions import CompressionError


logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of content for specialized processing."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONVERSATION = "conversation"
    SCIENTIFIC_PAPER = "scientific_paper"
    NEWS_ARTICLE = "news_article"
    LEGAL_DOCUMENT = "legal_document"
    TECHNICAL_MANUAL = "technical_manual"
    GENERAL_TEXT = "general_text"


class CompressionStrategy(str, Enum):
    """Available compression strategies."""
    HIERARCHICAL = "hierarchical"
    SEMANTIC_CLUSTERING = "semantic_clustering"
    FREQUENCY_BASED = "frequency_based"
    TEMPLATE_MATCHING = "template_matching"
    INCREMENTAL = "incremental"
    STREAMING = "streaming"


@dataclass
class ContentCharacteristics:
    """Characteristics of content for adaptive compression."""
    content_type: ContentType
    length: int
    complexity_score: float
    repetition_ratio: float
    structure_score: float
    vocabulary_size: int
    avg_sentence_length: float
    technical_terms_ratio: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionParameters:
    """Parameters for compression algorithms."""
    strategy: CompressionStrategy
    compression_ratio: float
    chunk_size: int
    overlap_ratio: float
    clustering_k: int | None = None
    min_frequency: int = 2
    template_threshold: float = 0.8
    incremental_threshold: float = 0.1
    stream_window_size: int = 1000


@dataclass
class StreamingState:
    """State for streaming compression."""
    buffer: deque
    window_tokens: list[MegaToken]
    processed_count: int
    total_compression_ratio: float
    last_update: float
    checksum: str = ""


class ContentAnalyzer:
    """Analyzes content to determine optimal compression strategy."""

    def __init__(self):
        # Patterns for content type detection
        self.code_patterns = [
            r'\b(def|class|import|function|var|let|const)\b',
            r'[{}\[\]();]',
            r'\/\/|\/\*|\*\/|#.*$',
            r'\b(if|else|while|for|return)\b'
        ]

        self.doc_patterns = [
            r'# .+|## .+|### .+',  # Markdown headers
            r'\*\*.+\*\*|__.+__',  # Bold text
            r'`[^`]+`',  # Code blocks
            r'http[s]?://\S+'  # URLs
        ]

        self.scientific_patterns = [
            r'\b(abstract|introduction|methodology|results|conclusion)\b',
            r'\b(et al\.|i\.e\.|e\.g\.)\b',
            r'\b\d+\.\d+\b',  # Decimal numbers
            r'\([^)]*\d{4}[^)]*\)'  # Citations with years
        ]

    def analyze_content(self, text: str) -> ContentCharacteristics:
        """Analyze content to determine characteristics."""
        # Basic statistics
        length = len(text)
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        vocabulary = set(word.lower().strip('.,!?;:') for word in words)

        # Calculate metrics
        vocabulary_size = len(vocabulary)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        repetition_ratio = 1.0 - (vocabulary_size / max(len(words), 1))

        # Detect content type
        content_type = self._detect_content_type(text)

        # Calculate complexity score
        complexity_score = self._calculate_complexity(text, words, vocabulary_size)

        # Calculate structure score
        structure_score = self._calculate_structure_score(text)

        # Technical terms ratio
        technical_terms_ratio = self._calculate_technical_ratio(words)

        return ContentCharacteristics(
            content_type=content_type,
            length=length,
            complexity_score=complexity_score,
            repetition_ratio=repetition_ratio,
            structure_score=structure_score,
            vocabulary_size=vocabulary_size,
            avg_sentence_length=avg_sentence_length,
            technical_terms_ratio=technical_terms_ratio,
            metadata={
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': vocabulary_size
            }
        )

    def _detect_content_type(self, text: str) -> ContentType:
        """Detect the type of content."""
        text_lower = text.lower()

        # Count pattern matches
        code_score = sum(len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
                        for pattern in self.code_patterns)

        doc_score = sum(len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
                       for pattern in self.doc_patterns)

        sci_score = sum(len(re.findall(pattern, text, re.MULTILINE | re.IGNORECASE))
                       for pattern in self.scientific_patterns)

        # Normalize scores by text length
        text_length = len(text.split())
        code_score /= max(text_length, 1)
        doc_score /= max(text_length, 1)
        sci_score /= max(text_length, 1)

        # Determine content type
        if code_score > 0.05:
            return ContentType.CODE
        elif doc_score > 0.02:
            return ContentType.DOCUMENTATION
        elif sci_score > 0.01:
            return ContentType.SCIENTIFIC_PAPER
        elif 'legal' in text_lower or 'pursuant' in text_lower:
            return ContentType.LEGAL_DOCUMENT
        elif len(re.findall(r'\n\w+:', text)) > 3:  # Conversation format
            return ContentType.CONVERSATION
        else:
            return ContentType.GENERAL_TEXT

    def _calculate_complexity(self, text: str, words: list[str], vocab_size: int) -> float:
        """Calculate text complexity score (0-1)."""
        # Lexical diversity
        lexical_diversity = vocab_size / max(len(words), 1)

        # Average word length
        avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
        word_length_score = min(avg_word_length / 10, 1.0)

        # Sentence structure complexity
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_words = len(words) / max(len(sentences), 1)
        sentence_complexity = min(avg_sentence_words / 20, 1.0)

        # Combined complexity score
        complexity = (lexical_diversity * 0.4 +
                     word_length_score * 0.3 +
                     sentence_complexity * 0.3)

        return min(complexity, 1.0)

    def _calculate_structure_score(self, text: str) -> float:
        """Calculate structural organization score (0-1)."""
        # Headers and sections
        headers = len(re.findall(r'^#+\s+.+$|^[A-Z][^.]*:$', text, re.MULTILINE))

        # Lists and bullet points
        lists = len(re.findall(r'^\s*[-*+]\s+|^\s*\d+\.\s+', text, re.MULTILINE))

        # Paragraphs
        paragraphs = len(re.split(r'\n\s*\n', text.strip()))

        # Code blocks
        code_blocks = len(re.findall(r'```[\s\S]*?```|`[^`]+`', text))

        # Normalize by text length
        text_lines = len(text.split('\n'))
        structure_elements = headers + lists + code_blocks

        structure_score = min(structure_elements / max(text_lines / 10, 1), 1.0)
        return structure_score

    def _calculate_technical_ratio(self, words: list[str]) -> float:
        """Calculate ratio of technical terms."""
        # Simple heuristic: words with unusual patterns
        technical_patterns = [
            r'^[a-z]+[A-Z][a-z]+',  # camelCase
            r'^[A-Z_]+$',  # CONSTANTS
            r'^\w+\.\w+',  # dot notation
            r'^\w+_\w+',  # snake_case
            r'^\w{10,}$',  # Very long words
        ]

        technical_count = 0
        for word in words:
            for pattern in technical_patterns:
                if re.match(pattern, word):
                    technical_count += 1
                    break

        return technical_count / max(len(words), 1)


class StrategySelector:
    """Selects optimal compression strategy based on content characteristics."""

    def __init__(self):
        # Strategy effectiveness matrix
        self.strategy_matrix = {
            ContentType.CODE: {
                CompressionStrategy.TEMPLATE_MATCHING: 0.9,
                CompressionStrategy.FREQUENCY_BASED: 0.8,
                CompressionStrategy.HIERARCHICAL: 0.6,
                CompressionStrategy.SEMANTIC_CLUSTERING: 0.4
            },
            ContentType.DOCUMENTATION: {
                CompressionStrategy.HIERARCHICAL: 0.9,
                CompressionStrategy.SEMANTIC_CLUSTERING: 0.8,
                CompressionStrategy.TEMPLATE_MATCHING: 0.7,
                CompressionStrategy.FREQUENCY_BASED: 0.5
            },
            ContentType.SCIENTIFIC_PAPER: {
                CompressionStrategy.SEMANTIC_CLUSTERING: 0.9,
                CompressionStrategy.HIERARCHICAL: 0.8,
                CompressionStrategy.FREQUENCY_BASED: 0.6,
                CompressionStrategy.TEMPLATE_MATCHING: 0.4
            },
            ContentType.GENERAL_TEXT: {
                CompressionStrategy.HIERARCHICAL: 0.8,
                CompressionStrategy.SEMANTIC_CLUSTERING: 0.7,
                CompressionStrategy.FREQUENCY_BASED: 0.6,
                CompressionStrategy.TEMPLATE_MATCHING: 0.5
            }
        }

    def select_strategy(
        self,
        characteristics: ContentCharacteristics,
        target_ratio: float = 8.0,
        prefer_speed: bool = False
    ) -> CompressionParameters:
        """Select optimal compression strategy."""

        # Get effectiveness scores for content type
        content_strategies = self.strategy_matrix.get(
            characteristics.content_type,
            self.strategy_matrix[ContentType.GENERAL_TEXT]
        )

        # Adjust scores based on characteristics
        adjusted_scores = {}
        for strategy, base_score in content_strategies.items():
            score = base_score

            # Adjust for complexity
            if characteristics.complexity_score > 0.7:
                if strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
                    score *= 1.2  # Better for complex content
                else:
                    score *= 0.9

            # Adjust for repetition
            if characteristics.repetition_ratio > 0.3:
                if strategy == CompressionStrategy.FREQUENCY_BASED:
                    score *= 1.3  # Better for repetitive content

            # Adjust for structure
            if characteristics.structure_score > 0.6:
                if strategy == CompressionStrategy.HIERARCHICAL:
                    score *= 1.2  # Better for structured content

            # Speed preference adjustment
            if prefer_speed:
                speed_multipliers = {
                    CompressionStrategy.FREQUENCY_BASED: 1.3,
                    CompressionStrategy.TEMPLATE_MATCHING: 1.2,
                    CompressionStrategy.HIERARCHICAL: 1.0,
                    CompressionStrategy.SEMANTIC_CLUSTERING: 0.8
                }
                score *= speed_multipliers.get(strategy, 1.0)

            adjusted_scores[strategy] = score

        # Select best strategy
        best_strategy = max(adjusted_scores.keys(), key=lambda k: adjusted_scores[k])

        # Generate parameters
        return self._generate_parameters(best_strategy, characteristics, target_ratio)

    def _generate_parameters(
        self,
        strategy: CompressionStrategy,
        characteristics: ContentCharacteristics,
        target_ratio: float
    ) -> CompressionParameters:
        """Generate parameters for selected strategy."""

        # Base parameters
        chunk_size = 512
        overlap_ratio = 0.1

        # Strategy-specific adjustments
        if strategy == CompressionStrategy.HIERARCHICAL:
            if characteristics.structure_score > 0.6:
                chunk_size = 1024  # Larger chunks for structured content
            overlap_ratio = 0.15

        elif strategy == CompressionStrategy.SEMANTIC_CLUSTERING:
            clustering_k = max(10, min(100, int(characteristics.length / 1000)))
            chunk_size = 256  # Smaller chunks for clustering

        elif strategy == CompressionStrategy.FREQUENCY_BASED:
            min_frequency = 3 if characteristics.repetition_ratio > 0.4 else 2
            chunk_size = 256

        elif strategy == CompressionStrategy.TEMPLATE_MATCHING:
            template_threshold = 0.9 if characteristics.content_type == ContentType.CODE else 0.8
            chunk_size = 128  # Small chunks for template detection

        # Content type adjustments
        if characteristics.content_type == ContentType.CODE:
            overlap_ratio = 0.05  # Less overlap for code
        elif characteristics.content_type == ContentType.SCIENTIFIC_PAPER:
            chunk_size = 768  # Larger chunks for academic content

        return CompressionParameters(
            strategy=strategy,
            compression_ratio=target_ratio,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio,
            clustering_k=getattr(locals(), 'clustering_k', None),
            min_frequency=getattr(locals(), 'min_frequency', 2),
            template_threshold=getattr(locals(), 'template_threshold', 0.8)
        )


class AdaptiveCompressor(CompressorBase):
    """Adaptive compressor that selects optimal strategies."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_caching: bool = True
    ):
        super().__init__(model_name)

        self.content_analyzer = ContentAnalyzer()
        self.strategy_selector = StrategySelector()
        self.enable_caching = enable_caching

        # Strategy implementations
        self.strategies = {
            CompressionStrategy.HIERARCHICAL: self._hierarchical_compression,
            CompressionStrategy.SEMANTIC_CLUSTERING: self._semantic_clustering_compression,
            CompressionStrategy.FREQUENCY_BASED: self._frequency_based_compression,
            CompressionStrategy.TEMPLATE_MATCHING: self._template_matching_compression
        }

        # Performance cache
        self.strategy_performance = {}

    def compress(self, text: str, **kwargs) -> CompressionResult:
        """Adaptive compression with strategy selection."""
        start_time = time.time()

        # Analyze content
        characteristics = self.content_analyzer.analyze_content(text)

        # Select optimal strategy
        target_ratio = kwargs.get('compression_ratio', 8.0)
        prefer_speed = kwargs.get('prefer_speed', False)

        parameters = self.strategy_selector.select_strategy(
            characteristics, target_ratio, prefer_speed
        )

        # Apply selected strategy
        strategy_impl = self.strategies[parameters.strategy]
        result = strategy_impl(text, parameters, characteristics)

        # Update performance tracking
        processing_time = time.time() - start_time
        self._update_strategy_performance(
            parameters.strategy,
            characteristics.content_type,
            result.compression_ratio,
            processing_time
        )

        # Add adaptive metadata
        result.metadata.update({
            'content_type': characteristics.content_type.value,
            'selected_strategy': parameters.strategy.value,
            'content_complexity': characteristics.complexity_score,
            'adaptive_compression': True
        })

        return result

    def _hierarchical_compression(
        self,
        text: str,
        params: CompressionParameters,
        characteristics: ContentCharacteristics
    ) -> CompressionResult:
        """Hierarchical compression implementation."""
        from .core import ContextCompressor

        # Create hierarchical compressor with adaptive parameters
        compressor = ContextCompressor(
            model_name=self.model_name,
            chunk_size=params.chunk_size,
            compression_ratio=params.compression_ratio,
            overlap_ratio=params.overlap_ratio
        )

        return compressor.compress(text)

    def _semantic_clustering_compression(
        self,
        text: str,
        params: CompressionParameters,
        characteristics: ContentCharacteristics
    ) -> CompressionResult:
        """Semantic clustering-based compression."""
        start_time = time.time()

        # Chunk text into smaller segments
        chunks = self._chunk_text(text, params.chunk_size, params.overlap_ratio)

        if not chunks:
            raise CompressionError("No chunks generated")

        # Generate embeddings
        embeddings = self._encode_chunks(chunks)

        # Perform clustering
        k = params.clustering_k or max(5, min(50, len(chunks) // 10))
        k = min(k, len(chunks))  # Ensure k doesn't exceed number of chunks

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Create mega-tokens from cluster centers
            mega_tokens = []
            for i in range(k):
                cluster_mask = cluster_labels == i
                if not np.any(cluster_mask):
                    continue

                cluster_chunks = [chunks[j] for j in range(len(chunks)) if cluster_mask[j]]
                cluster_embeddings = [embeddings[j] for j in range(len(embeddings)) if cluster_mask[j]]

                # Average embeddings in cluster
                cluster_center = np.mean(cluster_embeddings, axis=0)

                # Create representative text
                representative_text = self._create_cluster_summary(cluster_chunks)

                # Calculate confidence based on cluster coherence
                coherence = self._calculate_cluster_coherence(cluster_embeddings)

                mega_token = MegaToken(
                    vector=cluster_center,
                    metadata={
                        'cluster_id': i,
                        'cluster_size': len(cluster_chunks),
                        'representative_text': representative_text,
                        'compression_method': 'semantic_clustering'
                    },
                    confidence=coherence
                )
                mega_tokens.append(mega_token)

        except ImportError:
            # Fallback to simple sampling if sklearn not available
            step = max(1, len(chunks) // int(len(chunks) / params.compression_ratio))
            selected_indices = range(0, len(chunks), step)

            mega_tokens = []
            for i, idx in enumerate(selected_indices):
                if idx < len(chunks):
                    mega_token = MegaToken(
                        vector=embeddings[idx],
                        metadata={
                            'index': i,
                            'source_chunk': chunks[idx],
                            'compression_method': 'simple_sampling'
                        },
                        confidence=0.7
                    )
                    mega_tokens.append(mega_token)

        processing_time = time.time() - start_time
        original_length = self.count_tokens(text)

        return CompressionResult(
            mega_tokens=mega_tokens,
            original_length=original_length,
            compressed_length=len(mega_tokens),
            compression_ratio=original_length / max(len(mega_tokens), 1),
            processing_time=processing_time,
            metadata={
                'strategy': 'semantic_clustering',
                'clusters': k if 'kmeans' in locals() else len(mega_tokens),
                'original_chunks': len(chunks)
            }
        )

    def _frequency_based_compression(
        self,
        text: str,
        params: CompressionParameters,
        characteristics: ContentCharacteristics
    ) -> CompressionResult:
        """Frequency-based compression for repetitive content."""
        start_time = time.time()

        # Use TF-IDF to identify important terms/phrases
        chunks = self._chunk_text(text, params.chunk_size, params.overlap_ratio)

        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=1000,
                min_df=params.min_frequency,
                ngram_range=(1, 3),
                stop_words='english'
            )

            tfidf_matrix = vectorizer.fit_transform(chunks)
            feature_names = vectorizer.get_feature_names_out()

            # Get top features by TF-IDF score
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[::-1]

            # Select chunks based on feature importance
            target_chunks = int(len(chunks) / params.compression_ratio)
            target_chunks = max(1, min(target_chunks, len(chunks)))

            # Score chunks by their TF-IDF feature coverage
            chunk_scores = np.sum(tfidf_matrix[:, top_indices[:50]], axis=1).A1
            selected_indices = np.argsort(chunk_scores)[::-1][:target_chunks]

            # Create mega-tokens
            mega_tokens = []
            embeddings = self._encode_chunks(chunks)

            for i, chunk_idx in enumerate(selected_indices):
                chunk = chunks[chunk_idx]
                embedding = embeddings[chunk_idx]

                # Get top features for this chunk
                chunk_features = tfidf_matrix[chunk_idx].toarray()[0]
                top_feature_indices = np.argsort(chunk_features)[::-1][:10]
                top_features = [feature_names[idx] for idx in top_feature_indices if chunk_features[idx] > 0]

                mega_token = MegaToken(
                    vector=embedding,
                    metadata={
                        'chunk_index': chunk_idx,
                        'tfidf_score': float(chunk_scores[chunk_idx]),
                        'top_features': top_features,
                        'source_text': chunk[:200] + "..." if len(chunk) > 200 else chunk,
                        'compression_method': 'frequency_based'
                    },
                    confidence=min(1.0, chunk_scores[chunk_idx] / 10.0)
                )
                mega_tokens.append(mega_token)

        except ImportError:
            # Fallback if sklearn not available
            mega_tokens = self._simple_frequency_fallback(text, chunks, params)

        processing_time = time.time() - start_time
        original_length = self.count_tokens(text)

        return CompressionResult(
            mega_tokens=mega_tokens,
            original_length=original_length,
            compressed_length=len(mega_tokens),
            compression_ratio=original_length / max(len(mega_tokens), 1),
            processing_time=processing_time,
            metadata={
                'strategy': 'frequency_based',
                'min_frequency': params.min_frequency,
                'selected_chunks': len(mega_tokens)
            }
        )

    def _template_matching_compression(
        self,
        text: str,
        params: CompressionParameters,
        characteristics: ContentCharacteristics
    ) -> CompressionResult:
        """Template-based compression for structured content."""
        start_time = time.time()

        # Detect templates and patterns
        chunks = self._chunk_text(text, params.chunk_size, params.overlap_ratio)
        templates = self._extract_templates(chunks, params.template_threshold)

        # Group chunks by template similarity
        template_groups = {}
        for i, chunk in enumerate(chunks):
            best_template = self._find_best_template(chunk, templates, params.template_threshold)

            if best_template:
                template_id = best_template['id']
                if template_id not in template_groups:
                    template_groups[template_id] = []
                template_groups[template_id].append((i, chunk))
            else:
                # Create new template group for unique content
                unique_id = f"unique_{i}"
                template_groups[unique_id] = [(i, chunk)]

        # Create mega-tokens from template groups
        mega_tokens = []
        embeddings = self._encode_chunks(chunks)

        for template_id, group_chunks in template_groups.items():
            if not group_chunks:
                continue

            # Average embeddings for template group
            group_indices = [idx for idx, _ in group_chunks]
            group_embeddings = [embeddings[idx] for idx in group_indices]
            avg_embedding = np.mean(group_embeddings, axis=0)

            # Create representative text
            if len(group_chunks) > 1:
                representative_text = self._create_template_summary(
                    [chunk for _, chunk in group_chunks]
                )
            else:
                representative_text = group_chunks[0][1]

            mega_token = MegaToken(
                vector=avg_embedding,
                metadata={
                    'template_id': template_id,
                    'instances': len(group_chunks),
                    'representative_text': representative_text[:200] + "..." if len(representative_text) > 200 else representative_text,
                    'compression_method': 'template_matching',
                    'chunk_indices': group_indices
                },
                confidence=min(1.0, len(group_chunks) / 10.0)
            )
            mega_tokens.append(mega_token)

        processing_time = time.time() - start_time
        original_length = self.count_tokens(text)

        return CompressionResult(
            mega_tokens=mega_tokens,
            original_length=original_length,
            compressed_length=len(mega_tokens),
            compression_ratio=original_length / max(len(mega_tokens), 1),
            processing_time=processing_time,
            metadata={
                'strategy': 'template_matching',
                'templates_found': len(templates),
                'template_groups': len(template_groups)
            }
        )

    def _chunk_text(self, text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
        """Chunk text with overlapping segments."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        overlap_size = int(chunk_size * overlap_ratio)
        step_size = chunk_size - overlap_size

        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))

        return chunks

    def _encode_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        """Encode text chunks into embeddings."""
        if hasattr(self.model, 'encode'):
            embeddings = self.model.encode(chunks, convert_to_numpy=True)
            return [emb for emb in embeddings]
        else:
            # Fallback implementation for transformers models
            embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    embeddings.append(embedding[0])

            return embeddings

    def _create_cluster_summary(self, chunks: list[str]) -> str:
        """Create summary text for a cluster of chunks."""
        if len(chunks) == 1:
            return chunks[0]

        # Simple approach: take first chunk as representative
        # In practice, could use more sophisticated summarization
        return chunks[0]

    def _calculate_cluster_coherence(self, embeddings: list[np.ndarray]) -> float:
        """Calculate coherence score for a cluster."""
        if len(embeddings) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    def _simple_frequency_fallback(
        self,
        text: str,
        chunks: list[str],
        params: CompressionParameters
    ) -> list[MegaToken]:
        """Simple frequency-based fallback without sklearn."""
        # Count word frequencies
        word_counts = {}
        for chunk in chunks:
            words = chunk.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Score chunks by high-frequency word coverage
        chunk_scores = []
        high_freq_words = set(word for word, count in word_counts.items()
                             if count >= params.min_frequency)

        for chunk in chunks:
            words = set(chunk.lower().split())
            score = len(words.intersection(high_freq_words))
            chunk_scores.append(score)

        # Select top chunks
        target_count = max(1, int(len(chunks) / params.compression_ratio))
        top_indices = sorted(range(len(chunk_scores)),
                           key=lambda i: chunk_scores[i], reverse=True)[:target_count]

        # Create mega-tokens
        mega_tokens = []
        embeddings = self._encode_chunks(chunks)

        for i, chunk_idx in enumerate(top_indices):
            mega_token = MegaToken(
                vector=embeddings[chunk_idx],
                metadata={
                    'chunk_index': chunk_idx,
                    'frequency_score': chunk_scores[chunk_idx],
                    'source_text': chunks[chunk_idx][:200] + "..." if len(chunks[chunk_idx]) > 200 else chunks[chunk_idx],
                    'compression_method': 'simple_frequency'
                },
                confidence=min(1.0, chunk_scores[chunk_idx] / 20.0)
            )
            mega_tokens.append(mega_token)

        return mega_tokens

    def _extract_templates(self, chunks: list[str], threshold: float) -> list[dict[str, Any]]:
        """Extract templates from chunks."""
        templates = []

        for i, chunk in enumerate(chunks):
            # Simple pattern extraction (could be more sophisticated)
            # Look for common structures in code or documentation

            # Code patterns
            if re.search(r'\b(def|class|function|import)\b', chunk):
                template_type = 'code_definition'
            elif re.search(r'^\s*#.*$', chunk, re.MULTILINE):
                template_type = 'comment_block'
            elif re.search(r'^#{1,6}\s+', chunk, re.MULTILINE):
                template_type = 'markdown_header'
            else:
                continue  # No clear template pattern

            templates.append({
                'id': f"{template_type}_{len(templates)}",
                'type': template_type,
                'example': chunk,
                'pattern': self._create_pattern(chunk, template_type)
            })

        return templates

    def _create_pattern(self, text: str, template_type: str) -> str:
        """Create a pattern from template text."""
        # Simplified pattern creation
        if template_type == 'code_definition':
            return re.sub(r'\b[a-zA-Z_]\w*\b', '<IDENTIFIER>', text)
        elif template_type == 'markdown_header':
            return re.sub(r'#{1,6}\s+.*$', '# <HEADER>', text, flags=re.MULTILINE)
        else:
            return text

    def _find_best_template(
        self,
        chunk: str,
        templates: list[dict[str, Any]],
        threshold: float
    ) -> dict[str, Any] | None:
        """Find best matching template for chunk."""
        best_match = None
        best_score = 0.0

        for template in templates:
            # Simple similarity based on common words
            template_words = set(template['example'].lower().split())
            chunk_words = set(chunk.lower().split())

            if len(template_words) == 0:
                continue

            intersection = len(template_words.intersection(chunk_words))
            union = len(template_words.union(chunk_words))

            if union == 0:
                continue

            similarity = intersection / union

            if similarity > threshold and similarity > best_score:
                best_score = similarity
                best_match = template

        return best_match

    def _create_template_summary(self, chunks: list[str]) -> str:
        """Create summary for template group."""
        if len(chunks) == 1:
            return chunks[0]

        # Find common structure and create summary
        # For now, return the shortest chunk as representative
        return min(chunks, key=len)

    def _update_strategy_performance(
        self,
        strategy: CompressionStrategy,
        content_type: ContentType,
        compression_ratio: float,
        processing_time: float
    ):
        """Update strategy performance tracking."""
        key = (strategy, content_type)

        if key not in self.strategy_performance:
            self.strategy_performance[key] = {
                'count': 0,
                'total_ratio': 0.0,
                'total_time': 0.0,
                'best_ratio': 0.0,
                'best_time': float('inf')
            }

        perf = self.strategy_performance[key]
        perf['count'] += 1
        perf['total_ratio'] += compression_ratio
        perf['total_time'] += processing_time
        perf['best_ratio'] = max(perf['best_ratio'], compression_ratio)
        perf['best_time'] = min(perf['best_time'], processing_time)

    def get_strategy_performance(self) -> dict[str, Any]:
        """Get strategy performance statistics."""
        results = {}

        for (strategy, content_type), perf in self.strategy_performance.items():
            key = f"{strategy.value}_{content_type.value}"

            results[key] = {
                'count': perf['count'],
                'avg_compression_ratio': perf['total_ratio'] / perf['count'],
                'avg_processing_time': perf['total_time'] / perf['count'],
                'best_compression_ratio': perf['best_ratio'],
                'best_processing_time': perf['best_time']
            }

        return results

    def decompress(self, mega_tokens: list[MegaToken], **kwargs) -> str:
        """Decompress mega-tokens back to text."""
        if not mega_tokens:
            return ""

        # Reconstruct based on metadata
        parts = []
        for token in mega_tokens:
            if 'representative_text' in token.metadata:
                parts.append(token.metadata['representative_text'])
            elif 'source_text' in token.metadata:
                parts.append(token.metadata['source_text'])
            elif 'source_chunk' in token.metadata:
                parts.append(token.metadata['source_chunk'])

        return " ".join(parts) if parts else ""


class StreamingCompressor:
    """Streaming compressor for real-time data processing."""

    def __init__(
        self,
        base_compressor: AdaptiveCompressor,
        window_size: int = 1000,
        compression_interval: int = 100
    ):
        self.base_compressor = base_compressor
        self.window_size = window_size
        self.compression_interval = compression_interval

        self.stream_state = StreamingState(
            buffer=deque(maxlen=window_size),
            window_tokens=[],
            processed_count=0,
            total_compression_ratio=1.0,
            last_update=time.time()
        )

        self._lock = threading.RLock()

    def add_text(self, text: str) -> list[MegaToken] | None:
        """Add text to stream and return compressed tokens if threshold reached."""
        with self._lock:
            self.stream_state.buffer.append(text)
            self.stream_state.processed_count += 1

            # Check if compression is needed
            if len(self.stream_state.buffer) >= self.compression_interval:
                return self._compress_window()

            return None

    def _compress_window(self) -> list[MegaToken]:
        """Compress current window of text."""
        # Combine buffered text
        combined_text = " ".join(self.stream_state.buffer)

        # Compress using adaptive compressor
        result = self.base_compressor.compress(combined_text)

        # Update stream state
        self.stream_state.window_tokens = result.mega_tokens
        self.stream_state.total_compression_ratio = result.compression_ratio
        self.stream_state.last_update = time.time()

        # Create checksum for incremental updates
        text_hash = hashlib.md5(combined_text.encode()).hexdigest()
        self.stream_state.checksum = text_hash

        # Clear buffer (keep some overlap)
        overlap_size = min(50, len(self.stream_state.buffer) // 4)
        for _ in range(len(self.stream_state.buffer) - overlap_size):
            self.stream_state.buffer.popleft()

        return result.mega_tokens

    def get_current_tokens(self) -> list[MegaToken]:
        """Get current compressed representation."""
        with self._lock:
            return self.stream_state.window_tokens.copy()

    def get_stream_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        with self._lock:
            return {
                'buffer_size': len(self.stream_state.buffer),
                'processed_count': self.stream_state.processed_count,
                'current_tokens': len(self.stream_state.window_tokens),
                'compression_ratio': self.stream_state.total_compression_ratio,
                'last_update': self.stream_state.last_update,
                'checksum': self.stream_state.checksum
            }
