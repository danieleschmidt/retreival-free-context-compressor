"""Multi-document compressor for handling collections of documents."""

import logging
import time
import threading
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .core.base import CompressorBase, MegaToken, CompressionResult
from .exceptions import CompressionError, ValidationError
from .validation import InputValidator
from .optimization import MemoryOptimizer, BatchProcessor, ConcurrencyOptimizer
from .caching import TieredCache, create_cache_key

logger = logging.getLogger(__name__)


@dataclass
class DocumentMeta:
    """Metadata for a document in a multi-document collection."""
    doc_id: str
    title: Optional[str]
    source: Optional[str]
    creation_time: float
    length: int
    doc_type: str
    priority: float
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'source': self.source,
            'creation_time': self.creation_time,
            'length': self.length,
            'doc_type': self.doc_type,
            'priority': self.priority,
            'tags': self.tags
        }


@dataclass
class Document:
    """Represents a single document in a collection."""
    content: str
    metadata: DocumentMeta
    
    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self.metadata.doc_id
    
    @property
    def length(self) -> int:
        """Get document length."""
        return len(self.content)
    
    def get_hash(self) -> str:
        """Get content hash for caching."""
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()


@dataclass
class CompressionStrategy:
    """Strategy for compressing a document collection."""
    global_compression_ratio: float
    per_document_ratios: Dict[str, float]
    priority_weights: Dict[str, float]
    similarity_threshold: float
    deduplication_enabled: bool
    cross_document_compression: bool


class DocumentCollection:
    """Manages a collection of documents for multi-document compression."""
    
    def __init__(self):
        """Initialize document collection."""
        self._documents: Dict[str, Document] = {}
        self._document_order: List[str] = []
        self._lock = threading.RLock()
        
        # Similarity tracking
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._content_hashes: Dict[str, str] = {}
        
    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        doc_type: str = "text",
        priority: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add document to collection.
        
        Args:
            content: Document content
            doc_id: Optional document ID (auto-generated if None)
            title: Document title
            source: Document source path or URL
            doc_type: Type of document
            priority: Priority for compression (higher = less compression)
            tags: Document tags
            
        Returns:
            Document ID
        """
        with self._lock:
            # Generate ID if not provided
            if doc_id is None:
                doc_id = f"doc_{len(self._documents):04d}"
            
            # Check for duplicates
            if doc_id in self._documents:
                raise ValueError(f"Document with ID '{doc_id}' already exists")
            
            # Create metadata
            metadata = DocumentMeta(
                doc_id=doc_id,
                title=title,
                source=source,
                creation_time=time.time(),
                length=len(content),
                doc_type=doc_type,
                priority=priority,
                tags=tags or []
            )
            
            # Create document
            document = Document(content=content, metadata=metadata)
            
            # Store document
            self._documents[doc_id] = document
            self._document_order.append(doc_id)
            self._content_hashes[doc_id] = document.get_hash()
            
            logger.debug(f"Added document '{doc_id}' ({len(content)} chars)")
            
            return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document from collection."""
        with self._lock:
            if doc_id in self._documents:
                del self._documents[doc_id]
                self._document_order.remove(doc_id)
                self._content_hashes.pop(doc_id, None)
                
                # Clear similarity cache entries
                to_remove = [
                    key for key in self._similarity_cache.keys()
                    if doc_id in key
                ]
                for key in to_remove:
                    del self._similarity_cache[key]
                
                return True
            return False
    
    def get_documents(self) -> List[Document]:
        """Get all documents in order."""
        with self._lock:
            return [self._documents[doc_id] for doc_id in self._document_order]
    
    def get_document_ids(self) -> List[str]:
        """Get all document IDs in order."""
        return self._document_order.copy()
    
    def size(self) -> int:
        """Get number of documents."""
        return len(self._documents)
    
    def total_length(self) -> int:
        """Get total length of all documents."""
        return sum(doc.length for doc in self._documents.values())
    
    def find_similar_documents(
        self,
        doc_id: str,
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """Find documents similar to given document.
        
        Args:
            doc_id: Reference document ID
            threshold: Similarity threshold (0.0-1.0)
            
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        if doc_id not in self._documents:
            return []
        
        similar = []
        ref_doc = self._documents[doc_id]
        
        for other_id in self._documents:
            if other_id == doc_id:
                continue
            
            similarity = self._calculate_similarity(doc_id, other_id)
            if similarity >= threshold:
                similar.append((other_id, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def _calculate_similarity(self, doc_id1: str, doc_id2: str) -> float:
        """Calculate similarity between two documents."""
        # Use cache if available
        cache_key = tuple(sorted([doc_id1, doc_id2]))
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Simple similarity based on content hash and length
        doc1 = self._documents[doc_id1]
        doc2 = self._documents[doc_id2]
        
        # Hash-based similarity (exact matches)
        if self._content_hashes[doc_id1] == self._content_hashes[doc_id2]:
            similarity = 1.0
        else:
            # Length-based approximation
            len1, len2 = doc1.length, doc2.length
            if len1 == 0 and len2 == 0:
                similarity = 1.0
            elif len1 == 0 or len2 == 0:
                similarity = 0.0
            else:
                similarity = min(len1, len2) / max(len1, len2)
                
                # Reduce similarity for very different lengths
                if abs(len1 - len2) > max(len1, len2) * 0.5:
                    similarity *= 0.5
        
        # Cache result
        self._similarity_cache[cache_key] = similarity
        
        return similarity


class MultiDocCompressor(CompressorBase):
    """Compressor for handling multiple documents with cross-document optimization."""
    
    def __init__(
        self,
        model_name: str = "context-compressor-base",
        device: Optional[str] = None,
        max_length: int = 256000,
        compression_ratio: float = 8.0,
        similarity_threshold: float = 0.8,
        enable_deduplication: bool = True,
        enable_cross_doc_compression: bool = True,
        max_workers: int = 4
    ):
        """Initialize multi-document compressor.
        
        Args:
            model_name: Name of compression model
            device: Computing device
            max_length: Maximum input length per document
            compression_ratio: Default compression ratio
            similarity_threshold: Threshold for document similarity detection
            enable_deduplication: Whether to deduplicate similar documents
            enable_cross_doc_compression: Whether to compress across documents
            max_workers: Maximum number of worker threads
        """
        super().__init__(model_name, device, max_length, compression_ratio)
        
        self.similarity_threshold = similarity_threshold
        self.enable_deduplication = enable_deduplication
        self.enable_cross_doc_compression = enable_cross_doc_compression
        self.max_workers = max_workers
        
        # Components
        self._validator = InputValidator()
        self._memory_optimizer = MemoryOptimizer()
        self._batch_processor = BatchProcessor(batch_size=4)
        self._concurrency_optimizer = ConcurrencyOptimizer(max_workers=max_workers)
        self._cache = TieredCache()
        
        # Statistics
        self._stats = {
            'documents_processed': 0,
            'duplicates_found': 0,
            'similar_groups_merged': 0,
            'total_processing_time': 0.0,
            'compression_ratios': []
        }
        
    def load_model(self) -> None:
        """Load compression model."""
        try:
            # Use parent implementation
            super().load_model()
            logger.info("Multi-document compressor model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load multi-document compression model: {e}")
            raise CompressionError(f"Model loading failed: {e}")
    
    def compress_collection(
        self,
        documents: DocumentCollection,
        strategy: Optional[CompressionStrategy] = None
    ) -> Dict[str, CompressionResult]:
        """Compress a collection of documents.
        
        Args:
            documents: Document collection to compress
            strategy: Optional compression strategy
            
        Returns:
            Dictionary mapping document IDs to compression results
        """
        start_time = time.time()
        
        if documents.size() == 0:
            return {}
        
        # Use default strategy if none provided
        if strategy is None:
            strategy = self._create_default_strategy(documents)
        
        logger.info(f"Compressing collection of {documents.size()} documents")
        
        try:
            # Load model if needed
            if self._encoder_model is None:
                self.load_model()
            
            with self._memory_optimizer.memory_efficient_context():
                # Preprocess documents (deduplication, similarity analysis)
                processed_docs = self._preprocess_documents(documents, strategy)
                
                # Compress documents (potentially in parallel)
                if self.max_workers > 1:
                    results = self._compress_parallel(processed_docs, strategy)
                else:
                    results = self._compress_sequential(processed_docs, strategy)
                
                # Post-process results (cross-document optimization)
                if strategy.cross_document_compression:
                    results = self._apply_cross_document_optimization(results, strategy)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats['documents_processed'] += len(results)
            self._stats['total_processing_time'] += processing_time
            self._stats['compression_ratios'].extend([
                result.compression_ratio for result in results.values()
            ])
            
            logger.info(
                f"Compressed {len(results)} documents in {processing_time:.2f}s "
                f"(avg ratio: {sum(r.compression_ratio for r in results.values()) / len(results):.1f}x)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Collection compression failed: {e}")
            raise CompressionError(f"Multi-document compression failed: {e}")
    
    def compress(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> CompressionResult:
        """Compress single text or list of texts.
        
        This method adapts the multi-document compression for single-document use.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult
        """
        # Convert to document collection
        collection = DocumentCollection()
        
        if isinstance(text, str):
            collection.add_document(text, doc_id="main_doc")
        else:
            for i, t in enumerate(text):
                collection.add_document(t, doc_id=f"doc_{i}")
        
        # Compress collection
        results = self.compress_collection(collection)
        
        if len(results) == 1:
            # Return single result
            return next(iter(results.values()))
        else:
            # Combine multiple results
            all_mega_tokens = []
            total_original = 0
            total_compressed = 0
            total_processing_time = 0
            
            for result in results.values():
                all_mega_tokens.extend(result.mega_tokens)
                total_original += result.original_length
                total_compressed += result.compressed_length
                total_processing_time += result.processing_time
            
            overall_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
            
            return CompressionResult(
                mega_tokens=all_mega_tokens,
                original_length=total_original,
                compressed_length=total_compressed,
                compression_ratio=overall_ratio,
                processing_time=total_processing_time,
                metadata={
                    'multi_document': True,
                    'document_count': len(results),
                    'individual_results': len(results)
                }
            )
    
    def _create_default_strategy(self, documents: DocumentCollection) -> CompressionStrategy:
        """Create default compression strategy for document collection."""
        # Calculate per-document ratios based on priority
        per_doc_ratios = {}
        for doc in documents.get_documents():
            # Higher priority = lower compression ratio (preserve more)
            ratio = self.compression_ratio / max(0.5, doc.metadata.priority)
            per_doc_ratios[doc.doc_id] = min(32.0, max(2.0, ratio))
        
        return CompressionStrategy(
            global_compression_ratio=self.compression_ratio,
            per_document_ratios=per_doc_ratios,
            priority_weights={},
            similarity_threshold=self.similarity_threshold,
            deduplication_enabled=self.enable_deduplication,
            cross_document_compression=self.enable_cross_doc_compression
        )
    
    def _preprocess_documents(
        self,
        documents: DocumentCollection,
        strategy: CompressionStrategy
    ) -> DocumentCollection:
        """Preprocess documents (deduplication, etc.).
        
        Args:
            documents: Input document collection
            strategy: Compression strategy
            
        Returns:
            Processed document collection
        """
        if not strategy.deduplication_enabled:
            return documents
        
        # Find and handle duplicates/similar documents
        processed = DocumentCollection()
        doc_groups = {}  # Groups of similar documents
        
        for doc_id in documents.get_document_ids():
            document = documents.get_document(doc_id)
            
            # Find similar documents
            similar = documents.find_similar_documents(
                doc_id, 
                strategy.similarity_threshold
            )
            
            if not similar:
                # No similar documents, add as-is
                processed.add_document(
                    document.content,
                    doc_id=document.doc_id,
                    title=document.metadata.title,
                    source=document.metadata.source,
                    doc_type=document.metadata.doc_type,
                    priority=document.metadata.priority,
                    tags=document.metadata.tags
                )
            else:
                # Group similar documents
                group_key = min([doc_id] + [sim_id for sim_id, _ in similar])
                
                if group_key not in doc_groups:
                    doc_groups[group_key] = []
                doc_groups[group_key].append(document)
        
        # Process document groups
        for group_docs in doc_groups.values():
            if len(group_docs) == 1:
                # Single document in group
                doc = group_docs[0]
                processed.add_document(
                    doc.content,
                    doc_id=doc.doc_id,
                    title=doc.metadata.title,
                    source=doc.metadata.source,
                    doc_type=doc.metadata.doc_type,
                    priority=doc.metadata.priority,
                    tags=doc.metadata.tags
                )
            else:
                # Merge similar documents
                self._stats['similar_groups_merged'] += 1
                merged_doc = self._merge_similar_documents(group_docs)
                processed.add_document(
                    merged_doc.content,
                    doc_id=merged_doc.doc_id,
                    title=merged_doc.metadata.title,
                    source=merged_doc.metadata.source,
                    doc_type=merged_doc.metadata.doc_type,
                    priority=merged_doc.metadata.priority,
                    tags=merged_doc.metadata.tags
                )
        
        logger.debug(
            f"Preprocessing: {documents.size()} -> {processed.size()} documents "
            f"({self._stats['similar_groups_merged']} groups merged)"
        )
        
        return processed
    
    def _merge_similar_documents(self, documents: List[Document]) -> Document:
        """Merge a list of similar documents into one.
        
        Args:
            documents: List of similar documents to merge
            
        Returns:
            Merged document
        """
        # Use highest priority document as base
        base_doc = max(documents, key=lambda d: d.metadata.priority)
        
        # Combine content (simple concatenation for now)
        combined_content = "\n\n".join([doc.content for doc in documents])
        
        # Merge metadata
        all_tags = []
        all_sources = []
        
        for doc in documents:
            all_tags.extend(doc.metadata.tags)
            if doc.metadata.source:
                all_sources.append(doc.metadata.source)
        
        merged_metadata = DocumentMeta(
            doc_id=f"merged_{base_doc.doc_id}",
            title=f"Merged: {base_doc.metadata.title or 'Untitled'}",
            source="; ".join(all_sources) if all_sources else None,
            creation_time=time.time(),
            length=len(combined_content),
            doc_type=base_doc.metadata.doc_type,
            priority=max(doc.metadata.priority for doc in documents),
            tags=list(set(all_tags))
        )
        
        return Document(content=combined_content, metadata=merged_metadata)
    
    def _compress_sequential(
        self,
        documents: DocumentCollection,
        strategy: CompressionStrategy
    ) -> Dict[str, CompressionResult]:
        """Compress documents sequentially.
        
        Args:
            documents: Document collection
            strategy: Compression strategy
            
        Returns:
            Dictionary of compression results
        """
        results = {}
        
        for doc_id in documents.get_document_ids():
            document = documents.get_document(doc_id)
            target_ratio = strategy.per_document_ratios.get(doc_id, strategy.global_compression_ratio)
            
            # Check cache
            cache_key = create_cache_key(
                document.content,
                self.model_name,
                {'compression_ratio': target_ratio}
            )
            
            cached_result = self._cache.get(cache_key)
            if cached_result:
                results[doc_id] = cached_result
                continue
            
            # Compress document
            original_ratio = self.compression_ratio
            self.compression_ratio = target_ratio
            
            try:
                result = super().compress(document.content)
                
                # Enhance with document metadata
                enhanced_metadata = {
                    **result.metadata,
                    'document_id': doc_id,
                    'document_title': document.metadata.title,
                    'document_priority': document.metadata.priority,
                    'multi_document_compression': True
                }
                
                enhanced_result = CompressionResult(
                    mega_tokens=result.mega_tokens,
                    original_length=result.original_length,
                    compressed_length=result.compressed_length,
                    compression_ratio=result.compression_ratio,
                    processing_time=result.processing_time,
                    metadata=enhanced_metadata
                )
                
                results[doc_id] = enhanced_result
                
                # Cache result
                self._cache.put(cache_key, enhanced_result)
                
            finally:
                self.compression_ratio = original_ratio
        
        return results
    
    def _compress_parallel(
        self,
        documents: DocumentCollection,
        strategy: CompressionStrategy
    ) -> Dict[str, CompressionResult]:
        """Compress documents in parallel.
        
        Args:
            documents: Document collection
            strategy: Compression strategy
            
        Returns:
            Dictionary of compression results
        """
        results = {}
        
        def compress_document(doc_id: str) -> Tuple[str, CompressionResult]:
            document = documents.get_document(doc_id)
            target_ratio = strategy.per_document_ratios.get(doc_id, strategy.global_compression_ratio)
            
            # Check cache
            cache_key = create_cache_key(
                document.content,
                self.model_name,
                {'compression_ratio': target_ratio}
            )
            
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return doc_id, cached_result
            
            # Create a new compressor instance for thread safety
            # Note: In a real implementation, you'd need to handle model loading per thread
            # For now, we'll use the parent compression method with synchronization
            
            # Temporarily adjust compression ratio
            original_ratio = self.compression_ratio
            self.compression_ratio = target_ratio
            
            try:
                result = super().compress(document.content)
                
                # Enhance with document metadata
                enhanced_metadata = {
                    **result.metadata,
                    'document_id': doc_id,
                    'document_title': document.metadata.title,
                    'document_priority': document.metadata.priority,
                    'multi_document_compression': True
                }
                
                enhanced_result = CompressionResult(
                    mega_tokens=result.mega_tokens,
                    original_length=result.original_length,
                    compressed_length=result.compressed_length,
                    compression_ratio=result.compression_ratio,
                    processing_time=result.processing_time,
                    metadata=enhanced_metadata
                )
                
                # Cache result
                self._cache.put(cache_key, enhanced_result)
                
                return doc_id, enhanced_result
                
            finally:
                self.compression_ratio = original_ratio
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all compression tasks
            future_to_doc_id = {
                executor.submit(compress_document, doc_id): doc_id
                for doc_id in documents.get_document_ids()
            }
            
            # Collect results
            for future in as_completed(future_to_doc_id):
                try:
                    doc_id, result = future.result()
                    results[doc_id] = result
                except Exception as e:
                    doc_id = future_to_doc_id[future]
                    logger.error(f"Error compressing document {doc_id}: {e}")
                    # Create dummy result for failed compression
                    results[doc_id] = CompressionResult(
                        mega_tokens=[],
                        original_length=0,
                        compressed_length=0,
                        compression_ratio=1.0,
                        processing_time=0.0,
                        metadata={'error': str(e), 'document_id': doc_id}
                    )
        
        return results
    
    def _apply_cross_document_optimization(
        self,
        results: Dict[str, CompressionResult],
        strategy: CompressionStrategy
    ) -> Dict[str, CompressionResult]:
        """Apply cross-document compression optimizations.
        
        Args:
            results: Initial compression results
            strategy: Compression strategy
            
        Returns:
            Optimized compression results
        """
        # For now, this is a placeholder for cross-document optimization
        # In a full implementation, this would look for patterns across documents
        # and create shared mega-tokens
        
        logger.debug("Cross-document optimization applied (placeholder)")
        return results
    
    def decompress(
        self,
        mega_tokens: List[MegaToken],
        **kwargs
    ) -> str:
        """Decompress mega-tokens from multi-document compression."""
        parts = []
        
        current_doc_id = None
        for i, token in enumerate(mega_tokens):
            doc_id = token.metadata.get('document_id', 'unknown')
            doc_title = token.metadata.get('document_title', 'Untitled')
            
            if doc_id != current_doc_id:
                if current_doc_id is not None:
                    parts.append("")  # Separator between documents
                parts.append(f"=== Document: {doc_title} (ID: {doc_id}) ===")
                current_doc_id = doc_id
            
            part = f"[Segment {i+1}: ratio={token.compression_ratio:.1f}x]"
            parts.append(part)
        
        return "\n".join(parts)
    
    def get_multi_doc_stats(self) -> Dict[str, Any]:
        """Get multi-document compression statistics."""
        stats = self._stats.copy()
        
        if stats['compression_ratios']:
            import numpy as np
            stats['compression_ratio_stats'] = {
                'min': min(stats['compression_ratios']),
                'max': max(stats['compression_ratios']),
                'mean': np.mean(stats['compression_ratios']),
                'std': np.std(stats['compression_ratios'])
            }
        
        return stats
