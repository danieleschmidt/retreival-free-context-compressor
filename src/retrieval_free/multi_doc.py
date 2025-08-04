"""Multi-document compression with cross-document deduplication."""

import hashlib
import time
from typing import Dict, List, Set, Tuple

import numpy as np

from .core import CompressorBase, CompressionResult, ContextCompressor, MegaToken
from .observability import monitor_performance


class MultiDocCompressor(CompressorBase):
    """Multi-document compressor with deduplication and cross-document attention."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        deduplication: bool = True,
        cross_doc_attention: bool = True,
        similarity_threshold: float = 0.85,
        compression_ratio: float = 8.0
    ):
        super().__init__(model_name)
        self.deduplication = deduplication
        self.cross_doc_attention = cross_doc_attention
        self.similarity_threshold = similarity_threshold
        self.compression_ratio = compression_ratio
        
        # Cache for seen content
        self.content_hashes: Set[str] = set()
        self.document_embeddings: List[Tuple[str, np.ndarray]] = []
        
    @monitor_performance  
    def compress(self, documents: List[str], **kwargs) -> CompressionResult:
        """Compress multiple documents with cross-document optimization."""
        if isinstance(documents, str):
            # Single document passed as string
            documents = [documents]
        
        start_time = time.time()
        
        # Step 1: Deduplicate documents if enabled
        if self.deduplication:
            documents = self._deduplicate_documents(documents)
        
        # Step 2: Individual document compression
        all_mega_tokens = []
        total_original_length = 0
        document_results = []
        
        for doc_id, document in enumerate(documents):
            compressor = ContextCompressor(
                model_name=self.model_name,
                compression_ratio=self.compression_ratio
            )
            
            doc_result = compressor.compress(document)
            
            # Add document metadata
            for token in doc_result.mega_tokens:
                token.metadata["document_id"] = doc_id
                token.metadata["document_hash"] = self._hash_text(document[:100])
            
            document_results.append(doc_result)
            all_mega_tokens.extend(doc_result.mega_tokens)
            total_original_length += doc_result.original_length
        
        # Step 3: Cross-document deduplication and attention
        if self.cross_doc_attention and len(documents) > 1:
            all_mega_tokens = self._apply_cross_document_compression(
                all_mega_tokens, documents
            )
        
        processing_time = time.time() - start_time
        compressed_length = len(all_mega_tokens)
        
        return CompressionResult(
            mega_tokens=all_mega_tokens,
            original_length=total_original_length,
            compressed_length=compressed_length,
            compression_ratio=self.get_compression_ratio(total_original_length, compressed_length),
            processing_time=processing_time,
            metadata={
                "multi_document": True,
                "documents_processed": len(documents),
                "deduplication_enabled": self.deduplication,
                "cross_doc_attention": self.cross_doc_attention,
                "similarity_threshold": self.similarity_threshold,
                "document_results": [
                    {
                        "doc_id": i,
                        "original_length": result.original_length,
                        "tokens": len(result.mega_tokens)
                    }
                    for i, result in enumerate(document_results)
                ]
            }
        )
    
    def decompress(self, mega_tokens: List[MegaToken], **kwargs) -> List[str]:
        """Decompress mega-tokens back to individual documents."""
        if not mega_tokens:
            return []
        
        # Group tokens by document_id
        documents_dict = {}
        for token in mega_tokens:
            doc_id = token.metadata.get("document_id", 0)
            if doc_id not in documents_dict:
                documents_dict[doc_id] = []
            documents_dict[doc_id].append(token)
        
        # Reconstruct each document
        reconstructed_docs = []
        for doc_id in sorted(documents_dict.keys()):
            doc_tokens = documents_dict[doc_id]
            doc_parts = []
            
            for token in doc_tokens:
                if "source_text" in token.metadata:
                    doc_parts.append(token.metadata["source_text"])
            
            reconstructed_docs.append(" ".join(doc_parts))
        
        return reconstructed_docs
    
    def compress_collection(
        self,
        documents: List[str],
        preserve_citations: bool = True,
        create_index: bool = True
    ) -> CompressionResult:
        """Compress an entire document collection with advanced features."""
        # Add citation preservation and indexing
        if preserve_citations:
            documents = self._preserve_citations(documents)
        
        result = self.compress(documents)
        
        if create_index:
            result.metadata["document_index"] = self._create_document_index(documents)
        
        return result
    
    def _deduplicate_documents(self, documents: List[str]) -> List[str]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        seen_hashes = set()
        
        for document in documents:
            # Fast hash-based deduplication
            doc_hash = self._hash_text(document)
            if doc_hash in seen_hashes:
                continue
            
            # Semantic similarity check against existing documents
            is_duplicate = False
            if self.cross_doc_attention:
                for existing_doc in unique_docs:
                    similarity = self._calculate_document_similarity(document, existing_doc)
                    if similarity > self.similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_docs.append(document)
                seen_hashes.add(doc_hash)
        
        return unique_docs
    
    def _apply_cross_document_compression(
        self,
        mega_tokens: List[MegaToken],
        documents: List[str]
    ) -> List[MegaToken]:
        """Apply cross-document compression to reduce redundancy."""
        if len(mega_tokens) <= 1:
            return mega_tokens
        
        # Group tokens by similarity
        token_vectors = np.array([token.vector for token in mega_tokens])
        
        try:
            from sklearn.cluster import DBSCAN
            
            # Use DBSCAN to find similar token clusters
            clustering = DBSCAN(
                eps=1 - self.similarity_threshold,  # Convert similarity to distance
                min_samples=2,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(token_vectors)
            
            # Merge similar tokens within clusters
            compressed_tokens = []
            processed_clusters = set()
            
            for i, (token, cluster_id) in enumerate(zip(mega_tokens, cluster_labels)):
                if cluster_id == -1:  # Noise (unique token)
                    compressed_tokens.append(token)
                elif cluster_id not in processed_clusters:
                    # Find all tokens in this cluster
                    cluster_tokens = [
                        mega_tokens[j] for j, cid in enumerate(cluster_labels) 
                        if cid == cluster_id
                    ]
                    
                    # Create a merged token
                    merged_token = self._merge_similar_tokens(cluster_tokens)
                    compressed_tokens.append(merged_token)
                    processed_clusters.add(cluster_id)
            
            return compressed_tokens
            
        except ImportError:
            # Fallback: simple pairwise similarity check
            return self._simple_cross_doc_deduplication(mega_tokens)
    
    def _merge_similar_tokens(self, tokens: List[MegaToken]) -> MegaToken:
        """Merge similar tokens into a single representative token."""
        if len(tokens) == 1:
            return tokens[0]
        
        # Average the vectors
        avg_vector = np.mean([token.vector for token in tokens], axis=0)
        
        # Combine source texts
        source_texts = []
        document_ids = set()
        confidences = []
        
        for token in tokens:
            if "source_text" in token.metadata:
                source_texts.append(token.metadata["source_text"])
            document_ids.add(token.metadata.get("document_id", "unknown"))
            confidences.append(token.confidence)
        
        # Create merged metadata
        merged_metadata = {
            "merged_from": len(tokens),
            "source_texts": source_texts[:3],  # Keep first 3 for space
            "document_ids": list(document_ids),
            "cross_document": True,
            "source_text": " | ".join(source_texts[:2])  # Combined preview
        }
        
        return MegaToken(
            vector=avg_vector,
            metadata=merged_metadata,
            confidence=np.mean(confidences)
        )
    
    def _simple_cross_doc_deduplication(self, mega_tokens: List[MegaToken]) -> List[MegaToken]:
        """Simple pairwise deduplication fallback."""
        deduplicated = []
        
        for token in mega_tokens:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self._cosine_similarity(token.vector, existing.vector)
                if similarity > self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(token)
        
        return deduplicated
    
    def _calculate_document_similarity(self, doc1: str, doc2: str) -> float:
        """Calculate semantic similarity between two documents."""
        # Simple implementation using token overlap
        words1 = set(doc1.lower().split())
        words2 = set(doc2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _preserve_citations(self, documents: List[str]) -> List[str]:
        """Preserve citation information in documents."""
        processed_docs = []
        
        for i, document in enumerate(documents):
            # Add document ID as citation marker
            cited_doc = f"[DOC-{i}] {document}"
            processed_docs.append(cited_doc)
        
        return processed_docs
    
    def _create_document_index(self, documents: List[str]) -> Dict:
        """Create an index of document metadata."""
        index = {
            "total_documents": len(documents),
            "document_metadata": []
        }
        
        for i, document in enumerate(documents):
            metadata = {
                "doc_id": i,
                "length": len(document),
                "word_count": len(document.split()),
                "hash": self._hash_text(document),
                "preview": document[:100] + "..." if len(document) > 100 else document
            }
            index["document_metadata"].append(metadata)
        
        return index