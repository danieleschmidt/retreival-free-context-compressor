"""Multi-document compression with cross-document analysis."""

import logging
from typing import List, Dict, Any, Union, Optional, Set
from dataclasses import dataclass
import hashlib
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .core.base import CompressorBase, MegaToken, CompressionResult
from .core.context_compressor import ContextCompressor

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    """Information about a document in the collection."""
    
    doc_id: str
    title: Optional[str]
    length: int
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class CrossDocumentRelation:
    """Relationship between documents."""
    
    doc1_id: str
    doc2_id: str
    similarity: float
    relation_type: str  # "duplicate", "similar", "citation", etc.
    shared_concepts: List[str]


class MultiDocCompressor(CompressorBase):
    """Compressor for multiple related documents with deduplication."""
    
    def __init__(
        self,
        model_name: str = "rfcc-multi-doc",
        device: Optional[str] = None,
        max_length: int = 1000000,  # Higher limit for multi-doc
        compression_ratio: float = 8.0,
        deduplication: bool = True,
        cross_doc_attention: bool = True,
        similarity_threshold: float = 0.8
    ):
        """Initialize multi-document compressor.
        
        Args:
            model_name: Base model name
            device: Computing device
            max_length: Maximum total input length
            compression_ratio: Target compression ratio
            deduplication: Whether to deduplicate content
            cross_doc_attention: Whether to use cross-document attention
            similarity_threshold: Threshold for similarity detection
        """
        super().__init__(model_name, device, max_length, compression_ratio)
        
        self.deduplication = deduplication
        self.cross_doc_attention = cross_doc_attention
        self.similarity_threshold = similarity_threshold
        
        # Internal components
        self._base_compressor = ContextCompressor(
            model_name=model_name,
            device=device,
            compression_ratio=compression_ratio
        )
        self._document_registry: Dict[str, DocumentInfo] = {}
        self._content_hashes: Set[str] = set()
        self._document_embeddings: Dict[str, torch.Tensor] = {}
    
    def load_model(self) -> None:
        """Load base compression model."""
        self._base_compressor.load_model()
    
    def compress_collection(
        self,
        documents: Union[List[str], Dict[str, str]],
        preserve_citations: bool = True,
        create_index: bool = True,
        **kwargs
    ) -> CompressionResult:
        """Compress a collection of related documents.
        
        Args:
            documents: List of texts or dict mapping doc_ids to texts
            preserve_citations: Whether to preserve citation relationships
            create_index: Whether to create searchable index
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult for the entire collection
        """
        # Convert to standardized format
        if isinstance(documents, list):
            doc_dict = {f"doc_{i}": text for i, text in enumerate(documents)}
        else:
            doc_dict = documents
        
        logger.info(f"Compressing collection of {len(doc_dict)} documents")
        
        # Step 1: Register and analyze documents
        self._register_documents(doc_dict)
        
        # Step 2: Detect relationships and duplicates
        relations = self._analyze_relationships(doc_dict)
        
        # Step 3: Perform deduplication if enabled
        if self.deduplication:
            doc_dict = self._deduplicate_documents(doc_dict, relations)
        
        # Step 4: Compress documents with cross-attention
        all_mega_tokens = []
        collection_metadata = {
            "documents": {},
            "relations": [],
            "deduplication_stats": {},
            "cross_doc_attention": self.cross_doc_attention
        }
        
        total_original = 0
        total_compressed = 0
        
        for doc_id, text in doc_dict.items():
            # Compress individual document
            result = self._base_compressor.compress(text)
            
            # Enhance mega-tokens with document metadata
            for token in result.mega_tokens:
                token.metadata.update({
                    "document_id": doc_id,
                    "document_title": self._document_registry[doc_id].title,
                    "multi_doc": True
                })
            
            all_mega_tokens.extend(result.mega_tokens)
            total_original += result.original_length
            total_compressed += result.compressed_length
            
            # Store document info
            collection_metadata["documents"][doc_id] = {
                "original_length": result.original_length,
                "compressed_length": result.compressed_length,
                "compression_ratio": result.compression_ratio,
                "content_hash": self._document_registry[doc_id].content_hash
            }
        
        # Step 5: Apply cross-document compression if enabled
        if self.cross_doc_attention and len(all_mega_tokens) > 1:
            all_mega_tokens = self._apply_cross_doc_compression(all_mega_tokens)
            total_compressed = len(all_mega_tokens)  # Update after cross-compression
        
        # Add relationship metadata
        collection_metadata["relations"] = [
            {
                "doc1": rel.doc1_id,
                "doc2": rel.doc2_id,
                "similarity": rel.similarity,
                "type": rel.relation_type,
                "concepts": rel.shared_concepts
            }
            for rel in relations
        ]
        
        # Calculate final metrics
        overall_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
        
        result = CompressionResult(
            mega_tokens=all_mega_tokens,
            original_length=total_original,
            compressed_length=total_compressed,
            compression_ratio=overall_ratio,
            processing_time=0,  # TODO: Track actual time
            metadata=collection_metadata
        )
        
        # Create index if requested
        if create_index:
            self._create_searchable_index(result)
        
        logger.info(
            f"Multi-document compression complete: {len(doc_dict)} docs -> "
            f"{len(all_mega_tokens)} mega-tokens ({overall_ratio:.1f}x ratio)"
        )
        
        return result
    
    def compress(
        self, 
        text: Union[str, List[str]], 
        **kwargs
    ) -> CompressionResult:
        """Compress text using multi-document approach.
        
        Args:
            text: Input text or list of texts
            **kwargs: Additional parameters
            
        Returns:
            CompressionResult
        """
        if isinstance(text, str):
            return self.compress_collection([text], **kwargs)
        else:
            return self.compress_collection(text, **kwargs)
    
    def decompress(
        self, 
        mega_tokens: List[MegaToken],
        **kwargs
    ) -> str:
        """Decompress multi-document mega-tokens."""
        # Group tokens by document
        doc_tokens = {}
        for token in mega_tokens:
            doc_id = token.metadata.get("document_id", "unknown")
            if doc_id not in doc_tokens:
                doc_tokens[doc_id] = []
            doc_tokens[doc_id].append(token)
        
        # Decompress each document separately
        decompressed_docs = {}
        for doc_id, tokens in doc_tokens.items():
            decompressed_docs[doc_id] = self._base_compressor.decompress(tokens, **kwargs)
        
        # Combine documents
        if len(decompressed_docs) == 1:
            return list(decompressed_docs.values())[0]
        else:
            # Format as multi-document output
            result = []
            for doc_id, content in decompressed_docs.items():
                result.append(f"=== Document: {doc_id} ===\n{content}\n")
            return "\n".join(result)
    
    def _register_documents(self, documents: Dict[str, str]) -> None:
        """Register documents in the internal registry.
        
        Args:
            documents: Dictionary mapping doc_ids to text content
        """
        for doc_id, text in documents.items():
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            doc_info = DocumentInfo(
                doc_id=doc_id,
                title=self._extract_title(text),
                length=len(text.split()),
                content_hash=content_hash,
                metadata={}
            )
            
            self._document_registry[doc_id] = doc_info
            self._content_hashes.add(content_hash)
    
    def _extract_title(self, text: str) -> Optional[str]:
        """Extract title from document text.
        
        Args:
            text: Document text
            
        Returns:
            Extracted title or None
        """
        # Simple heuristic: first line if it's short
        lines = text.strip().split('\n')
        if lines and len(lines[0].split()) <= 10:
            return lines[0].strip()
        
        # Fallback: first few words
        words = text.split()
        if len(words) >= 5:
            return " ".join(words[:5]) + "..."
        
        return None
    
    def _analyze_relationships(
        self, 
        documents: Dict[str, str]
    ) -> List[CrossDocumentRelation]:
        """Analyze relationships between documents.
        
        Args:
            documents: Dictionary mapping doc_ids to text content
            
        Returns:
            List of detected relationships
        """
        relations = []
        doc_ids = list(documents.keys())
        
        # Generate embeddings for similarity analysis
        if len(doc_ids) > 1:
            self._generate_document_embeddings(documents)
        
        # Compare all pairs of documents
        for i, doc1_id in enumerate(doc_ids):
            for doc2_id in doc_ids[i+1:]:
                similarity = self._calculate_similarity(doc1_id, doc2_id)
                
                # Determine relationship type
                relation_type = "similar"
                if similarity > 0.95:
                    relation_type = "duplicate"
                elif similarity < 0.3:
                    relation_type = "unrelated"
                
                # Extract shared concepts (simplified)
                shared_concepts = self._extract_shared_concepts(
                    documents[doc1_id], 
                    documents[doc2_id]
                )
                
                relation = CrossDocumentRelation(
                    doc1_id=doc1_id,
                    doc2_id=doc2_id,
                    similarity=similarity,
                    relation_type=relation_type,
                    shared_concepts=shared_concepts
                )
                
                relations.append(relation)
        
        return relations
    
    def _generate_document_embeddings(self, documents: Dict[str, str]) -> None:
        """Generate embeddings for documents.
        
        Args:
            documents: Dictionary mapping doc_ids to text content
        """
        for doc_id, text in documents.items():
            # Use first 1000 words for embedding (efficiency)
            sample_text = " ".join(text.split()[:1000])
            
            # Generate embedding using base compressor
            result = self._base_compressor.compress(sample_text)
            
            if result.mega_tokens:
                # Average embeddings of mega-tokens as document embedding
                embeddings = torch.stack([token.embedding for token in result.mega_tokens])
                doc_embedding = torch.mean(embeddings, dim=0)
                self._document_embeddings[doc_id] = doc_embedding
    
    def _calculate_similarity(self, doc1_id: str, doc2_id: str) -> float:
        """Calculate similarity between two documents.
        
        Args:
            doc1_id: First document ID
            doc2_id: Second document ID
            
        Returns:
            Similarity score between 0 and 1
        """
        if doc1_id not in self._document_embeddings or doc2_id not in self._document_embeddings:
            return 0.0
        
        emb1 = self._document_embeddings[doc1_id].cpu().numpy()
        emb2 = self._document_embeddings[doc2_id].cpu().numpy()
        
        # Cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return max(0.0, float(similarity))  # Ensure non-negative
    
    def _extract_shared_concepts(self, text1: str, text2: str) -> List[str]:
        """Extract shared concepts between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            List of shared important terms
        """
        # Simple word frequency approach
        words1 = set(word.lower() for word in text1.split() if len(word) > 3)
        words2 = set(word.lower() for word in text2.split() if len(word) > 3)
        
        shared = words1.intersection(words2)
        
        # Return most frequent shared terms (limited)
        return sorted(list(shared))[:10]
    
    def _deduplicate_documents(
        self, 
        documents: Dict[str, str],
        relations: List[CrossDocumentRelation]
    ) -> Dict[str, str]:
        """Remove duplicate documents based on relationships.
        
        Args:
            documents: Original documents
            relations: Detected relationships
            
        Returns:
            Deduplicated documents
        """
        duplicates_to_remove = set()
        
        for relation in relations:
            if relation.relation_type == "duplicate":
                # Keep the document with the shorter ID (arbitrary choice)
                if relation.doc1_id < relation.doc2_id:
                    duplicates_to_remove.add(relation.doc2_id)
                else:
                    duplicates_to_remove.add(relation.doc1_id)
        
        # Remove duplicates
        deduplicated = {
            doc_id: text for doc_id, text in documents.items()
            if doc_id not in duplicates_to_remove
        }
        
        if duplicates_to_remove:
            logger.info(f"Removed {len(duplicates_to_remove)} duplicate documents")
        
        return deduplicated
    
    def _apply_cross_doc_compression(
        self, 
        mega_tokens: List[MegaToken]
    ) -> List[MegaToken]:
        """Apply cross-document compression to reduce redundancy.
        
        Args:
            mega_tokens: All mega-tokens from documents
            
        Returns:
            Compressed mega-tokens with cross-document optimization
        """
        if len(mega_tokens) <= 1:
            return mega_tokens
        
        # Group tokens by similarity
        embeddings = torch.stack([token.embedding for token in mega_tokens])
        
        # Calculate pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())
        
        # Find highly similar tokens for merging
        merged_tokens = []
        used_indices = set()
        
        for i, token in enumerate(mega_tokens):
            if i in used_indices:
                continue
            
            # Find similar tokens
            similar_indices = []
            for j in range(i+1, len(mega_tokens)):
                if j not in used_indices and similarities[i][j] > self.similarity_threshold:
                    similar_indices.append(j)
            
            if similar_indices:
                # Merge similar tokens
                all_indices = [i] + similar_indices
                merged_embedding = torch.mean(
                    torch.stack([mega_tokens[idx].embedding for idx in all_indices]),
                    dim=0
                )
                
                # Create merged token with combined metadata
                merged_token = MegaToken(
                    embedding=merged_embedding,
                    metadata={
                        "merged_from": [mega_tokens[idx].metadata.get("document_id", "unknown") 
                                      for idx in all_indices],
                        "cross_doc_compression": True,
                        "merge_count": len(all_indices)
                    },
                    source_range=(0, 0),  # Cross-document range
                    compression_ratio=len(all_indices)
                )
                
                merged_tokens.append(merged_token)
                used_indices.update(all_indices)
            else:
                merged_tokens.append(token)
                used_indices.add(i)
        
        reduction = len(mega_tokens) - len(merged_tokens)
        if reduction > 0:
            logger.info(f"Cross-document compression reduced tokens by {reduction}")
        
        return merged_tokens
    
    def _create_searchable_index(self, result: CompressionResult) -> None:
        """Create searchable index for compressed collection.
        
        Args:
            result: Compression result to index
        """
        # This would create a searchable index structure
        # For now, just log the creation
        logger.info(f"Created searchable index for {len(result.mega_tokens)} mega-tokens")
        
        # In a real implementation, this would:
        # 1. Build FAISS index from embeddings
        # 2. Create keyword search index
        # 3. Store document metadata for retrieval