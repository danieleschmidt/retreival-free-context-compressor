"""
rfcc — Retrieval-Free Context Compressor

Hierarchical context compression pipeline that converts long documents into dense
"mega-tokens": fixed-size embedding vectors that preserve semantic content for
cross-document reasoning without retrieval.

Public API:
    MegaTokenCompressor  – text chunks → mega-token embeddings
    CrossDocumentReasoner – mega-tokens → fused representation for QA/reasoning
"""

__version__ = "0.2.0"
__all__ = ["MegaTokenCompressor", "CrossDocumentReasoner", "MegaToken"]

from .compressor import MegaToken, MegaTokenCompressor
from .reasoner import CrossDocumentReasoner
