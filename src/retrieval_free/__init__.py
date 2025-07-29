"""Retrieval-Free Context Compressor.

A transformer plug-in that compresses long documents into dense "mega-tokens,"
enabling 256k-token context without external RAG.
"""

__version__ = "--help"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core compression interfaces
from .core import ContextCompressor, AutoCompressor
from .streaming import StreamingCompressor
from .selective import SelectiveCompressor
from .multi_doc import MultiDocCompressor

# Integration plugins
from .plugins import CompressorPlugin

__all__ = [
    "ContextCompressor",
    "AutoCompressor", 
    "StreamingCompressor",
    "SelectiveCompressor",
    "MultiDocCompressor",
    "CompressorPlugin",
]