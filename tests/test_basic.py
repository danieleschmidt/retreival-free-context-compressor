"""Basic tests for retrieval_free package."""

import pytest
from retrieval_free import __version__


def test_version():
    """Test package version is accessible."""
    assert __version__ == "0.1.0"


def test_imports():
    """Test core imports work."""
    from retrieval_free import (
        ContextCompressor,
        AutoCompressor,
        StreamingCompressor,
        SelectiveCompressor,
        MultiDocCompressor,
        CompressorPlugin,
    )
    
    # Basic smoke test - classes should be importable
    assert ContextCompressor is not None
    assert AutoCompressor is not None
    assert StreamingCompressor is not None
    assert SelectiveCompressor is not None
    assert MultiDocCompressor is not None
    assert CompressorPlugin is not None