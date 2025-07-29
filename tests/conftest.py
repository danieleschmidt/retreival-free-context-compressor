"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
from unittest.mock import Mock


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_document() -> str:
    """Sample document for testing compression."""
    return """
    This is a sample document for testing compression algorithms.
    It contains multiple paragraphs with various information.
    
    The document discusses various topics including machine learning,
    natural language processing, and document compression techniques.
    
    Some technical details are included to make the document more realistic
    for testing purposes. The compression algorithm should be able to
    handle this type of content effectively.
    """ * 10  # Make it longer for realistic testing


@pytest.fixture
def sample_questions() -> list[str]:
    """Sample questions for testing Q&A functionality."""
    return [
        "What does this document discuss?",
        "What are the main topics covered?",
        "How does compression work in this context?",
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = list(range(100))  # Mock token IDs
    tokenizer.decode.return_value = "decoded text"
    tokenizer.vocab_size = 50000
    return tokenizer


@pytest.fixture
def mock_model():
    """Mock transformer model for testing."""
    model = Mock()
    model.config.hidden_size = 768
    model.config.num_attention_heads = 12
    model.config.max_position_embeddings = 2048
    return model


@pytest.fixture
def compression_config() -> Dict[str, Any]:
    """Default compression configuration for testing."""
    return {
        "compression_ratio": 8.0,
        "max_input_length": 4096,
        "hidden_size": 768,
        "num_layers": 6,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "dropout": 0.1,
    }


@pytest.fixture(scope="session")
def device() -> str:
    """Determine device for testing (CPU/CUDA)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def benchmark_data():
    """Large dataset for benchmarking tests."""
    return {
        "documents": ["Long document content"] * 1000,
        "questions": ["Test question"] * 100,
        "expected_compression_ratio": 8.0,
        "max_latency_ms": 1000,
    }


# Performance test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests
        if "benchmark" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.benchmark)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)


# Environment setup
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
    yield
    # Cleanup if needed