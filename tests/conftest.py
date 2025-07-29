"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator


@pytest.fixture(scope="session")
def sample_documents() -> dict[str, str]:
    """Sample documents for testing compression."""
    return {
        "short": "This is a short document for testing.",
        "medium": " ".join([
            "This is a medium-length document.",
            "It contains multiple sentences and paragraphs.",
            "The content is designed to test compression algorithms.",
            "It should provide a good balance of content diversity.",
        ] * 50),
        "long": " ".join([
            "This is a very long document designed for testing.",
            "It contains repetitive patterns and diverse content.",
            "The document should stress-test compression algorithms.",
            "Various topics are covered to ensure information diversity.",
        ] * 1000),
    }


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Temporary directory for model files during testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing without requiring actual models."""
    class MockTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))
        
        def decode(self, tokens: list[int]) -> str:
            return " ".join([f"token_{i}" for i in tokens])
        
        def count_tokens(self, text: str) -> int:
            return len(text.split())
    
    return MockTokenizer()


@pytest.fixture
def compression_test_data() -> dict:
    """Test data for compression evaluation."""
    return {
        "documents": [
            "Machine learning is transforming industries.",
            "Natural language processing enables computers to understand text.",
            "Artificial intelligence systems require large amounts of data.",
        ],
        "questions": [
            "What is transforming industries?",
            "What enables computers to understand text?", 
            "What do AI systems require?",
        ],
        "expected_answers": [
            "Machine learning",
            "Natural language processing",
            "Large amounts of data",
        ],
    }