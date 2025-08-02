"""Example tests demonstrating best practices."""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from tests.fixtures.test_data import TestDataLoader, TEST_CONFIGS


class TestExamples:
    """Example test implementations following best practices."""
    
    def test_basic_functionality_example(self):
        """Example of a basic functionality test."""
        # Arrange
        input_text = "This is a test document."
        expected_output_length = len(input_text) // 2
        
        # Act
        result = self._mock_process_text(input_text)
        
        # Assert
        assert len(result) <= expected_output_length
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("input_size,expected_ratio", [
        (1000, 8.0),
        (5000, 8.0),
        (10000, 8.0),
    ])
    def test_parameterized_example(self, input_size, expected_ratio):
        """Example of parameterized testing."""
        # Arrange
        test_text = "word " * input_size
        
        # Act
        compressed = self._mock_compress(test_text, expected_ratio)
        actual_ratio = len(test_text.split()) / len(compressed.split())
        
        # Assert
        assert abs(actual_ratio - expected_ratio) < 1.0
    
    def test_fixture_usage_example(self, sample_documents, temp_model_dir):
        """Example of using fixtures in tests."""
        # Arrange
        doc = sample_documents["short"]
        model_path = temp_model_dir / "test_model.bin"
        
        # Act
        self._mock_save_model(model_path)
        result = self._mock_process_with_model(doc, model_path)
        
        # Assert
        assert model_path.exists()
        assert result is not None
    
    @patch('builtins.open', new_callable=MagicMock)
    def test_mocking_example(self, mock_open):
        """Example of mocking external dependencies."""
        # Arrange
        mock_open.return_value.__enter__.return_value.read.return_value = "test content"
        
        # Act
        result = self._mock_read_file("test.txt")
        
        # Assert
        mock_open.assert_called_once_with("test.txt", 'r')
        assert result == "test content"
    
    @pytest.mark.slow
    def test_performance_example(self):
        """Example of performance testing."""
        # Arrange
        large_text = "word " * 100000
        max_time_seconds = 5.0
        
        # Act
        start_time = time.time()
        result = self._mock_slow_operation(large_text)
        end_time = time.time()
        
        # Assert
        execution_time = end_time - start_time
        assert execution_time < max_time_seconds
        assert result is not None
    
    def test_error_handling_example(self):
        """Example of testing error conditions."""
        # Test empty input
        with pytest.raises(ValueError, match="Input cannot be empty"):
            self._mock_validate_input("")
        
        # Test invalid type
        with pytest.raises(TypeError, match="Input must be a string"):
            self._mock_validate_input(None)
        
        # Test out of range
        with pytest.raises(ValueError, match="Ratio must be between"):
            self._mock_compress_with_ratio("text", ratio=-1)
    
    def test_context_manager_example(self):
        """Example of testing context managers."""
        with self._mock_context_manager() as manager:
            assert manager.is_active
            result = manager.process("test")
            assert result is not None
        
        # Context manager should be cleaned up
        assert not manager.is_active
    
    @pytest.mark.integration
    def test_integration_example(self):
        """Example of integration testing."""
        # Test multiple components working together
        components = {
            "tokenizer": Mock(),
            "encoder": Mock(), 
            "compressor": Mock()
        }
        
        # Configure mocks
        components["tokenizer"].encode.return_value = [1, 2, 3, 4, 5]
        components["encoder"].encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        components["compressor"].compress.return_value = [[0.25, 0.35]]
        
        # Test integration
        result = self._mock_integration_pipeline(components, "test text")
        
        # Verify component interactions
        components["tokenizer"].encode.assert_called_once()
        components["encoder"].encode.assert_called_once()
        components["compressor"].compress.assert_called_once()
        assert result is not None
    
    def test_data_driven_example(self):
        """Example of data-driven testing."""
        test_cases = TestDataLoader.load_qa_pairs()
        
        for test_case in test_cases:
            # Act
            answer = self._mock_answer_question(
                test_case["question"],
                test_case["context"]
            )
            
            # Assert
            assert answer is not None
            assert len(answer) > 0
            # Could add more sophisticated answer validation here
    
    def test_property_based_example(self):
        """Example of property-based testing concepts."""
        # Property: compression should always reduce size
        for _ in range(10):  # Test multiple random inputs
            text = self._generate_random_text()
            compressed = self._mock_compress(text, ratio=8.0)
            
            # Property assertion
            assert len(compressed) < len(text)
    
    def test_async_example(self):
        """Example of testing async functionality (mock)."""
        import asyncio
        
        async def mock_async_operation():
            await asyncio.sleep(0.1)  # Simulate async work
            return "async result"
        
        # Test async function
        result = asyncio.run(mock_async_operation())
        assert result == "async result"
    
    # Helper methods for testing
    
    def _mock_process_text(self, text: str) -> str:
        """Mock text processing function."""
        return text[:len(text)//2]
    
    def _mock_compress(self, text: str, ratio: float = 8.0) -> str:
        """Mock compression function."""
        words = text.split()
        compressed_size = max(1, int(len(words) / ratio))
        return " ".join(words[:compressed_size])
    
    def _mock_save_model(self, path: Path) -> None:
        """Mock model saving."""
        path.touch()
    
    def _mock_process_with_model(self, text: str, model_path: Path) -> str:
        """Mock processing with model."""
        if not model_path.exists():
            raise FileNotFoundError("Model not found")
        return f"processed_{text[:10]}"
    
    def _mock_read_file(self, filename: str) -> str:
        """Mock file reading."""
        with open(filename, 'r') as f:
            return f.read()
    
    def _mock_slow_operation(self, text: str) -> str:
        """Mock slow operation."""
        time.sleep(0.1)  # Simulate processing time
        return f"processed_{len(text)}_words"
    
    def _mock_validate_input(self, input_data):
        """Mock input validation."""
        if input_data == "":
            raise ValueError("Input cannot be empty")
        if not isinstance(input_data, str):
            raise TypeError("Input must be a string")
        return True
    
    def _mock_compress_with_ratio(self, text: str, ratio: float) -> str:
        """Mock compression with ratio validation."""
        if ratio < 1.0 or ratio > 32.0:
            raise ValueError("Ratio must be between 1.0 and 32.0")
        return self._mock_compress(text, ratio)
    
    def _mock_context_manager(self):
        """Mock context manager."""
        class MockContextManager:
            def __init__(self):
                self.is_active = False
            
            def __enter__(self):
                self.is_active = True
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.is_active = False
            
            def process(self, data):
                if not self.is_active:
                    raise RuntimeError("Context manager not active")
                return f"processed_{data}"
        
        return MockContextManager()
    
    def _mock_integration_pipeline(self, components: dict, text: str) -> list:
        """Mock integration pipeline."""
        tokens = components["tokenizer"].encode(text)
        embeddings = components["encoder"].encode(tokens)
        compressed = components["compressor"].compress(embeddings)
        return compressed
    
    def _mock_answer_question(self, question: str, context: str) -> str:
        """Mock question answering."""
        # Simple mock: return first few words of context
        words = context.split()
        return " ".join(words[:5])
    
    def _generate_random_text(self) -> str:
        """Generate random text for property testing."""
        import random
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        length = random.randint(10, 100)
        return " ".join(random.choices(words, k=length))


class TestAdvancedExamples:
    """Advanced testing examples and patterns."""
    
    def test_snapshot_testing_example(self, tmp_path):
        """Example of snapshot testing pattern."""
        # Generate some output
        result = self._generate_complex_output()
        
        # Save to snapshot file
        snapshot_file = tmp_path / "snapshot.txt"
        snapshot_file.write_text(str(result))
        
        # In real implementation, you'd compare with stored snapshot
        assert snapshot_file.exists()
        assert len(snapshot_file.read_text()) > 0
    
    def test_monkey_patching_example(self, monkeypatch):
        """Example of monkey patching for testing."""
        # Patch environment variable
        monkeypatch.setenv("TEST_MODE", "true")
        
        # Patch attribute
        monkeypatch.setattr("sys.platform", "test_platform")
        
        # Test functionality that depends on patched values
        result = self._mock_environment_dependent_function()
        assert "test_platform" in str(result)
    
    def test_custom_fixture_example(self, compression_test_data):
        """Example using custom fixture from conftest.py."""
        documents = compression_test_data["documents"]
        questions = compression_test_data["questions"]
        
        for doc, question in zip(documents, questions):
            result = self._mock_qa_system(question, doc)
            assert result is not None
    
    def _generate_complex_output(self) -> dict:
        """Generate complex output for snapshot testing."""
        return {
            "compression_ratio": 8.0,
            "quality_score": 0.95,
            "processing_time": 123.45,
            "metadata": {
                "model_version": "1.0.0",
                "timestamp": "2025-01-01T00:00:00Z"
            }
        }
    
    def _mock_environment_dependent_function(self) -> str:
        """Mock function that depends on environment."""
        import os
        import sys
        test_mode = os.getenv("TEST_MODE", "false")
        platform = sys.platform
        return f"Running on {platform} in test mode: {test_mode}"
    
    def _mock_qa_system(self, question: str, document: str) -> str:
        """Mock QA system."""
        return f"Answer to '{question[:20]}...' from document"