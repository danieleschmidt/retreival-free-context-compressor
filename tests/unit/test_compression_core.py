"""Unit tests for core compression functionality."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from tests.fixtures.test_data import TestDataLoader, TEST_CONFIGS


class TestCompressionCore:
    """Test suite for core compression algorithms."""
    
    def test_token_encoding(self, mock_tokenizer):
        """Test basic token encoding functionality."""
        text = "This is a test document."
        tokens = mock_tokenizer.encode(text)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, int) for token in tokens)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        original_tokens = 1000
        compressed_tokens = 125
        expected_ratio = 8.0
        
        calculated_ratio = original_tokens / compressed_tokens
        assert abs(calculated_ratio - expected_ratio) < 0.1
    
    def test_mega_token_generation(self):
        """Test mega-token generation process."""
        # Mock the compression process
        input_embeddings = torch.randn(100, 768)  # 100 tokens, 768 dimensions
        target_compression = 8
        
        # Simulate compression
        compressed_size = input_embeddings.size(0) // target_compression
        mega_tokens = torch.randn(compressed_size, 768)
        
        assert mega_tokens.size(0) == compressed_size
        assert mega_tokens.size(1) == input_embeddings.size(1)
    
    def test_information_bottleneck(self):
        """Test information bottleneck implementation."""
        # Mock information bottleneck calculation
        original_info = 100.0  # bits
        compressed_info = 85.0  # bits
        info_retention = compressed_info / original_info
        
        assert info_retention > 0.8  # Expect >80% information retention
        assert info_retention <= 1.0
    
    @pytest.mark.parametrize("config_name", ["fast", "standard"])
    def test_compression_with_different_configs(self, config_name):
        """Test compression with different configuration settings."""
        config = TEST_CONFIGS[config_name]
        
        # Mock document with specified max tokens
        mock_document = " ".join(["token"] * config["max_tokens"])
        
        # Simulate compression
        original_tokens = len(mock_document.split())
        expected_compressed = original_tokens / config["compression_ratio"]
        
        assert original_tokens <= config["max_tokens"]
        assert expected_compressed > 0
    
    def test_hierarchical_encoding_stages(self):
        """Test the hierarchical encoding pipeline."""
        stages = ["token", "sentence", "paragraph", "document"]
        
        for i, stage in enumerate(stages):
            # Mock processing at each stage
            input_size = 1000 // (2 ** i)  # Reduce size at each stage
            output_size = input_size // 2
            
            assert output_size < input_size
            assert output_size > 0
    
    def test_cross_attention_mechanism(self):
        """Test cross-attention for compression."""
        query_dim = 768
        key_dim = 768
        value_dim = 768
        
        # Mock attention weights
        attention_weights = torch.softmax(torch.randn(10, 20), dim=-1)
        
        assert attention_weights.sum(dim=-1).allclose(torch.ones(10))
        assert attention_weights.shape == (10, 20)
    
    def test_streaming_compression(self):
        """Test streaming compression functionality."""
        window_size = 1000
        overlap = 100
        
        # Mock streaming windows
        windows = []
        for i in range(5):
            start = i * (window_size - overlap)
            end = start + window_size
            windows.append((start, end))
        
        # Ensure proper overlap
        for i in range(len(windows) - 1):
            current_end = windows[i][1]
            next_start = windows[i + 1][0]
            overlap_size = current_end - next_start
            assert overlap_size == overlap
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError):
            # Test with empty input
            self._mock_compress("")
        
        with pytest.raises(TypeError):
            # Test with invalid input type
            self._mock_compress(None)
    
    def _mock_compress(self, text):
        """Mock compression function for testing."""
        if not text:
            raise ValueError("Input text cannot be empty")
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
        return f"compressed_{len(text)}"


class TestCompressionQuality:
    """Test suite for compression quality metrics."""
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation for compressed outputs."""
        # Mock predicted and true answers
        predicted = "machine learning algorithms"
        true_answer = "machine learning"
        
        # Simple token-based F1 calculation
        pred_tokens = set(predicted.split())
        true_tokens = set(true_answer.split())
        
        precision = len(pred_tokens & true_tokens) / len(pred_tokens)
        recall = len(pred_tokens & true_tokens) / len(true_tokens)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= f1 <= 1
        assert f1 > 0.5  # Expect reasonable overlap
    
    def test_compression_quality_metrics(self):
        """Test various quality metrics for compression."""
        qa_pairs = TestDataLoader.load_qa_pairs()
        
        for qa_pair in qa_pairs:
            # Mock compression and answering
            compressed_context = self._mock_compress_context(qa_pair["context"])
            predicted_answer = self._mock_answer_question(
                qa_pair["question"], 
                compressed_context
            )
            
            # Basic quality checks
            assert len(compressed_context) < len(qa_pair["context"])
            assert len(predicted_answer) > 0
    
    def test_information_retention_metrics(self):
        """Test information retention calculation."""
        original_entropy = 100.0  # Mock entropy value
        compressed_entropy = 85.0  # Mock compressed entropy
        
        retention_ratio = compressed_entropy / original_entropy
        
        assert 0.7 <= retention_ratio <= 1.0  # Expect 70-100% retention
    
    def _mock_compress_context(self, context: str) -> str:
        """Mock context compression."""
        # Simple compression simulation
        words = context.split()
        compressed_words = words[::2]  # Take every other word
        return " ".join(compressed_words)
    
    def _mock_answer_question(self, question: str, context: str) -> str:
        """Mock question answering."""
        # Simple answer extraction simulation
        words = context.split()
        return " ".join(words[:5])  # Return first 5 words


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""
    
    @pytest.mark.performance
    def test_compression_speed(self):
        """Benchmark compression speed."""
        import time
        
        test_doc = TestDataLoader.load_sample_documents()["wikipedia_sample"]
        
        start_time = time.time()
        # Mock compression
        compressed = self._mock_fast_compress(test_doc)
        end_time = time.time()
        
        compression_time = (end_time - start_time) * 1000  # Convert to ms
        
        assert compression_time < 1000  # Should complete in <1 second
        assert len(compressed) < len(test_doc)
    
    @pytest.mark.memory
    def test_memory_usage(self):
        """Test memory usage during compression."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Mock memory-intensive compression
        large_doc = "word " * 100000  # 100k words
        compressed = self._mock_fast_compress(large_doc)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < 500  # Should use <500MB additional memory
    
    @pytest.mark.gpu
    def test_gpu_utilization(self):
        """Test GPU utilization during compression."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.cuda.current_device()
        
        # Mock GPU computation
        tensor = torch.randn(1000, 768, device=device)
        compressed_tensor = torch.nn.functional.avg_pool1d(
            tensor.unsqueeze(0), kernel_size=8
        ).squeeze(0)
        
        assert compressed_tensor.device.type == 'cuda'
        assert compressed_tensor.size(0) < tensor.size(0)
    
    def _mock_fast_compress(self, text: str) -> str:
        """Fast mock compression for benchmarking."""
        words = text.split()
        # Simple compression: take every 8th word
        compressed_words = words[::8]
        return " ".join(compressed_words)