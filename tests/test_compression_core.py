"""Core compression functionality tests."""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import torch
import numpy as np

# Mock torch and dependencies for testing without actual installs
mock_torch = MagicMock()
mock_torch.cuda.is_available.return_value = False
mock_torch.randn.return_value = MagicMock()
mock_torch.tensor.return_value = MagicMock()

mock_transformers = MagicMock()
mock_sklearn = MagicMock()

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies for testing."""
    with patch.dict('sys.modules', {
        'torch': mock_torch,
        'torch.nn': MagicMock(),
        'transformers': mock_transformers,
        'sklearn': mock_sklearn,
        'sklearn.cluster': MagicMock(),
        'sentence_transformers': MagicMock(),
        'psutil': MagicMock(),
    }):
        yield


class TestCompressorBase:
    """Test base compressor functionality."""
    
    def test_compressor_initialization(self):
        """Test compressor can be initialized."""
        from src.retrieval_free.core.base import CompressorBase
        
        # Create mock concrete implementation
        class MockCompressor(CompressorBase):
            def compress(self, text, **kwargs):
                return MagicMock()
            
            def decompress(self, mega_tokens, **kwargs):
                return "decompressed text"
                
            def load_model(self):
                pass
        
        compressor = MockCompressor(
            model_name="test-model",
            device="cpu",
            compression_ratio=8.0
        )
        
        assert compressor.model_name == "test-model"
        assert compressor.device == "cpu"
        assert compressor.compression_ratio == 8.0
    
    def test_token_counting(self):
        """Test token counting functionality."""
        from src.retrieval_free.core.base import CompressorBase
        
        class MockCompressor(CompressorBase):
            def compress(self, text, **kwargs):
                return MagicMock()
            def decompress(self, mega_tokens, **kwargs):
                return "test"
            def load_model(self):
                pass
        
        compressor = MockCompressor("test")
        
        # Test rough token counting
        text = "This is a test sentence with multiple words."
        count = compressor.count_tokens(text)
        
        # Should be roughly text length / 4
        expected = len(text) // 4
        assert abs(count - expected) <= 2
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        from src.retrieval_free.core.base import CompressorBase
        
        class MockCompressor(CompressorBase):
            def compress(self, text, **kwargs):
                return MagicMock()
            def decompress(self, mega_tokens, **kwargs):
                return "test"
            def load_model(self):
                pass
        
        compressor = MockCompressor("test", compression_ratio=8.0)
        
        # Test memory estimation
        text_length = 1000  # tokens
        memory_est = compressor.estimate_memory_usage(text_length)
        
        assert 'input_mb' in memory_est
        assert 'compressed_mb' in memory_est
        assert 'peak_mb' in memory_est
        assert 'savings_mb' in memory_est
        
        # Compressed should be smaller than input
        assert memory_est['compressed_mb'] < memory_est['input_mb']


class TestMegaToken:
    """Test MegaToken data structure."""
    
    def test_mega_token_creation(self):
        """Test MegaToken creation and validation."""
        from src.retrieval_free.core.base import MegaToken
        
        # Mock embedding tensor
        mock_embedding = MagicMock()
        mock_embedding.dim.return_value = 1
        
        token = MegaToken(
            embedding=mock_embedding,
            metadata={'test': 'value'},
            source_range=(0, 100),
            compression_ratio=8.0
        )
        
        assert token.embedding == mock_embedding
        assert token.metadata['test'] == 'value'
        assert token.source_range == (0, 100)  
        assert token.compression_ratio == 8.0
    
    def test_mega_token_validation(self):
        """Test MegaToken validation logic."""
        from src.retrieval_free.core.base import MegaToken
        
        # Test invalid compression ratio
        with pytest.raises(ValueError, match="Compression ratio must be positive"):
            mock_embedding = MagicMock()
            mock_embedding.dim.return_value = 1
            
            MegaToken(
                embedding=mock_embedding,
                metadata={},
                source_range=(0, 100),
                compression_ratio=-1.0
            )


class TestCompressionResult:
    """Test CompressionResult data structure."""
    
    def test_compression_result_properties(self):
        """Test CompressionResult calculated properties."""
        from src.retrieval_free.core.base import CompressionResult, MegaToken
        
        # Create mock mega tokens
        mock_tokens = [MagicMock() for _ in range(5)]
        
        result = CompressionResult(
            mega_tokens=mock_tokens,
            original_length=1000,
            compressed_length=125,
            compression_ratio=8.0,
            processing_time=1.5,
            metadata={'test': 'data'}
        )
        
        assert result.total_tokens == 5
        assert result.memory_savings == 0.875  # 1 - 125/1000
        assert result.compression_ratio == 8.0


class TestContextCompressor:
    """Test ContextCompressor implementation."""
    
    @patch('src.retrieval_free.core.context_compressor.AutoTokenizer')
    @patch('src.retrieval_free.core.context_compressor.AutoModel')
    def test_context_compressor_creation(self, mock_model, mock_tokenizer):
        """Test ContextCompressor can be created."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        
        compressor = ContextCompressor(
            model_name="test-model",
            compression_ratio=8.0,
            chunk_size=512
        )
        
        assert compressor.model_name == "test-model"
        assert compressor.compression_ratio == 8.0
        assert compressor.chunk_size == 512
    
    def test_text_chunking(self):
        """Test text chunking functionality."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        
        compressor = ContextCompressor(
            chunk_size=10,  # Small for testing
            overlap=2
        )
        
        # Test chunking with word-based splitting
        text = "This is a test sentence with many words for chunking test"
        chunks = compressor._chunk_text(text)
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should be reasonable size
        for chunk in chunks:
            words = chunk.split()
            assert len(words) <= compressor.chunk_size + compressor.overlap
    
    @patch('src.retrieval_free.core.context_compressor.validate_compression_request')
    def test_compression_with_validation(self, mock_validate):
        """Test compression with input validation."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        from src.retrieval_free.validation import ValidationResult
        
        # Mock successful validation
        mock_validate.return_value = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            sanitized_input={'text': 'test text'},
            risk_score=0.0
        )
        
        compressor = ContextCompressor()
        
        # Mock the model loading and processing
        with patch.object(compressor, 'load_model'):
            with patch.object(compressor, '_chunk_text', return_value=['test']):
                with patch.object(compressor, '_encode_chunks_optimized') as mock_encode:
                    with patch.object(compressor, '_cluster_embeddings') as mock_cluster:
                        # Setup mocks
                        mock_encode.return_value = MagicMock()
                        mock_cluster.return_value = ([MagicMock()], [0])
                        
                        # Test compression
                        result = compressor.compress("test text")
                        
                        # Verify validation was called
                        mock_validate.assert_called_once()
                        
                        # Verify result structure
                        assert hasattr(result, 'mega_tokens')
                        assert hasattr(result, 'compression_ratio')


class TestAutoCompressor:
    """Test AutoCompressor factory."""
    
    def test_model_registry(self):
        """Test model registry functionality."""
        from src.retrieval_free.core.auto_compressor import ModelRegistry
        
        # Test listing models
        models = ModelRegistry.list_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Test getting model info
        model_info = ModelRegistry.get_model_info("rfcc-base-8x")
        assert model_info is not None
        assert 'class' in model_info
        assert 'compression_ratio' in model_info
    
    def test_model_registration(self):
        """Test custom model registration."""
        from src.retrieval_free.core.auto_compressor import ModelRegistry
        
        # Register custom model
        custom_config = {
            'class': 'ContextCompressor',
            'compression_ratio': 16.0,
            'description': 'Test model'
        }
        
        ModelRegistry.register_model('test-model', custom_config)
        
        # Verify registration
        registered = ModelRegistry.get_model_info('test-model')
        assert registered == custom_config
    
    @patch('src.retrieval_free.core.auto_compressor.ContextCompressor')
    def test_create_custom_compressor(self, mock_compressor_class):
        """Test creating custom compressor."""
        from src.retrieval_free.core.auto_compressor import AutoCompressor
        
        # Mock compressor instance
        mock_instance = MagicMock()
        mock_compressor_class.return_value = mock_instance
        
        # Create custom compressor
        compressor = AutoCompressor.create_custom_compressor(
            compressor_type="context",
            compression_ratio=16.0,
            device="cpu"
        )
        
        # Verify creation
        mock_compressor_class.assert_called_once()
        mock_instance.load_model.assert_called_once()
        assert compressor == mock_instance


class TestValidationIntegration:
    """Test validation integration."""
    
    def test_validation_error_handling(self):
        """Test validation error handling in compression."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        from src.retrieval_free.exceptions import ValidationError
        
        compressor = ContextCompressor()
        
        # Mock validation failure
        with patch('src.retrieval_free.core.context_compressor.validate_compression_request') as mock_validate:
            mock_validate.return_value = MagicMock(
                is_valid=False,
                errors=['Test validation error']
            )
            
            # Should raise ValidationError
            with pytest.raises(ValidationError):
                compressor.compress("test text")
    
    def test_security_validation(self):
        """Test security validation integration."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test malicious pattern detection
        malicious_text = "<script>alert('xss')</script>"
        result = validator.validate_text(malicious_text)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.risk_score > 0


class TestCachingIntegration:
    """Test caching system integration."""
    
    def test_cache_key_generation(self):
        """Test cache key generation.""" 
        from src.retrieval_free.caching import create_cache_key
        
        key = create_cache_key(
            text="test text",
            model_name="test-model",
            parameters={'ratio': 8.0}
        )
        
        assert isinstance(key, str)
        assert 'compress' in key
        assert 'test-model' in key
    
    def test_memory_cache_operations(self):
        """Test memory cache operations."""
        from src.retrieval_free.caching import MemoryCache
        
        cache = MemoryCache(max_size=10)
        
        # Test put/get
        cache.put("test_key", "test_value")
        value = cache.get("test_key")
        assert value == "test_value"
        
        # Test miss
        missing = cache.get("missing_key")
        assert missing is None
        
        # Test stats
        stats = cache.get_stats()
        assert stats['size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1


class TestOptimizationIntegration:
    """Test optimization features integration."""
    
    def test_batch_processor(self):
        """Test batch processing functionality.""" 
        from src.retrieval_free.optimization import BatchProcessor
        
        processor = BatchProcessor(batch_size=2)
        
        # Test processing
        items = ['item1', 'item2', 'item3', 'item4']
        
        def process_batch(batch):
            return [f"processed_{item}" for item in batch]
        
        results = processor.process_batch(items, process_batch)
        
        assert len(results) == 4
        assert all('processed_' in result for result in results)
        
        processor.shutdown()
    
    def test_memory_optimizer(self):
        """Test memory optimization."""
        from src.retrieval_free.optimization import MemoryOptimizer
        
        optimizer = MemoryOptimizer()
        
        # Test memory usage stats
        stats = optimizer.get_memory_usage()
        assert isinstance(stats, dict)
        
        # Test memory context
        with optimizer.memory_efficient_context():
            # Should not raise any errors
            pass


# Integration test for the full pipeline
class TestFullPipeline:
    """Test complete compression pipeline."""
    
    @patch('src.retrieval_free.core.context_compressor.validate_compression_request')
    @patch('src.retrieval_free.core.context_compressor.TieredCache')
    def test_end_to_end_compression(self, mock_cache_class, mock_validate):
        """Test complete compression pipeline."""
        from src.retrieval_free.core.context_compressor import ContextCompressor
        from src.retrieval_free.validation import ValidationResult
        
        # Setup mocks
        mock_validate.return_value = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            sanitized_input={'text': 'test text'},
            risk_score=0.0
        )
        
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss
        mock_cache_class.return_value = mock_cache
        
        compressor = ContextCompressor(chunk_size=20, overlap=5)
        
        # Mock all the heavy operations
        with patch.object(compressor, 'load_model'):
            with patch.object(compressor, '_encode_chunks_optimized') as mock_encode:
                with patch.object(compressor, '_cluster_embeddings') as mock_cluster:
                    # Setup encoding mock
                    mock_embedding = MagicMock()
                    mock_embedding.shape = [2, 384]  # 2 chunks, 384 dims
                    mock_encode.return_value = mock_embedding
                    
                    # Setup clustering mock
                    mock_centers = [MagicMock(), MagicMock()]
                    mock_labels = [0, 1]
                    mock_cluster.return_value = (mock_centers, mock_labels)
                    
                    # Run compression
                    result = compressor.compress("This is a longer test document that should be split into multiple chunks for processing.")
                    
                    # Verify results
                    assert hasattr(result, 'mega_tokens')
                    assert hasattr(result, 'compression_ratio')
                    assert hasattr(result, 'processing_time')
                    assert len(result.mega_tokens) == 2  # Should match mock clusters
                    
                    # Verify caching was attempted
                    mock_cache.get.assert_called_once()
                    mock_cache.put.assert_called_once()


if __name__ == "__main__":
    # Run tests with minimal output
    pytest.main([__file__, "-v", "--tb=short"])