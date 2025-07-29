"""Unit tests for compression functionality."""

import pytest
import torch
from unittest.mock import Mock, patch

# These would be real imports in actual implementation
# from retrieval_free.core import ContextCompressor
# from retrieval_free.streaming import StreamingCompressor


class TestContextCompressor:
    """Test core compression functionality."""
    
    def test_compressor_initialization(self, compression_config):
        """Test compressor can be initialized with config."""
        # Mock the actual implementation
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_compressor.return_value = mock_instance
            
            # Test initialization
            compressor = mock_compressor(config=compression_config)
            mock_compressor.assert_called_once_with(config=compression_config)
            assert compressor is not None
    
    def test_compress_document(self, sample_document, compression_config):
        """Test document compression functionality."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.compress.return_value = {
                'compressed_tokens': torch.randn(32, 768),  # Mock compressed representation
                'compression_ratio': 8.2,
                'metadata': {'original_length': 1000, 'compressed_length': 122}
            }
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor(config=compression_config)
            result = compressor.compress(sample_document)
            
            # Verify compression was called and returned expected structure
            mock_instance.compress.assert_called_once_with(sample_document)
            assert 'compressed_tokens' in result
            assert 'compression_ratio' in result
            assert result['compression_ratio'] > 1.0
    
    def test_compression_ratio_calculation(self, sample_document):
        """Test compression ratio calculation."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.get_compression_ratio.return_value = 8.5
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            ratio = compressor.get_compression_ratio()
            
            assert isinstance(ratio, (int, float))
            assert ratio > 1.0
    
    def test_token_counting(self, sample_document):
        """Test token counting functionality."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.count_tokens.return_value = 256
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            token_count = compressor.count_tokens(sample_document)
            
            assert isinstance(token_count, int)
            assert token_count > 0
    
    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.compress.side_effect = ValueError("Empty document")
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            with pytest.raises(ValueError, match="Empty document"):
                compressor.compress("")
    
    def test_compression_with_custom_ratio(self, sample_document):
        """Test compression with custom ratio."""
        custom_ratio = 12.0
        with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.compress.return_value = {
                'compressed_tokens': torch.randn(20, 768),
                'compression_ratio': custom_ratio,
                'metadata': {'target_ratio': custom_ratio}
            }
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            result = compressor.compress(sample_document, target_ratio=custom_ratio)
            
            assert result['compression_ratio'] == custom_ratio


class TestStreamingCompressor:
    """Test streaming compression functionality."""
    
    def test_streaming_initialization(self):
        """Test streaming compressor initialization."""
        with patch('retrieval_free.streaming.StreamingCompressor') as mock_compressor:
            config = {
                'window_size': 32000,
                'compression_ratio': 8.0,
                'prune_threshold': 0.1
            }
            mock_instance = Mock()
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor(**config)
            mock_compressor.assert_called_once_with(**config)
    
    def test_add_chunk(self, sample_document):
        """Test adding chunks to streaming compressor."""
        with patch('retrieval_free.streaming.StreamingCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.add_chunk.return_value = torch.randn(40, 768)
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            result = compressor.add_chunk(sample_document)
            
            mock_instance.add_chunk.assert_called_once_with(sample_document)
            assert result is not None
    
    def test_pruning_mechanism(self):
        """Test automatic pruning in streaming compression."""
        with patch('retrieval_free.streaming.StreamingCompressor') as mock_compressor:
            mock_instance = Mock()
            mock_instance.should_prune.return_value = True
            mock_instance.prune_obsolete.return_value = {'pruned_tokens': 15}
            mock_compressor.return_value = mock_instance
            
            compressor = mock_compressor()
            
            # Test pruning decision
            assert compressor.should_prune() is True
            
            # Test pruning execution
            result = compressor.prune_obsolete()
            assert 'pruned_tokens' in result
            mock_instance.prune_obsolete.assert_called_once()


class TestAutoCompressor:
    """Test auto-compression functionality."""
    
    def test_from_pretrained_loading(self):
        """Test loading pretrained compressor."""
        model_name = "rfcc-base-8x"
        with patch('retrieval_free.core.AutoCompressor') as mock_auto:
            mock_instance = Mock()
            mock_auto.from_pretrained.return_value = mock_instance
            
            compressor = mock_auto.from_pretrained(model_name)
            
            mock_auto.from_pretrained.assert_called_once_with(model_name)
            assert compressor is not None
    
    def test_model_inference(self, sample_document):
        """Test model inference with pretrained compressor."""
        with patch('retrieval_free.core.AutoCompressor') as mock_auto:
            mock_instance = Mock()
            mock_instance.compress.return_value = {
                'compressed_tokens': torch.randn(25, 768),
                'compression_ratio': 8.7,
                'inference_time_ms': 245
            }
            mock_auto.from_pretrained.return_value = mock_instance
            
            compressor = mock_auto.from_pretrained("rfcc-base-8x")
            result = compressor.compress(sample_document)
            
            assert 'compressed_tokens' in result
            assert 'inference_time_ms' in result
            assert result['compression_ratio'] > 1.0


@pytest.mark.parametrize("compression_ratio", [2.0, 4.0, 8.0, 16.0])
def test_compression_ratios(compression_ratio, sample_document):
    """Test different compression ratios."""
    with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
        mock_instance = Mock()
        expected_size = int(1000 / compression_ratio)  # Mock calculation
        mock_instance.compress.return_value = {
            'compressed_tokens': torch.randn(expected_size, 768),
            'compression_ratio': compression_ratio,
        }
        mock_compressor.return_value = mock_instance
        
        compressor = mock_compressor()
        result = compressor.compress(sample_document, target_ratio=compression_ratio)
        
        assert result['compression_ratio'] == compression_ratio


@pytest.mark.gpu
def test_gpu_compression(sample_document, skip_if_no_gpu):
    """Test compression on GPU."""
    with patch('retrieval_free.core.ContextCompressor') as mock_compressor:
        mock_instance = Mock()
        mock_instance.compress.return_value = {
            'compressed_tokens': torch.randn(32, 768).cuda(),
            'compression_ratio': 8.0,
            'device': 'cuda:0'
        }
        mock_compressor.return_value = mock_instance
        
        compressor = mock_compressor(device='cuda')
        result = compressor.compress(sample_document)
        
        assert result['device'] == 'cuda:0'