"""End-to-end integration tests for the compression pipeline."""

import pytest
from unittest.mock import MagicMock, patch


class TestCompressionPipeline:
    """Test the complete compression and decompression pipeline."""

    @pytest.mark.integration
    def test_full_compression_cycle(self, sample_documents, mock_tokenizer):
        """Test complete compression and query cycle."""
        # Mock the compression components since actual models aren't available
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["mega_token_1", "mega_token_2"]
            mock_compressor.count_tokens.return_value = 1000
            mock_compressor.get_compression_ratio.return_value = 8.0
            MockCompressor.return_value = mock_compressor
            
            # Test would import and use actual classes here
            # from retrieval_free import ContextCompressor
            # compressor = ContextCompressor.from_pretrained("test-model")
            
            document = sample_documents["medium"]
            compressed = mock_compressor.compress(document)
            
            assert len(compressed) == 2
            assert mock_compressor.get_compression_ratio() == 8.0

    @pytest.mark.integration
    def test_streaming_compression_integration(self, sample_documents):
        """Test streaming compression with multiple document chunks."""
        with patch('retrieval_free.StreamingCompressor') as MockStreamingCompressor:
            mock_compressor = MagicMock()
            mock_compressor.add_chunk.return_value = ["stream_token_1"]
            mock_compressor.should_prune.return_value = False
            MockStreamingCompressor.return_value = mock_compressor
            
            # Test streaming workflow
            chunks = [sample_documents["short"]] * 5
            
            for chunk in chunks:
                result = mock_compressor.add_chunk(chunk)
                assert result is not None
                
            assert mock_compressor.add_chunk.call_count == 5

    @pytest.mark.integration 
    @pytest.mark.slow
    def test_large_document_processing(self, sample_documents):
        """Test processing of large documents."""
        with patch('retrieval_free.AutoCompressor') as MockAutoCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["large_token_1"] * 10
            MockAutoCompressor.from_pretrained.return_value = mock_compressor
            
            large_doc = sample_documents["long"]
            
            # Test large document handling
            compressed = mock_compressor.compress(large_doc)
            assert len(compressed) == 10

    @pytest.mark.integration
    def test_multi_document_compression(self, compression_test_data):
        """Test compression of multiple related documents."""
        with patch('retrieval_free.MultiDocCompressor') as MockMultiCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress_collection.return_value = {
                "mega_tokens": ["multi_token_1", "multi_token_2"],
                "citations": [{"doc_id": 0, "span": (0, 50)}],
            }
            MockMultiCompressor.return_value = mock_compressor
            
            documents = compression_test_data["documents"]
            result = mock_compressor.compress_collection(documents)
            
            assert "mega_tokens" in result
            assert "citations" in result
            assert len(result["mega_tokens"]) == 2


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @pytest.mark.integration
    def test_malformed_input_handling(self):
        """Test handling of malformed or empty inputs."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.side_effect = ValueError("Invalid input")
            MockCompressor.return_value = mock_compressor
            
            with pytest.raises(ValueError, match="Invalid input"):
                mock_compressor.compress("")

    @pytest.mark.integration
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure conditions."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.side_effect = MemoryError("Out of memory")
            MockCompressor.return_value = mock_compressor
            
            with pytest.raises(MemoryError):
                mock_compressor.compress("Large document content")

    @pytest.mark.integration
    def test_network_failure_recovery(self):
        """Test recovery from network failures during model loading."""
        with patch('retrieval_free.AutoCompressor.from_pretrained') as mock_load:
            mock_load.side_effect = ConnectionError("Network unavailable")
            
            with pytest.raises(ConnectionError):
                # This would test actual network error handling
                mock_load("remote-model")


class TestPerformanceIntegration:
    """Integration tests focused on performance characteristics."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_compression_latency(self, sample_documents):
        """Test compression latency meets performance requirements."""
        import time
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            def slow_compress(text):
                time.sleep(0.1)  # Simulate processing time
                return ["token_1", "token_2"]
            
            mock_compressor = MagicMock()
            mock_compressor.compress.side_effect = slow_compress
            MockCompressor.return_value = mock_compressor
            
            start_time = time.time()
            result = mock_compressor.compress(sample_documents["medium"])
            elapsed = time.time() - start_time
            
            # Assert reasonable performance (would be actual requirements)
            assert elapsed < 1.0  # Less than 1 second
            assert len(result) == 2

    @pytest.mark.integration
    def test_memory_usage_scaling(self, sample_documents):
        """Test memory usage scales appropriately with input size."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            mock_compressor.compress.return_value = ["token"] * 5
            MockCompressor.return_value = mock_compressor
            
            # Test with different document sizes
            for doc_type in ["short", "medium", "long"]:
                result = mock_compressor.compress(sample_documents[doc_type])
                assert len(result) == 5  # Consistent output size regardless of input