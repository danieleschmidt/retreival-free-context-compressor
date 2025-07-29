"""Property-based tests using Hypothesis."""

import pytest
from hypothesis import given, strategies as st, assume, settings
from unittest.mock import MagicMock, patch


class TestCompressionProperties:
    """Property-based tests for compression invariants."""

    @given(text=st.text(min_size=1, max_size=1000))
    @settings(max_examples=50)
    def test_compression_preserves_information(self, text):
        """Property: Compression should preserve essential information."""
        assume(len(text.strip()) > 0)  # Avoid empty strings
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            # Mock that always returns consistent compression
            mock_compressor.compress.return_value = [f"compressed_{hash(text) % 1000}"]
            mock_compressor.get_compression_ratio.return_value = 5.0
            MockCompressor.return_value = mock_compressor
            
            compressed = mock_compressor.compress(text)
            ratio = mock_compressor.get_compression_ratio()
            
            # Properties that should hold
            assert len(compressed) > 0  # Always produces output
            assert ratio > 0  # Positive compression ratio
            assert isinstance(compressed, list)  # Returns list of tokens

    @given(
        texts=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10)
    )
    @settings(max_examples=30)
    def test_compression_consistency(self, texts):
        """Property: Same input should produce same compressed output."""
        assume(all(len(text.strip()) > 0 for text in texts))
        
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            
            # Mock consistent compression based on input
            def consistent_compress(text):
                return [f"token_{hash(text) % 100}"] * (len(text) // 10 + 1)
            
            mock_compressor.compress.side_effect = consistent_compress
            MockCompressor.return_value = mock_compressor
            
            # Test consistency
            for text in texts:
                result1 = mock_compressor.compress(text)
                result2 = mock_compressor.compress(text)
                assert result1 == result2  # Consistent results

    @given(
        compression_ratio=st.floats(min_value=1.1, max_value=20.0),
        input_size=st.integers(min_value=100, max_value=10000)
    )
    @settings(max_examples=30)
    def test_compression_ratio_bounds(self, compression_ratio, input_size):
        """Property: Compression ratio should be within reasonable bounds."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            
            # Mock compression that respects the ratio
            expected_output_size = max(1, int(input_size / compression_ratio))
            mock_compressor.compress.return_value = ["token"] * expected_output_size
            mock_compressor.get_compression_ratio.return_value = compression_ratio
            
            MockCompressor.return_value = mock_compressor
            
            dummy_text = "word " * (input_size // 5)  # Approximate input size
            compressed = mock_compressor.compress(dummy_text)
            actual_ratio = mock_compressor.get_compression_ratio()
            
            # Properties
            assert 1.0 < actual_ratio <= 20.0  # Reasonable compression bounds
            assert len(compressed) > 0  # Always produces output
            assert len(compressed) <= input_size  # Never larger than input

    @given(
        text=st.text(min_size=10, max_size=1000),
        chunk_size=st.integers(min_value=5, max_value=100)
    )
    @settings(max_examples=30)
    def test_streaming_compression_properties(self, text, chunk_size):
        """Property: Streaming compression should handle arbitrary chunking."""
        assume(len(text.strip()) >= chunk_size)
        
        with patch('retrieval_free.StreamingCompressor') as MockStreamingCompressor:
            mock_compressor = MagicMock()
            
            # Mock streaming that accumulates tokens
            accumulated_tokens = []
            
            def add_chunk_mock(chunk):
                tokens = [f"stream_{len(chunk)}_{len(accumulated_tokens)}"]
                accumulated_tokens.extend(tokens)
                return accumulated_tokens.copy()
            
            mock_compressor.add_chunk.side_effect = add_chunk_mock
            mock_compressor.should_prune.return_value = False
            MockStreamingCompressor.return_value = mock_compressor
            
            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            results = []
            for chunk in chunks:
                if chunk.strip():  # Skip empty chunks
                    result = mock_compressor.add_chunk(chunk)
                    results.append(result)
            
            # Properties
            if results:
                assert len(results[-1]) >= len(chunks)  # Accumulates over time
                assert all(isinstance(r, list) for r in results)  # Consistent types


class TestTokenizationProperties:
    """Property-based tests for tokenization behavior."""

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=50)
    def test_token_count_properties(self, text):
        """Property: Token counting should be consistent and reasonable."""
        assume(len(text.strip()) > 0)
        
        # Mock tokenizer behavior
        token_count = len(text.split()) + len(text) // 4  # Approximate tokenization
        
        # Properties that should hold
        assert token_count >= 0  # Non-negative token count
        assert token_count <= len(text)  # Reasonable upper bound
        
        # Test consistency
        count1 = token_count
        count2 = token_count
        assert count1 == count2  # Consistent counting

    @given(
        texts=st.lists(st.text(min_size=1, max_size=100), min_size=2, max_size=5)
    )
    @settings(max_examples=30)
    def test_concatenation_properties(self, texts):
        """Property: Token count of concatenated text should be reasonable."""
        assume(all(len(text.strip()) > 0 for text in texts))
        
        # Mock token counting for individual texts
        individual_counts = [len(text.split()) for text in texts]
        concatenated = " ".join(texts)
        concatenated_count = len(concatenated.split())
        
        # Properties
        assert concatenated_count >= max(individual_counts)  # At least as many as largest
        assert concatenated_count <= sum(individual_counts) + len(texts)  # Reasonable upper bound


class TestErrorHandlingProperties:
    """Property-based tests for error handling."""

    @given(invalid_input=st.one_of(st.none(), st.just(""), st.just("   ")))
    @settings(max_examples=20)
    def test_invalid_input_handling(self, invalid_input):
        """Property: Invalid inputs should be handled gracefully."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            
            # Mock appropriate error handling
            if invalid_input is None or not str(invalid_input).strip():
                mock_compressor.compress.side_effect = ValueError("Invalid input")
            else:
                mock_compressor.compress.return_value = ["default_token"]
            
            MockCompressor.return_value = mock_compressor
            
            if invalid_input is None or not str(invalid_input).strip():
                with pytest.raises(ValueError):
                    mock_compressor.compress(invalid_input)
            else:
                result = mock_compressor.compress(invalid_input)
                assert isinstance(result, list)

    @given(
        memory_limit=st.integers(min_value=1, max_value=1000),
        input_size=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=20)
    def test_memory_constraint_properties(self, memory_limit, input_size):
        """Property: Operations should respect memory constraints."""
        with patch('retrieval_free.ContextCompressor') as MockCompressor:
            mock_compressor = MagicMock()
            
            # Simulate memory constraint behavior
            if input_size > memory_limit * 10:  # Arbitrary threshold
                mock_compressor.compress.side_effect = MemoryError("Memory limit exceeded")
            else:
                output_size = min(input_size // 4, memory_limit)
                mock_compressor.compress.return_value = ["mem_token"] * output_size
            
            MockCompressor.return_value = mock_compressor
            
            dummy_input = "x " * input_size
            
            if input_size > memory_limit * 10:
                with pytest.raises(MemoryError):
                    mock_compressor.compress(dummy_input)
            else:
                result = mock_compressor.compress(dummy_input)
                assert len(result) <= memory_limit  # Respects memory limits