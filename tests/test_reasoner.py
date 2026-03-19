"""Tests for CrossDocumentReasoner."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rfcc import MegaTokenCompressor, CrossDocumentReasoner
from rfcc.reasoner import ReasoningResult


CHUNKS = [
    "Knowledge graphs store facts as triples of subject, predicate, object.",
    "Context compression encodes long text into compact mega-token vectors.",
    "Self-attention computes pairwise affinities between all token positions.",
    "Document retrieval fetches relevant passages given a natural language query.",
    "Neural networks learn distributed representations through gradient descent.",
]


@pytest.fixture(scope="module")
def mega_tokens():
    comp = MegaTokenCompressor(mega_dim=32, d_model=64, n_heads=2, n_layers=1)
    return comp.compress(CHUNKS)


@pytest.fixture(scope="module")
def reasoner():
    return CrossDocumentReasoner(mega_dim=32)


class TestCrossDocumentReasoner:
    def test_reason_returns_result_type(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert isinstance(result, ReasoningResult)

    def test_fused_vector_shape(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert result.fused_vector.ndim == 1
        assert len(result.fused_vector) == 32

    def test_fused_vector_dtype(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert result.fused_vector.dtype == np.float32

    def test_attention_weights_shape(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert result.attention_weights.shape == (len(CHUNKS),)

    def test_attention_weights_sum_to_one(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        total = float(result.attention_weights.sum())
        assert abs(total - 1.0) < 1e-5

    def test_attention_weights_non_negative(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert (result.attention_weights >= 0).all()

    def test_ranked_chunks_length(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        assert len(result.ranked_chunks) == len(CHUNKS)

    def test_ranked_chunks_descending(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        weights = [w for _, w, _ in result.ranked_chunks]
        assert weights == sorted(weights, reverse=True)

    def test_top_k(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        top3 = result.top_k(3)
        assert len(top3) == 3

    def test_top_k_larger_than_n(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens)
        top_all = result.top_k(100)
        assert len(top_all) == len(CHUNKS)

    def test_with_query(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens, query="How are knowledge graphs structured?")
        assert result.query == "How are knowledge graphs structured?"
        assert isinstance(result, ReasoningResult)

    def test_with_query_weights_sum_to_one(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens, query="attention mechanism in transformers")
        total = float(result.attention_weights.sum())
        assert abs(total - 1.0) < 1e-5

    def test_query_none_vs_with_query_different(self, mega_tokens, reasoner):
        """Query-guided reasoning should produce different weights than mean pooling."""
        result_no_q = reasoner.reason(mega_tokens, query=None)
        result_q = reasoner.reason(mega_tokens, query="knowledge graph triples")
        # Weights should differ (different attention)
        assert not np.allclose(result_no_q.attention_weights, result_q.attention_weights)

    def test_empty_mega_tokens_raises(self, reasoner):
        with pytest.raises(ValueError, match="empty"):
            reasoner.reason([])

    def test_single_mega_token(self, reasoner):
        comp = MegaTokenCompressor(mega_dim=32, d_model=64, n_heads=2, n_layers=1)
        tokens = comp.compress(["Only one document."])
        result = reasoner.reason(tokens)
        assert len(result.attention_weights) == 1
        assert abs(float(result.attention_weights[0]) - 1.0) < 1e-5

    def test_similarity_method(self, mega_tokens, reasoner):
        sim = reasoner.similarity(mega_tokens[0], mega_tokens[1])
        assert -1.0 <= sim <= 1.0

    def test_repr(self, mega_tokens, reasoner):
        result = reasoner.reason(mega_tokens, query="test")
        r = repr(result)
        assert "ReasoningResult" in r
        assert "test" in r
