"""Tests for MegaTokenCompressor."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from rfcc import MegaTokenCompressor, MegaToken


SAMPLE_CHUNKS = [
    "Knowledge graphs represent entities and their relationships in a structured way.",
    "Context compression reduces long documents to compact dense vectors.",
    "The transformer attention mechanism enables context-aware representations.",
    "Retrieval-augmented generation fetches relevant passages at inference time.",
]


# ---------------------------------------------------------------------------
# MegaTokenCompressor tests
# ---------------------------------------------------------------------------

class TestMegaTokenCompressor:
    def setup_method(self):
        self.comp = MegaTokenCompressor(mega_dim=32, d_model=64, n_heads=2, n_layers=1)

    def test_compress_returns_correct_count(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        assert len(result) == len(SAMPLE_CHUNKS)

    def test_each_result_is_megatoken(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        for mt in result:
            assert isinstance(mt, MegaToken)

    def test_vector_shape(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        for mt in result:
            assert mt.vector.ndim == 1
            assert len(mt.vector) == 32  # mega_dim

    def test_vector_dtype(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        for mt in result:
            assert mt.vector.dtype == np.float32

    def test_chunk_indices(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        for i, mt in enumerate(result):
            assert mt.chunk_index == i

    def test_source_text_preserved(self):
        result = self.comp.compress(SAMPLE_CHUNKS)
        for mt, original in zip(result, SAMPLE_CHUNKS):
            assert mt.source_text == original

    def test_vectors_are_normalised(self):
        """Default normalize=True → unit vectors."""
        result = self.comp.compress(SAMPLE_CHUNKS)
        for mt in result:
            norm = float(np.linalg.norm(mt.vector))
            assert abs(norm - 1.0) < 1e-5, f"Expected unit vector, got norm={norm}"

    def test_compress_empty_returns_empty(self):
        assert self.comp.compress([]) == []

    def test_compress_single_chunk(self):
        result = self.comp.compress(["Just one sentence here."])
        assert len(result) == 1
        assert result[0].chunk_index == 0

    def test_similarity_matrix_shape(self):
        tokens = self.comp.compress(SAMPLE_CHUNKS)
        sim = self.comp.similarity_matrix(tokens)
        N = len(SAMPLE_CHUNKS)
        assert sim.shape == (N, N)

    def test_similarity_matrix_diagonal_ones(self):
        tokens = self.comp.compress(SAMPLE_CHUNKS)
        sim = self.comp.similarity_matrix(tokens)
        for i in range(len(SAMPLE_CHUNKS)):
            assert abs(sim[i, i] - 1.0) < 1e-5

    def test_similarity_matrix_symmetric(self):
        tokens = self.comp.compress(SAMPLE_CHUNKS)
        sim = self.comp.similarity_matrix(tokens)
        np.testing.assert_allclose(sim, sim.T, atol=1e-5)

    def test_megatoken_similarity(self):
        tokens = self.comp.compress(SAMPLE_CHUNKS)
        sim = tokens[0].similarity(tokens[1])
        assert -1.0 <= sim <= 1.0

    def test_different_chunks_produce_different_vectors(self):
        """Deterministic encoder: same vocab → same output."""
        tokens = self.comp.compress(["apple orange", "quantum mechanics"])
        # They should not be identical (different words → different vectors)
        assert not np.allclose(tokens[0].vector, tokens[1].vector, atol=1e-4)

    def test_no_normalize_flag(self):
        comp = MegaTokenCompressor(mega_dim=32, d_model=64, n_heads=2, n_layers=1, normalize=False)
        result = comp.compress(["hello world"])
        assert result[0].vector.ndim == 1

    def test_large_batch(self):
        chunks = [f"Document number {i} with some text content." for i in range(50)]
        result = self.comp.compress(chunks, batch_size=16)
        assert len(result) == 50

    def test_repr(self):
        tokens = self.comp.compress(["test"])
        r = repr(tokens[0])
        assert "MegaToken" in r
        assert "chunk=0" in r
