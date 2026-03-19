"""
MegaTokenCompressor — turns a list of text chunks into mega-token embeddings.

Usage
-----
    from rfcc import MegaTokenCompressor

    compressor = MegaTokenCompressor()
    chunks = ["Document A intro ...", "Document A body ...", "Document B ..."]
    mega_tokens = compressor.compress(chunks)
    # mega_tokens is a list of MegaToken, each with .vector (np.ndarray, shape (mega_dim,))

Design
------
  1. Build / update a word-level vocabulary from the input texts.
  2. Tokenise and batch-encode with the built-in WordTokenizer.
  3. Feed through HierarchicalEncoder (Transformer + bottleneck).
  4. L2-normalise the output vectors for cosine similarity compatibility.
  5. Wrap each vector in a MegaToken dataclass with metadata.

The model is deliberately small and initialised randomly — it works out-of-the-box
for structural/similarity tasks and can be fine-tuned further.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import HierarchicalEncoder, WordTokenizer


# ---------------------------------------------------------------------------
# MegaToken dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MegaToken:
    """Dense compressed representation of a text chunk."""
    vector: np.ndarray          # shape (mega_dim,)
    source_text: str            # original chunk text
    chunk_index: int            # index in the input list
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"MegaToken(chunk={self.chunk_index}, "
            f"dim={len(self.vector)}, "
            f"text={self.source_text[:40]!r}...)"
        )

    def similarity(self, other: "MegaToken") -> float:
        """Cosine similarity to another MegaToken (assumes L2-normalised vectors)."""
        a = self.vector / (np.linalg.norm(self.vector) + 1e-8)
        b = other.vector / (np.linalg.norm(other.vector) + 1e-8)
        return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# MegaTokenCompressor
# ---------------------------------------------------------------------------

class MegaTokenCompressor:
    """
    Compresses a list of text chunks into fixed-size mega-token embeddings.

    Parameters
    ----------
    mega_dim   : int   – dimension of each output mega-token vector  (default 64)
    d_model    : int   – internal Transformer dimension               (default 128)
    n_heads    : int   – number of attention heads                    (default 4)
    n_layers   : int   – number of Transformer encoder layers         (default 2)
    max_seq_len: int   – maximum tokens per chunk                     (default 256)
    device     : str   – 'cpu' or 'cuda' (auto-detected if None)
    normalize  : bool  – L2-normalise output vectors (default True)
    """

    def __init__(
        self,
        mega_dim: int = 64,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.mega_dim = mega_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = WordTokenizer()
        self._encoder: Optional[HierarchicalEncoder] = None  # lazy init

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_encoder(self) -> None:
        """(Re)initialise encoder after vocabulary is known."""
        self._encoder = HierarchicalEncoder(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads if hasattr(self, "n_heads") else 4,
            n_layers=self.n_layers if hasattr(self, "n_layers") else 2,
            mega_dim=self.mega_dim,
            max_seq_len=self.max_seq_len + 2,  # +2 for BOS/EOS
        ).to(self.device)
        self._encoder.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, chunks: List[str], batch_size: int = 32) -> List[MegaToken]:
        """
        Compress text chunks into mega-token embeddings.

        Parameters
        ----------
        chunks     : list of text strings (one per document/paragraph/chunk)
        batch_size : inference batch size

        Returns
        -------
        List of MegaToken objects (same length as chunks)
        """
        if not chunks:
            return []

        # Extend vocabulary and (re)init encoder
        self.tokenizer.build_vocab(chunks)
        self._init_encoder()

        all_vectors: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(chunks), batch_size):
                batch_texts = chunks[start : start + batch_size]
                token_ids = self.tokenizer.batch_encode(
                    batch_texts, max_len=self.max_seq_len, device=self.device
                )
                vecs = self._encoder(token_ids)  # (B, mega_dim)

                if self.normalize:
                    vecs = F.normalize(vecs, p=2, dim=-1)

                all_vectors.extend(vecs.cpu().numpy())

        return [
            MegaToken(
                vector=vec,
                source_text=text,
                chunk_index=i,
            )
            for i, (vec, text) in enumerate(zip(all_vectors, chunks))
        ]

    def similarity_matrix(self, mega_tokens: List[MegaToken]) -> np.ndarray:
        """
        Return (N, N) cosine-similarity matrix for a list of mega-tokens.
        """
        vecs = np.stack([mt.vector for mt in mega_tokens])  # (N, D)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        normed = vecs / norms
        return normed @ normed.T
