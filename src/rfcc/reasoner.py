"""
CrossDocumentReasoner — fuses multiple mega-tokens into a single representation
for downstream QA or reasoning tasks.

Given a collection of mega-tokens (one per document or chunk) and an optional
query, the reasoner:
  1. Computes query-guided attention weights over the mega-tokens.
  2. Produces a weighted sum (fused vector).
  3. Optionally ranks mega-tokens by relevance to the query.

When no query is provided, mean pooling is used.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .compressor import MegaToken


# ---------------------------------------------------------------------------
# CrossAttentionFuser (small learnable module)
# ---------------------------------------------------------------------------

class CrossAttentionFuser(nn.Module):
    """
    Single-head cross-attention: query vector attends over mega-token keys/values.

    Parameters
    ----------
    mega_dim   : dimension of mega-token vectors (and query)
    hidden_dim : intermediate projection dimension
    """

    def __init__(self, mega_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.mega_dim = mega_dim

        self.q_proj = nn.Linear(mega_dim, hidden_dim)
        self.k_proj = nn.Linear(mega_dim, hidden_dim)
        self.v_proj = nn.Linear(mega_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, mega_dim)
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,        # (1, mega_dim)
        keys: torch.Tensor,         # (N, mega_dim)
        values: torch.Tensor,       # (N, mega_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        fused  : (1, mega_dim) fused vector
        weights: (N,) attention weights
        """
        q = self.q_proj(query)          # (1, hidden)
        k = self.k_proj(keys)           # (N, hidden)
        v = self.v_proj(values)         # (N, hidden)

        scores = (q @ k.T) * self.scale  # (1, N)
        weights = F.softmax(scores, dim=-1)  # (1, N)

        attended = weights @ v           # (1, hidden)
        fused = self.out_proj(attended)  # (1, mega_dim)
        return fused, weights.squeeze(0)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ReasoningResult:
    """Output of CrossDocumentReasoner.reason()."""
    fused_vector: np.ndarray                    # (mega_dim,) fused representation
    attention_weights: np.ndarray               # (N,) weight per mega-token
    ranked_chunks: List[Tuple[int, float, str]] # (chunk_index, weight, text)
    query: Optional[str]

    def top_k(self, k: int = 3) -> List[Tuple[int, float, str]]:
        """Return top-k most attended chunks."""
        return self.ranked_chunks[:k]

    def __repr__(self) -> str:
        top = self.ranked_chunks[:3]
        top_str = "\n".join(
            f"  [{i}] w={w:.3f} | {t[:60]!r}" for i, w, t in top
        )
        return f"ReasoningResult(query={self.query!r}, top-3:\n{top_str}\n)"


# ---------------------------------------------------------------------------
# CrossDocumentReasoner
# ---------------------------------------------------------------------------

class CrossDocumentReasoner:
    """
    Fuses a collection of mega-tokens into a single representation for
    downstream QA or reasoning.

    Parameters
    ----------
    mega_dim   : must match the MegaTokenCompressor's mega_dim
    hidden_dim : internal attention projection size
    device     : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        mega_dim: int = 64,
        hidden_dim: int = 128,
        device: Optional[str] = None,
    ):
        self.mega_dim = mega_dim

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.fuser = CrossAttentionFuser(mega_dim=mega_dim, hidden_dim=hidden_dim).to(device)
        self.fuser.eval()

        # Tiny word-level query encoder (bag-of-chars hashing → dense vector)
        self._query_proj = nn.Linear(mega_dim, mega_dim).to(device)
        self._query_proj.eval()

    # ------------------------------------------------------------------
    # Query encoding
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> torch.Tensor:
        """
        Encode a text query into a mega_dim vector using character n-gram hashing.
        This is purely offline — no vocabulary or downloads needed.
        """
        # Character trigram hashing into a fixed-size float vector
        vec = np.zeros(self.mega_dim, dtype=np.float32)
        words = query.lower().split()
        for word in words:
            for i in range(len(word)):
                gram = word[max(0, i-1): i+2]  # trigram window
                h = hash(gram) % self.mega_dim
                vec[h] += 1.0
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm
        t = torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self._query_proj(t)  # (1, mega_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        mega_tokens: List[MegaToken],
        query: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Fuse mega-tokens, optionally guided by a text query.

        Parameters
        ----------
        mega_tokens : list of MegaToken from MegaTokenCompressor.compress()
        query       : optional natural-language question or topic

        Returns
        -------
        ReasoningResult with fused_vector, attention_weights, ranked_chunks
        """
        if not mega_tokens:
            raise ValueError("mega_tokens list is empty")

        vecs = np.stack([mt.vector for mt in mega_tokens])       # (N, D)
        keys = torch.tensor(vecs, dtype=torch.float32, device=self.device)
        values = keys  # share keys and values

        with torch.no_grad():
            if query is not None:
                q_vec = self._encode_query(query)                 # (1, mega_dim)
            else:
                # Mean of all mega-tokens as the "query"
                q_vec = keys.mean(dim=0, keepdim=True)            # (1, mega_dim)

            fused, weights = self.fuser(q_vec, keys, values)     # (1,D), (N,)

        fused_np = fused.squeeze(0).cpu().numpy()
        weights_np = weights.cpu().numpy()

        # Build ranked list
        order = np.argsort(weights_np)[::-1]
        ranked = [
            (mega_tokens[i].chunk_index, float(weights_np[i]), mega_tokens[i].source_text)
            for i in order
        ]

        return ReasoningResult(
            fused_vector=fused_np,
            attention_weights=weights_np,
            ranked_chunks=ranked,
            query=query,
        )

    def similarity(self, a: MegaToken, b: MegaToken) -> float:
        """Cosine similarity between two mega-tokens."""
        return float(a.similarity(b))
