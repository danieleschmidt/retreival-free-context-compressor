"""
Word-level encoder that works offline — no downloads required.

Architecture:
  1. Tokenise text into word-id sequences (vocabulary built from data or fixed)
  2. Embed tokens with a learned embedding table
  3. Apply positional encoding + Transformer encoder layers
  4. Pool to a fixed-size vector

The vocabulary is optionally bootstrapped from a small built-in word list so the
model is immediately usable for the demo without training.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

class WordTokenizer:
    """Minimal word-level tokeniser with fixed vocab."""

    PAD, UNK, BOS, EOS = 0, 1, 2, 3
    _SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_vocab: int = 8192):
        self.max_vocab = max_vocab
        if vocab is not None:
            self.word2id = vocab
        else:
            self.word2id = {w: i for i, w in enumerate(self._SPECIAL)}
        self.id2word = {v: k for k, v in self.word2id.items()}

    # ------------------------------------------------------------------
    def _split(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+|[^\s\w]", text.lower())

    def build_vocab(self, texts: List[str]) -> None:
        """Extend vocabulary from a list of texts."""
        counts: Counter = Counter()
        for t in texts:
            counts.update(self._split(t))
        for word, _ in counts.most_common(self.max_vocab - len(self.word2id)):
            if word not in self.word2id:
                idx = len(self.word2id)
                self.word2id[word] = idx
                self.id2word[idx] = word

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        ids = [self.word2id.get(w, self.UNK) for w in self._split(text)]
        ids = ids[:max_len]                         # truncate
        ids = [self.BOS] + ids + [self.EOS]
        return ids

    def batch_encode(
        self,
        texts: List[str],
        max_len: int = 256,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Return (batch, seq_len) int tensor, zero-padded."""
        encoded = [self.encode(t, max_len) for t in texts]
        max_seq = max(len(e) for e in encoded)
        padded = [e + [self.PAD] * (max_seq - len(e)) for e in encoded]
        return torch.tensor(padded, dtype=torch.long, device=device)

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Hierarchical encoder
# ---------------------------------------------------------------------------

class HierarchicalEncoder(nn.Module):
    """
    Two-level encoder:
      Level 1 – sentence/chunk Transformer → sentence vector
      Level 2 – bottleneck projection → mega-token vector

    Parameters
    ----------
    vocab_size : int
    d_model    : token embedding dimension (also Transformer d_model)
    n_heads    : Transformer attention heads
    n_layers   : Transformer encoder layers
    mega_dim   : output mega-token dimension
    dropout    : dropout rate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        mega_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 258,
    ):
        super().__init__()
        self.d_model = d_model
        self.mega_dim = mega_dim

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,      # pre-norm for stability without big datasets
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck: chunk vector → mega-token vector
        self.bottleneck = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mega_dim),
        )

    # ------------------------------------------------------------------
    def _key_padding_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """True where padding (id==0)."""
        return token_ids == 0  # (batch, seq)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        token_ids : (batch, seq_len) long tensor

        Returns
        -------
        mega_tokens : (batch, mega_dim) float tensor
        """
        mask = self._key_padding_mask(token_ids)
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        # Mean-pool over non-padding positions
        lengths = (~mask).float().sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)
        x = (x * (~mask).unsqueeze(-1).float()).sum(dim=1) / lengths       # (B, d_model)

        return self.bottleneck(x)  # (B, mega_dim)
