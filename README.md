# Retrieval-Free Context Compressor

A hierarchical context compression pipeline that converts long documents into dense **mega-tokens** — fixed-size embedding vectors that preserve semantic content for cross-document inference, without any retrieval step at query time.

## Motivation

Large language models have finite context windows. RAG (Retrieval-Augmented Generation) works around this by fetching relevant chunks at query time, but it adds latency, requires an index, and can miss cross-document dependencies. This library takes a different approach:

> **Compress everything upfront.** Encode each document chunk into a compact mega-token vector. At query time, fuse the relevant mega-tokens using cross-attention — no retrieval index needed.

This makes it a natural complement to graph-based reasoning systems (e.g., DocGraph) where documents are already structured as a graph and you want to reason over *all* nodes without re-reading raw text.

## Architecture

```
Documents
   │
   ▼
┌──────────────────────┐
│  WordTokenizer        │  character-clean word-level tokenisation
│  HierarchicalEncoder  │  Transformer (2L × 4H) + bottleneck projection
└──────────────────────┘
   │
   ▼  mega-tokens (N × D vectors, D=64 by default)
   │
┌──────────────────────┐
│ CrossDocumentReasoner │  query-guided cross-attention over mega-tokens
└──────────────────────┘
   │
   ▼  fused representation + ranked chunk list
```

Key design choices:
- **Offline-first** — no external model downloads required; vocabulary is built from input data
- **Pure PyTorch** — no `sentence-transformers`, no `torch_geometric`, no external APIs
- **Fixed output size** — mega-token dimension is a hyperparameter (default 64); output is always the same shape regardless of input length
- **L2-normalised vectors** — cosine similarity works directly via dot product

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0.

## Quick Start

```python
from rfcc import MegaTokenCompressor, CrossDocumentReasoner

# 1. Compress documents → mega-tokens
compressor = MegaTokenCompressor(mega_dim=64)
chunks = [
    "Knowledge graphs represent entities and relationships in a structured graph.",
    "Context compression reduces long documents to compact dense vectors.",
    "Self-attention allows each token to attend to all other positions.",
]
mega_tokens = compressor.compress(chunks)

print(mega_tokens[0])
# MegaToken(chunk=0, dim=64, text='Knowledge graphs represent entiti'...)

# 2. Cross-document similarity
sim = compressor.similarity_matrix(mega_tokens)
print(sim.round(3))

# 3. Reason over mega-tokens with a query
reasoner = CrossDocumentReasoner(mega_dim=64)
result = reasoner.reason(mega_tokens, query="How are knowledge graphs structured?")
for rank, (idx, weight, text) in enumerate(result.top_k(2), 1):
    print(f"#{rank} [doc {idx}] w={weight:.3f} — {text[:60]}")
```

## Demo

```bash
python demo.py
```

Expected output: similarity matrix over 5 sample documents + query-guided rankings.

## API Reference

### `MegaTokenCompressor`

| Parameter   | Default | Description                          |
|-------------|---------|--------------------------------------|
| `mega_dim`  | 64      | Output vector dimension              |
| `d_model`   | 128     | Internal Transformer dimension       |
| `n_heads`   | 4       | Attention heads                      |
| `n_layers`  | 2       | Transformer encoder layers           |
| `max_seq_len`| 256    | Max tokens per chunk (truncated)     |
| `normalize` | True    | L2-normalise output vectors          |
| `device`    | auto    | `'cpu'` or `'cuda'`                 |

**Methods:**
- `compress(chunks, batch_size=32) → List[MegaToken]`
- `similarity_matrix(mega_tokens) → np.ndarray`  shape `(N, N)`

### `MegaToken`

| Attribute      | Type           | Description               |
|----------------|----------------|---------------------------|
| `vector`       | `np.ndarray`   | shape `(mega_dim,)`       |
| `source_text`  | `str`          | original chunk text       |
| `chunk_index`  | `int`          | position in input list    |
| `metadata`     | `dict`         | free-form extra info      |

**Methods:**
- `similarity(other: MegaToken) → float`  cosine similarity

### `CrossDocumentReasoner`

| Parameter   | Default | Description                       |
|-------------|---------|-----------------------------------|
| `mega_dim`  | 64      | Must match compressor's mega_dim  |
| `hidden_dim`| 128     | Cross-attention projection size   |
| `device`    | auto    | `'cpu'` or `'cuda'`              |

**Methods:**
- `reason(mega_tokens, query=None) → ReasoningResult`
- `similarity(a, b) → float`

### `ReasoningResult`

| Attribute          | Type                          | Description                      |
|--------------------|-------------------------------|----------------------------------|
| `fused_vector`     | `np.ndarray` `(mega_dim,)`   | Weighted sum of mega-token values|
| `attention_weights`| `np.ndarray` `(N,)`          | Softmax weights per mega-token   |
| `ranked_chunks`    | `List[(idx, weight, text)]`  | Sorted by descending weight      |
| `query`            | `str or None`                | The query used                   |

**Methods:**
- `top_k(k=3) → List[(idx, weight, text)]`

## Tests

```bash
pytest tests/ -v
```

34 tests, all passing.

## Connection to DocGraph

This library is designed as a companion to graph-based document reasoning (DocGraph). In that setting:

- Each graph **node** (paper/section/claim) becomes a set of text chunks
- Chunks are compressed to **mega-tokens** and stored on the node
- At query time, the **CrossDocumentReasoner** fuses mega-tokens from relevant nodes without re-reading raw text
- The fused vector can be passed to a downstream LLM or classifier

This decouples the graph traversal (which nodes are relevant?) from the language reasoning (what do those nodes say?).

## License

MIT
