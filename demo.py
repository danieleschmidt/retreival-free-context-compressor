#!/usr/bin/env python3
"""
demo.py — Retrieval-Free Context Compressor demo

Demonstrates hierarchical compression of 5 documents into mega-tokens
and cross-document reasoning with a similarity query.

Usage:
    python demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from rfcc import MegaTokenCompressor, CrossDocumentReasoner

# ---------------------------------------------------------------------------
# Sample documents (diverse topics to test cross-document reasoning)
# ---------------------------------------------------------------------------

DOCUMENTS = [
    # Doc 0 — Knowledge graphs
    """Knowledge graphs represent information as a network of entities and relationships.
    They enable structured reasoning over large factual datasets. Nodes represent entities
    such as people, places, and concepts. Edges represent typed relationships between them.
    Graph-based inference allows systems to answer multi-hop questions by traversing edges.""",

    # Doc 1 — Context compression (the topic of this repo)
    """Context compression reduces the length of text sequences passed to language models.
    Long documents consume many tokens, which is expensive and slow. Hierarchical compression
    first encodes sentences, then paragraphs, then whole documents into compact representations.
    Mega-tokens are fixed-size dense vectors that summarise the semantic content of a chunk.""",

    # Doc 2 — Transformer attention
    """The attention mechanism in Transformers allows each token to attend to all other tokens.
    Self-attention computes query, key, and value projections for every position in the sequence.
    Cross-attention extends this to attend over a different sequence, enabling encoder-decoder
    architectures. Attention weights reveal which tokens are most relevant for a given position.""",

    # Doc 3 — DocGraph (Daniel's dissertation topic)
    """DocGraph is a framework for building knowledge graphs from scientific literature.
    It extracts entities, relations, and claims from papers and links them into a unified graph.
    Downstream tasks such as literature review, hypothesis generation, and citation analysis
    benefit from traversing the DocGraph structure rather than re-reading raw documents.""",

    # Doc 4 — Retrieval-augmented generation
    """Retrieval-augmented generation (RAG) combines a retrieval system with a generative model.
    A query is used to retrieve relevant documents from a large corpus. The retrieved text is
    then prepended to the prompt as context. RAG improves factual accuracy but requires an
    index and a retrieval step at inference time, adding latency and complexity.""",
]

QUERIES = [
    "How do knowledge graphs enable multi-hop reasoning?",
    "What is the role of attention in compressing context?",
    "How does DocGraph relate to literature understanding?",
]


def separator(title: str = "") -> None:
    line = "─" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def main() -> None:
    print("=" * 60)
    print("  Retrieval-Free Context Compressor — Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Compress documents to mega-tokens
    # ------------------------------------------------------------------
    separator("Step 1: Compressing documents → mega-tokens")

    compressor = MegaTokenCompressor(mega_dim=64, d_model=128, n_heads=4, n_layers=2)
    mega_tokens = compressor.compress(DOCUMENTS)

    print(f"\nCompressed {len(DOCUMENTS)} documents → {len(mega_tokens)} mega-tokens")
    print(f"Each mega-token: {mega_tokens[0].vector.shape[0]}-dimensional vector")
    print()
    for mt in mega_tokens:
        print(f"  {mt}")

    # ------------------------------------------------------------------
    # Step 2: Similarity matrix
    # ------------------------------------------------------------------
    separator("Step 2: Cross-document similarity matrix")

    sim = compressor.similarity_matrix(mega_tokens)
    titles = ["KnowledgeGraph", "CtxCompressor", "Attention", "DocGraph", "RAG"]
    header = f"{'':14s}" + "".join(f"{t:>14s}" for t in titles)
    print(f"\n{header}")
    for i, row in enumerate(sim):
        vals = "".join(f"{v:14.3f}" for v in row)
        print(f"{titles[i]:14s}{vals}")

    # ------------------------------------------------------------------
    # Step 3: Cross-document reasoning with queries
    # ------------------------------------------------------------------
    separator("Step 3: Cross-document reasoning (query-guided)")

    reasoner = CrossDocumentReasoner(mega_dim=64)

    for query in QUERIES:
        print(f"\nQuery: {query!r}")
        result = reasoner.reason(mega_tokens, query=query)
        print(f"  Top-3 most relevant chunks:")
        for rank, (idx, weight, text) in enumerate(result.top_k(3), 1):
            print(f"    #{rank} [doc {idx}] weight={weight:.3f} — {text[:70].strip()!r}")

    # ------------------------------------------------------------------
    # Step 4: Fused representation
    # ------------------------------------------------------------------
    separator("Step 4: Fused representation (mean pooling, no query)")

    result = reasoner.reason(mega_tokens)
    print(f"\nFused vector shape: {result.fused_vector.shape}")
    print(f"Fused vector norm:  {(result.fused_vector ** 2).sum() ** 0.5:.4f}")
    print(f"Attention weights:  {[f'{w:.3f}' for w in result.attention_weights]}")

    separator()
    print("\n✓ Demo complete. All steps passed.\n")


if __name__ == "__main__":
    main()
