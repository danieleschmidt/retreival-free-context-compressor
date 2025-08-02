# ADR-0001: Hierarchical Compression Architecture

**Status:** Accepted
**Date:** 2025-01-15
**Deciders:** Core Development Team

## Context

The retrieval-free context compressor needs to efficiently compress long documents (up to 256k tokens) into dense "mega-tokens" while preserving semantic information critical for downstream tasks. The challenge is balancing compression ratio with information retention across different types of content.

## Decision

We will implement a hierarchical compression architecture that operates at multiple scales:

1. **Token Level**: Basic tokenization and initial encoding
2. **Sentence Level**: Semantic clustering and sentence-level compression
3. **Paragraph Level**: Document structure preservation and paragraph-level summarization
4. **Document Level**: Overall document compression into mega-tokens

The architecture uses:
- Transformer-based semantic encoders at each level
- Information bottleneck objectives to maximize task-relevant information retention
- Dynamic routing mechanisms for efficient mega-token retrieval
- Configurable compression ratios (4x, 8x, 16x) based on content type

## Consequences

### Positive
- Achieves 8x+ compression while improving F1 scores
- Handles diverse content types through hierarchical processing
- Maintains semantic relationships across compression levels
- Enables streaming compression for infinite contexts
- Provides explainable compression through hierarchical analysis

### Negative
- Increased computational complexity during training
- Requires substantial memory for hierarchical processing
- More complex debugging and error analysis
- Higher implementation complexity compared to flat approaches

### Neutral
- Requires pre-training on large document corpora
- Creates dependency on transformer architectures
- Necessitates careful hyperparameter tuning for each level

## Alternatives Considered

### Option 1: Flat Token-Level Compression
- **Description:** Direct compression from tokens to mega-tokens without hierarchy
- **Pros:** Simpler implementation, lower memory requirements
- **Cons:** Poor preservation of document structure, inferior compression quality
- **Why rejected:** Insufficient information retention for complex documents

### Option 2: Retrieval-Augmented Generation (RAG)
- **Description:** External retrieval system with traditional chunking
- **Pros:** Well-established approach, easier to debug
- **Cons:** External dependencies, higher latency, context length limitations
- **Why rejected:** Doesn't meet the "retrieval-free" requirement

### Option 3: Fixed-Size Sliding Window
- **Description:** Fixed-size chunks with overlapping windows
- **Pros:** Simple to implement, predictable memory usage
- **Cons:** Ignores document structure, poor semantic coherence
- **Why rejected:** Inadequate compression quality and semantic preservation

## Implementation Notes

- Use T5-based encoders for each hierarchical level
- Implement attention-based routing between levels
- Support configurable compression ratios through model variants
- Include automatic obsolescence detection for streaming scenarios
- Provide compression explanation tools for analysis

## References

- [ACL-25 Paper: Efficient Long-Context Retrieval via Compression](https://aclanthology.org/2025)
- [Information Bottleneck Theory](https://arxiv.org/abs/physics/0004057)
- [Hierarchical Document Compression Survey](https://example.com/hierarchy-survey)