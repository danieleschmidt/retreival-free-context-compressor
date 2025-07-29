# Architecture Overview

This document provides a comprehensive architectural overview of the Retrieval-Free Context Compressor.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval-Free Context Compressor             │
├─────────────────────────────────────────────────────────────────┤
│  API Layer                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ REST API        │  │ Python SDK      │  │ CLI Interface   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Core Compression Engine                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Context         │  │ Streaming       │  │ Auto            │ │
│  │ Compressor      │  │ Compressor      │  │ Compressor      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Neural Network Components                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Hierarchical    │  │ Information     │  │ Dynamic         │ │
│  │ Encoder         │  │ Bottleneck      │  │ Router          │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Model Storage   │  │ Caching Layer   │  │ Monitoring      │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Context Compressor (`src/retrieval_free/core/`)

The main compression engine responsible for:
- Document tokenization and preprocessing
- Hierarchical encoding from tokens → sentences → paragraphs → mega-tokens
- Information bottleneck optimization
- Compression ratio control

**Key Classes:**
- `ContextCompressor`: Main compression interface
- `AutoCompressor`: Pre-trained model loader
- `CompressionConfig`: Configuration management

### 2. Streaming Compressor (`src/retrieval_free/streaming/`)

Handles continuous document streams:
- Sliding window processing
- Automatic obsolescence detection
- Memory-efficient chunk processing
- Real-time compression updates

**Key Classes:**
- `StreamingCompressor`: Main streaming interface
- `ChunkProcessor`: Individual chunk handler
- `ObsolescenceDetector`: Information relevance tracker

### 3. Neural Network Architecture

#### Hierarchical Encoder
```
Input Tokens → Sentence Encoder → Paragraph Encoder → Document Encoder
     ↓              ↓                   ↓                    ↓
   [768d]         [768d]             [768d]              [768d]
  Token Emb    Sent Attention    Para Attention     Doc Attention
```

#### Information Bottleneck
- **Compression Objective**: Minimize I(X;Z) while maximizing I(Z;Y)
- **Learnable Parameters**: Compression rates per layer
- **Adaptive Mechanism**: Content-aware compression ratios

#### Dynamic Router
- **Cross-Attention**: Query-to-mega-token attention
- **Relevance Scoring**: Information utility metrics  
- **Memory Management**: Automated pruning decisions

### 4. Plugin System (`src/retrieval_free/plugins/`)

Framework integrations:
- **HuggingFace Transformers**: `CompressorPlugin`
- **LangChain**: `CompressionChain`
- **OpenAI Compatible**: API wrapper
- **Custom Integrations**: Extensible plugin architecture

## Data Flow

### Standard Compression Flow
```
1. Document Input
   ↓
2. Tokenization (via transformers tokenizer)
   ↓
3. Hierarchical Encoding
   - Token-level: Self-attention + position encoding
   - Sentence-level: Cross-attention between tokens
   - Paragraph-level: Cross-attention between sentences
   - Document-level: Global representation
   ↓
4. Information Bottleneck
   - Compression target calculation
   - Learnable compression ratios
   - Information preservation optimization
   ↓
5. Mega-token Generation
   - Dense vector representations
   - Metadata preservation
   - Attention masks
   ↓
6. Output: Compressed Context
```

### Streaming Flow
```
1. Document Chunk Input
   ↓
2. Incremental Processing
   - Add to sliding window
   - Update mega-tokens
   - Check obsolescence threshold
   ↓
3. Pruning Decision
   - Relevance scoring
   - Memory pressure evaluation
   - Automatic cleanup
   ↓
4. Output: Updated Mega-tokens
```

## Model Architecture Details

### Compression Transformer
```python
class CompressionTransformer(nn.Module):
    def __init__(self, config):
        self.hierarchical_encoder = HierarchicalEncoder(config)
        self.bottleneck = InformationBottleneck(config)
        self.dynamic_router = DynamicRouter(config)
        
    def forward(self, input_ids, attention_mask):
        # Multi-scale encoding
        token_repr = self.hierarchical_encoder.token_layer(input_ids)
        sent_repr = self.hierarchical_encoder.sentence_layer(token_repr)
        para_repr = self.hierarchical_encoder.paragraph_layer(sent_repr)
        
        # Information bottleneck compression
        compressed = self.bottleneck(para_repr)
        
        # Dynamic routing for query processing
        mega_tokens = self.dynamic_router(compressed)
        
        return mega_tokens
```

### Training Objectives

1. **Reconstruction Loss**: L2 distance between original and reconstructed embeddings
2. **Information Bottleneck**: Mutual information optimization
3. **Question Answering**: Task-specific fine-tuning
4. **Compression Ratio**: Learnable compression targets

```python
total_loss = (
    α * reconstruction_loss +
    β * information_bottleneck_loss +
    γ * qa_loss +
    δ * compression_ratio_loss
)
```

## Performance Characteristics

### Compression Ratios
- **Standard**: 8.0× (256k → 32k tokens)
- **High Quality**: 4.0× (256k → 64k tokens)  
- **High Compression**: 16.0× (256k → 16k tokens)
- **Adaptive**: 4.0×-16.0× based on content complexity

### Latency Targets
- **Single Document**: <500ms (256k tokens)
- **Streaming Chunk**: <50ms (4k token chunks)
- **Batch Processing**: <100ms per document (batch size 8)

### Memory Usage
- **Model Size**: ~2.5GB (base), ~7GB (large)
- **Runtime Memory**: ~4GB peak for 256k token documents
- **GPU Memory**: ~8GB for training, ~3GB for inference

## Scalability Considerations

### Horizontal Scaling
- **Stateless Design**: Each compression request is independent
- **Load Balancing**: Round-robin across compression workers
- **Caching**: Compressed representations cached by content hash

### Vertical Scaling  
- **GPU Acceleration**: CUDA optimizations for attention operations
- **Mixed Precision**: FP16 training and inference
- **Model Parallelism**: Large models split across multiple GPUs

### Storage Architecture
```
Model Storage/
├── pretrained/
│   ├── rfcc-base-8x/
│   ├── rfcc-large-8x/
│   └── rfcc-streaming/
├── cache/
│   ├── compressed_docs/
│   └── embeddings/
└── checkpoints/
    └── training/
```

## Security Architecture

### Input Validation
- Content sanitization
- Token limit enforcement  
- Malicious input detection

### Model Security
- Model integrity verification
- Signed model checkpoints
- Secure model loading

### Output Protection
- Information leakage prevention
- Differential privacy (optional)
- Audit logging

## Monitoring and Observability

### Metrics Collection
- **Performance**: Latency, throughput, memory usage
- **Quality**: Compression ratios, information retention
- **Errors**: Failed compressions, model errors
- **Usage**: API calls, model loads, cache hits

### Logging
```python
# Structured logging example
logger.info("compression_completed", extra={
    "document_id": doc_id,
    "original_tokens": 15420,
    "compressed_tokens": 1927,
    "compression_ratio": 8.0,
    "latency_ms": 234,
    "model_version": "rfcc-base-8x-v1.2"
})
```

### Health Checks
- Model availability
- GPU memory status
- Cache system health
- API endpoint status

## Extension Points

### Custom Compression Algorithms
```python
class CustomCompressor(BaseCompressor):
    def compress(self, document: str) -> CompressedOutput:
        # Custom compression logic
        pass
```

### Plugin Development
```python
class MyFrameworkPlugin(BasePlugin):
    def integrate(self, framework_model):
        # Framework-specific integration
        pass
```

### Evaluation Metrics
```python
class CustomEvaluator(BaseEvaluator):
    def evaluate(self, original, compressed, questions):
        # Custom evaluation logic
        pass
```

## Future Architecture Considerations

### Multimodal Support
- Image and text co-compression
- Cross-modal attention mechanisms
- Unified embedding spaces

### Federated Learning
- Distributed model training
- Privacy-preserving aggregation
- Client-side compression

### Edge Deployment
- Mobile-optimized models
- Quantization strategies
- Progressive loading

This architecture supports the core mission of providing efficient, high-quality document compression while maintaining extensibility and performance at scale.