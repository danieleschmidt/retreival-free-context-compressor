"""Test data fixtures for compression testing."""

import json
from pathlib import Path
from typing import Dict, List, Any


class TestDataLoader:
    """Utility for loading test datasets."""
    
    @staticmethod
    def load_sample_documents() -> Dict[str, str]:
        """Load sample documents for testing."""
        return {
            "wikipedia_sample": """
                Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Machine learning algorithms build a model based on training data in order to make predictions or decisions without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, such as in medicine, email filtering, speech recognition, agriculture, and computer vision, where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.
                
                A subset of machine learning is closely related to computational statistics, which focuses on making predictions using computers, but not all machine learning is statistical learning. The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning. Data mining is a related field of study, focusing on exploratory data analysis through unsupervised learning.
                
                Some types of machine learning algorithms include supervised learning, unsupervised learning, and reinforcement learning. Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. Unsupervised learning algorithms take a set of data that contains only inputs, and finds structure in the data, like grouping or clustering of data points.
            """,
            
            "technical_documentation": """
                The Retrieval-Free Context Compressor implements a novel hierarchical compression architecture. The system processes documents through multiple encoding stages: token-level encoding using transformer models, sentence-level aggregation through pooling operations, paragraph-level compression using cross-attention mechanisms, and document-level mega-token generation.
                
                The information bottleneck principle guides the compression process, ensuring that task-relevant information is preserved while reducing dimensionality. The compression ratio can be adjusted from 2× to 32× depending on the specific use case and quality requirements.
                
                Training involves multi-objective optimization with reconstruction loss, compression loss, and downstream task performance. The model supports both batch processing for large document collections and streaming processing for real-time applications.
            """,
            
            "conversational_text": """
                User: How does the compression algorithm work?
                Assistant: The compression algorithm uses a hierarchical approach that processes text at multiple levels. First, it encodes individual tokens using a transformer model. Then it aggregates sentences using pooling operations. Next, it compresses paragraphs using cross-attention mechanisms. Finally, it generates document-level mega-tokens that represent the entire content.
                
                User: What's the compression ratio?
                Assistant: The compression ratio typically ranges from 8× to 16×, meaning a 256k token document can be compressed to 16k-32k mega-tokens while preserving most of the semantic information needed for question answering and text generation tasks.
                
                User: How is quality maintained?
                Assistant: Quality is maintained through the information bottleneck principle, which ensures that task-relevant information is preserved during compression. The system is trained using multi-objective optimization that balances compression efficiency with downstream task performance.
            """,
            
            "scientific_paper": """
                Abstract: We present a novel approach to long-context processing through hierarchical document compression. Our method achieves 8× compression ratios while maintaining 95% of the original performance on question-answering benchmarks.
                
                Introduction: Large language models face significant challenges when processing long documents due to computational and memory constraints. Traditional approaches rely on external retrieval systems, which introduce latency and complexity. Our approach eliminates the need for external retrieval by compressing long contexts into dense representations.
                
                Methodology: Our compression architecture consists of four main components: (1) Token-level encoding using pre-trained transformers, (2) Sentence-level aggregation through learned pooling, (3) Paragraph-level compression via cross-attention, and (4) Document-level mega-token generation.
                
                Results: Experimental evaluation on Natural Questions, TriviaQA, and MS MARCO datasets shows that our approach achieves superior performance compared to retrieval-based baselines while requiring 50% less memory and 60% less computational time.
                
                Conclusion: The proposed hierarchical compression approach offers a promising alternative to retrieval-augmented generation, enabling efficient processing of long documents within the model's context window.
            """
        }
    
    @staticmethod
    def load_qa_pairs() -> List[Dict[str, Any]]:
        """Load question-answer pairs for evaluation."""
        return [
            {
                "question": "What is machine learning?",
                "context": "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data.",
                "answer": "A field of study in artificial intelligence concerned with developing statistical algorithms that learn from data",
                "answer_start": 34
            },
            {
                "question": "How does hierarchical compression work?",
                "context": "The Retrieval-Free Context Compressor implements a novel hierarchical compression architecture. The system processes documents through multiple encoding stages: token-level encoding, sentence-level aggregation, paragraph-level compression, and document-level mega-token generation.",
                "answer": "It processes documents through multiple encoding stages: token-level, sentence-level, paragraph-level, and document-level",
                "answer_start": 95
            },
            {
                "question": "What compression ratios are achieved?",
                "context": "The compression ratio typically ranges from 8× to 16×, meaning a 256k token document can be compressed to 16k-32k mega-tokens while preserving semantic information.",
                "answer": "8× to 16× compression ratios",
                "answer_start": 42
            }
        ]
    
    @staticmethod
    def load_benchmark_data() -> Dict[str, Any]:
        """Load standardized benchmark data for performance testing."""
        return {
            "compression_benchmarks": {
                "small_docs": [doc[:1000] for doc in TestDataLoader.load_sample_documents().values()],
                "medium_docs": [doc[:5000] for doc in TestDataLoader.load_sample_documents().values()],
                "large_docs": list(TestDataLoader.load_sample_documents().values())
            },
            "performance_targets": {
                "compression_ratio": 8.0,
                "f1_retention": 0.95,
                "latency_ms": 500,
                "memory_mb": 1000
            },
            "quality_metrics": [
                "f1_score",
                "exact_match",
                "rouge_l",
                "bleu_score",
                "compression_ratio",
                "information_retention"
            ]
        }
    
    @staticmethod
    def save_test_results(results: Dict[str, Any], output_path: Path) -> None:
        """Save test results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    @staticmethod
    def load_test_results(results_path: Path) -> Dict[str, Any]:
        """Load test results from JSON file."""
        with open(results_path, 'r') as f:
            return json.load(f)


# Pre-generated test data constants
SAMPLE_TOKENS = list(range(1000))
SAMPLE_EMBEDDINGS = [[0.1, 0.2, 0.3] for _ in range(100)]
SAMPLE_MEGA_TOKENS = [{"embedding": emb, "metadata": {"position": i}} for i, emb in enumerate(SAMPLE_EMBEDDINGS)]

# Configuration for different test scenarios
TEST_CONFIGS = {
    "fast": {
        "max_tokens": 1000,
        "compression_ratio": 4.0,
        "timeout": 30
    },
    "standard": {
        "max_tokens": 10000,
        "compression_ratio": 8.0,
        "timeout": 120
    },
    "comprehensive": {
        "max_tokens": 100000,
        "compression_ratio": 16.0,
        "timeout": 600
    }
}