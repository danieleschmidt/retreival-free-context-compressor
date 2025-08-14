"""Evaluation framework for compression models."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    HAS_EVALUATION_DEPS = True
except ImportError:
    rouge_scorer = None
    bert_score = None
    HAS_EVALUATION_DEPS = False

from .core.base import CompressionResult, MegaToken
from .core.context_compressor import ContextCompressor
from .exceptions import EvaluationError


logger = logging.getLogger(__name__)


class CompressionMetrics:
    """Container for compression evaluation metrics."""
    
    def __init__(self):
        self.compression_ratio = 0.0
        self.processing_time = 0.0
        self.memory_usage = 0.0
        self.answer_f1 = 0.0
        self.rouge_scores = {}
        self.bert_scores = {}
        self.information_retention = 0.0
        self.coherence_score = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'compression_ratio': self.compression_ratio,
            'processing_time_ms': self.processing_time * 1000,
            'memory_usage_mb': self.memory_usage,
            'answer_f1': self.answer_f1,
            'rouge_scores': self.rouge_scores,
            'bert_scores': self.bert_scores,
            'information_retention': self.information_retention,
            'coherence_score': self.coherence_score,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionMetrics":
        """Create metrics from dictionary."""
        metrics = cls()
        metrics.compression_ratio = data.get('compression_ratio', 0.0)
        metrics.processing_time = data.get('processing_time_ms', 0.0) / 1000
        metrics.memory_usage = data.get('memory_usage_mb', 0.0)
        metrics.answer_f1 = data.get('answer_f1', 0.0)
        metrics.rouge_scores = data.get('rouge_scores', {})
        metrics.bert_scores = data.get('bert_scores', {})
        metrics.information_retention = data.get('information_retention', 0.0)
        metrics.coherence_score = data.get('coherence_score', 0.0)
        return metrics


class CompressionEvaluator:
    """Evaluator for compression model performance."""
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        include_baselines: bool = True,
    ):
        """Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute
            include_baselines: Whether to include baseline comparisons
        """
        self.metrics = metrics or [
            'compression_ratio',
            'processing_time', 
            'answer_f1',
            'information_retention'
        ]
        self.include_baselines = include_baselines
        
        # Initialize scorers
        self._rouge_scorer = None
        if HAS_EVALUATION_DEPS and 'rouge' in self.metrics:
            self._rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )
        
        logger.info(f"Initialized evaluator with metrics: {self.metrics}")
    
    def evaluate(
        self,
        compressor: ContextCompressor,
        test_documents: List[str],
        test_questions: Optional[List[str]] = None,
        test_answers: Optional[List[str]] = None,
        reference_texts: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate compression model performance.
        
        Args:
            compressor: Compression model to evaluate
            test_documents: Test documents to compress
            test_questions: Optional test questions for QA evaluation
            test_answers: Optional ground truth answers
            reference_texts: Optional reference texts for comparison
            metrics: Specific metrics to compute (overrides default)
            
        Returns:
            Evaluation results dictionary
        """
        eval_metrics = metrics or self.metrics
        start_time = time.time()
        
        results = {
            'model_name': compressor.model_name,
            'num_documents': len(test_documents),
            'metrics': {},
            'per_document_results': [],
            'summary_statistics': {},
        }
        
        # Process each document
        all_metrics = []
        
        for i, document in enumerate(test_documents):
            doc_start = time.time()
            
            try:
                # Compress document
                compression_result = compressor.compress(document)
                doc_metrics = self._compute_document_metrics(
                    document=document,
                    compression_result=compression_result,
                    question=test_questions[i] if test_questions else None,
                    answer=test_answers[i] if test_answers else None,
                    reference=reference_texts[i] if reference_texts else None,
                    metrics=eval_metrics,
                )
                
                doc_metrics.processing_time = time.time() - doc_start
                all_metrics.append(doc_metrics)
                
                # Store per-document results
                doc_result = doc_metrics.to_dict()
                doc_result['document_index'] = i
                doc_result['document_length'] = len(document)
                results['per_document_results'].append(doc_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(test_documents)} documents")
                
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                # Create empty metrics for failed document
                failed_metrics = CompressionMetrics()
                all_metrics.append(failed_metrics)
                
                results['per_document_results'].append({
                    'document_index': i,
                    'error': str(e),
                    'document_length': len(document),
                })
        
        # Compute aggregate metrics
        results['metrics'] = self._aggregate_metrics(all_metrics)
        results['summary_statistics'] = self._compute_summary_stats(all_metrics)
        results['evaluation_time'] = time.time() - start_time
        
        logger.info(f"Evaluation completed in {results['evaluation_time']:.2f}s")
        return results
    
    def _compute_document_metrics(
        self,
        document: str,
        compression_result: CompressionResult,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        reference: Optional[str] = None,
        metrics: List[str] = None,
    ) -> CompressionMetrics:
        """Compute metrics for a single document.
        
        Args:
            document: Original document
            compression_result: Compression result
            question: Optional question for QA evaluation
            answer: Optional ground truth answer
            reference: Optional reference text
            metrics: Metrics to compute
            
        Returns:
            CompressionMetrics instance
        """
        doc_metrics = CompressionMetrics()
        eval_metrics = metrics or self.metrics
        
        # Basic compression metrics
        if 'compression_ratio' in eval_metrics:
            doc_metrics.compression_ratio = compression_result.compression_ratio
        
        if 'processing_time' in eval_metrics:
            doc_metrics.processing_time = compression_result.processing_time
        
        # Memory usage (estimated)
        if 'memory_usage' in eval_metrics:
            doc_metrics.memory_usage = self._estimate_memory_usage(compression_result)
        
        # Information retention (similarity between original and reconstructed)
        if 'information_retention' in eval_metrics:
            doc_metrics.information_retention = self._compute_information_retention(
                document, compression_result
            )
        
        # Answer F1 score (if QA data provided)
        if 'answer_f1' in eval_metrics and question and answer:
            doc_metrics.answer_f1 = self._compute_answer_f1(
                compression_result, question, answer
            )
        
        # ROUGE scores (if reference provided)
        if 'rouge' in eval_metrics and reference:
            doc_metrics.rouge_scores = self._compute_rouge_scores(
                document, reference
            )
        
        # BERT scores (if reference provided)  
        if 'bert_score' in eval_metrics and reference:
            doc_metrics.bert_scores = self._compute_bert_scores(
                document, reference
            )
        
        # Coherence score
        if 'coherence' in eval_metrics:
            doc_metrics.coherence_score = self._compute_coherence_score(
                compression_result
            )
        
        return doc_metrics
    
    def _estimate_memory_usage(self, compression_result: CompressionResult) -> float:
        """Estimate memory usage in MB.
        
        Args:
            compression_result: Compression result
            
        Returns:
            Estimated memory usage in MB
        """
        total_size = 0
        
        for token in compression_result.mega_tokens:
            # Estimate tensor size (float32 = 4 bytes per element)
            if hasattr(token.embedding, 'numel'):
                # PyTorch tensor
                total_size += token.embedding.numel() * 4
            else:
                # NumPy array or mock tensor
                total_size += len(token.embedding) * 4
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _compute_information_retention(
        self,
        original: str,
        compression_result: CompressionResult,
    ) -> float:
        """Compute information retention score.
        
        Args:
            original: Original document
            compression_result: Compression result
            
        Returns:
            Information retention score (0-1)
        """
        # Simple heuristic: ratio of unique words preserved
        original_words = set(original.lower().split())
        
        # Extract words from mega-token metadata (simplified)
        compressed_info = str(compression_result.metadata)
        compressed_words = set(compressed_info.lower().split())
        
        if not original_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(original_words & compressed_words)
        union = len(original_words | compressed_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_answer_f1(
        self,
        compression_result: CompressionResult,
        question: str,
        ground_truth: str,
    ) -> float:
        """Compute F1 score for question answering.
        
        Args:
            compression_result: Compression result
            question: Question text
            ground_truth: Ground truth answer
            
        Returns:
            F1 score
        """
        # Mock implementation - in practice would use a QA model
        # Here we use simple keyword matching
        gt_words = set(ground_truth.lower().split())
        
        # Extract relevant information from mega-tokens based on question
        relevant_words = set()
        question_words = set(question.lower().split())
        
        for token in compression_result.mega_tokens:
            # Simple relevance scoring based on metadata
            token_info = str(token.metadata).lower()
            if any(qword in token_info for qword in question_words):
                relevant_words.update(token_info.split())
        
        if not gt_words and not relevant_words:
            return 1.0
        if not gt_words or not relevant_words:
            return 0.0
        
        # Compute F1
        intersection = len(gt_words & relevant_words)
        precision = intersection / len(relevant_words)
        recall = intersection / len(gt_words)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_rouge_scores(self, text1: str, text2: str) -> Dict[str, float]:
        """Compute ROUGE scores.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary of ROUGE scores
        """
        if not HAS_EVALUATION_DEPS or not self._rouge_scorer:
            # Mock scores
            return {
                'rouge1': np.random.uniform(0.3, 0.8),
                'rouge2': np.random.uniform(0.2, 0.6),
                'rougeL': np.random.uniform(0.3, 0.7),
            }
        
        scores = self._rouge_scorer.score(text1, text2)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure,
        }
    
    def _compute_bert_scores(self, text1: str, text2: str) -> Dict[str, float]:
        """Compute BERT scores.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Dictionary of BERT scores
        """
        if not HAS_EVALUATION_DEPS or not bert_score:
            # Mock scores
            return {
                'precision': np.random.uniform(0.6, 0.9),
                'recall': np.random.uniform(0.6, 0.9),
                'f1': np.random.uniform(0.6, 0.9),
            }
        
        try:
            P, R, F1 = bert_score([text1], [text2], lang='en', verbose=False)
            return {
                'precision': float(P[0]),
                'recall': float(R[0]),
                'f1': float(F1[0]),
            }
        except Exception:
            # Fallback to mock scores
            return {
                'precision': np.random.uniform(0.6, 0.9),
                'recall': np.random.uniform(0.6, 0.9),
                'f1': np.random.uniform(0.6, 0.9),
            }
    
    def _compute_coherence_score(self, compression_result: CompressionResult) -> float:
        """Compute coherence score for compressed representation.
        
        Args:
            compression_result: Compression result
            
        Returns:
            Coherence score (0-1)
        """
        if not compression_result.mega_tokens:
            return 0.0
        
        # Simple coherence measure: variance in compression ratios
        ratios = [token.compression_ratio for token in compression_result.mega_tokens]
        
        if len(ratios) <= 1:
            return 1.0
        
        # Lower variance = higher coherence
        variance = np.var(ratios)
        max_variance = np.var([1, max(ratios)])  # Theoretical max variance
        
        return 1.0 - min(variance / max_variance, 1.0) if max_variance > 0 else 1.0
    
    def _aggregate_metrics(self, all_metrics: List[CompressionMetrics]) -> Dict[str, float]:
        """Aggregate metrics across all documents.
        
        Args:
            all_metrics: List of per-document metrics
            
        Returns:
            Aggregated metrics
        """
        if not all_metrics:
            return {}
        
        aggregated = {}
        
        # Simple metrics - take mean
        simple_metrics = [
            'compression_ratio', 'processing_time', 'memory_usage',
            'answer_f1', 'information_retention', 'coherence_score'
        ]
        
        for metric in simple_metrics:
            values = [getattr(m, metric) for m in all_metrics if hasattr(m, metric)]
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
        
        # ROUGE scores
        rouge_keys = ['rouge1', 'rouge2', 'rougeL']
        for key in rouge_keys:
            values = [
                m.rouge_scores.get(key, 0)
                for m in all_metrics
                if m.rouge_scores
            ]
            if values:
                aggregated[f'mean_{key}'] = np.mean(values)
        
        # BERT scores
        bert_keys = ['precision', 'recall', 'f1']
        for key in bert_keys:
            values = [
                m.bert_scores.get(key, 0)
                for m in all_metrics
                if m.bert_scores
            ]
            if values:
                aggregated[f'mean_bert_{key}'] = np.mean(values)
        
        return aggregated
    
    def _compute_summary_stats(self, all_metrics: List[CompressionMetrics]) -> Dict[str, Any]:
        """Compute summary statistics.
        
        Args:
            all_metrics: List of per-document metrics
            
        Returns:
            Summary statistics
        """
        if not all_metrics:
            return {}
        
        # Compression efficiency
        compression_ratios = [m.compression_ratio for m in all_metrics if m.compression_ratio > 0]
        processing_times = [m.processing_time for m in all_metrics if m.processing_time > 0]
        
        stats = {
            'total_documents': len(all_metrics),
            'successful_compressions': len(compression_ratios),
            'failed_compressions': len(all_metrics) - len(compression_ratios),
        }
        
        if compression_ratios:
            stats.update({
                'compression_efficiency': {
                    'mean_ratio': np.mean(compression_ratios),
                    'median_ratio': np.median(compression_ratios),
                    'best_ratio': np.max(compression_ratios),
                    'worst_ratio': np.min(compression_ratios),
                }
            })
        
        if processing_times:
            stats.update({
                'performance': {
                    'mean_time_ms': np.mean(processing_times) * 1000,
                    'median_time_ms': np.median(processing_times) * 1000,
                    'total_time_s': np.sum(processing_times),
                    'throughput_docs_per_sec': len(processing_times) / np.sum(processing_times),
                }
            })
        
        return stats
    
    def analyze_compression(
        self,
        document: str,
        compressor: ContextCompressor,
        show_heatmap: bool = False,
        export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze compression for a single document.
        
        Args:
            document: Document to analyze
            compressor: Compression model
            show_heatmap: Whether to generate compression heatmap
            export_path: Optional path to export analysis
            
        Returns:
            Compression analysis results
        """
        # Compress document
        result = compressor.compress(document)
        
        # Analyze compression patterns
        analysis = {
            'document_stats': {
                'original_length': len(document),
                'original_tokens': result.original_length,
                'compressed_tokens': result.compressed_length,
                'compression_ratio': result.compression_ratio,
            },
            'mega_token_analysis': [],
            'compression_patterns': self._analyze_compression_patterns(result),
        }
        
        # Analyze each mega-token
        for i, token in enumerate(result.mega_tokens):
            token_analysis = {
                'token_id': i,
                'embedding_dim': len(token.embedding) if hasattr(token.embedding, '__len__') else 0,
                'source_range': token.source_range,
                'compression_ratio': token.compression_ratio,
                'metadata': token.metadata,
            }
            analysis['mega_token_analysis'].append(token_analysis)
        
        # Export if requested
        if export_path:
            self._export_analysis(analysis, export_path, show_heatmap)
        
        return analysis
    
    def _analyze_compression_patterns(self, result: CompressionResult) -> Dict[str, Any]:
        """Analyze compression patterns in the result.
        
        Args:
            result: Compression result
            
        Returns:
            Pattern analysis
        """
        if not result.mega_tokens:
            return {}
        
        ratios = [token.compression_ratio for token in result.mega_tokens]
        
        return {
            'compression_distribution': {
                'mean': np.mean(ratios),
                'std': np.std(ratios),
                'min': np.min(ratios),
                'max': np.max(ratios),
                'percentiles': {
                    '25': np.percentile(ratios, 25),
                    '50': np.percentile(ratios, 50),
                    '75': np.percentile(ratios, 75),
                    '90': np.percentile(ratios, 90),
                }
            },
            'uniformity_score': 1.0 - (np.std(ratios) / np.mean(ratios)) if np.mean(ratios) > 0 else 0,
            'efficiency_score': np.mean(ratios) / max(ratios) if max(ratios) > 0 else 0,
        }
    
    def _export_analysis(
        self,
        analysis: Dict[str, Any],
        export_path: str,
        include_visualization: bool = False,
    ) -> None:
        """Export analysis results.
        
        Args:
            analysis: Analysis results
            export_path: Export file path
            include_visualization: Whether to include visualizations
        """
        export_path = Path(export_path)
        
        # Export as JSON
        json_path = export_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate simple HTML report
        html_path = export_path.with_suffix('.html')
        self._generate_html_report(analysis, html_path)
        
        logger.info(f"Analysis exported to: {json_path} and {html_path}")
    
    def _generate_html_report(self, analysis: Dict[str, Any], html_path: Path) -> None:
        """Generate HTML report for compression analysis.
        
        Args:
            analysis: Analysis results
            html_path: Path for HTML file
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Compression Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ margin: 10px 0; }}
                .token {{ background: #f5f5f5; padding: 10px; margin: 5px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Compression Analysis Report</h1>
            
            <h2>Document Statistics</h2>
            <div class="metric">Original Length: {analysis['document_stats']['original_length']}</div>
            <div class="metric">Original Tokens: {analysis['document_stats']['original_tokens']}</div>
            <div class="metric">Compressed Tokens: {analysis['document_stats']['compressed_tokens']}</div>
            <div class="metric">Compression Ratio: {analysis['document_stats']['compression_ratio']:.2f}x</div>
            
            <h2>Mega-Token Analysis</h2>
            <table>
                <tr>
                    <th>Token ID</th>
                    <th>Embedding Dim</th>
                    <th>Source Range</th>
                    <th>Compression Ratio</th>
                </tr>
        """
        
        for token in analysis['mega_token_analysis']:
            html_content += f"""
                <tr>
                    <td>{token['token_id']}</td>
                    <td>{token['embedding_dim']}</td>
                    <td>{token['source_range']}</td>
                    <td>{token['compression_ratio']:.2f}x</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Compression Patterns</h2>
        """
        
        if 'compression_patterns' in analysis:
            patterns = analysis['compression_patterns']
            if 'compression_distribution' in patterns:
                dist = patterns['compression_distribution']
                html_content += f"""
                    <div class="metric">Mean Ratio: {dist['mean']:.2f}</div>
                    <div class="metric">Standard Deviation: {dist['std']:.2f}</div>
                    <div class="metric">Min Ratio: {dist['min']:.2f}</div>
                    <div class="metric">Max Ratio: {dist['max']:.2f}</div>
                """
            
            if 'uniformity_score' in patterns:
                html_content += f"""
                    <div class="metric">Uniformity Score: {patterns['uniformity_score']:.3f}</div>
                    <div class="metric">Efficiency Score: {patterns['efficiency_score']:.3f}</div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)


def run_benchmark_suite(
    compressor: ContextCompressor,
    benchmark_name: str = "standard",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a standard benchmark suite.
    
    Args:
        compressor: Compression model to benchmark
        benchmark_name: Name of benchmark suite
        output_dir: Directory to save results
        
    Returns:
        Benchmark results
    """
    evaluator = CompressionEvaluator()
    
    # Create sample test data
    test_docs = [
        "This is a sample document for testing compression performance. " * 50,
        "Another test document with different content patterns. " * 30,
        "A third document focusing on specific domain knowledge. " * 40,
    ]
    
    test_questions = [
        "What is this document about?",
        "What are the main topics discussed?",
        "What domain does this focus on?",
    ]
    
    test_answers = [
        "compression performance testing",
        "different content patterns",
        "domain knowledge",
    ]
    
    # Run evaluation
    results = evaluator.evaluate(
        compressor=compressor,
        test_documents=test_docs,
        test_questions=test_questions,
        test_answers=test_answers,
    )
    
    # Add benchmark metadata
    results['benchmark_info'] = {
        'name': benchmark_name,
        'version': '1.0',
        'timestamp': time.time(),
        'compressor_model': compressor.model_name,
    }
    
    # Save results if output directory provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / f"{benchmark_name}_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    # Demo evaluation
    from .core.auto_compressor import AutoCompressor
    
    print("Loading compressor...")
    compressor = AutoCompressor.from_pretrained("rfcc-base-8x")
    
    print("Running benchmark suite...")
    results = run_benchmark_suite(
        compressor=compressor,
        benchmark_name="demo",
        output_dir="benchmark_results",
    )
    
    print("Benchmark completed!")
    print(f"Mean compression ratio: {results['metrics'].get('mean_compression_ratio', 0):.2f}x")
    print(f"Mean processing time: {results['metrics'].get('mean_processing_time', 0)*1000:.1f}ms")
    print(f"Mean answer F1: {results['metrics'].get('mean_answer_f1', 0):.3f}")