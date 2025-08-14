"""Command-line interface for the retrieval-free compressor."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.auto_compressor import AutoCompressor
from .evaluation import CompressionEvaluator, run_benchmark_suite
from .training import CompressionDataset, CompressionTrainer, create_sample_dataset


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compress_command(args: argparse.Namespace) -> None:
    """Handle compress command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Load compressor
        logger.info(f"Loading compressor: {args.model}")
        compressor = AutoCompressor.from_pretrained(
            args.model,
            device=args.device,
        )
        
        # Read input
        if args.input == '-':
            text = sys.stdin.read()
        else:
            with open(args.input, 'r', encoding='utf-8') as f:
                text = f.read()
        
        logger.info(f"Input length: {len(text)} characters")
        
        # Compress
        start_time = time.time()
        result = compressor.compress(text)
        compression_time = time.time() - start_time
        
        # Display results
        print(f"Original tokens: {result.original_length}")
        print(f"Compressed tokens: {result.compressed_length}")
        print(f"Compression ratio: {result.compression_ratio:.1f}x")
        print(f"Processing time: {compression_time:.2f}s")
        print(f"Mega-tokens created: {len(result.mega_tokens)}")
        
        # Save output if specified
        if args.output:
            output_data = {
                'original_length': result.original_length,
                'compressed_length': result.compressed_length,
                'compression_ratio': result.compression_ratio,
                'processing_time': compression_time,
                'meta_tokens': [
                    {
                        'id': i,
                        'embedding_dim': len(token.embedding) if hasattr(token.embedding, '__len__') else 0,
                        'source_range': token.source_range,
                        'compression_ratio': token.compression_ratio,
                        'metadata': token.metadata,
                    }
                    for i, token in enumerate(result.mega_tokens)
                ],
                'metadata': result.metadata,
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"Results saved to: {args.output}")
        
        # Show attention weights if query provided
        if args.query:
            attention_weights = compressor.get_attention_weights(args.query, result.mega_tokens)
            print(f"\nAttention weights for query '{args.query}':")
            for i, weight in enumerate(attention_weights):
                print(f"  Token {i}: {float(weight):.3f}")
        
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        sys.exit(1)


def evaluate_command(args: argparse.Namespace) -> None:
    """Handle evaluate command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Load compressor
        logger.info(f"Loading compressor: {args.model}")
        compressor = AutoCompressor.from_pretrained(
            args.model,
            device=args.device,
        )
        
        # Load test data
        if args.test_data:
            if args.test_data.endswith('.json'):
                with open(args.test_data, 'r') as f:
                    test_data = json.load(f)
                documents = test_data.get('documents', [])
                questions = test_data.get('questions', [])
                answers = test_data.get('answers', [])
            else:
                # Assume plain text file with one document per line
                with open(args.test_data, 'r', encoding='utf-8') as f:
                    documents = [line.strip() for line in f if line.strip()]
                questions = []
                answers = []
        else:
            # Use sample data
            logger.info("No test data provided, using sample data")
            documents = [
                "This is a sample document for testing. " * 20,
                "Another test document with different content. " * 15,
                "A third document for evaluation purposes. " * 25,
            ]
            questions = []
            answers = []
        
        logger.info(f"Evaluating on {len(documents)} documents")
        
        # Set up evaluator
        evaluator = CompressionEvaluator(
            metrics=args.metrics.split(',') if args.metrics else None
        )
        
        # Run evaluation
        results = evaluator.evaluate(
            compressor=compressor,
            test_documents=documents,
            test_questions=questions if questions else None,
            test_answers=answers if answers else None,
        )
        
        # Display results
        print(f"\nEvaluation Results for {args.model}")
        print("=" * 50)
        print(f"Documents processed: {results['num_documents']}")
        print(f"Evaluation time: {results['evaluation_time']:.2f}s")
        
        metrics = results['metrics']
        if 'mean_compression_ratio' in metrics:
            print(f"Mean compression ratio: {metrics['mean_compression_ratio']:.2f}x")
        if 'mean_processing_time' in metrics:
            print(f"Mean processing time: {metrics['mean_processing_time']*1000:.1f}ms")
        if 'mean_answer_f1' in metrics:
            print(f"Mean answer F1: {metrics['mean_answer_f1']:.3f}")
        if 'mean_information_retention' in metrics:
            print(f"Mean information retention: {metrics['mean_information_retention']:.3f}")
        
        # Save detailed results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def train_command(args: argparse.Namespace) -> None:
    """Handle train command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Load or create training data
        if args.train_data:
            logger.info(f"Loading training data from: {args.train_data}")
            
            if args.train_data.endswith('.json'):
                with open(args.train_data, 'r') as f:
                    data = json.load(f)
                
                train_dataset = CompressionDataset(
                    documents=data.get('documents', []),
                    questions=data.get('questions', []),
                    answers=data.get('answers', []),
                    compression_ratio=args.compression_ratio,
                )
            else:
                # Assume text file
                train_dataset = CompressionDataset.from_files(
                    document_path=args.train_data,
                    compression_ratio=args.compression_ratio,
                )
        else:
            # Create sample dataset
            logger.info("No training data provided, creating sample dataset")
            train_dataset = create_sample_dataset(
                num_samples=args.num_samples,
                compression_ratio=args.compression_ratio,
            )
        
        # Create eval dataset if provided
        eval_dataset = None
        if args.eval_data:
            if args.eval_data.endswith('.json'):
                with open(args.eval_data, 'r') as f:
                    data = json.load(f)
                eval_dataset = CompressionDataset(
                    documents=data.get('documents', []),
                    questions=data.get('questions', []),
                    answers=data.get('answers', []),
                    compression_ratio=args.compression_ratio,
                )
            else:
                eval_dataset = CompressionDataset.from_files(
                    document_path=args.eval_data,
                    compression_ratio=args.compression_ratio,
                )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Initialize trainer
        trainer = CompressionTrainer(
            model_name=args.model_name,
            base_model=args.base_model,
            compression_objective=args.objective,
            device=args.device,
        )
        
        # Training configuration
        output_dir = args.output_dir or f"./models/{args.model_name}"
        
        # Train model
        logger.info("Starting training...")
        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
        )
        
        # Save trained model
        trainer.save_pretrained(output_dir)
        
        # Display results
        print(f"\nTraining completed!")
        print(f"Epochs: {results['epochs_completed']}")
        print(f"Total steps: {results['total_steps']}")
        print(f"Final training loss: {results['final_train_loss']:.4f}")
        if 'best_eval_loss' in results:
            print(f"Best evaluation loss: {results['best_eval_loss']:.4f}")
        print(f"Model saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def benchmark_command(args: argparse.Namespace) -> None:
    """Handle benchmark command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Load compressor
        logger.info(f"Loading compressor: {args.model}")
        compressor = AutoCompressor.from_pretrained(
            args.model,
            device=args.device,
        )
        
        # Run benchmark
        results = run_benchmark_suite(
            compressor=compressor,
            benchmark_name=args.benchmark,
            output_dir=args.output_dir,
        )
        
        # Display summary
        print(f"\nBenchmark Results: {args.benchmark}")
        print("=" * 50)
        
        summary = results['summary_statistics']
        if 'compression_efficiency' in summary:
            eff = summary['compression_efficiency']
            print(f"Mean compression ratio: {eff['mean_ratio']:.2f}x")
            print(f"Best compression ratio: {eff['best_ratio']:.2f}x")
        
        if 'performance' in summary:
            perf = summary['performance']
            print(f"Mean processing time: {perf['mean_time_ms']:.1f}ms")
            print(f"Throughput: {perf['throughput_docs_per_sec']:.1f} docs/sec")
        
        metrics = results['metrics']
        if 'mean_answer_f1' in metrics:
            print(f"Mean answer F1: {metrics['mean_answer_f1']:.3f}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


def list_models_command(args: argparse.Namespace) -> None:
    """Handle list-models command.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        models = AutoCompressor.list_available_models()
        
        print("Available Models:")
        print("=" * 50)
        
        for name, description in models.items():
            print(f"{name:<20} {description}")
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Retrieval-Free Context Compressor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, auto-detected if not specified)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compress command
    compress_parser = subparsers.add_parser("compress", help="Compress text or documents")
    compress_parser.add_argument("input", help="Input file ('-' for stdin)")
    compress_parser.add_argument("--model", "-m", default="rfcc-base-8x", help="Compression model to use")
    compress_parser.add_argument("--output", "-o", help="Output file for results")
    compress_parser.add_argument("--query", "-q", help="Query for attention weight analysis")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate compression model")
    eval_parser.add_argument("model", help="Model to evaluate")
    eval_parser.add_argument("--test-data", help="Test data file (JSON or text)")
    eval_parser.add_argument("--metrics", help="Comma-separated list of metrics")
    eval_parser.add_argument("--output", "-o", help="Output file for results")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train compression model")
    train_parser.add_argument("--model-name", default="custom-compressor", help="Name for trained model")
    train_parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2", help="Base model")
    train_parser.add_argument("--train-data", help="Training data file")
    train_parser.add_argument("--eval-data", help="Evaluation data file")
    train_parser.add_argument("--compression-ratio", type=float, default=8.0, help="Target compression ratio")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--num-samples", type=int, default=100, help="Number of sample documents to generate")
    train_parser.add_argument("--objective", default="info_bottleneck", help="Training objective")
    train_parser.add_argument("--output-dir", help="Output directory for trained model")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    benchmark_parser.add_argument("model", help="Model to benchmark")
    benchmark_parser.add_argument("--benchmark", default="standard", help="Benchmark suite to run")
    benchmark_parser.add_argument("--output-dir", help="Output directory for results")
    
    # List models command
    subparsers.add_parser("list-models", help="List available models")
    
    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Dispatch command
    if args.command == "compress":
        compress_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "benchmark":
        benchmark_command(args)
    elif args.command == "list-models":
        list_models_command(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
