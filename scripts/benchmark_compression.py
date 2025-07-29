#!/usr/bin/env python3
"""Compression benchmarking script."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
import psutil
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID


console = Console()


class CompressionBenchmark:
    """Comprehensive compression benchmarking suite."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.results: List[Dict[str, Any]] = []
        
        console.print(f"[bold blue]Initializing benchmark for {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine optimal device for benchmarking."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def benchmark_latency(self, documents: List[str], num_runs: int = 5) -> Dict[str, float]:
        """Benchmark compression latency."""
        console.print("[yellow]Running latency benchmarks...")
        
        latencies = []
        
        with Progress() as progress:
            task = progress.add_task("Benchmarking latency...", total=len(documents) * num_runs)
            
            for doc in documents:
                doc_latencies = []
                for _ in range(num_runs):
                    # Mock compression timing
                    start_time = time.perf_counter()
                    
                    # Simulate compression work
                    _ = self._mock_compress(doc)
                    
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    doc_latencies.append(latency_ms)
                    
                    progress.advance(task)
                
                latencies.extend(doc_latencies)
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
        }
    
    def benchmark_throughput(self, documents: List[str], duration_seconds: int = 60) -> Dict[str, float]:
        """Benchmark compression throughput."""
        console.print("[yellow]Running throughput benchmarks...")
        
        start_time = time.time()
        processed_docs = 0
        total_tokens = 0
        
        with Progress() as progress:
            task = progress.add_task("Measuring throughput...", total=duration_seconds)
            
            while time.time() - start_time < duration_seconds:
                for doc in documents:
                    if time.time() - start_time >= duration_seconds:
                        break
                    
                    result = self._mock_compress(doc)
                    processed_docs += 1
                    total_tokens += len(doc.split())  # Rough token estimate
                
                elapsed = time.time() - start_time
                progress.update(task, completed=min(elapsed, duration_seconds))
        
        total_time = time.time() - start_time
        
        return {
            "docs_per_second": processed_docs / total_time,
            "tokens_per_second": total_tokens / total_time,
            "total_processed": processed_docs,
            "total_time_seconds": total_time,
        }
    
    def benchmark_memory(self, documents: List[str]) -> Dict[str, float]:
        """Benchmark memory usage during compression."""
        console.print("[yellow]Running memory benchmarks...")
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        peak_memory = initial_memory
        peak_gpu_memory = initial_gpu_memory if torch.cuda.is_available() else 0
        
        with Progress() as progress:
            task = progress.add_task("Measuring memory usage...", total=len(documents))
            
            for doc in documents:
                # Compress document
                _ = self._mock_compress(doc)
                
                # Track memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                if torch.cuda.is_available():
                    current_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    peak_gpu_memory = max(peak_gpu_memory, current_gpu_memory)
                
                progress.advance(task)
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "memory_increase_mb": peak_memory - initial_memory,
            "peak_gpu_memory_mb": peak_gpu_memory,
            "gpu_memory_increase_mb": peak_gpu_memory - (initial_gpu_memory if torch.cuda.is_available() else 0),
        }
    
    def benchmark_compression_quality(self, documents: List[str]) -> Dict[str, float]:
        """Benchmark compression quality metrics."""
        console.print("[yellow]Running compression quality benchmarks...")
        
        compression_ratios = []
        processing_times = []
        
        with Progress() as progress:
            task = progress.add_task("Measuring compression quality...", total=len(documents))
            
            for doc in documents:
                start_time = time.perf_counter()
                result = self._mock_compress(doc)
                end_time = time.perf_counter()
                
                compression_ratios.append(result["compression_ratio"])
                processing_times.append(end_time - start_time)
                
                progress.advance(task)
        
        return {
            "mean_compression_ratio": np.mean(compression_ratios),
            "median_compression_ratio": np.median(compression_ratios),
            "min_compression_ratio": np.min(compression_ratios),
            "max_compression_ratio": np.max(compression_ratios),
            "compression_ratio_std": np.std(compression_ratios),
            "mean_processing_time": np.mean(processing_times),
        }
    
    def _mock_compress(self, document: str) -> Dict[str, Any]:
        """Mock compression for benchmarking (replace with real implementation)."""
        # Simulate compression work
        time.sleep(0.01 + len(document) / 100000)  # Scale with document length
        
        # Simulate realistic compression results
        original_tokens = len(document.split())
        compression_ratio = np.random.uniform(6.0, 10.0)  # Realistic range
        compressed_size = int(original_tokens / compression_ratio)
        
        return {
            "compressed_tokens": torch.randn(compressed_size, 768),
            "compression_ratio": compression_ratio,
            "original_tokens": original_tokens,
            "compressed_tokens_count": compressed_size,
        }
    
    def run_full_benchmark(self, documents: List[str]) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        console.print("[bold green]Running comprehensive benchmark suite...")
        
        results = {
            "model_name": self.model_name,
            "device": self.device,
            "timestamp": time.time(),
            "num_documents": len(documents),
        }
        
        # Run individual benchmarks
        results["latency"] = self.benchmark_latency(documents)
        results["throughput"] = self.benchmark_throughput(documents, duration_seconds=30)
        results["memory"] = self.benchmark_memory(documents)
        results["quality"] = self.benchmark_compression_quality(documents)
        
        self.results.append(results)
        return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display benchmark results in a formatted table."""
        console.print("\n[bold green]Benchmark Results")
        
        # Latency table
        latency_table = Table(title="Latency Metrics (ms)")
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Value", style="magenta")
        
        for key, value in results["latency"].items():
            latency_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
        
        console.print(latency_table)
        
        # Throughput table
        throughput_table = Table(title="Throughput Metrics")
        throughput_table.add_column("Metric", style="cyan")
        throughput_table.add_column("Value", style="magenta")
        
        for key, value in results["throughput"].items():
            throughput_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
        
        console.print(throughput_table)
        
        # Memory table
        memory_table = Table(title="Memory Usage (MB)")
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="magenta")
        
        for key, value in results["memory"].items():
            memory_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
        
        console.print(memory_table)
        
        # Quality table
        quality_table = Table(title="Compression Quality")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="magenta")
        
        for key, value in results["quality"].items():
            quality_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
        
        console.print(quality_table)
    
    def save_results(self, output_path: Path):
        """Save benchmark results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        console.print(f"[green]Results saved to {output_path}")


def load_test_documents(data_path: Path) -> List[str]:
    """Load test documents for benchmarking."""
    if data_path.exists():
        # Load real documents
        documents = []
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        return documents
    else:
        # Generate synthetic documents
        console.print("[yellow]Using synthetic test documents")
        base_content = """
        This is a sample document for compression benchmarking.
        It contains multiple paragraphs with technical content about
        machine learning, natural language processing, and AI systems.
        
        The document includes various types of information including
        definitions, examples, code snippets, and detailed explanations
        of complex algorithms and methodologies used in modern AI.
        """
        
        documents = []
        for i in range(50):
            # Create documents of varying lengths
            multiplier = np.random.randint(1, 20)
            doc = base_content * multiplier
            documents.append(doc)
        
        return documents


def main():
    """Main benchmarking script."""
    parser = argparse.ArgumentParser(description="Compression benchmarking suite")
    parser.add_argument("--model", default="rfcc-base-8x", help="Model name to benchmark")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--data-path", type=Path, help="Path to test documents")
    parser.add_argument("--output", type=Path, default="benchmark_results.json", help="Output file")
    parser.add_argument("--duration", type=int, default=60, help="Throughput test duration (seconds)")
    
    args = parser.parse_args()
    
    # Load test documents
    data_path = args.data_path or Path("test_data")
    documents = load_test_documents(data_path)
    
    console.print(f"[blue]Loaded {len(documents)} test documents")
    
    # Initialize benchmark
    benchmark = CompressionBenchmark(args.model, args.device)
    
    # Run benchmarks
    results = benchmark.run_full_benchmark(documents)
    
    # Display results
    benchmark.display_results(results)
    
    # Save results
    benchmark.save_results(args.output)
    
    console.print("[bold green]Benchmarking complete!")


if __name__ == "__main__":
    main()