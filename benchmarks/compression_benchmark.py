"""Comprehensive benchmarking suite for compression performance."""

import time
import json
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from unittest.mock import MagicMock


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    test_name: str
    input_size: int
    output_size: int
    compression_ratio: float
    processing_time: float
    memory_usage_mb: float
    throughput_tokens_per_sec: float
    cpu_percent: float
    timestamp: float
    metadata: Dict[str, Any]


class SystemProfiler:
    """Profile system resources during benchmarks."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
    def start_profiling(self):
        """Start resource profiling."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024
        self.start_cpu = self.process.cpu_percent()
        
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        end_cpu = self.process.cpu_percent()
        
        return {
            "duration": end_time - self.start_time,
            "memory_usage_mb": end_memory,
            "memory_delta_mb": end_memory - self.start_memory,
            "cpu_percent": end_cpu,
        }


class MockCompressor:
    """Mock compressor for benchmarking without actual models."""
    
    def __init__(self, compression_ratio: float = 8.0, processing_delay: float = 0.001):
        self.compression_ratio = compression_ratio
        self.processing_delay = processing_delay
        self.model_name = f"mock-{compression_ratio}x"
        
    def compress(self, text: str) -> List[str]:
        """Mock compression with simulated processing time."""
        # Simulate processing time proportional to input size
        processing_time = len(text) * self.processing_delay / 1000
        time.sleep(processing_time)
        
        # Generate mock compressed tokens
        input_tokens = len(text.split())
        output_tokens = max(1, int(input_tokens / self.compression_ratio))
        
        return [f"token_{i}" for i in range(output_tokens)]
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(text.split())
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio."""
        return self.compression_ratio


class CompressionBenchmark:
    """Main benchmarking suite."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiler = SystemProfiler()
        self.results: List[BenchmarkResult] = []
        
    def generate_test_documents(self) -> Dict[str, str]:
        """Generate test documents of various sizes."""
        base_text = (
            "Machine learning and artificial intelligence are transforming "
            "how we process and understand large amounts of textual data. "
            "Context compression techniques enable efficient handling of "
            "long documents without losing essential information. "
        )
        
        return {
            "tiny": base_text,
            "small": base_text * 10,
            "medium": base_text * 100,
            "large": base_text * 1000,
            "xlarge": base_text * 5000,
            "xxlarge": base_text * 10000,
        }
    
    def run_compression_benchmark(
        self, 
        compressor: MockCompressor,
        document: str,
        test_name: str,
        iterations: int = 1
    ) -> BenchmarkResult:
        """Run a single compression benchmark."""
        print(f"Running {test_name} with {compressor.model_name}...")
        
        # Warm-up run
        compressor.compress(document[:100])
        
        total_time = 0
        memory_usage = []
        cpu_usage = []
        
        for i in range(iterations):
            self.profiler.start_profiling()
            
            compressed = compressor.compress(document)
            
            metrics = self.profiler.stop_profiling()
            total_time += metrics["duration"]
            memory_usage.append(metrics["memory_usage_mb"])
            cpu_usage.append(metrics["cpu_percent"])
        
        # Calculate averages
        avg_time = total_time / iterations
        avg_memory = np.mean(memory_usage)
        avg_cpu = np.mean(cpu_usage)
        
        # Calculate metrics
        input_size = compressor.count_tokens(document)
        output_size = len(compressed)
        compression_ratio = input_size / output_size if output_size > 0 else 1.0
        throughput = input_size / avg_time if avg_time > 0 else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            input_size=input_size,
            output_size=output_size,
            compression_ratio=compression_ratio,
            processing_time=avg_time,
            memory_usage_mb=avg_memory,
            throughput_tokens_per_sec=throughput,
            cpu_percent=avg_cpu,
            timestamp=time.time(),
            metadata={
                "model": compressor.model_name,
                "iterations": iterations,
                "document_length": len(document),
            }
        )
        
        self.results.append(result)
        return result
    
    def run_scaling_benchmark(self) -> None:
        """Benchmark compression scaling with different input sizes."""
        print("Running scaling benchmark...")
        
        documents = self.generate_test_documents()
        compressor = MockCompressor(compression_ratio=8.0)
        
        for size_name, document in documents.items():
            self.run_compression_benchmark(
                compressor, document, f"scaling_{size_name}", iterations=3
            )
    
    def run_compression_ratio_benchmark(self) -> None:
        """Benchmark different compression ratios."""
        print("Running compression ratio benchmark...")
        
        document = self.generate_test_documents()["medium"]
        compression_ratios = [2.0, 4.0, 8.0, 16.0, 32.0]
        
        for ratio in compression_ratios:
            compressor = MockCompressor(compression_ratio=ratio)
            self.run_compression_benchmark(
                compressor, document, f"ratio_{ratio}x", iterations=5
            )
    
    def run_throughput_benchmark(self) -> None:
        """Benchmark throughput with various processing delays."""
        print("Running throughput benchmark...")
        
        document = self.generate_test_documents()["small"]
        processing_delays = [0.0001, 0.001, 0.01, 0.1]  # Simulated delays
        
        for delay in processing_delays:
            compressor = MockCompressor(processing_delay=delay)
            self.run_compression_benchmark(
                compressor, document, f"throughput_delay_{delay}", iterations=10
            )
    
    def run_memory_benchmark(self) -> None:
        """Benchmark memory usage patterns."""
        print("Running memory benchmark...")
        
        documents = self.generate_test_documents()
        compressor = MockCompressor(compression_ratio=8.0)
        
        # Test memory scaling
        for size_name, document in documents.items():
            if size_name in ["large", "xlarge", "xxlarge"]:
                self.run_compression_benchmark(
                    compressor, document, f"memory_{size_name}", iterations=1
                )
    
    def save_results(self) -> None:
        """Save benchmark results to files."""
        timestamp = int(time.time())
        
        # Save raw results as JSON
        json_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        print(f"Results saved to {json_file}")
        
        # Save summary CSV
        csv_file = self.output_dir / f"benchmark_summary_{timestamp}.csv"
        import pandas as pd
        
        df = pd.DataFrame([asdict(result) for result in self.results])
        df.to_csv(csv_file, index=False)
        
        print(f"Summary saved to {csv_file}")
    
    def generate_plots(self) -> None:
        """Generate visualization plots."""
        print("Generating plots...")
        
        if not self.results:
            print("No results to plot")
            return
        
        timestamp = int(time.time())
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Plot 1: Scaling performance
        scaling_results = [r for r in self.results if r.test_name.startswith("scaling_")]
        if scaling_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            sizes = [r.input_size for r in scaling_results]
            times = [r.processing_time for r in scaling_results]
            throughputs = [r.throughput_tokens_per_sec for r in scaling_results]
            
            ax1.loglog(sizes, times, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Input Size (tokens)')
            ax1.set_ylabel('Processing Time (seconds)')
            ax1.set_title('Compression Time vs Input Size')
            ax1.grid(True, alpha=0.3)
            
            ax2.semilogx(sizes, throughputs, 's-', linewidth=2, markersize=8)
            ax2.set_xlabel('Input Size (tokens)')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput vs Input Size')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"scaling_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 2: Compression ratio comparison
        ratio_results = [r for r in self.results if r.test_name.startswith("ratio_")]
        if ratio_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ratios = [r.compression_ratio for r in ratio_results]
            times = [r.processing_time for r in ratio_results]
            throughputs = [r.throughput_tokens_per_sec for r in ratio_results]
            
            ax1.plot(ratios, times, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Compression Ratio')
            ax1.set_ylabel('Processing Time (seconds)')
            ax1.set_title('Processing Time vs Compression Ratio')
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(ratios, throughputs, 's-', linewidth=2, markersize=8)
            ax2.set_xlabel('Compression Ratio')
            ax2.set_ylabel('Throughput (tokens/sec)')
            ax2.set_title('Throughput vs Compression Ratio')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"ratio_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot 3: Memory usage
        memory_results = [r for r in self.results if r.test_name.startswith("memory_")]
        if memory_results:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            sizes = [r.input_size for r in memory_results]
            memory = [r.memory_usage_mb for r in memory_results]
            
            ax.loglog(sizes, memory, 'o-', linewidth=2, markersize=8)
            ax.set_xlabel('Input Size (tokens)')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Usage vs Input Size')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"memory_plot_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {self.output_dir}")
    
    def print_summary(self) -> None:
        """Print benchmark summary."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Group results by benchmark type
        by_type = {}
        for result in self.results:
            benchmark_type = result.test_name.split('_')[0]
            if benchmark_type not in by_type:
                by_type[benchmark_type] = []
            by_type[benchmark_type].append(result)
        
        for benchmark_type, results in by_type.items():
            print(f"\n{benchmark_type.upper()} BENCHMARK:")
            print("-" * 40)
            
            for result in results:
                print(f"  {result.test_name}:")
                print(f"    Compression Ratio: {result.compression_ratio:.2f}x")
                print(f"    Processing Time: {result.processing_time:.4f}s")
                print(f"    Throughput: {result.throughput_tokens_per_sec:.0f} tokens/sec")
                print(f"    Memory Usage: {result.memory_usage_mb:.1f} MB")
                print()
        
        # Overall statistics
        all_ratios = [r.compression_ratio for r in self.results]
        all_throughputs = [r.throughput_tokens_per_sec for r in self.results]
        all_memory = [r.memory_usage_mb for r in self.results]
        
        print("OVERALL STATISTICS:")
        print("-" * 40)
        print(f"Average Compression Ratio: {np.mean(all_ratios):.2f}x")
        print(f"Average Throughput: {np.mean(all_throughputs):.0f} tokens/sec")
        print(f"Average Memory Usage: {np.mean(all_memory):.1f} MB")
        print(f"Total Benchmark Time: {sum(r.processing_time for r in self.results):.2f}s")


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Run compression benchmarks")
    parser.add_argument("--output", "-o", default="benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--plot", action="store_true", 
                       help="Generate visualization plots")
    parser.add_argument("--benchmarks", nargs="+", 
                       choices=["scaling", "ratio", "throughput", "memory", "all"],
                       default=["all"], help="Benchmarks to run")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = CompressionBenchmark(output_dir=args.output)
    
    print("Starting compression benchmarks...")
    print(f"Output directory: {benchmark.output_dir}")
    
    # Run selected benchmarks
    if "all" in args.benchmarks or "scaling" in args.benchmarks:
        benchmark.run_scaling_benchmark()
    
    if "all" in args.benchmarks or "ratio" in args.benchmarks:
        benchmark.run_compression_ratio_benchmark()
    
    if "all" in args.benchmarks or "throughput" in args.benchmarks:
        benchmark.run_throughput_benchmark()
    
    if "all" in args.benchmarks or "memory" in args.benchmarks:
        benchmark.run_memory_benchmark()
    
    # Save results and generate plots
    benchmark.save_results()
    
    if args.plot:
        try:
            benchmark.generate_plots()
        except ImportError:
            print("Warning: matplotlib/seaborn not available, skipping plots")
    
    # Print summary
    benchmark.print_summary()
    
    print(f"\nBenchmark completed! Results saved to {benchmark.output_dir}")


if __name__ == "__main__":
    main()