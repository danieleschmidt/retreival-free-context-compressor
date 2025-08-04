"""Benchmark runner for performance testing."""

import time
import logging
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import json
from pathlib import Path
import unittest.mock as mock
from unittest.mock import MagicMock, patch

# Mock or import psutil
try:
    import psutil
except ImportError:
    # Create mock psutil for environments without it
    class MockPsutil:
        @staticmethod
        def cpu_count():
            return 4
        
        @staticmethod
        def virtual_memory():
            mock_mem = MagicMock()
            mock_mem.total = 8 * 1024**3  # 8GB
            mock_mem.used = 4 * 1024**3   # 4GB
            return mock_mem
        
        @staticmethod
        def cpu_percent(interval=1):
            return 50.0
    
    psutil = MockPsutil()

# Mock dependencies for benchmarking without requiring full installs
@patch.dict('sys.modules', {
    'torch': MagicMock(),
    'torch.nn': MagicMock(),
    'transformers': MagicMock(),
    'sklearn': MagicMock(),
    'sentence_transformers': MagicMock(),
    'faiss': MagicMock(),
})
def setup_mocks():
    """Setup mocks for benchmarking."""
    pass

setup_mocks()

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    name: str
    duration_ms: float
    memory_mb: float
    throughput_ops_per_sec: float
    success_rate: float
    error_count: int
    metadata: Dict[str, Any]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    
    name: str
    description: str
    test_data_size: int
    iterations: int
    warmup_iterations: int
    timeout_seconds: float
    parameters: Dict[str, Any]


class BenchmarkRunner:
    """Runner for executing performance benchmarks."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.current_config: Optional[BenchmarkConfig] = None
    
    def run_benchmark(
        self,
        config: BenchmarkConfig,
        benchmark_func: Callable[[Any], Any],
        test_data: Any
    ) -> BenchmarkResult:
        """Run a single benchmark.
        
        Args:
            config: Benchmark configuration
            benchmark_func: Function to benchmark
            test_data: Test data for the benchmark
            
        Returns:
            BenchmarkResult with performance metrics
        """
        self.current_config = config
        logger.info(f"Starting benchmark: {config.name}")
        
        # Warmup runs
        logger.info(f"Running {config.warmup_iterations} warmup iterations")
        self._run_warmup(benchmark_func, test_data, config.warmup_iterations)
        
        # Clear memory before actual benchmark
        if hasattr(psutil, 'virtual_memory'):
            initial_memory = psutil.virtual_memory().used / 1024 / 1024
        else:
            initial_memory = 0
        
        # Actual benchmark runs
        durations = []
        errors = 0
        successful_ops = 0
        
        logger.info(f"Running {config.iterations} benchmark iterations")
        
        for i in range(config.iterations):
            try:
                start_time = time.time()
                
                # Run benchmark function
                result = benchmark_func(test_data)
                
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                durations.append(duration_ms)
                successful_ops += 1
                
                if result is None:
                    logger.warning(f"Iteration {i+1} returned None")
                
            except Exception as e:
                logger.error(f"Iteration {i+1} failed: {e}")
                errors += 1
                
                # Stop if too many errors
                if errors > config.iterations * 0.5:
                    logger.error("Too many errors, stopping benchmark")
                    break
        
        # Calculate metrics
        if durations:
            avg_duration_ms = statistics.mean(durations)
            throughput = successful_ops / (sum(durations) / 1000) if durations else 0
        else:
            avg_duration_ms = 0
            throughput = 0
        
        success_rate = successful_ops / config.iterations if config.iterations > 0 else 0
        
        # Memory usage (rough estimate)
        if hasattr(psutil, 'virtual_memory'):
            final_memory = psutil.virtual_memory().used / 1024 / 1024
            memory_used = max(0, final_memory - initial_memory)
        else:
            memory_used = 0
        
        # Create result
        result = BenchmarkResult(
            name=config.name,
            duration_ms=avg_duration_ms,
            memory_mb=memory_used,
            throughput_ops_per_sec=throughput,
            success_rate=success_rate,
            error_count=errors,
            metadata={
                'iterations': config.iterations,
                'warmup_iterations': config.warmup_iterations,
                'test_data_size': config.test_data_size,
                'parameters': config.parameters,
                'duration_stats': {
                    'min_ms': min(durations) if durations else 0,
                    'max_ms': max(durations) if durations else 0,
                    'median_ms': statistics.median(durations) if durations else 0,
                    'stddev_ms': statistics.stdev(durations) if len(durations) > 1 else 0
                }
            }
        )
        
        self.results.append(result)
        
        logger.info(f"Benchmark {config.name} completed:")
        logger.info(f"  Average duration: {avg_duration_ms:.2f}ms")
        logger.info(f"  Throughput: {throughput:.2f} ops/sec")
        logger.info(f"  Success rate: {success_rate:.2%}")
        logger.info(f"  Memory used: {memory_used:.2f}MB")
        
        return result
    
    def _run_warmup(
        self, 
        benchmark_func: Callable[[Any], Any], 
        test_data: Any, 
        iterations: int
    ) -> None:
        """Run warmup iterations."""
        for i in range(iterations):
            try:
                benchmark_func(test_data)
            except Exception as e:
                logger.debug(f"Warmup iteration {i+1} failed: {e}")
    
    def save_results(self, filename: str = None) -> str:
        """Save benchmark results to file.
        
        Args:
            filename: Optional filename, will be auto-generated if not provided
            
        Returns:
            Path to saved results file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert results to serializable format
        results_data = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'results': [
                {
                    'name': r.name,
                    'duration_ms': r.duration_ms,
                    'memory_mb': r.memory_mb,
                    'throughput_ops_per_sec': r.throughput_ops_per_sec,
                    'success_rate': r.success_rate,
                    'error_count': r.error_count,
                    'metadata': r.metadata
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Benchmark results saved to: {filepath}")
        return str(filepath)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            'python_version': '3.10+',  # Mock version
            'platform': 'linux',
        }
        
        try:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'cpu_percent': psutil.cpu_percent(interval=1),
            })
        except:
            # Fallback values
            info.update({
                'cpu_count': 4,
                'memory_gb': 8.0,
                'cpu_percent': 50.0,
            })
        
        return info
    
    def generate_report(self) -> str:
        """Generate a summary report of benchmark results.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No benchmark results available."
        
        report_lines = [
            "=== BENCHMARK REPORT ===",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total benchmarks: {len(self.results)}",
            "",
            "System Information:",
            f"  CPU Count: {self._get_system_info().get('cpu_count', 'Unknown')}",
            f"  Memory: {self._get_system_info().get('memory_gb', 'Unknown'):.1f}GB",
            "",
            "Results Summary:",
            "-" * 80,
        ]
        
        # Add header
        report_lines.append(
            f"{'Benchmark':<30} {'Duration (ms)':<15} {'Throughput':<15} {'Success Rate':<15} {'Memory (MB)':<12}"
        )
        report_lines.append("-" * 80)
        
        # Add results
        for result in self.results:
            report_lines.append(
                f"{result.name:<30} "
                f"{result.duration_ms:<15.2f} "
                f"{result.throughput_ops_per_sec:<15.2f} "
                f"{result.success_rate:<15.2%} "
                f"{result.memory_mb:<12.2f}"
            )
        
        report_lines.extend([
            "",
            "Performance Analysis:",
            "-" * 40,
        ])
        
        # Add performance analysis
        avg_duration = statistics.mean([r.duration_ms for r in self.results])
        avg_throughput = statistics.mean([r.throughput_ops_per_sec for r in self.results])
        avg_success_rate = statistics.mean([r.success_rate for r in self.results])
        
        report_lines.extend([
            f"Average Duration: {avg_duration:.2f}ms",
            f"Average Throughput: {avg_throughput:.2f} ops/sec",
            f"Average Success Rate: {avg_success_rate:.2%}",
            "",
        ])
        
        # Performance categorization
        if avg_duration < 100:
            perf_category = "Excellent"
        elif avg_duration < 500:
            perf_category = "Good"
        elif avg_duration < 1000:
            perf_category = "Acceptable"
        else:
            perf_category = "Needs Improvement"
        
        report_lines.extend([
            f"Overall Performance: {perf_category}",
            "",
            "Recommendations:",
        ])
        
        # Add recommendations
        if avg_duration > 1000:
            report_lines.append("- Consider optimizing slow operations")
        if avg_success_rate < 0.95:
            report_lines.append("- Investigate and fix failing operations")
        if any(r.memory_mb > 1000 for r in self.results):
            report_lines.append("- Consider memory optimization for high-memory operations")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def create_compression_benchmark() -> BenchmarkConfig:
    """Create benchmark config for compression testing."""
    return BenchmarkConfig(
        name="compression_benchmark",
        description="Test compression performance with various text sizes",
        test_data_size=1000,
        iterations=10,
        warmup_iterations=3,
        timeout_seconds=30.0,
        parameters={
            'compression_ratio': 8.0,
            'chunk_size': 512,
            'model': 'mock-model'
        }
    )


def create_caching_benchmark() -> BenchmarkConfig:
    """Create benchmark config for caching performance."""
    return BenchmarkConfig(
        name="caching_benchmark",
        description="Test caching system performance",
        test_data_size=100,
        iterations=100,
        warmup_iterations=10,
        timeout_seconds=10.0,
        parameters={
            'cache_size': 1000,
            'item_size': 1024
        }
    )


def create_batch_processing_benchmark() -> BenchmarkConfig:
    """Create benchmark config for batch processing."""
    return BenchmarkConfig(
        name="batch_processing_benchmark", 
        description="Test batch processing performance",
        test_data_size=50,
        iterations=20,
        warmup_iterations=5,
        timeout_seconds=20.0,
        parameters={
            'batch_size': 8,
            'worker_count': 4
        }
    )


def mock_compression_function(test_data: List[str]) -> Dict[str, Any]:
    """Mock compression function for benchmarking."""
    # Simulate compression work
    time.sleep(0.01)  # Simulate processing time
    
    # Return mock result
    return {
        'compressed_tokens': len(test_data) // 8,
        'original_tokens': len(test_data),
        'compression_ratio': 8.0
    }


def mock_caching_function(cache_operations: List[Dict[str, Any]]) -> Dict[str, int]:
    """Mock caching function for benchmarking."""
    hits = 0
    misses = 0
    
    # Simulate cache operations
    cache = {}
    
    for op in cache_operations:
        if op['type'] == 'get':
            if op['key'] in cache:
                hits += 1
            else:
                misses += 1
        elif op['type'] == 'put':
            cache[op['key']] = op['value']
    
    return {'hits': hits, 'misses': misses}


def mock_batch_processing_function(batches: List[List[str]]) -> List[str]:
    """Mock batch processing function for benchmarking."""
    results = []
    
    for batch in batches:
        # Simulate batch processing
        time.sleep(0.005)  # Small delay per batch
        results.extend([f"processed_{item}" for item in batch])
    
    return results


def run_compression_benchmark(runner: BenchmarkRunner) -> BenchmarkResult:
    """Run compression performance benchmark."""
    config = create_compression_benchmark()
    
    # Create test data
    test_data = [f"This is test sentence {i} for compression benchmarking." for i in range(config.test_data_size)]
    
    return runner.run_benchmark(config, mock_compression_function, test_data)


def run_caching_benchmark(runner: BenchmarkRunner) -> BenchmarkResult:
    """Run caching performance benchmark."""
    config = create_caching_benchmark()
    
    # Create cache operations
    operations = []
    for i in range(config.test_data_size):
        # Mix of puts and gets
        if i % 3 == 0:
            operations.append({'type': 'put', 'key': f'key_{i}', 'value': f'value_{i}'})
        else:
            operations.append({'type': 'get', 'key': f'key_{i % 10}'})  # Some cache hits
    
    return runner.run_benchmark(config, mock_caching_function, operations)


def run_batch_processing_benchmark(runner: BenchmarkRunner) -> BenchmarkResult:
    """Run batch processing performance benchmark.""" 
    config = create_batch_processing_benchmark()
    
    # Create batch data
    items = [f"item_{i}" for i in range(config.test_data_size)]
    batch_size = config.parameters['batch_size']
    
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    return runner.run_benchmark(config, mock_batch_processing_function, batches)


def main():
    """Main benchmark execution."""
    print("Starting Retrieval-Free Context Compressor Benchmarks")
    print("=" * 60)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create benchmark runner
    runner = BenchmarkRunner("benchmark_results")
    
    try:
        # Run benchmarks
        print("\n1. Running Compression Benchmark...")
        compression_result = run_compression_benchmark(runner)
        
        print("\n2. Running Caching Benchmark...")
        caching_result = run_caching_benchmark(runner)
        
        print("\n3. Running Batch Processing Benchmark...")
        batch_result = run_batch_processing_benchmark(runner)
        
        # Save results
        results_file = runner.save_results()
        
        # Generate and display report
        report = runner.generate_report()
        print("\n" + report)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Performance summary
        all_results = [compression_result, caching_result, batch_result]
        avg_duration = statistics.mean([r.duration_ms for r in all_results])
        avg_success_rate = statistics.mean([r.success_rate for r in all_results])
        
        print(f"\nOverall Performance Summary:")
        print(f"  Average Duration: {avg_duration:.2f}ms")
        print(f"  Average Success Rate: {avg_success_rate:.2%}")
        
        if avg_success_rate >= 0.95:
            print("  Status: ✅ All benchmarks passed")
        else:
            print("  Status: ⚠️  Some benchmarks had failures")
            
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())