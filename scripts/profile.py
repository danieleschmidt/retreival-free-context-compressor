#!/usr/bin/env python3
"""Profiling script for performance analysis of compression operations."""

import cProfile
import pstats
import io
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from retrieval_free.observability import monitor_performance, metrics_collector
except ImportError:
    print("Warning: Could not import observability module, using mocks")
    
    def monitor_performance(func):
        return func
    
    class MockMetricsCollector:
        def get_all_metrics(self):
            return {"mock": True}
    
    metrics_collector = MockMetricsCollector()


class MockCompressor:
    """Mock compressor for profiling without actual models."""
    
    def __init__(self, compression_ratio: float = 8.0):
        self.compression_ratio = compression_ratio
        self.model_name = f"mock-{compression_ratio}x"
    
    @monitor_performance
    def compress(self, text: str):
        """Mock compression with realistic processing simulation."""
        # Simulate text processing
        tokens = text.split()
        processed_tokens = []
        
        # Simulate some computational work
        for i, token in enumerate(tokens):
            if i % 100 == 0:
                # Simulate periodic heavy computation
                time.sleep(0.0001)
            
            # Simulate token processing
            processed_token = f"proc_{hash(token) % 1000}"
            processed_tokens.append(processed_token)
        
        # Simulate compression
        compressed_size = max(1, len(processed_tokens) // int(self.compression_ratio))
        return processed_tokens[:compressed_size]
    
    @monitor_performance
    def count_tokens(self, text: str) -> int:
        """Count tokens with processing simulation."""
        return len(text.split())


class ProfileRunner:
    """Main profiling runner with different analysis modes."""
    
    def __init__(self, output_dir: str = "profiling_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.compressor = MockCompressor()
    
    def generate_test_text(self, size: str) -> str:
        """Generate test text of specified size."""
        base_text = (
            "Artificial intelligence and machine learning systems require "
            "efficient processing of large textual datasets. Context compression "
            "techniques enable better utilization of computational resources "
            "while maintaining information quality and semantic coherence. "
        )
        
        multipliers = {
            "small": 100,
            "medium": 1000,
            "large": 5000,
            "xlarge": 20000,
        }
        
        multiplier = multipliers.get(size, 1000)
        return base_text * multiplier
    
    def run_cpu_profiling(self, text_size: str = "medium") -> str:
        """Run CPU profiling using cProfile."""
        print(f"Running CPU profiling on {text_size} text...")
        
        text = self.generate_test_text(text_size)
        
        # Set up profiler
        profiler = cProfile.Profile()
        
        # Profile the compression operation
        profiler.enable()
        
        # Run multiple iterations for better profiling data
        for _ in range(10):
            result = self.compressor.compress(text)
            token_count = self.compressor.count_tokens(text)
        
        profiler.disable()
        
        # Save profiling data
        timestamp = int(time.time())
        profile_file = self.output_dir / f"cpu_profile_{text_size}_{timestamp}.prof"
        profiler.dump_stats(str(profile_file))
        
        # Generate human-readable report
        report_file = self.output_dir / f"cpu_profile_{text_size}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            
            f.write(s.getvalue())
            
            # Add summary statistics
            f.write("\n\nPROFILING SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Text size: {text_size}\n")
            f.write(f"Input length: {len(text)} characters\n")
            f.write(f"Token count: {token_count}\n")
            f.write(f"Compressed size: {len(result)}\n")
            f.write(f"Total function calls: {ps.total_calls}\n")
            f.write(f"Total time: {ps.total_tt:.4f} seconds\n")
        
        print(f"CPU profile saved to {report_file}")
        return str(report_file)
    
    def run_memory_profiling(self, text_size: str = "medium") -> str:
        """Run memory profiling using tracemalloc."""
        import tracemalloc
        
        print(f"Running memory profiling on {text_size} text...")
        
        text = self.generate_test_text(text_size)
        
        # Start memory tracing
        tracemalloc.start()
        
        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run compression operations
        results = []
        for i in range(5):
            result = self.compressor.compress(text)
            results.append(result)
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Analyze memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Generate memory report
        timestamp = int(time.time())
        report_file = self.output_dir / f"memory_profile_{text_size}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("MEMORY PROFILING REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Text size: {text_size}\n")
            f.write(f"Input length: {len(text)} characters\n")
            f.write(f"Iterations: 5\n\n")
            
            f.write("TOP MEMORY ALLOCATIONS:\n")
            f.write("-" * 30 + "\n")
            
            for index, stat in enumerate(top_stats[:20]):
                f.write(f"{index + 1}. {stat}\n")
            
            # Current memory usage
            current_usage = tracemalloc.get_traced_memory()
            f.write(f"\nCurrent memory usage: {current_usage[0] / 1024 / 1024:.2f} MB\n")
            f.write(f"Peak memory usage: {current_usage[1] / 1024 / 1024:.2f} MB\n")
        
        tracemalloc.stop()
        
        print(f"Memory profile saved to {report_file}")
        return str(report_file)
    
    def run_line_profiling(self, text_size: str = "medium") -> str:
        """Run line-by-line profiling (requires line_profiler)."""
        try:
            import line_profiler
        except ImportError:
            print("Warning: line_profiler not available, skipping line profiling")
            return ""
        
        print(f"Running line profiling on {text_size} text...")
        
        text = self.generate_test_text(text_size)
        
        # Create line profiler
        profiler = line_profiler.LineProfiler()
        
        # Add functions to profile
        profiler.add_function(self.compressor.compress)
        profiler.add_function(self.compressor.count_tokens)
        
        # Run profiling
        profiler.enable_by_count()
        
        for _ in range(3):
            result = self.compressor.compress(text)
            token_count = self.compressor.count_tokens(text)
        
        profiler.disable_by_count()
        
        # Generate report
        timestamp = int(time.time())
        report_file = self.output_dir / f"line_profile_{text_size}_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            profiler.print_stats(stream=f)
        
        print(f"Line profile saved to {report_file}")
        return str(report_file)
    
    def run_observability_analysis(self) -> str:
        """Run analysis using the observability module."""
        print("Running observability analysis...")
        
        text = self.generate_test_text("medium")
        
        # Reset metrics
        metrics_collector.metrics.clear()
        metrics_collector.counters.clear()
        metrics_collector.timers.clear()
        
        # Run operations to collect metrics
        for i in range(10):
            result = self.compressor.compress(text)
        
        # Get metrics
        all_metrics = metrics_collector.get_all_metrics()
        
        # Save metrics report
        timestamp = int(time.time())
        report_file = self.output_dir / f"observability_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Generate human-readable summary
        summary_file = self.output_dir / f"observability_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("OBSERVABILITY ANALYSIS\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis timestamp: {all_metrics.get('timestamp', 'N/A')}\n\n")
            
            f.write("COUNTERS:\n")
            for name, value in all_metrics.get('counters', {}).items():
                f.write(f"  {name}: {value}\n")
            
            f.write("\nGAUGES:\n")
            for name, value in all_metrics.get('gauges', {}).items():
                f.write(f"  {name}: {value}\n")
            
            f.write("\nTIMERS:\n")
            for name, value in all_metrics.get('timers', {}).items():
                f.write(f"  {name}: {value:.4f}s\n")
        
        print(f"Observability analysis saved to {summary_file}")
        return str(summary_file)
    
    def generate_combined_report(self, profile_files: list) -> str:
        """Generate a combined profiling report."""
        timestamp = int(time.time())
        combined_file = self.output_dir / f"combined_profile_report_{timestamp}.md"
        
        with open(combined_file, 'w') as f:
            f.write("# Comprehensive Profiling Report\n\n")
            f.write(f"**Generated**: {time.ctime()}\n")
            f.write(f"**Profiler**: retrieval-free context compressor\n\n")
            
            f.write("## Summary\n\n")
            f.write("This report contains comprehensive profiling results including:\n")
            f.write("- CPU profiling (cProfile)\n")
            f.write("- Memory profiling (tracemalloc)\n")
            f.write("- Line-by-line profiling (line_profiler)\n")
            f.write("- Observability metrics\n\n")
            
            f.write("## Generated Files\n\n")
            for profile_file in profile_files:
                if profile_file:  # Skip empty strings
                    f.write(f"- `{Path(profile_file).name}`\n")
            
            f.write("\n## Key Findings\n\n")
            f.write("Review the individual profiling reports for detailed analysis:\n")
            f.write("1. **CPU Usage**: Check CPU profile for hotspots and optimization opportunities\n")
            f.write("2. **Memory Usage**: Review memory profile for leaks and allocation patterns\n")
            f.write("3. **Line Performance**: Examine line profile for slow code sections\n")
            f.write("4. **System Metrics**: Monitor observability data for resource utilization\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("Based on profiling results, consider:\n")
            f.write("- Optimizing functions with highest cumulative time\n")
            f.write("- Reducing memory allocations in critical paths\n")
            f.write("- Implementing caching for frequently accessed data\n")
            f.write("- Adding monitoring for production performance tracking\n")
        
        print(f"Combined report saved to {combined_file}")
        return str(combined_file)


def main():
    """Main profiling script."""
    parser = argparse.ArgumentParser(description="Run profiling analysis")
    parser.add_argument("--output", "-o", default="profiling_results",
                       help="Output directory for results")
    parser.add_argument("--size", "-s", choices=["small", "medium", "large", "xlarge"],
                       default="medium", help="Text size for profiling")
    parser.add_argument("--modes", "-m", nargs="+",
                       choices=["cpu", "memory", "line", "observability", "all"],
                       default=["all"], help="Profiling modes to run")
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = ProfileRunner(output_dir=args.output)
    
    print("Starting profiling analysis...")
    print(f"Output directory: {profiler.output_dir}")
    print(f"Text size: {args.size}")
    
    profile_files = []
    
    # Run selected profiling modes
    if "all" in args.modes or "cpu" in args.modes:
        profile_files.append(profiler.run_cpu_profiling(args.size))
    
    if "all" in args.modes or "memory" in args.modes:
        profile_files.append(profiler.run_memory_profiling(args.size))
    
    if "all" in args.modes or "line" in args.modes:
        profile_files.append(profiler.run_line_profiling(args.size))
    
    if "all" in args.modes or "observability" in args.modes:
        profile_files.append(profiler.run_observability_analysis())
    
    # Generate combined report
    combined_report = profiler.generate_combined_report(profile_files)
    
    print(f"\nProfiling completed! Results saved to {profiler.output_dir}")
    print(f"Combined report: {combined_report}")


if __name__ == "__main__":
    main()