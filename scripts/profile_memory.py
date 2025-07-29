#!/usr/bin/env python3
"""Memory profiling script for compression operations."""

import argparse
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any
import psutil
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import matplotlib.pyplot as plt


console = Console()


class MemoryProfiler:
    """Memory profiling utilities for compression operations."""
    
    def __init__(self, model_name: str = "rfcc-base-8x"):
        self.model_name = model_name
        self.process = psutil.Process()
        self.snapshots = []
        console.print(f"[blue]Memory profiler initialized for {model_name}")
    
    def start_tracing(self):
        """Start memory tracing."""
        tracemalloc.start()
        console.print("[green]Memory tracing started")
    
    def take_snapshot(self, label: str):
        """Take a memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        memory_info = self.process.memory_info()
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
        
        self.snapshots.append({
            'label': label,
            'timestamp': time.time(),
            'tracemalloc_snapshot': snapshot,
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'gpu_mb': gpu_memory / 1024 / 1024 if gpu_memory else 0,
        })
        
        console.print(f"[yellow]Snapshot taken: {label} (RSS: {memory_info.rss / 1024 / 1024:.1f} MB)")
    
    def profile_compression_pipeline(self, documents: List[str]) -> Dict[str, Any]:
        """Profile memory usage throughout compression pipeline."""
        console.print("[blue]Profiling compression pipeline memory usage...")
        
        self.start_tracing()
        self.take_snapshot("Initial")
        
        # Simulate model loading
        console.print("Loading model...")
        time.sleep(0.5)  # Simulate loading time
        mock_model_weights = torch.randn(1000, 768) * 0.01  # Simulate model weights
        self.take_snapshot("Model Loaded")
        
        total_processed = 0
        peak_memory = 0
        
        with Progress() as progress:
            task = progress.add_task("Processing documents...", total=len(documents))
            
            for i, document in enumerate(documents):
                # Simulate document processing
                self._mock_process_document(document)
                total_processed += 1
                
                # Track peak memory
                current_memory = self.process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                
                # Take periodic snapshots
                if i % 10 == 0:
                    self.take_snapshot(f"Document {i}")
                
                progress.advance(task)
        
        self.take_snapshot("Processing Complete")
        
        # Clean up
        del mock_model_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.take_snapshot("Cleanup Complete")
        
        return {
            'total_documents': len(documents),
            'peak_memory_mb': peak_memory,
            'final_memory_mb': self.snapshots[-1]['rss_mb'],
            'memory_increase_mb': self.snapshots[-1]['rss_mb'] - self.snapshots[0]['rss_mb'],
        }
    
    def _mock_process_document(self, document: str):
        """Mock document processing with realistic memory patterns."""
        # Simulate tokenization
        tokens = document.split()  # Simple tokenization
        
        # Simulate embedding lookup (creates temporary tensors)
        temp_embeddings = torch.randn(len(tokens), 768) * 0.01
        
        # Simulate compression processing
        time.sleep(0.01)  # Small delay to simulate processing
        
        # Simulate compressed output
        compression_ratio = np.random.uniform(6.0, 10.0)
        compressed_size = max(1, int(len(tokens) / compression_ratio))
        compressed_tokens = torch.randn(compressed_size, 768) * 0.01
        
        # Simulate some memory cleanup
        del temp_embeddings
        
        return compressed_tokens
    
    def analyze_memory_growth(self) -> Dict[str, Any]:
        """Analyze memory growth patterns across snapshots."""
        if len(self.snapshots) < 2:
            return {}
        
        growth_rates = []
        for i in range(1, len(self.snapshots)):
            prev_memory = self.snapshots[i-1]['rss_mb']
            curr_memory = self.snapshots[i]['rss_mb']
            growth = curr_memory - prev_memory
            growth_rates.append(growth)
        
        return {
            'total_growth_mb': self.snapshots[-1]['rss_mb'] - self.snapshots[0]['rss_mb'],
            'max_growth_step_mb': max(growth_rates) if growth_rates else 0,
            'min_growth_step_mb': min(growth_rates) if growth_rates else 0,
            'avg_growth_per_step_mb': np.mean(growth_rates) if growth_rates else 0,
            'growth_steps_count': len(growth_rates),
        }
    
    def find_memory_leaks(self) -> List[Dict[str, Any]]:
        """Identify potential memory leaks by comparing snapshots."""
        if len(self.snapshots) < 2:
            return []
        
        leaks = []
        
        # Compare first and last snapshots
        first_snapshot = self.snapshots[0]['tracemalloc_snapshot']
        last_snapshot = self.snapshots[-1]['tracemalloc_snapshot']
        
        # Get top differences
        top_stats = last_snapshot.compare_to(first_snapshot, 'lineno')
        
        for stat in top_stats[:10]:  # Top 10 memory differences
            if stat.size_diff > 1024 * 1024:  # Only report differences > 1MB
                leaks.append({
                    'file': stat.traceback.format()[-1] if stat.traceback else 'Unknown',
                    'size_diff_mb': stat.size_diff / 1024 / 1024,
                    'count_diff': stat.count_diff,
                })
        
        return leaks
    
    def generate_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        return {
            'profiler_info': {
                'model_name': self.model_name,
                'total_snapshots': len(self.snapshots),
                'profiling_duration': self.snapshots[-1]['timestamp'] - self.snapshots[0]['timestamp'] if self.snapshots else 0,
            },
            'memory_usage': {
                'initial_mb': self.snapshots[0]['rss_mb'] if self.snapshots else 0,
                'final_mb': self.snapshots[-1]['rss_mb'] if self.snapshots else 0,
                'peak_mb': max(s['rss_mb'] for s in self.snapshots) if self.snapshots else 0,
                'gpu_peak_mb': max(s['gpu_mb'] for s in self.snapshots) if self.snapshots else 0,
            },
            'growth_analysis': self.analyze_memory_growth(),
            'potential_leaks': self.find_memory_leaks(),
        }
    
    def plot_memory_usage(self, output_path: Path):
        """Plot memory usage over time."""
        if not self.snapshots:
            console.print("[red]No snapshots available for plotting")
            return
        
        timestamps = [s['timestamp'] - self.snapshots[0]['timestamp'] for s in self.snapshots]
        rss_memory = [s['rss_mb'] for s in self.snapshots]
        gpu_memory = [s['gpu_mb'] for s in self.snapshots]
        labels = [s['label'] for s in self.snapshots]
        
        plt.figure(figsize=(12, 8))
        
        # Plot RSS memory
        plt.subplot(2, 1, 1)
        plt.plot(timestamps, rss_memory, 'b-', marker='o', label='RSS Memory')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.title('RSS Memory Usage Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add labels for key points
        for i, (t, m, label) in enumerate(zip(timestamps, rss_memory, labels)):
            if i % max(1, len(labels) // 5) == 0:  # Show every 5th label
                plt.annotate(label, (t, m), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot GPU memory if available
        plt.subplot(2, 1, 2)
        if any(gpu > 0 for gpu in gpu_memory):
            plt.plot(timestamps, gpu_memory, 'r-', marker='s', label='GPU Memory')
            plt.xlabel('Time (seconds)')
            plt.ylabel('GPU Memory (MB)')
            plt.title('GPU Memory Usage Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No GPU memory data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=12)
            plt.title('GPU Memory Usage Over Time')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        console.print(f"[green]Memory usage plot saved to {output_path}")
    
    def display_report(self, report: Dict[str, Any]):
        """Display memory profiling report."""
        console.print("\n[bold green]Memory Profiling Report")
        
        # Basic info
        info_table = Table(title="Profiling Information")
        info_table.add_column("Metric", style="cyan")
        info_table.add_column("Value", style="magenta")
        
        for key, value in report['profiler_info'].items():
            info_table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(info_table)
        
        # Memory usage
        memory_table = Table(title="Memory Usage (MB)")
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="magenta")
        
        for key, value in report['memory_usage'].items():
            memory_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
        
        console.print(memory_table)
        
        # Growth analysis
        if report['growth_analysis']:
            growth_table = Table(title="Memory Growth Analysis")
            growth_table.add_column("Metric", style="cyan")
            growth_table.add_column("Value", style="magenta")
            
            for key, value in report['growth_analysis'].items():
                growth_table.add_row(key.replace('_', ' ').title(), f"{value:.2f}")
            
            console.print(growth_table)
        
        # Potential leaks
        if report['potential_leaks']:
            console.print("\n[bold red]Potential Memory Leaks Detected:")
            leak_table = Table()
            leak_table.add_column("File", style="yellow")
            leak_table.add_column("Size Diff (MB)", style="red")
            leak_table.add_column("Count Diff", style="blue")
            
            for leak in report['potential_leaks']:
                leak_table.add_row(
                    leak['file'][:50] + "..." if len(leak['file']) > 50 else leak['file'],
                    f"{leak['size_diff_mb']:.2f}",
                    str(leak['count_diff'])
                )
            
            console.print(leak_table)


def load_test_documents(data_path: Path, num_docs: int = 100) -> List[str]:
    """Load test documents for profiling."""
    if data_path and data_path.exists():
        documents = []
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                documents.append(f.read())
        return documents[:num_docs]
    else:
        # Generate synthetic documents
        console.print(f"[yellow]Generating {num_docs} synthetic documents for profiling")
        base_content = "This is test content for memory profiling. " * 100
        
        documents = []
        for i in range(num_docs):
            # Vary document sizes
            multiplier = np.random.randint(1, 10)
            doc = base_content * multiplier
            documents.append(doc)
        
        return documents


def main():
    """Main memory profiling script."""
    parser = argparse.ArgumentParser(description="Memory profiling for compression")
    parser.add_argument("--model", default="rfcc-base-8x", help="Model name to profile")
    parser.add_argument("--data-path", type=Path, help="Path to test documents")
    parser.add_argument("--num-docs", type=int, default=100, help="Number of documents to process")
    parser.add_argument("--output-plot", type=Path, default="memory_profile.png", help="Output plot file")
    parser.add_argument("--output-report", type=Path, default="memory_report.json", help="Output report file")
    
    args = parser.parse_args()
    
    # Load test documents
    documents = load_test_documents(args.data_path, args.num_docs)
    console.print(f"[blue]Loaded {len(documents)} documents for profiling")
    
    # Initialize profiler
    profiler = MemoryProfiler(args.model)
    
    # Run profiling
    pipeline_results = profiler.profile_compression_pipeline(documents)
    
    # Generate report
    report = profiler.generate_memory_report()
    report['pipeline_results'] = pipeline_results
    
    # Display results
    profiler.display_report(report)
    
    # Save plot
    profiler.plot_memory_usage(args.output_plot)
    
    # Save report
    import json
    with open(args.output_report, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print(f"[green]Memory profiling complete! Report saved to {args.output_report}")


if __name__ == "__main__":
    main()