#!/usr/bin/env python3
"""Basic functionality test without heavy ML dependencies."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports without ML dependencies."""
    print("Testing basic imports...")
    
    # Test version import
    from retrieval_free import __version__
    print(f"✓ Version: {__version__}")
    
    # Test observability module (no ML deps)
    from retrieval_free.observability import (
        MetricsCollector, 
        PerformanceMonitor, 
        StructuredLogger,
        get_observability_status
    )
    print("✓ Observability module imported")
    
    # Test metrics collector
    collector = MetricsCollector()
    collector.increment("test_metric", 5)
    collector.set_gauge("test_gauge", 42.0)
    
    with collector.timer("test_timer"):
        import time
        time.sleep(0.01)
    
    metrics = collector.get_all_metrics()
    assert "test_metric" in metrics["counters"]
    assert metrics["counters"]["test_metric"] == 5
    assert "test_gauge" in metrics["gauges"]
    assert metrics["gauges"]["test_gauge"] == 42.0
    assert "test_timer" in metrics["timers"]
    print("✓ MetricsCollector functionality works")
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    sys_metrics = monitor.get_system_metrics()
    assert "memory_mb" in sys_metrics
    assert "cpu_percent" in sys_metrics
    print("✓ PerformanceMonitor functionality works")
    
    # Test structured logger
    logger = StructuredLogger("test", "INFO")
    logger.log_compression_start(1000, "test-model")
    logger.log_compression_complete(1000, 100, 10.0, 0.5)
    print("✓ StructuredLogger functionality works")
    
    print("✓ All basic imports and functionality tests passed!")


def test_core_classes_structure():
    """Test that core classes can be imported (structure test only)."""
    print("\nTesting core class structures...")
    
    try:
        # This will fail due to missing torch, but we can catch and verify structure
        from retrieval_free.core import MegaToken, CompressionResult
        print("✓ Core data classes imported successfully")
        
        # Test MegaToken creation
        import numpy as np
        token = MegaToken(
            vector=np.array([1.0, 2.0, 3.0]),
            metadata={"test": "data"},
            confidence=0.95
        )
        assert token.confidence == 0.95
        assert token.metadata["test"] == "data"
        assert len(token.vector) == 3
        print("✓ MegaToken creation works")
        
    except ImportError as e:
        if "torch" in str(e) or "transformers" in str(e):
            print("⚠ Core classes not tested (missing ML dependencies - expected)")
        else:
            raise


def test_project_structure():
    """Test that project structure is correct."""
    print("\nTesting project structure...")
    
    required_files = [
        "src/retrieval_free/__init__.py",
        "src/retrieval_free/core.py", 
        "src/retrieval_free/streaming.py",
        "src/retrieval_free/selective.py",
        "src/retrieval_free/multi_doc.py",
        "src/retrieval_free/plugins.py",
        "src/retrieval_free/cli.py",
        "src/retrieval_free/observability.py",
        "pyproject.toml",
        "README.md"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        assert os.path.exists(full_path), f"Missing required file: {file_path}"
        print(f"✓ {file_path}")
    
    print("✓ All required files present")


def test_cli_structure():
    """Test CLI module structure."""
    print("\nTesting CLI structure...")
    
    from retrieval_free.cli import main
    print("✓ CLI main function imported")
    
    # Test that CLI interface class exists
    from retrieval_free.plugins import CLIInterface
    cli = CLIInterface()
    assert hasattr(cli, 'main')
    print("✓ CLI interface class structure correct")


if __name__ == "__main__":
    try:
        test_imports()
        test_core_classes_structure() 
        test_project_structure()
        test_cli_structure()
        print("\n🎉 All basic tests passed! Core implementation structure is working.")
        print("\nNote: Full ML functionality tests require torch/transformers installation.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)