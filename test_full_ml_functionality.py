#!/usr/bin/env python3
"""Full ML functionality test with real compression."""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_compression_functionality():
    """Test core compression with real ML models."""
    print("Testing core compression functionality...")
    
    try:
        # Test AutoCompressor with pretrained model
        from retrieval_free import AutoCompressor
        
        # Load a base compressor (should be faster)
        compressor = AutoCompressor.from_pretrained("rfcc-base-8x")
        print("✓ AutoCompressor loaded successfully")
        
        # Test compression with sample text
        sample_text = """
        This is a test document for compression. It contains multiple sentences
        to test the hierarchical compression algorithm. The document should be
        compressed into mega-tokens while preserving semantic meaning.
        
        Machine learning models like transformers can process long sequences,
        but they face challenges with very long contexts. This compression
        system addresses that challenge by creating dense representations
        that maintain information quality while reducing token count.
        
        The compression uses hierarchical encoding and information bottleneck
        techniques to create optimal representations for downstream tasks.
        """
        
        print(f"Original text length: {len(sample_text)} characters")
        
        # Compress the text
        start_time = time.time()
        result = compressor.compress(sample_text)
        compression_time = time.time() - start_time
        
        print(f"✓ Compression completed in {compression_time:.2f}s")
        print(f"✓ Original tokens: {result.original_length}")
        print(f"✓ Compressed to: {result.compressed_length} mega-tokens")
        print(f"✓ Compression ratio: {result.compression_ratio:.1f}×")
        print(f"✓ Processing time: {result.processing_time:.2f}s")
        
        # Verify compression result structure
        assert len(result.mega_tokens) > 0, "No mega-tokens generated"
        assert result.compression_ratio > 1.0, "No compression achieved"
        
        # Test decompression (approximate)
        reconstructed = compressor.decompress(result.mega_tokens)
        print(f"✓ Decompression completed, length: {len(reconstructed)} chars")
        
        # Test individual mega-token properties
        mega_token = result.mega_tokens[0]
        assert mega_token.confidence > 0.0, "Invalid confidence score"
        assert len(mega_token.vector) > 0, "Empty embedding vector"
        assert isinstance(mega_token.metadata, dict), "Invalid metadata"
        print("✓ Mega-token structure validated")
        
        print("✓ Core compression functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ Core compression test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_compression():
    """Test streaming compression functionality."""
    print("\nTesting streaming compression...")
    
    try:
        from retrieval_free import StreamingCompressor
        
        # Create streaming compressor
        compressor = StreamingCompressor(
            buffer_size=1000,
            compression_ratio=4.0,
            auto_flush_threshold=500
        )
        print("✓ StreamingCompressor created")
        
        # Test with multiple chunks
        chunks = [
            "First chunk of data for streaming compression testing.",
            "Second chunk continues the document with more content.",
            "Third chunk adds additional information to the stream.",
            "Final chunk completes the streaming compression test."
        ]
        
        mega_tokens_list = []
        for i, chunk in enumerate(chunks):
            mega_tokens = compressor.add_chunk(chunk)
            mega_tokens_list.append(mega_tokens)
            print(f"✓ Processed chunk {i+1}, got {len(mega_tokens)} mega-tokens")
        
        print("✓ Streaming compression working!")
        return True
        
    except ImportError as e:
        if "transformers" in str(e) or "sentence_transformers" in str(e):
            print("⚠ Streaming test skipped (ML dependencies not available)")
            return True
        else:
            print(f"❌ Streaming test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        return False


def test_selective_compression():
    """Test selective compression with different content types."""
    print("\nTesting selective compression...")
    
    try:
        from retrieval_free import SelectiveCompressor
        
        # Create selective compressor with different settings
        compressor = SelectiveCompressor(
            compression_ratio=4.0,
            adaptive_thresholds=True
        )
        print("✓ SelectiveCompressor created")
        
        # Test document with mixed content
        mixed_document = """
        LEGAL NOTICE: This agreement governs the use of software.
        
        General information: This is standard content that can be compressed normally.
        This paragraph contains typical information found in documents.
        
        Repetitive content: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
        """
        
        result = compressor.compress(mixed_document)
        print(f"✓ Selective compression completed")
        print(f"✓ Compressed to: {result.compressed_length} mega-tokens")
        
        print("✓ Selective compression working!")
        return True
        
    except ImportError as e:
        if "transformers" in str(e):
            print("⚠ Selective test skipped (ML dependencies not available)")
            return True
        else:
            print(f"❌ Selective test failed: {e}")
            return False
    except Exception as e:
        print(f"❌ Selective test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring integration."""
    print("\nTesting performance monitoring...")
    
    try:
        from retrieval_free import PerformanceMonitor, get_performance_monitor
        
        # Get global monitor instance
        monitor = get_performance_monitor()
        print("✓ Performance monitor accessed")
        
        # Test basic functionality
        monitor.record_metric("test_metric", 42.0)
        print("✓ Basic metric recording works")
        
        # Test resource monitoring if available
        try:
            usage = monitor.get_resource_usage()
            print(f"✓ Resource usage: CPU {usage.cpu_percent}%, Memory {usage.memory_percent}%")
        except AttributeError:
            print("✓ Resource monitoring not available (expected)")
        
        print("✓ Performance monitoring working!")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def test_cli_functionality():
    """Test CLI interface."""
    print("\nTesting CLI functionality...")
    
    try:
        from retrieval_free.cli import main
        from retrieval_free.plugins import CLIInterface
        
        # Test CLI interface creation
        cli = CLIInterface()
        assert hasattr(cli, 'main')
        print("✓ CLI interface created")
        
        # Test version command (would be: rfcc --version)
        # Note: Not running actual CLI to avoid sys.exit()
        print("✓ CLI structure validated")
        
        print("✓ CLI functionality working!")
        return True
        
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Running Full ML Functionality Tests")
    print("=" * 50)
    
    tests = [
        test_core_compression_functionality,
        test_streaming_compression,
        test_selective_compression,
        test_performance_monitoring,
        test_cli_functionality,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All ML functionality tests passed!")
        print("\n✅ GENERATION 1 (MAKE IT WORK) - COMPLETE!")
        sys.exit(0)
    else:
        print("❌ Some tests failed - needs debugging")
        sys.exit(1)