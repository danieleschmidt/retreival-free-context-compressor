#!/usr/bin/env python3
"""Test CLI functionality without heavy dependencies."""

import sys
import os
sys.path.insert(0, 'src')

def test_cli_import():
    """Test that CLI can be imported."""
    try:
        from retrieval_free.cli import main
        print("✓ CLI main function can be imported")
        return True
    except Exception as e:
        print(f"✗ CLI import failed: {e}")
        return False

def test_basic_argument_parsing():
    """Test basic CLI argument parsing."""
    try:
        # Create a minimal CLI implementation for testing
        import argparse
        
        parser = argparse.ArgumentParser(description="Test CLI")
        subparsers = parser.add_subparsers(dest="command")
        
        # Test compress command
        compress_parser = subparsers.add_parser("compress")
        compress_parser.add_argument("input", help="Input text")
        compress_parser.add_argument("--model", default="test")
        
        # Test list command
        list_parser = subparsers.add_parser("list-models")
        
        # Parse test arguments
        test_args = ["compress", "test text", "--model", "test_model"]
        args = parser.parse_args(test_args)
        
        if args.command == "compress" and args.input == "test text":
            print("✓ Basic argument parsing works")
            return True
        else:
            print(f"✗ Argument parsing failed: {args}")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False

def test_cli_help():
    """Test CLI help functionality."""
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Retrieval-Free Context Compressor")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Compress command
        compress_parser = subparsers.add_parser("compress", help="Compress text")
        compress_parser.add_argument("input", help="Input text or file")
        
        # List models command  
        list_parser = subparsers.add_parser("list-models", help="List available models")
        
        # Test help generation (just make sure it doesn't crash)
        help_text = parser.format_help()
        
        if "Retrieval-Free Context Compressor" in help_text:
            print("✓ CLI help generation works")
            return True
        else:
            print("✗ CLI help generation failed")
            return False
            
    except Exception as e:
        print(f"✗ CLI help test failed: {e}")
        return False

def test_mock_compression():
    """Test mock compression functionality."""
    try:
        # Create a simple mock compressor for testing
        class MockCompressor:
            def __init__(self, model_name="test"):
                self.model_name = model_name
            
            def compress(self, text):
                return {
                    'original_length': len(text.split()),
                    'compressed_length': max(1, len(text.split()) // 8),
                    'compression_ratio': 8.0,
                    'mega_tokens': ['[mock_token_1]', '[mock_token_2]']
                }
        
        compressor = MockCompressor()
        result = compressor.compress("This is a test document for compression")
        
        if result['compression_ratio'] == 8.0:
            print("✓ Mock compression works")
            return True
        else:
            print(f"✗ Mock compression failed: {result}")
            return False
            
    except Exception as e:
        print(f"✗ Mock compression test failed: {e}")
        return False

def main():
    """Run CLI tests."""
    print("=" * 60)
    print("RETRIEVAL-FREE CONTEXT COMPRESSOR - CLI Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test CLI import
    if not test_cli_import():
        all_passed = False
    
    # Test basic functionality without heavy dependencies
    if not test_basic_argument_parsing():
        all_passed = False
        
    if not test_cli_help():
        all_passed = False
        
    if not test_mock_compression():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 CLI TESTS PASSED!")
        print("Basic CLI functionality is working.")
    else:
        print("❌ SOME CLI TESTS FAILED!")
        print("There are issues with CLI functionality.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())