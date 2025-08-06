#!/usr/bin/env python3
"""Basic test script to verify module structure without heavy dependencies."""

import sys
import os
sys.path.insert(0, 'src')

def test_module_imports():
    """Test that modules can be imported without errors."""
    print("Testing basic module structure...")
    
    # Test exception classes
    try:
        from retrieval_free.exceptions import RetrievalFreeError, CompressionError
        print("✓ Exception classes imported successfully")
    except Exception as e:
        print(f"✗ Exception import failed: {e}")
        return False
    
    # Test validation module
    try:
        from retrieval_free.validation import InputValidator, ValidationResult
        print("✓ Validation classes imported successfully")
    except Exception as e:
        print(f"✗ Validation import failed: {e}")
        return False
    
    # Test caching module
    try:
        from retrieval_free.caching import MemoryCache, create_cache_key
        print("✓ Caching classes imported successfully")
    except Exception as e:
        print(f"✗ Caching import failed: {e}")
        return False
    
    return True

def test_validation_functionality():
    """Test validation functionality."""
    print("\nTesting validation functionality...")
    
    try:
        from retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test valid input
        result = validator.validate_text_input("This is a test document.")
        if result.is_valid:
            print("✓ Valid input validation passed")
        else:
            print(f"✗ Valid input failed: {result.errors}")
            return False
        
        # Test invalid input
        result = validator.validate_text_input("")
        if not result.is_valid:
            print("✓ Invalid input validation correctly failed")
        else:
            print("✗ Invalid input should have failed validation")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Validation functionality test failed: {e}")
        return False

def test_caching_functionality():
    """Test caching functionality."""
    print("\nTesting caching functionality...")
    
    try:
        from retrieval_free.caching import MemoryCache, create_cache_key
        
        cache = MemoryCache(max_size=10)
        
        # Test cache operations
        key = create_cache_key("test text", "test_model", {"ratio": 8.0})
        cache.put(key, "test_value")
        
        value = cache.get(key)
        if value == "test_value":
            print("✓ Cache put/get operations work")
        else:
            print("✗ Cache operations failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Caching functionality test failed: {e}")
        return False

def test_exception_functionality():
    """Test exception functionality."""
    print("\nTesting exception functionality...")
    
    try:
        from retrieval_free.exceptions import CompressionError, ValidationError
        
        # Test exception creation
        error = CompressionError("Test error", input_length=100)
        if error.input_length == 100:
            print("✓ Exception attributes work")
        else:
            print("✗ Exception attributes failed")
            return False
        
        # Test exception dictionary conversion
        error_dict = error.to_dict()
        if "error_type" in error_dict and "message" in error_dict:
            print("✓ Exception dictionary conversion works")
        else:
            print("✗ Exception dictionary conversion failed")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Exception functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RETRIEVAL-FREE CONTEXT COMPRESSOR - Basic Structure Test")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_module_imports():
        all_passed = False
    
    # Test functionality
    if not test_validation_functionality():
        all_passed = False
        
    if not test_caching_functionality():
        all_passed = False
        
    if not test_exception_functionality():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("The core module structure is working correctly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("There are issues with the basic module structure.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())