#!/usr/bin/env python3
"""Test error handling and validation functionality."""

import sys
import os
from unittest.mock import MagicMock, patch
import numpy as np

# Mock ML libraries
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_validation_functionality():
    """Test input validation and security measures."""
    print("Testing validation functionality...")
    
    from retrieval_free.validation import (
        SecurityValidator, 
        ParameterValidator,
        ValidationResult,
        RateLimiter
    )
    
    # Test security validator
    validator = SecurityValidator()
    
    # Test normal text
    result = validator.validate_input("This is normal text for testing.")
    assert result.is_valid
    assert not result.has_errors
    print("✓ Normal text validation passed")
    
    # Test dangerous patterns
    dangerous_text = "<script>alert('xss')</script>"
    result = validator.validate_input(dangerous_text)
    assert not result.is_valid
    assert result.has_errors
    print("✓ Dangerous pattern detection works")
    
    # Test large input
    large_text = "x" * 20_000_000  # 20MB
    result = validator.validate_input(large_text)
    assert not result.is_valid
    assert "too large" in str(result.errors).lower()
    print("✓ Large input detection works")
    
    # Test parameter validator
    param_validator = ParameterValidator()
    
    # Valid compression ratio
    result = param_validator.validate_compression_ratio(8.0)
    assert result.is_valid
    print("✓ Valid compression ratio accepted")
    
    # Invalid compression ratio
    result = param_validator.validate_compression_ratio(0.5)
    assert not result.is_valid
    print("✓ Invalid compression ratio rejected")
    
    # Test rate limiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    
    # Should allow first 3 requests
    for i in range(3):
        allowed, remaining = limiter.is_allowed("test_client")
        assert allowed
        assert remaining == 2 - i
    
    # Should reject 4th request
    allowed, remaining = limiter.is_allowed("test_client")
    assert not allowed
    assert remaining == 0
    print("✓ Rate limiter works correctly")


def test_custom_exceptions():
    """Test custom exception classes."""
    print("\nTesting custom exceptions...")
    
    from retrieval_free.exceptions import (
        ValidationError,
        SecurityError, 
        CompressionError,
        ModelError,
        create_error_response,
        is_recoverable_error,
        is_critical_error
    )
    
    # Test ValidationError
    error = ValidationError("Invalid input", field="text")
    assert error.field == "text"
    assert error.error_code == "VALIDATION_ERROR"
    print("✓ ValidationError created correctly")
    
    # Test error response creation
    response = create_error_response(error)
    assert not response["success"]
    assert response["error"]["error_type"] == "ValidationError"
    assert response["is_user_error"]
    print("✓ Error response creation works")
    
    # Test error classification
    security_error = SecurityError("Malicious input detected")
    assert is_critical_error(security_error)
    assert not is_recoverable_error(security_error)
    print("✓ Error classification works")


def test_mega_token_validation():
    """Test MegaToken validation."""
    print("\nTesting MegaToken validation...")
    
    from retrieval_free.core import MegaToken
    from retrieval_free.exceptions import ValidationError
    
    # Valid MegaToken
    token = MegaToken(
        vector=np.array([1.0, 2.0, 3.0]),
        metadata={"test": "data"},
        confidence=0.95
    )
    assert token.confidence == 0.95
    print("✓ Valid MegaToken created")
    
    # Test invalid confidence
    try:
        MegaToken(
            vector=np.array([1.0, 2.0, 3.0]),
            metadata={"test": "data"},
            confidence=1.5  # Invalid
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "confidence" in e.message.lower()
        print("✓ Invalid confidence rejected")
    
    # Test empty vector
    try:
        MegaToken(
            vector=np.array([]),
            metadata={"test": "data"},
            confidence=0.8
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "empty" in e.message.lower()
        print("✓ Empty vector rejected")
    
    # Test non-finite values
    try:
        MegaToken(
            vector=np.array([1.0, np.inf, 3.0]),
            metadata={"test": "data"},
            confidence=0.8
        )
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "finite" in e.message.lower()
        print("✓ Non-finite values rejected")


def test_compressor_error_handling():
    """Test compressor error handling with mocks."""
    print("\nTesting compressor error handling...")
    
    from retrieval_free.core import ContextCompressor
    from retrieval_free.exceptions import ValidationError, ModelError
    
    # Test invalid model name
    try:
        ContextCompressor(model_name="")
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "model name" in e.message.lower()
        print("✓ Empty model name rejected")
    
    # Test invalid chunk size
    try:
        ContextCompressor(chunk_size=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "chunk size" in str(e).lower()
        print("✓ Invalid chunk size rejected")
    
    # Test invalid compression ratio
    try:
        ContextCompressor(compression_ratio=0.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "compression ratio" in str(e).lower()
        print("✓ Invalid compression ratio rejected")
    
    # Test invalid overlap ratio
    try:
        ContextCompressor(overlap_ratio=0.8)
        assert False, "Should have raised ValidationError" 
    except ValidationError as e:
        assert "overlap ratio" in e.message.lower()
        print("✓ Invalid overlap ratio rejected")
    
    print("✓ Compressor parameter validation works")


def test_resource_monitoring():
    """Test resource monitoring capabilities."""
    print("\nTesting resource monitoring...")
    
    from retrieval_free.observability import (
        PerformanceMonitor,
        HealthChecker,
        MetricsCollector
    )
    
    # Test performance monitor
    monitor = PerformanceMonitor()
    metrics = monitor.get_system_metrics()
    
    assert "memory_mb" in metrics
    assert "cpu_percent" in metrics
    assert metrics["memory_mb"] > 0
    print("✓ Performance monitoring works")
    
    # Test health checker
    health = HealthChecker()
    
    # Test memory check
    def memory_check():
        return metrics["memory_mb"] < 32768  # Less than 32GB
    
    health.register_check("memory", memory_check)
    results = health.run_checks()
    
    assert "memory" in results["checks"]
    print("✓ Health checking works")
    
    # Test metrics collector with error handling
    collector = MetricsCollector()
    
    # Should handle timer context properly
    try:
        with collector.timer("test_operation"):
            raise Exception("Test exception")
    except Exception:
        pass  # Expected
    
    metrics = collector.get_all_metrics()
    assert "test_operation" in metrics["timers"]
    assert metrics["timers"]["test_operation"] > 0
    print("✓ Metrics collection handles errors correctly")


if __name__ == "__main__":
    try:
        test_validation_functionality()
        test_custom_exceptions()
        test_mega_token_validation()
        test_compressor_error_handling()
        test_resource_monitoring()
        
        print("\n✅ GENERATION 2 (MAKE IT ROBUST) - ERROR HANDLING COMPLETE!")
        print("All error handling, validation, and security measures implemented and tested.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)