"""Security-focused tests."""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import tempfile
import os
from pathlib import Path


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock heavy dependencies for testing."""
    with patch.dict('sys.modules', {
        'torch': MagicMock(),
        'transformers': MagicMock(),
        'sklearn': MagicMock(),
        'psutil': MagicMock(),
        'bandit': MagicMock(),
    }):
        yield


class TestInputValidation:
    """Test input validation security."""
    
    def test_malicious_script_detection(self):
        """Test detection of malicious script content."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test various malicious patterns
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "eval(malicious_code)",
            "exec(harmful_command)",
        ]
        
        for malicious_input in malicious_inputs:
            result = validator.validate_text(malicious_input)
            
            assert not result.is_valid, f"Should reject: {malicious_input}"
            assert result.risk_score > 0.5, f"Should have high risk score for: {malicious_input}"
            assert len(result.errors) > 0, f"Should have errors for: {malicious_input}"
    
    def test_suspicious_pattern_detection(self):
        """Test detection of suspicious patterns."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test suspicious patterns (warnings, not errors)
        suspicious_inputs = [
            "password=secret123",
            "api_key=sk-1234567890",
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # Excessive repetition
        ]
        
        for suspicious_input in suspicious_inputs:
            result = validator.validate_text(suspicious_input)
            
            # Should be valid but with warnings
            assert result.is_valid or len(result.warnings) > 0, f"Should warn about: {suspicious_input}"
            assert result.risk_score > 0, f"Should have some risk score for: {suspicious_input}"
    
    def test_parameter_validation(self):
        """Test parameter validation security."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test invalid parameters
        invalid_params = {
            'compression_ratio': -1.0,  # Negative
            'max_length': 'invalid',    # Wrong type
            'device': 123,              # Wrong type
        }
        
        result = validator.validate_parameters(invalid_params)
        
        assert not result.is_valid
        assert len(result.errors) >= 2  # Should catch multiple issues
    
    def test_file_validation(self):
        """Test file input validation."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test non-existent file
        result = validator.validate_file_input("/non/existent/file.txt")
        assert not result.is_valid
        assert "does not exist" in result.errors[0]
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("test content")
            tmp_path = tmp.name
        
        try:
            result = validator.validate_file_input(tmp_path)
            assert result.is_valid
        finally:
            os.unlink(tmp_path)
    
    def test_text_sanitization(self):
        """Test text sanitization."""
        from src.retrieval_free.validation import InputValidator
        
        validator = InputValidator()
        
        # Test text with problematic content
        dirty_text = "Normal text\x00null byte\n\n\n\n\nexcessive newlines     excessive spaces"
        
        result = validator.validate_text(dirty_text)
        sanitized = result.sanitized_input
        
        # Should remove null bytes
        assert '\x00' not in sanitized
        
        # Should reduce excessive whitespace
        assert '\n\n\n\n\n' not in sanitized


class TestModelSecurity:
    """Test model security validation."""
    
    def test_model_source_validation(self):
        """Test model source validation."""
        from src.retrieval_free.security import ModelSecurityValidator
        
        validator = ModelSecurityValidator()
        
        # Test trusted source
        result = validator.validate_model_source("https://huggingface.co/model")
        assert result.passed
        
        # Test untrusted source
        result = validator.validate_model_source("https://malicious-site.com/model")
        assert not result.passed
        assert len(result.vulnerabilities) > 0
    
    def test_model_checksum_verification(self):
        """Test model checksum verification."""
        from src.retrieval_free.security import ModelSecurityValidator
        
        validator = ModelSecurityValidator()
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write("test model content")
            tmp_path = tmp.name
        
        try:
            # Test without expected hash (should pass)
            result = validator.verify_model_checksum(tmp_path)
            assert result is True
            
            # Test with wrong hash (should fail)  
            wrong_hash = "wrong_hash"
            result = validator.verify_model_checksum(tmp_path, wrong_hash)
            assert result is False
        finally:
            os.unlink(tmp_path)
    
    def test_local_path_validation(self):
        """Test local path security validation."""
        from src.retrieval_free.security import ModelSecurityValidator
        
        validator = ModelSecurityValidator()
        
        # Test non-existent path
        result = validator.validate_model_source("/non/existent/model")
        assert not result.passed
        assert any(vuln['type'] == 'missing_file' for vuln in result.vulnerabilities)


class TestSandboxedExecution:
    """Test sandboxed execution security."""
    
    def test_import_restrictions(self):
        """Test import restrictions in sandbox."""
        from src.retrieval_free.security import SandboxedExecution
        
        sandbox = SandboxedExecution(allowed_imports={'os', 'sys'})
        
        with sandbox:
            # Should allow safe imports
            import os
            import sys
            
            # Should block dangerous imports
            with pytest.raises(ImportError):
                import subprocess


class TestSecureStorage:
    """Test secure storage functionality."""
    
    def test_api_key_storage(self):
        """Test API key storage and retrieval."""
        from src.retrieval_free.security import SecureStorage
        
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = SecureStorage(temp_dir)
            
            # Store API key
            success = storage.store_api_key("test_service", "secret_key_123")
            assert success
            
            # Retrieve API key
            retrieved = storage.retrieve_api_key("test_service")
            assert retrieved == "secret_key_123"
            
            # Test non-existent key
            missing = storage.retrieve_api_key("missing_service")
            assert missing is None


class TestAuditLogging:
    """Test audit logging security."""
    
    def test_audit_logger_creation(self):
        """Test audit logger creation."""
        from src.retrieval_free.security import AuditLogger
        
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "audit.log")
            logger = AuditLogger(log_file)
            
            # Test logging events
            logger.log_model_load("test-model", "test-source", "test-user")
            logger.log_compression_request(1000, "test-model", {}, "test-user")
            logger.log_security_event("test_event", "test description", "warning")
            
            # Verify log file exists
            assert os.path.exists(log_file)
            
            # Verify log content
            with open(log_file, 'r') as f:
                content = f.read()
                assert "MODEL_LOAD" in content
                assert "COMPRESSION_REQUEST" in content
                assert "SECURITY_EVENT" in content


class TestVulnerabilityScanning:
    """Test vulnerability scanning."""
    
    @patch('subprocess.run')
    def test_vulnerability_scan_with_bandit(self, mock_run):
        """Test vulnerability scanning with bandit."""
        from src.retrieval_free.security import scan_for_vulnerabilities
        
        # Mock successful bandit run
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = '{"results": [{"issue_severity": "HIGH", "issue_text": "Test issue", "filename": "test.py", "line_number": 10}]}'
        mock_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = scan_for_vulnerabilities(temp_dir)
            
            assert not result.passed  # Should fail due to mock issue
            assert len(result.vulnerabilities) == 1
            assert result.vulnerabilities[0]['severity'] == 'high'
    
    @patch('subprocess.run')
    def test_vulnerability_scan_fallback(self, mock_run):
        """Test vulnerability scanning fallback when bandit not available."""
        from src.retrieval_free.security import scan_for_vulnerabilities
        
        # Mock bandit not available
        mock_run.side_effect = FileNotFoundError()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test Python file with suspicious content
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, 'w') as f:
                f.write("eval(user_input)")  # Suspicious pattern
            
            result = scan_for_vulnerabilities(temp_dir)
            
            # Should still detect patterns even without bandit
            assert len(result.warnings) > 0  # Should warn about bandit not available


class TestSecurityIntegration:
    """Test security integration with main components."""
    
    def test_compression_security_validation(self):
        """Test security validation in compression pipeline."""
        from src.retrieval_free.validation import validate_compression_request
        
        # Test with malicious input
        malicious_text = "<script>alert('xss')</script>"
        parameters = {'compression_ratio': 8.0}
        
        result = validate_compression_request(malicious_text, parameters)
        
        assert not result.is_valid
        assert result.risk_score > 0.5
    
    def test_model_loading_security(self):
        """Test security checks during model loading."""
        from src.retrieval_free.security import ModelSecurityValidator
        
        validator = ModelSecurityValidator()
        
        # Test various model sources
        test_cases = [
            ("https://huggingface.co/model", True),   # Should pass
            ("https://evil-site.com/model", False),   # Should fail
            ("file:///etc/passwd", False),            # Should fail
        ]
        
        for model_source, should_pass in test_cases:
            result = validator.validate_model_source(model_source)
            if should_pass:
                assert result.passed, f"Should pass: {model_source}"
            else:
                assert not result.passed, f"Should fail: {model_source}"
    
    def test_resource_limit_validation(self):
        """Test resource limit validation."""
        from src.retrieval_free.validation import MemoryValidator
        
        validator = MemoryValidator(max_memory_mb=1024)  # 1GB limit
        
        # Test within limits
        result = validator.check_memory_requirements(
            text_length=10000,  # 10k tokens
            compression_ratio=8.0
        )
        assert result.is_valid
        
        # Test exceeding limits
        result = validator.check_memory_requirements(
            text_length=10000000,  # 10M tokens
            compression_ratio=8.0
        )
        assert not result.is_valid
        assert len(result.errors) > 0


class TestExceptionSecurity:
    """Test security-related exception handling."""
    
    def test_security_exception_creation(self):
        """Test SecurityError exception creation."""
        from src.retrieval_free.exceptions import SecurityError
        
        error = SecurityError(
            "Security violation detected",
            security_check="input_validation",
            risk_score=0.9
        )
        
        assert error.error_code == "SECURITY_VIOLATION"
        assert error.details['security_check'] == "input_validation"
        assert error.details['risk_score'] == 0.9
    
    def test_exception_conversion(self):
        """Test exception handling and conversion."""
        from src.retrieval_free.exceptions import handle_exception, create_error_response
        
        # Test generic exception conversion
        original_error = ValueError("Invalid input")
        converted = handle_exception(original_error, "test_context", reraise=False)
        
        assert converted is not None
        assert "test_context" in str(converted)
        
        # Test error response creation
        response = create_error_response(converted)
        assert response['success'] is False
        assert 'error' in response
        assert 'timestamp' in response


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])