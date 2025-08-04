"""Input validation and sanitization."""

import re
import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[Any] = None
    risk_score: float = 0.0


class InputValidator:
    """Validator for text inputs and parameters."""
    
    def __init__(
        self,
        max_length: int = 1000000,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        allowed_encodings: List[str] = None
    ):
        """Initialize input validator.
        
        Args:
            max_length: Maximum text length in characters
            max_file_size: Maximum file size in bytes
            allowed_encodings: List of allowed text encodings
        """
        self.max_length = max_length
        self.max_file_size = max_file_size
        self.allowed_encodings = allowed_encodings or ['utf-8', 'ascii', 'latin-1']
        
        # Security patterns
        self.malicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',               # JavaScript URLs
            r'data:.*base64',            # Base64 data URLs
            r'\\x[0-9a-f]{2}',           # Hex escape sequences
            r'%[0-9a-f]{2}',             # URL encoding
            r'eval\s*\(',                # Eval function
            r'exec\s*\(',                # Exec function
        ]
        
        self.suspicious_patterns = [
            r'[^\x20-\x7E\x09\x0A\x0D]',  # Non-printable ASCII
            r'(.)\1{100,}',                # Excessive repetition
            r'(password|token|key|secret|api_key)',  # Sensitive data
        ]
    
    def validate_text(self, text: str, context: str = "input") -> ValidationResult:
        """Validate text input.
        
        Args:
            text: Text to validate
            context: Context description for logging
            
        Returns:
            ValidationResult with validation outcome
        """
        errors = []
        warnings = []
        risk_score = 0.0
        
        # Basic type check
        if not isinstance(text, str):
            errors.append(f"Input must be string, got {type(text)}")
            return ValidationResult(False, errors, warnings)
        
        # Length check
        if len(text) > self.max_length:
            errors.append(f"Text too long: {len(text)} > {self.max_length}")
        
        # Empty check
        if not text.strip():
            warnings.append("Input text is empty or whitespace-only")
        
        # Encoding validation
        try:
            text.encode('utf-8')
        except UnicodeEncodeError as e:
            errors.append(f"Invalid text encoding: {e}")
            risk_score += 0.3
        
        # Security pattern detection
        for pattern in self.malicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                errors.append(f"Potentially malicious pattern detected: {pattern}")
                risk_score += 0.5
        
        # Suspicious pattern detection
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                warnings.append(f"Suspicious pattern detected: {pattern}")
                risk_score += 0.2
        
        # Content quality checks
        word_count = len(text.split())
        if word_count < 10:
            warnings.append(f"Very short text: {word_count} words")
        
        # Sanitize input
        sanitized_text = self._sanitize_text(text)
        
        is_valid = len(errors) == 0 and risk_score < 0.8
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized_text,
            risk_score=risk_score
        )
        
        if not is_valid:
            logger.warning(f"Validation failed for {context}: {errors}")
        elif warnings:
            logger.info(f"Validation warnings for {context}: {warnings}")
        
        return result
    
    def validate_parameters(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate compression parameters.
        
        Args:
            params: Parameters to validate
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        sanitized_params = params.copy()
        
        # Compression ratio validation
        if 'compression_ratio' in params:
            ratio = params['compression_ratio']
            if not isinstance(ratio, (int, float)) or ratio <= 0:
                errors.append(f"Invalid compression_ratio: {ratio}")
            elif ratio > 100:
                warnings.append(f"Very high compression ratio: {ratio}")
                sanitized_params['compression_ratio'] = min(ratio, 32.0)
        
        # Device validation
        if 'device' in params:
            device = params['device']
            if device and not isinstance(device, str):
                errors.append(f"Device must be string, got {type(device)}")
            elif device and device not in ['cpu', 'cuda', 'auto']:
                if not device.startswith('cuda:'):
                    warnings.append(f"Unusual device specification: {device}")
        
        # Max length validation
        if 'max_length' in params:
            max_length = params['max_length']
            if not isinstance(max_length, int) or max_length <= 0:
                errors.append(f"Invalid max_length: {max_length}")
            elif max_length > 10000000:  # 10M tokens
                warnings.append(f"Very large max_length: {max_length}")
                sanitized_params['max_length'] = min(max_length, 1000000)
        
        # Chunk size validation
        if 'chunk_size' in params:
            chunk_size = params['chunk_size']
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                errors.append(f"Invalid chunk_size: {chunk_size}")
            elif chunk_size > 10000:
                warnings.append(f"Large chunk_size may cause memory issues: {chunk_size}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized_params
        )
    
    def validate_file_input(self, file_path: str) -> ValidationResult:
        """Validate file input.
        
        Args:
            file_path: Path to file to validate
            
        Returns:
            ValidationResult
        """
        import os
        
        errors = []
        warnings = []
        
        # Path validation
        if not isinstance(file_path, str):
            errors.append(f"File path must be string, got {type(file_path)}")
            return ValidationResult(False, errors, warnings)
        
        # Existence check
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            return ValidationResult(False, errors, warnings)
        
        # Size check
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                errors.append(f"File too large: {file_size} > {self.max_file_size}")
            elif file_size == 0:
                warnings.append("File is empty")
        except OSError as e:
            errors.append(f"Cannot read file size: {e}")
        
        # Extension check
        allowed_extensions = ['.txt', '.md', '.rst', '.py', '.json', '.xml', '.html']
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            warnings.append(f"Unusual file extension: {file_ext}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings
        )
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text by removing/replacing problematic content.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Replace excessive whitespace
        text = re.sub(r'\s{5,}', ' ', text)  # Multiple spaces
        text = re.sub(r'\n{4,}', '\n\n\n', text)  # Multiple newlines
        
        # Remove control characters (except common ones)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
            text += "\n[TRUNCATED]"
        
        return text
    
    def compute_content_hash(self, text: str) -> str:
        """Compute hash of content for deduplication.
        
        Args:
            text: Text to hash
            
        Returns:
            Content hash
        """
        # Normalize text for consistent hashing
        normalized = text.strip().lower()
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


class MemoryValidator:
    """Validator for memory usage and resource limits."""
    
    def __init__(self, max_memory_mb: int = 8192):
        """Initialize memory validator.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_mb = max_memory_mb
    
    def check_memory_requirements(
        self, 
        text_length: int, 
        compression_ratio: float = 8.0
    ) -> ValidationResult:
        """Check if operation fits within memory limits.
        
        Args:
            text_length: Input text length in tokens
            compression_ratio: Target compression ratio
            
        Returns:
            ValidationResult with memory assessment
        """
        errors = []
        warnings = []
        
        # Estimate memory requirements
        # Rough estimates based on transformer memory usage
        input_memory_mb = text_length * 0.004  # ~4KB per token
        processing_memory_mb = input_memory_mb * 2.5  # Peak during processing
        compressed_memory_mb = (text_length / compression_ratio) * 0.016  # Denser
        
        total_memory_mb = processing_memory_mb + compressed_memory_mb
        
        if total_memory_mb > self.max_memory_mb:
            errors.append(
                f"Estimated memory usage ({total_memory_mb:.1f}MB) "
                f"exceeds limit ({self.max_memory_mb}MB)"
            )
        elif total_memory_mb > self.max_memory_mb * 0.8:
            warnings.append(
                f"High memory usage estimated: {total_memory_mb:.1f}MB"
            )
        
        # Check for extremely large inputs
        if text_length > 1000000:  # 1M tokens
            warnings.append(f"Very large input: {text_length} tokens")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={
                'estimated_memory_mb': total_memory_mb,
                'input_memory_mb': input_memory_mb,
                'processing_memory_mb': processing_memory_mb,
                'compressed_memory_mb': compressed_memory_mb
            }
        )
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information.
        
        Returns:
            Dictionary with memory info in MB
        """
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            return {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'percent_used': memory.percent
            }
        except ImportError:
            logger.warning("psutil not available, cannot get memory info")
            return {}


def validate_compression_request(
    text: Union[str, List[str]],
    parameters: Dict[str, Any],
    file_path: Optional[str] = None
) -> ValidationResult:
    """Comprehensive validation of compression request.
    
    Args:
        text: Input text to validate
        parameters: Compression parameters
        file_path: Optional file path if reading from file
        
    Returns:
        Combined validation result
    """
    input_validator = InputValidator()
    memory_validator = MemoryValidator()
    
    all_errors = []
    all_warnings = []
    
    # Validate file input if provided
    if file_path:
        file_result = input_validator.validate_file_input(file_path)
        all_errors.extend(file_result.errors)
        all_warnings.extend(file_result.warnings)
        
        if not file_result.is_valid:
            return ValidationResult(False, all_errors, all_warnings)
    
    # Validate text input
    if isinstance(text, list):
        combined_text = " ".join(text)
    else:
        combined_text = text
    
    text_result = input_validator.validate_text(combined_text)
    all_errors.extend(text_result.errors)
    all_warnings.extend(text_result.warnings)
    
    # Validate parameters
    param_result = input_validator.validate_parameters(parameters)
    all_errors.extend(param_result.errors)
    all_warnings.extend(param_result.warnings)
    
    # Memory validation
    if text_result.is_valid:
        text_length = len(combined_text.split())
        compression_ratio = parameters.get('compression_ratio', 8.0)
        
        memory_result = memory_validator.check_memory_requirements(
            text_length, compression_ratio
        )
        all_errors.extend(memory_result.errors)
        all_warnings.extend(memory_result.warnings)
    
    # Risk assessment
    risk_score = text_result.risk_score
    if risk_score > 0.5:
        all_warnings.append(f"High risk content detected (score: {risk_score:.2f})")
    
    is_valid = len(all_errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=all_errors,
        warnings=all_warnings,
        sanitized_input={
            'text': text_result.sanitized_input,
            'parameters': param_result.sanitized_input
        },
        risk_score=risk_score
    )