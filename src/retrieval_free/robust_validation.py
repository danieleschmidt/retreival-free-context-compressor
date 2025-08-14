"""Robust validation and error handling for compression operations."""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .exceptions import ValidationError, SecurityError


logger = logging.getLogger(__name__)


class InputSanitizer:
    """Sanitize and validate input data for security and robustness."""
    
    # Suspicious patterns that might indicate injection attacks
    SUSPICIOUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'data:text/html',  # Data URLs
        r'eval\s*\(',  # eval() calls
        r'exec\s*\(',  # exec() calls
        r'import\s+\w+',  # Import statements
        r'__import__',  # __import__ calls
        r'subprocess',  # subprocess module
        r'os\.system',  # os.system calls
        r'pickle\.loads',  # pickle deserialization
    ]
    
    # Maximum safe input sizes
    MAX_TEXT_LENGTH = 10_000_000  # 10MB of text
    MAX_TOKEN_COUNT = 1_000_000   # 1M tokens
    MAX_BATCH_SIZE = 1000         # 1000 documents
    
    def __init__(self):
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for pattern in self.SUSPICIOUS_PATTERNS
        ]
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize input text by removing suspicious content.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
            
        Raises:
            SecurityError: If suspicious content is detected
        """
        if not isinstance(text, str):
            raise ValidationError(f"Expected string input, got {type(text)}")
        
        # Check for suspicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                logger.warning(f"Suspicious pattern detected: {pattern.pattern}")
                raise SecurityError(
                    f"Potentially malicious content detected in input text",
                    pattern=pattern.pattern
                )
        
        # Remove null bytes and control characters
        text = text.replace('\x00', '')
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def validate_input_size(self, text: str, max_length: Optional[int] = None) -> None:
        """Validate input text size.
        
        Args:
            text: Input text to validate
            max_length: Optional custom maximum length
            
        Raises:
            ValidationError: If input is too large
        """
        max_len = max_length or self.MAX_TEXT_LENGTH
        
        if len(text) > max_len:
            raise ValidationError(
                f"Input text too large: {len(text)} chars > {max_len} limit",
                input_size=len(text),
                max_size=max_len,
            )
    
    def validate_batch_size(self, batch: List[Any]) -> None:
        """Validate batch size.
        
        Args:
            batch: Input batch to validate
            
        Raises:
            ValidationError: If batch is too large
        """
        if len(batch) > self.MAX_BATCH_SIZE:
            raise ValidationError(
                f"Batch too large: {len(batch)} items > {self.MAX_BATCH_SIZE} limit",
                batch_size=len(batch),
                max_batch_size=self.MAX_BATCH_SIZE,
            )


class CompressionValidator:
    """Validate compression parameters and results."""
    
    MIN_COMPRESSION_RATIO = 1.1
    MAX_COMPRESSION_RATIO = 1000.0
    MIN_CHUNK_SIZE = 16
    MAX_CHUNK_SIZE = 8192
    MIN_OVERLAP = 0
    MAX_OVERLAP = 512
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
    
    def validate_compression_params(
        self,
        compression_ratio: float,
        chunk_size: int,
        overlap: int,
        max_length: int,
    ) -> Dict[str, Any]:
        """Validate compression parameters.
        
        Args:
            compression_ratio: Target compression ratio
            chunk_size: Text chunk size
            overlap: Chunk overlap size
            max_length: Maximum input length
            
        Returns:
            Validated and potentially adjusted parameters
            
        Raises:
            ValidationError: If parameters are invalid
        """
        errors = []
        
        # Validate compression ratio
        if not (self.MIN_COMPRESSION_RATIO <= compression_ratio <= self.MAX_COMPRESSION_RATIO):
            errors.append(
                f"Invalid compression ratio {compression_ratio}. "
                f"Must be between {self.MIN_COMPRESSION_RATIO} and {self.MAX_COMPRESSION_RATIO}"
            )
        
        # Validate chunk size
        if not (self.MIN_CHUNK_SIZE <= chunk_size <= self.MAX_CHUNK_SIZE):
            errors.append(
                f"Invalid chunk size {chunk_size}. "
                f"Must be between {self.MIN_CHUNK_SIZE} and {self.MAX_CHUNK_SIZE}"
            )
        
        # Validate overlap
        if not (self.MIN_OVERLAP <= overlap <= self.MAX_OVERLAP):
            errors.append(
                f"Invalid overlap {overlap}. "
                f"Must be between {self.MIN_OVERLAP} and {self.MAX_OVERLAP}"
            )
        
        # Validate overlap vs chunk size
        if overlap >= chunk_size:
            errors.append(
                f"Overlap {overlap} must be less than chunk size {chunk_size}"
            )
        
        # Validate max length
        if max_length < chunk_size:
            errors.append(
                f"Max length {max_length} must be >= chunk size {chunk_size}"
            )
        
        if errors:
            raise ValidationError(
                "Invalid compression parameters",
                validation_errors=errors,
            )
        
        return {
            'compression_ratio': float(compression_ratio),
            'chunk_size': int(chunk_size),
            'overlap': int(overlap),
            'max_length': int(max_length),
        }
    
    def validate_compression_result(
        self,
        original_length: int,
        compressed_length: int,
        compression_ratio: float,
        processing_time: float,
        mega_tokens: List[Any],
    ) -> Dict[str, Any]:
        """Validate compression result.
        
        Args:
            original_length: Original input length
            compressed_length: Compressed output length
            compression_ratio: Achieved compression ratio
            processing_time: Processing time in seconds
            mega_tokens: List of mega-tokens
            
        Returns:
            Validation result with warnings
        """
        warnings = []
        
        # Check if compression actually occurred
        if compressed_length >= original_length:
            warnings.append(
                f"No compression achieved: {compressed_length} >= {original_length}"
            )
        
        # Check compression ratio consistency
        expected_ratio = original_length / compressed_length if compressed_length > 0 else 0
        ratio_diff = abs(compression_ratio - expected_ratio)
        if ratio_diff > 0.1 * compression_ratio:  # 10% tolerance
            warnings.append(
                f"Compression ratio mismatch: reported {compression_ratio:.2f}, "
                f"calculated {expected_ratio:.2f}"
            )
        
        # Check processing time
        if processing_time > 60:  # More than 1 minute
            warnings.append(
                f"Long processing time: {processing_time:.2f}s"
            )
        
        # Check mega-token count
        if len(mega_tokens) == 0:
            warnings.append("No mega-tokens produced")
        elif len(mega_tokens) > original_length:
            warnings.append(
                f"Too many mega-tokens: {len(mega_tokens)} > {original_length}"
            )
        
        return {
            'is_valid': len(warnings) == 0 or all('mismatch' not in w.lower() for w in warnings),
            'warnings': warnings,
            'quality_score': self._calculate_quality_score(
                compression_ratio, processing_time, len(warnings)
            ),
        }
    
    def _calculate_quality_score(
        self,
        compression_ratio: float,
        processing_time: float,
        num_warnings: int,
    ) -> float:
        """Calculate quality score for compression result.
        
        Args:
            compression_ratio: Achieved compression ratio
            processing_time: Processing time in seconds
            num_warnings: Number of validation warnings
            
        Returns:
            Quality score between 0 and 1
        """
        # Base score from compression ratio
        ratio_score = min(compression_ratio / 10.0, 1.0)  # Cap at 10x
        
        # Time penalty (prefer faster compression)
        time_score = max(0, 1.0 - processing_time / 10.0)  # Penalty after 10s
        
        # Warning penalty
        warning_penalty = num_warnings * 0.1
        
        final_score = max(0, ratio_score * 0.5 + time_score * 0.3 - warning_penalty)
        
        return min(final_score, 1.0)


class RobustErrorHandler:
    """Robust error handling with retry logic and fallbacks."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.error_counts = {}
    
    def with_retry(
        self,
        func,
        *args,
        retry_exceptions: Tuple[Exception, ...] = (Exception,),
        **kwargs,
    ):
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_exceptions: Exceptions that trigger retries
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except retry_exceptions as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Track error counts
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed with {error_type}: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.max_retries + 1} attempts failed. "
                        f"Final error: {error_type}: {str(e)}"
                    )
        
        raise last_exception
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error statistics.
        
        Returns:
            Dictionary of error type counts
        """
        return self.error_counts.copy()


class MemoryMonitor:
    """Monitor memory usage during compression operations."""
    
    def __init__(self, max_memory_mb: Optional[int] = None):
        self.max_memory_mb = max_memory_mb or 8192  # 8GB default
        self.peak_usage = 0
        self.current_usage = 0
    
    def estimate_tensor_memory(self, tensor_shape: Tuple[int, ...], dtype_size: int = 4) -> int:
        """Estimate memory usage for a tensor.
        
        Args:
            tensor_shape: Shape of the tensor
            dtype_size: Size of data type in bytes (4 for float32)
            
        Returns:
            Estimated memory usage in bytes
        """
        elements = 1
        for dim in tensor_shape:
            elements *= dim
        return elements * dtype_size
    
    def check_memory_usage(self, additional_memory: int = 0) -> None:
        """Check if memory usage is within limits.
        
        Args:
            additional_memory: Additional memory to be allocated in bytes
            
        Raises:
            ValidationError: If memory limit would be exceeded
        """
        try:
            import psutil
            process = psutil.Process()
            current_mb = process.memory_info().rss / 1024 / 1024
            self.current_usage = current_mb
            self.peak_usage = max(self.peak_usage, current_mb)
            
            additional_mb = additional_memory / 1024 / 1024
            total_mb = current_mb + additional_mb
            
            if total_mb > self.max_memory_mb:
                raise ValidationError(
                    f"Memory limit exceeded: {total_mb:.1f}MB > {self.max_memory_mb}MB",
                    current_memory=current_mb,
                    additional_memory=additional_mb,
                    memory_limit=self.max_memory_mb,
                )
        
        except ImportError:
            # psutil not available, use simple heuristic
            if additional_memory > self.max_memory_mb * 1024 * 1024:
                logger.warning(
                    f"Large memory allocation requested: {additional_memory / 1024 / 1024:.1f}MB"
                )
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        return {
            'current_usage_mb': self.current_usage,
            'peak_usage_mb': self.peak_usage,
            'memory_limit_mb': self.max_memory_mb,
            'usage_percentage': (self.current_usage / self.max_memory_mb) * 100,
        }


class ConfigValidator:
    """Validate configuration parameters for robustness."""
    
    REQUIRED_FIELDS = {
        'model_name': str,
        'compression_ratio': (int, float),
        'max_length': int,
    }
    
    OPTIONAL_FIELDS = {
        'device': str,
        'chunk_size': int,
        'overlap': int,
        'batch_size': int,
        'timeout': (int, float),
    }
    
    DEFAULT_VALUES = {
        'device': 'cpu',
        'chunk_size': 512,
        'overlap': 64,
        'batch_size': 8,
        'timeout': 300.0,  # 5 minutes
    }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated and normalized configuration
            
        Raises:
            ValidationError: If configuration is invalid
        """
        errors = []
        validated_config = {}
        
        # Check required fields
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in config:
                errors.append(f"Missing required field: {field}")
                continue
            
            value = config[field]
            if not isinstance(value, expected_type):
                errors.append(
                    f"Invalid type for {field}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                continue
            
            validated_config[field] = value
        
        # Check optional fields and apply defaults
        for field, expected_type in self.OPTIONAL_FIELDS.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    errors.append(
                        f"Invalid type for {field}: expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                    continue
                validated_config[field] = value
            elif field in self.DEFAULT_VALUES:
                validated_config[field] = self.DEFAULT_VALUES[field]
        
        # Additional validation rules
        if 'compression_ratio' in validated_config:
            ratio = validated_config['compression_ratio']
            if ratio < 1.0:
                errors.append(f"Compression ratio must be >= 1.0, got {ratio}")
        
        if 'chunk_size' in validated_config and 'overlap' in validated_config:
            chunk_size = validated_config['chunk_size']
            overlap = validated_config['overlap']
            if overlap >= chunk_size:
                errors.append(f"Overlap {overlap} must be < chunk_size {chunk_size}")
        
        if 'timeout' in validated_config:
            timeout = validated_config['timeout']
            if timeout <= 0:
                errors.append(f"Timeout must be > 0, got {timeout}")
        
        if errors:
            raise ValidationError(
                "Configuration validation failed",
                validation_errors=errors,
            )
        
        return validated_config


# Utility functions for robust validation
def validate_compression_request(
    text: str,
    parameters: Dict[str, Any],
) -> 'ValidationResult':
    """Validate a compression request.
    
    Args:
        text: Input text to compress
        parameters: Compression parameters
        
    Returns:
        ValidationResult with sanitized inputs and validation status
    """
    sanitizer = InputSanitizer()
    validator = CompressionValidator()
    
    # Sanitize input text
    try:
        sanitized_text = sanitizer.sanitize_text(text)
        sanitizer.validate_input_size(sanitized_text)
    except (ValidationError, SecurityError) as e:
        return ValidationResult(
            is_valid=False,
            errors=[str(e)],
            sanitized_input={'text': text},  # Return original on error
        )
    
    # Validate parameters
    try:
        validated_params = validator.validate_compression_params(
            compression_ratio=parameters.get('compression_ratio', 8.0),
            chunk_size=parameters.get('chunk_size', 512),
            overlap=parameters.get('overlap', 64),
            max_length=parameters.get('max_length', 256000),
        )
    except ValidationError as e:
        return ValidationResult(
            is_valid=False,
            errors=e.validation_errors if hasattr(e, 'validation_errors') else [str(e)],
            sanitized_input={'text': sanitized_text},
        )
    
    return ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        sanitized_input={'text': sanitized_text},
        validated_parameters=validated_params,
    )


class ValidationResult:
    """Result of input validation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: List[str],
        warnings: List[str] = None,
        sanitized_input: Dict[str, Any] = None,
        validated_parameters: Dict[str, Any] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_input = sanitized_input or {}
        self.validated_parameters = validated_parameters or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'sanitized_input': self.sanitized_input,
            'validated_parameters': self.validated_parameters,
        }