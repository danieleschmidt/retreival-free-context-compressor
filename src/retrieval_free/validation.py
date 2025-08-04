"""Input validation and security measures."""

import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_input: Optional[str] = None
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class SecurityValidator:
    """Security validation for input content."""
    
    # Dangerous patterns that could indicate malicious content
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # JavaScript
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'eval\s*\(',                 # Eval calls
        r'exec\s*\(',                 # Exec calls
        r'__import__\s*\(',           # Python imports
        r'subprocess\.',              # Subprocess calls
        r'os\.(system|popen|spawn)',  # OS commands
        r'\.\./.*\.\.',               # Directory traversal
        r'\.\.\\.*\.\.',              # Windows directory traversal
    ]
    
    # Suspicious keywords that might indicate injection attempts
    SUSPICIOUS_KEYWORDS = [
        'union select', 'drop table', 'delete from', 'insert into',
        'update set', 'grant all', 'revoke', 'alter table',
        '; drop', '; delete', '; insert', '; update'
    ]
    
    def __init__(self, max_input_size: int = 10_000_000):  # 10MB default
        self.max_input_size = max_input_size
        
    def validate_input(self, text: str, context: str = "input") -> ValidationResult:
        """Validate input text for security issues."""
        errors = []
        warnings = []
        
        # Check input size
        if len(text) > self.max_input_size:
            errors.append(f"Input too large: {len(text)} bytes (max: {self.max_input_size})")
        
        # Check for dangerous patterns
        text_lower = text.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                errors.append(f"Dangerous pattern detected in {context}: {pattern[:50]}...")
        
        # Check for suspicious SQL injection patterns
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in text_lower:
                warnings.append(f"Suspicious keyword detected in {context}: {keyword}")
        
        # Check for excessive special characters (potential encoding attacks)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_char_ratio > 0.5:
            warnings.append(f"High special character ratio in {context}: {special_char_ratio:.2f}")
        
        # Check for extremely long lines (potential buffer overflow attempts)
        lines = text.split('\n')
        max_line_length = max((len(line) for line in lines), default=0)
        if max_line_length > 100000:  # 100KB line
            warnings.append(f"Extremely long line detected in {context}: {max_line_length} chars")
        
        # Sanitize input if no critical errors
        sanitized = None
        if not errors:
            sanitized = self._sanitize_text(text)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input=sanitized
        )
    
    def _sanitize_text(self, text: str) -> str:
        """Basic text sanitization."""
        # Remove null bytes
        text = text.replace('\0', '')
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace
        text = re.sub(r'\n{5,}', '\n\n\n\n', text)  # Max 4 consecutive newlines
        text = re.sub(r' {10,}', ' ' * 5, text)     # Max 5 consecutive spaces
        
        # Remove BOM if present
        if text.startswith('\ufeff'):
            text = text[1:]
        
        return text
    
    def compute_content_hash(self, text: str) -> str:
        """Compute secure hash of content for integrity checking."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class ParameterValidator:
    """Validator for function parameters."""
    
    @staticmethod
    def validate_compression_ratio(ratio: float) -> ValidationResult:
        """Validate compression ratio parameter."""
        errors = []
        warnings = []
        
        if not isinstance(ratio, (int, float)):
            errors.append("Compression ratio must be a number")
        elif ratio <= 1.0:
            errors.append("Compression ratio must be greater than 1.0")
        elif ratio > 1000.0:
            errors.append("Compression ratio too high (max: 1000x)")
        elif ratio > 50.0:
            warnings.append(f"Very high compression ratio: {ratio}x may cause quality loss")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_chunk_size(size: int) -> ValidationResult:
        """Validate chunk size parameter."""
        errors = []
        warnings = []
        
        if not isinstance(size, int):
            errors.append("Chunk size must be an integer")
        elif size <= 0:
            errors.append("Chunk size must be positive")
        elif size > 100000:
            errors.append("Chunk size too large (max: 100,000)")
        elif size < 50:
            warnings.append(f"Very small chunk size: {size} may cause inefficient compression")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_confidence(confidence: float) -> ValidationResult:
        """Validate confidence score."""
        errors = []
        warnings = []
        
        if not isinstance(confidence, (int, float)):
            errors.append("Confidence must be a number")
        elif not 0.0 <= confidence <= 1.0:
            errors.append("Confidence must be between 0.0 and 1.0")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )


def validate_input(
    max_size: int = 10_000_000,
    require_sanitization: bool = True
):
    """Decorator for input validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            validator = SecurityValidator(max_size)
            
            # Validate text inputs in args
            for i, arg in enumerate(args):
                if isinstance(arg, str) and len(arg) > 100:  # Only validate substantial text
                    result = validator.validate_input(arg, f"argument {i}")
                    if not result.is_valid:
                        raise ValueError(f"Input validation failed: {'; '.join(result.errors)}")
                    
                    if result.has_warnings:
                        import warnings
                        warnings.warn(f"Input validation warnings: {'; '.join(result.warnings)}")
                    
                    # Replace with sanitized version if required
                    if require_sanitization and result.sanitized_input:
                        args = list(args)
                        args[i] = result.sanitized_input
                        args = tuple(args)
            
            # Validate text inputs in kwargs
            for key, value in kwargs.items():
                if isinstance(value, str) and len(value) > 100:
                    result = validator.validate_input(value, f"parameter {key}")
                    if not result.is_valid:
                        raise ValueError(f"Input validation failed for {key}: {'; '.join(result.errors)}")
                    
                    if result.has_warnings:
                        import warnings
                        warnings.warn(f"Input validation warnings for {key}: {'; '.join(result.warnings)}")
                    
                    # Replace with sanitized version if required
                    if require_sanitization and result.sanitized_input:
                        kwargs[key] = result.sanitized_input
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def validate_parameters(**param_validators):
    """Decorator for parameter validation."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to parameter names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate specified parameters
            for param_name, validator_func in param_validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    result = validator_func(value)
                    
                    if not result.is_valid:
                        raise ValueError(f"Parameter validation failed for {param_name}: {'; '.join(result.errors)}")
                    
                    if result.has_warnings:
                        import warnings
                        warnings.warn(f"Parameter validation warnings for {param_name}: {'; '.join(result.warnings)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class RateLimiter:
    """Simple rate limiter for API protection."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {client_id: [timestamps]}
    
    def is_allowed(self, client_id: str = "default") -> Tuple[bool, int]:
        """Check if request is allowed. Returns (allowed, remaining_requests)."""
        import time
        current_time = time.time()
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                ts for ts in self.requests[client_id]
                if current_time - ts < self.window_seconds
            ]
        else:
            self.requests[client_id] = []
        
        # Check if under limit
        current_count = len(self.requests[client_id])
        if current_count < self.max_requests:
            self.requests[client_id].append(current_time)
            return True, self.max_requests - current_count - 1
        
        return False, 0
    
    def reset(self, client_id: str = "default"):
        """Reset rate limit for a client."""
        if client_id in self.requests:
            del self.requests[client_id]


def rate_limit(max_requests: int = 100, window_seconds: int = 60):
    """Decorator for rate limiting."""
    limiter = RateLimiter(max_requests, window_seconds)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name as client_id (could be enhanced with actual client identification)
            client_id = f"{func.__module__}.{func.__name__}"
            
            allowed, remaining = limiter.is_allowed(client_id)
            if not allowed:
                raise RuntimeError(f"Rate limit exceeded for {func.__name__}. Try again later.")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-configured validators
security_validator = SecurityValidator()
param_validator = ParameterValidator()

# Global rate limiter
global_rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)  # 1000 requests per hour