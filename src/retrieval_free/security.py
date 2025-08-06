"""Security utilities and safeguards."""

import os
import sys
import logging
import hashlib
import tempfile
from typing import Dict, Any, List, Optional, Set, Callable
from pathlib import Path
import json
import time
from dataclasses import dataclass, field
import secrets
import hmac
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
from functools import wraps
import ipaddress
import re

logger = logging.getLogger(__name__)

# PII Detection patterns
PII_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    'passport': r'\b[A-Z]{1,2}\d{6,9}\b',
    'driver_license': r'\b[A-Z]{1,2}\d{6,8}\b',
    'bank_account': r'\b\d{8,17}\b',
    'date_of_birth': r'\b(0?[1-9]|1[0-2])[\\/\\-](0?[1-9]|[12][0-9]|3[01])[\\/\\-]\d{4}\b'
}

# Malicious pattern detection
MALICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # Script injection
    r'javascript:\s*[\w\(\)]+',   # JavaScript URLs
    r'data:.*base64',             # Base64 data URLs
    r'\beval\s*\(',              # eval() calls
    r'\bexec\s*\(',              # exec() calls
    r'\b__import__\s*\(',        # Dynamic imports
    r'\bos\.system\s*\(',        # System calls
    r'subprocess\.',             # Subprocess calls
    r'\bopen\s*\(',              # File operations
    r'pickle\.loads?\s*\(',      # Pickle deserialization
]


@dataclass
class SecurityScan:
    """Result of security scanning."""
    
    passed: bool
    vulnerabilities: List[Dict[str, Any]]
    warnings: List[str]
    scan_time: float
    scan_id: str


@dataclass
class APIKey:
    """API Key information."""
    key_id: str
    key_hash: str
    permissions: Set[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit_requests: int = 100  # requests per minute
    rate_limit_window: int = 60     # seconds
    is_active: bool = True
    

@dataclass 
class RateLimitInfo:
    """Rate limit tracking information."""
    requests: deque = field(default_factory=deque)
    last_request: Optional[datetime] = None
    blocked_until: Optional[datetime] = None
    total_requests: int = 0
    blocked_requests: int = 0


@dataclass
class PIIDetectionResult:
    """PII detection result."""
    detected_types: List[str]
    locations: Dict[str, List[int]]  # type -> list of character positions
    masked_text: str
    risk_level: str  # low, medium, high
    recommendations: List[str]


class ModelSecurityValidator:
    """Validator for model security and integrity."""
    
    def __init__(self):
        """Initialize model security validator."""
        self.trusted_sources = {
            'huggingface.co',
            'pytorch.org', 
            'tensorflow.org',
            'github.com',
            'localhost'  # For local development
        }
        
        self.known_model_hashes: Dict[str, str] = {
            # Add known good model hashes here
            'rfcc-base-8x': 'placeholder_hash',
            'context-compressor-base': 'placeholder_hash'
        }
        
        # Malicious model signatures (simplified examples)
        self.malicious_signatures = {
            'backdoor_trigger_1': 'deadbeef',
            'data_exfiltration_1': 'cafebabe',
            'model_poisoning_1': 'feedface'
        }
    
    def validate_model_source(self, model_path: str) -> SecurityScan:
        """Validate model source and integrity.
        
        Args:
            model_path: Path or URL to model
            
        Returns:
            SecurityScan result
        """
        scan_id = hashlib.md5(f"{model_path}{time.time()}".encode()).hexdigest()[:8]
        start_time = time.time()
        
        vulnerabilities = []
        warnings = []
        
        # Check if it's a URL or local path
        if model_path.startswith(('http://', 'https://')):
            # URL validation
            from urllib.parse import urlparse
            
            parsed = urlparse(model_path)
            domain = parsed.netloc.lower()
            
            # Check against trusted sources
            is_trusted = any(trusted in domain for trusted in self.trusted_sources)
            
            if not is_trusted:
                vulnerabilities.append({
                    'type': 'untrusted_source',
                    'severity': 'high',
                    'description': f"Model source not in trusted list: {domain}",
                    'recommendation': 'Use models from trusted sources only'
                })
            
            # Check for suspicious URLs
            if any(suspicious in model_path.lower() for suspicious in ['temp', 'tmp', 'cache']):
                warnings.append("Model URL contains temporary path indicators")
            
        else:
            # Local path validation
            if not os.path.exists(model_path):
                vulnerabilities.append({
                    'type': 'missing_file',
                    'severity': 'high', 
                    'description': f"Model file not found: {model_path}",
                    'recommendation': 'Verify model path and permissions'
                })
            else:
                # Check file permissions
                path_obj = Path(model_path)
                if path_obj.is_file():
                    # Check if file is world-writable (security risk)
                    stat_info = path_obj.stat()
                    if stat_info.st_mode & 0o002:  # World writable
                        warnings.append("Model file is world-writable")
        
        scan_time = time.time() - start_time
        passed = len(vulnerabilities) == 0
        
        return SecurityScan(
            passed=passed,
            vulnerabilities=vulnerabilities,
            warnings=warnings,
            scan_time=scan_time,
            scan_id=scan_id
        )
    
    def verify_model_checksum(self, model_path: str, expected_hash: Optional[str] = None) -> bool:
        """Verify model file integrity using checksums.
        
        Args:
            model_path: Path to model file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if checksum matches or no expected hash provided
        """
        if not os.path.exists(model_path):
            return False
        
        # Calculate SHA-256 hash
        sha256_hash = hashlib.sha256()
        try:
            with open(model_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            calculated_hash = sha256_hash.hexdigest()
            
            # If no expected hash, just log the calculated one
            if expected_hash is None:
                logger.info(f"Model hash for {model_path}: {calculated_hash}")
                return True
            
            # Compare with expected hash
            return calculated_hash == expected_hash
            
        except Exception as e:
            logger.error(f"Error calculating checksum for {model_path}: {e}")
            return False


class SandboxedExecution:
    """Sandboxed execution environment for model operations."""
    
    def __init__(
        self,
        max_memory_mb: int = 4096,
        max_execution_time: int = 300,
        allowed_imports: Optional[Set[str]] = None
    ):
        """Initialize sandboxed execution.
        
        Args:
            max_memory_mb: Maximum memory usage
            max_execution_time: Maximum execution time in seconds
            allowed_imports: Set of allowed import modules
        """
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.allowed_imports = allowed_imports or {
            'torch', 'transformers', 'numpy', 'sklearn', 
            'sentence_transformers', 'datasets', 'tqdm',
            'einops', 'logging', 'json', 'os', 'sys'
        }
        
        # Resource monitoring
        self._start_memory = None
        self._start_time = None
    
    def __enter__(self):
        """Enter sandbox context."""
        self._start_time = time.time()
        
        # Get initial memory usage
        try:
            import psutil
            process = psutil.Process()
            self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            self._start_memory = 0
        
        # Install import hook for validation
        self._original_import = __builtins__.__import__
        __builtins__.__import__ = self._safe_import
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit sandbox context."""
        # Restore original import
        __builtins__.__import__ = self._original_import
        
        # Check resource usage
        if self._start_time:
            execution_time = time.time() - self._start_time
            if execution_time > self.max_execution_time:
                logger.warning(f"Execution time exceeded limit: {execution_time:.1f}s")
        
        try:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - (self._start_memory or 0)
            
            if memory_used > self.max_memory_mb:
                logger.warning(f"Memory usage exceeded limit: {memory_used:.1f}MB")
        except ImportError:
            pass
    
    def _safe_import(self, name, *args, **kwargs):
        """Safe import function that validates allowed modules.
        
        Args:
            name: Module name to import
            *args: Additional import arguments
            **kwargs: Additional import keyword arguments
            
        Returns:
            Imported module
            
        Raises:
            ImportError: If module not in allowed list
        """
        # Check if module is allowed
        base_module = name.split('.')[0]
        
        if base_module not in self.allowed_imports:
            logger.warning(f"Blocked import attempt: {name}")
            raise ImportError(f"Import of '{name}' not allowed in sandbox")
        
        # Use original import
        return self._original_import(name, *args, **kwargs)


class SecureStorage:
    """Secure storage for sensitive data and keys."""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize secure storage.
        
        Args:
            storage_dir: Directory for secure storage
        """
        if storage_dir is None:
            storage_dir = os.path.join(tempfile.gettempdir(), 'retrieval_free_secure')
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(mode=0o700, parents=True, exist_ok=True)  # Owner only
    
    def store_api_key(self, service: str, api_key: str) -> bool:
        """Securely store API key.
        
        Args:
            service: Service name
            api_key: API key to store
            
        Returns:
            True if stored successfully
        """
        try:
            key_file = self.storage_dir / f"{service}_key"
            
            # Simple encryption (in production, use proper encryption)
            encrypted_key = self._simple_encrypt(api_key)
            
            with open(key_file, 'w', mode=0o600) as f:  # Owner read/write only
                f.write(encrypted_key)
            
            logger.info(f"Stored API key for {service}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store API key for {service}: {e}")
            return False
    
    def retrieve_api_key(self, service: str) -> Optional[str]:
        """Retrieve stored API key.
        
        Args:
            service: Service name
            
        Returns:
            Decrypted API key or None if not found
        """
        try:
            key_file = self.storage_dir / f"{service}_key"
            
            if not key_file.exists():
                return None
            
            with open(key_file, 'r') as f:
                encrypted_key = f.read().strip()
            
            return self._simple_decrypt(encrypted_key)
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {service}: {e}")
            return None
    
    def _simple_encrypt(self, data: str) -> str:
        """Simple encryption (placeholder - use proper encryption in production).
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        # This is a placeholder - use proper encryption like Fernet in production
        import base64
        return base64.b64encode(data.encode()).decode()
    
    def _simple_decrypt(self, encrypted_data: str) -> str:
        """Simple decryption (placeholder - use proper decryption in production).
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        # This is a placeholder - use proper decryption like Fernet in production
        import base64
        return base64.b64decode(encrypted_data.encode()).decode()


class AuditLogger:
    """Audit logger for security events."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Path to audit log file
        """
        if log_file is None:
            log_dir = Path.home() / '.retrieval_free' / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / 'audit.log'
        
        self.log_file = Path(log_file)
        
        # Set up audit logger
        self.audit_logger = logging.getLogger('retrieval_free.audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.audit_logger.addHandler(file_handler)
    
    def log_model_load(self, model_name: str, source: str, user: str = None) -> None:
        """Log model loading event.
        
        Args:
            model_name: Name of loaded model
            source: Source of model (path/URL)
            user: User identifier
        """
        self.audit_logger.info(
            f"MODEL_LOAD - model={model_name}, source={source}, user={user or 'unknown'}"
        )
    
    def log_compression_request(
        self, 
        text_length: int, 
        model: str, 
        parameters: Dict[str, Any],
        user: str = None
    ) -> None:
        """Log compression request.
        
        Args:
            text_length: Length of input text
            model: Model used for compression
            parameters: Compression parameters
            user: User identifier
        """
        self.audit_logger.info(
            f"COMPRESSION_REQUEST - length={text_length}, model={model}, "
            f"params={json.dumps(parameters)}, user={user or 'unknown'}"
        )
    
    def log_security_event(
        self, 
        event_type: str, 
        description: str, 
        severity: str = "info"
    ) -> None:
        """Log security event.
        
        Args:
            event_type: Type of security event
            description: Event description  
            severity: Event severity (info, warning, error, critical)
        """
        level_map = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        
        level = level_map.get(severity.lower(), logging.INFO)
        
        self.audit_logger.log(
            level,
            f"SECURITY_EVENT - type={event_type}, severity={severity}, desc={description}"
        )


class AuthenticationManager:
    """Manages API keys and authentication."""
    
    def __init__(self):
        """Initialize authentication manager."""
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self._lock = threading.RLock()
    
    def generate_api_key(
        self,
        permissions: Optional[Set[str]] = None,
        rate_limit_requests: int = 100,
        rate_limit_window: int = 60
    ) -> Tuple[str, str]:
        """Generate a new API key.
        
        Args:
            permissions: Set of permissions for this key
            rate_limit_requests: Requests per window
            rate_limit_window: Window size in seconds
            
        Returns:
            Tuple of (key_id, api_key)
        """
        with self._lock:
            key_id = secrets.token_urlsafe(8)
            api_key = secrets.token_urlsafe(32)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            permissions = permissions or {'compress', 'decompress', 'health'}
            
            self.api_keys[key_id] = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                permissions=permissions,
                created_at=datetime.now(),
                rate_limit_requests=rate_limit_requests,
                rate_limit_window=rate_limit_window
            )
            
            logger.info(f"Generated API key: {key_id}")
            return key_id, api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key.
        
        Args:
            api_key: The API key to validate
            
        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with self._lock:
            for key_info in self.api_keys.values():
                if key_info.key_hash == key_hash and key_info.is_active:
                    key_info.last_used = datetime.now()
                    key_info.usage_count += 1
                    return key_info
        
        logger.warning(f"Invalid API key attempted: {key_hash[:8]}...")
        return None
    
    def check_rate_limit(self, key_id: str, api_key_info: APIKey) -> bool:
        """Check if request is within rate limit.
        
        Args:
            key_id: API key ID
            api_key_info: API key information
            
        Returns:
            True if within rate limit, False if exceeded
        """
        with self._lock:
            rate_info = self.rate_limits[key_id]
            current_time = datetime.now()
            
            # Check if currently blocked
            if rate_info.blocked_until and current_time < rate_info.blocked_until:
                rate_info.blocked_requests += 1
                return False
            
            # Clean up old requests outside the window
            window_start = current_time - timedelta(seconds=api_key_info.rate_limit_window)
            while rate_info.requests and rate_info.requests[0] < window_start:
                rate_info.requests.popleft()
            
            # Check if over limit
            if len(rate_info.requests) >= api_key_info.rate_limit_requests:
                # Block for the rate limit window
                rate_info.blocked_until = current_time + timedelta(seconds=api_key_info.rate_limit_window)
                rate_info.blocked_requests += 1
                logger.warning(f"Rate limit exceeded for key: {key_id}")
                return False
            
            # Add current request
            rate_info.requests.append(current_time)
            rate_info.total_requests += 1
            rate_info.last_request = current_time
            
            return True
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key.
        
        Args:
            key_id: API key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        with self._lock:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = False
                logger.info(f"Revoked API key: {key_id}")
                return True
            return False
    
    def get_usage_stats(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            Usage statistics dictionary
        """
        with self._lock:
            if key_id not in self.api_keys:
                return None
            
            key_info = self.api_keys[key_id]
            rate_info = self.rate_limits[key_id]
            
            return {
                'key_id': key_id,
                'created_at': key_info.created_at.isoformat(),
                'last_used': key_info.last_used.isoformat() if key_info.last_used else None,
                'usage_count': key_info.usage_count,
                'is_active': key_info.is_active,
                'permissions': list(key_info.permissions),
                'rate_limit': {
                    'requests_per_window': key_info.rate_limit_requests,
                    'window_seconds': key_info.rate_limit_window,
                    'current_requests': len(rate_info.requests),
                    'total_requests': rate_info.total_requests,
                    'blocked_requests': rate_info.blocked_requests,
                    'blocked_until': rate_info.blocked_until.isoformat() if rate_info.blocked_until else None
                }
            }


class EnhancedInputSanitizer:
    """Enhanced input sanitization with PII detection and malicious content filtering."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        self.pii_patterns = PII_PATTERNS
        self.malicious_patterns = MALICIOUS_PATTERNS
    
    def sanitize_input(self, text: str, mask_pii: bool = True) -> Dict[str, Any]:
        """Sanitize input text.
        
        Args:
            text: Input text to sanitize
            mask_pii: Whether to mask detected PII
            
        Returns:
            Dictionary with sanitization results
        """
        # Detect PII
        pii_result = self.detect_pii(text)
        
        # Detect malicious content
        malicious_result = self.detect_malicious_content(text)
        
        # Sanitize text
        sanitized_text = text
        
        # Mask PII if requested
        if mask_pii:
            sanitized_text = pii_result.masked_text
        
        # Remove malicious content
        for pattern in self.malicious_patterns:
            sanitized_text = re.sub(pattern, '[REMOVED]', sanitized_text, flags=re.IGNORECASE | re.DOTALL)
        
        return {
            'original_length': len(text),
            'sanitized_text': sanitized_text,
            'sanitized_length': len(sanitized_text),
            'pii_detected': pii_result.detected_types,
            'pii_risk_level': pii_result.risk_level,
            'malicious_patterns_found': malicious_result['patterns_found'],
            'risk_score': self._calculate_risk_score(pii_result, malicious_result),
            'recommendations': pii_result.recommendations + malicious_result.get('recommendations', [])
        }
    
    def detect_pii(self, text: str) -> PIIDetectionResult:
        """Detect personally identifiable information.
        
        Args:
            text: Text to scan for PII
            
        Returns:
            PIIDetectionResult with detection details
        """
        detected_types = []
        locations = {}
        masked_text = text
        recommendations = []
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            if matches:
                detected_types.append(pii_type)
                locations[pii_type] = [match.start() for match in matches]
                
                # Mask the PII in text
                for match in reversed(matches):  # Reverse to maintain positions
                    mask_length = len(match.group())
                    mask = '[' + pii_type.upper() + '_MASKED]'
                    masked_text = masked_text[:match.start()] + mask + masked_text[match.end():]
                
                recommendations.append(f"Consider removing or further anonymizing {pii_type} data")
        
        # Determine risk level
        high_risk_types = {'ssn', 'credit_card', 'bank_account', 'passport'}
        medium_risk_types = {'email', 'phone', 'driver_license', 'date_of_birth'}
        
        risk_level = 'low'
        if any(pii_type in high_risk_types for pii_type in detected_types):
            risk_level = 'high'
        elif any(pii_type in medium_risk_types for pii_type in detected_types):
            risk_level = 'medium'
        
        return PIIDetectionResult(
            detected_types=detected_types,
            locations=locations,
            masked_text=masked_text,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def detect_malicious_content(self, text: str) -> Dict[str, Any]:
        """Detect potentially malicious content.
        
        Args:
            text: Text to scan
            
        Returns:
            Dictionary with detection results
        """
        patterns_found = []
        recommendations = []
        
        for pattern in self.malicious_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                patterns_found.append(pattern)
                recommendations.append(f"Suspicious pattern detected: {pattern}")
        
        return {
            'patterns_found': patterns_found,
            'risk_level': 'high' if patterns_found else 'low',
            'recommendations': recommendations
        }
    
    def _calculate_risk_score(self, pii_result: PIIDetectionResult, malicious_result: Dict[str, Any]) -> float:
        """Calculate overall risk score.
        
        Args:
            pii_result: PII detection result
            malicious_result: Malicious content detection result
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        score = 0.0
        
        # PII risk scoring
        pii_weights = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6
        }
        score += pii_weights.get(pii_result.risk_level, 0.0)
        
        # Malicious content scoring
        if malicious_result['patterns_found']:
            score += 0.4  # High penalty for malicious patterns
        
        return min(score, 1.0)


def require_authentication(permissions: Optional[Set[str]] = None):
    """Decorator to require API key authentication.
    
    Args:
        permissions: Required permissions for this endpoint
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from kwargs or environment
            api_key = kwargs.pop('api_key', None) or os.environ.get('RETRIEVAL_FREE_API_KEY')
            
            if not api_key:
                raise SecurityError("API key required", error_code="AUTH_MISSING_KEY")
            
            # Get authentication manager
            auth_manager = get_auth_manager()
            
            # Validate API key
            key_info = auth_manager.validate_api_key(api_key)
            if not key_info:
                raise SecurityError("Invalid API key", error_code="AUTH_INVALID_KEY")
            
            # Check permissions
            if permissions and not permissions.issubset(key_info.permissions):
                missing = permissions - key_info.permissions
                raise SecurityError(
                    f"Insufficient permissions. Missing: {missing}",
                    error_code="AUTH_INSUFFICIENT_PERMISSIONS"
                )
            
            # Check rate limit
            if not auth_manager.check_rate_limit(key_info.key_id, key_info):
                raise SecurityError("Rate limit exceeded", error_code="AUTH_RATE_LIMIT_EXCEEDED")
            
            # Add key info to kwargs for use in function
            kwargs['_api_key_info'] = key_info
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


class SecurityError(Exception):
    """Security-related error."""
    
    def __init__(self, message: str, error_code: str = "SECURITY_ERROR"):
        super().__init__(message)
        self.error_code = error_code


def scan_for_vulnerabilities(
    code_dir: str,
    exclude_patterns: Optional[List[str]] = None
) -> SecurityScan:
    """Scan codebase for security vulnerabilities.
    
    Args:
        code_dir: Directory to scan
        exclude_patterns: Patterns to exclude from scan
        
    Returns:
        SecurityScan result
    """
    scan_id = hashlib.md5(f"{code_dir}{time.time()}".encode()).hexdigest()[:8]
    start_time = time.time()
    
    vulnerabilities = []
    warnings = []
    
    exclude_patterns = exclude_patterns or ['*test*', '*__pycache__*', '*.pyc']
    
    try:
        # Use bandit for Python security scanning if available
        try:
            import subprocess
            import json
            
            result = subprocess.run([
                'bandit', '-r', code_dir, '-f', 'json',
                '--exclude', ','.join(exclude_patterns)
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Parse bandit output
                bandit_results = json.loads(result.stdout)
                
                for issue in bandit_results.get('results', []):
                    vulnerabilities.append({
                        'type': 'bandit_finding',
                        'severity': issue.get('issue_severity', 'medium').lower(),
                        'description': issue.get('issue_text', ''),
                        'file': issue.get('filename', ''),
                        'line': issue.get('line_number', 0),
                        'recommendation': 'Review flagged code for security issues'
                    })
            
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
            # Bandit not available, do basic checks
            warnings.append("Bandit security scanner not available")
            
            # Basic pattern-based checks
            for root, dirs, files in os.walk(code_dir):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        _check_file_for_issues(file_path, vulnerabilities, warnings)
    
    except Exception as e:
        warnings.append(f"Error during vulnerability scan: {e}")
    
    scan_time = time.time() - start_time
    passed = len(vulnerabilities) == 0
    
    return SecurityScan(
        passed=passed,
        vulnerabilities=vulnerabilities,
        warnings=warnings,
        scan_time=scan_time,
        scan_id=scan_id
    )


def _check_file_for_issues(
    file_path: str, 
    vulnerabilities: List[Dict[str, Any]], 
    warnings: List[str]
) -> None:
    """Basic security checks for a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for potential security issues using enhanced patterns
        for pattern in MALICIOUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                vulnerabilities.append({
                    'type': 'pattern_match',
                    'severity': 'high',
                    'description': f"Suspicious pattern '{pattern}' found in {file_path}",
                    'file': file_path,
                    'recommendation': 'Review and validate usage'
                })
    
    except Exception as e:
        warnings.append(f"Could not scan {file_path}: {e}")


# Global instances
_auth_manager: Optional[AuthenticationManager] = None
_input_sanitizer: Optional[EnhancedInputSanitizer] = None


def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager.
    
    Returns:
        AuthenticationManager instance
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


def get_input_sanitizer() -> EnhancedInputSanitizer:
    """Get global input sanitizer.
    
    Returns:
        EnhancedInputSanitizer instance
    """
    global _input_sanitizer
    if _input_sanitizer is None:
        _input_sanitizer = EnhancedInputSanitizer()
    return _input_sanitizer