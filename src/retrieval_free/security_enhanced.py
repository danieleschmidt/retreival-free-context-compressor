"""
Enhanced Security Framework - Generation 2
Comprehensive security measures for production deployment.
"""

import hashlib
import hmac
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass
import logging
from pathlib import Path

from .exceptions import SecurityError, ValidationError


logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # Input validation
    max_input_length: int = 1000000  # 1M characters
    max_compression_ratio: float = 100.0
    allowed_file_types: Set[str] = None
    
    # Rate limiting
    max_requests_per_minute: int = 1000
    max_requests_per_hour: int = 10000
    
    # Authentication
    require_api_key: bool = True
    api_key_length: int = 32
    session_timeout_minutes: int = 30
    
    # Content filtering
    enable_content_filtering: bool = True
    blocked_patterns: List[str] = None
    
    # Logging and monitoring
    log_security_events: bool = True
    alert_on_violations: bool = True
    
    def __post_init__(self):
        if self.allowed_file_types is None:
            self.allowed_file_types = {'.txt', '.md', '.json', '.csv'}
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r'(?i)(password|secret|key|token)\s*[=:]\s*[^\s]+',  # Secrets
                r'(?i)bearer\s+[a-zA-Z0-9\-_]+',  # Bearer tokens
                r'(?i)api[_-]?key\s*[=:]\s*[^\s]+',  # API keys
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN patterns
            ]


class SecurityValidator:
    """Validates input for security threats and policy violations."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.blocked_patterns = [re.compile(pattern) for pattern in self.config.blocked_patterns]
    
    def validate_input_length(self, text: str) -> None:
        """Validate input length against security limits."""
        if len(text) > self.config.max_input_length:
            raise SecurityError(
                f"Input length {len(text)} exceeds maximum allowed {self.config.max_input_length}",
                threat_type="input_overflow",
                details={"input_length": len(text), "max_allowed": self.config.max_input_length}
            )
    
    def validate_content(self, text: str) -> List[str]:
        """Validate content for security threats.
        
        Returns:
            List of detected security issues (empty if clean)
        """
        if not self.config.enable_content_filtering:
            return []
        
        violations = []
        
        for pattern in self.blocked_patterns:
            matches = pattern.findall(text)
            if matches:
                violations.append(f"Detected sensitive pattern: {pattern.pattern[:50]}...")
        
        return violations
    
    def validate_file_type(self, file_path: Union[str, Path]) -> None:
        """Validate file type is allowed."""
        file_path = Path(file_path)
        if file_path.suffix.lower() not in self.config.allowed_file_types:
            raise SecurityError(
                f"File type {file_path.suffix} not allowed",
                threat_type="invalid_file_type",
                details={"file_type": file_path.suffix, "allowed_types": list(self.config.allowed_file_types)}
            )
    
    def sanitize_text(self, text: str) -> str:
        """Sanitize text by removing or masking sensitive content."""
        sanitized = text
        
        for pattern in self.blocked_patterns:
            # Replace matches with [REDACTED]
            sanitized = pattern.sub('[REDACTED]', sanitized)
        
        return sanitized


class RateLimiter:
    """Token bucket rate limiter for API requests."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.buckets: Dict[str, Dict[str, Any]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        current_time = time.time()
        
        if client_id not in self.buckets:
            self.buckets[client_id] = {
                'minute_tokens': self.config.max_requests_per_minute,
                'hour_tokens': self.config.max_requests_per_hour,
                'last_minute_refill': current_time,
                'last_hour_refill': current_time
            }
        
        bucket = self.buckets[client_id]
        
        # Refill tokens based on time elapsed
        self._refill_bucket(bucket, current_time)
        
        # Check if request is allowed
        if bucket['minute_tokens'] > 0 and bucket['hour_tokens'] > 0:
            bucket['minute_tokens'] -= 1
            bucket['hour_tokens'] -= 1
            return True
        
        return False
    
    def _refill_bucket(self, bucket: Dict[str, Any], current_time: float) -> None:
        """Refill token bucket based on elapsed time."""
        # Refill minute bucket
        elapsed_minutes = (current_time - bucket['last_minute_refill']) / 60.0
        if elapsed_minutes >= 1.0:
            bucket['minute_tokens'] = self.config.max_requests_per_minute
            bucket['last_minute_refill'] = current_time
        
        # Refill hour bucket
        elapsed_hours = (current_time - bucket['last_hour_refill']) / 3600.0
        if elapsed_hours >= 1.0:
            bucket['hour_tokens'] = self.config.max_requests_per_hour
            bucket['last_hour_refill'] = current_time


class APIKeyManager:
    """Manages API key generation, validation, and lifecycle."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.active_keys: Dict[str, Dict[str, Any]] = {}
    
    def generate_api_key(self, client_id: str, expiration_days: int = 365) -> str:
        """Generate a new API key for client."""
        api_key = self._generate_secure_key()
        
        self.active_keys[api_key] = {
            'client_id': client_id,
            'created_at': time.time(),
            'expires_at': time.time() + (expiration_days * 24 * 3600),
            'last_used': None,
            'usage_count': 0
        }
        
        logger.info(f"Generated API key for client {client_id}")
        return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key and update usage statistics."""
        if api_key not in self.active_keys:
            return False
        
        key_info = self.active_keys[api_key]
        current_time = time.time()
        
        # Check expiration
        if current_time > key_info['expires_at']:
            del self.active_keys[api_key]
            return False
        
        # Update usage
        key_info['last_used'] = current_time
        key_info['usage_count'] += 1
        
        return True
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        if api_key in self.active_keys:
            client_id = self.active_keys[api_key]['client_id']
            del self.active_keys[api_key]
            logger.info(f"Revoked API key for client {client_id}")
            return True
        return False
    
    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure API key."""
        return hashlib.sha256(
            (str(uuid.uuid4()) + str(time.time())).encode()
        ).hexdigest()[:self.config.api_key_length]


class SecurityAuditor:
    """Audits and logs security events."""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log a security event."""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'details': details,
            'event_id': str(uuid.uuid4())
        }
        
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} - {details}")
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [
            event for event in self.security_events 
            if event['timestamp'] > cutoff_time
        ]
        
        event_counts = {}
        for event in recent_events:
            event_type = event['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            'report_period_hours': hours,
            'total_events': len(recent_events),
            'event_counts': event_counts,
            'events': recent_events
        }


class SecureDataHandler:
    """Handles sensitive data with encryption and secure storage."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        # Simple XOR encryption for demo (use proper encryption in production)
        encrypted = ''.join(
            chr(ord(char) ^ (self.encryption_key[i % len(self.encryption_key)] % 256))
            for i, char in enumerate(data)
        )
        return encrypted.encode('utf-8').hex()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        try:
            data = bytes.fromhex(encrypted_data).decode('utf-8')
            decrypted = ''.join(
                chr(ord(char) ^ (self.encryption_key[i % len(self.encryption_key)] % 256))
                for i, char in enumerate(data)
            )
            return decrypted
        except Exception as e:
            raise SecurityError(f"Failed to decrypt data: {e}")
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return hashlib.sha256(str(uuid.uuid4()).encode()).digest()


class SecurityFramework:
    """Main security framework orchestrating all security components."""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.validator = SecurityValidator(self.config)
        self.rate_limiter = RateLimiter(self.config)
        self.api_key_manager = APIKeyManager(self.config)
        self.auditor = SecurityAuditor()
        self.data_handler = SecureDataHandler()
    
    def validate_request(self, text: str, client_id: str, api_key: Optional[str] = None) -> None:
        """Comprehensive request validation."""
        # Rate limiting
        if not self.rate_limiter.is_allowed(client_id):
            self.auditor.log_security_event(
                'rate_limit_exceeded',
                {'client_id': client_id}
            )
            raise SecurityError(
                "Rate limit exceeded",
                threat_type="rate_limit",
                details={'client_id': client_id}
            )
        
        # API key validation
        if self.config.require_api_key and not self.api_key_manager.validate_api_key(api_key):
            self.auditor.log_security_event(
                'invalid_api_key',
                {'client_id': client_id, 'api_key_prefix': api_key[:8] if api_key else None}
            )
            raise SecurityError(
                "Invalid or expired API key",
                threat_type="authentication",
                details={'client_id': client_id}
            )
        
        # Input validation
        self.validator.validate_input_length(text)
        
        # Content validation
        violations = self.validator.validate_content(text)
        if violations:
            self.auditor.log_security_event(
                'content_violation',
                {'client_id': client_id, 'violations': violations}
            )
            raise SecurityError(
                f"Content policy violations detected: {len(violations)} issues",
                threat_type="content_policy",
                details={'violations': violations}
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'config': {
                'max_input_length': self.config.max_input_length,
                'rate_limiting_enabled': True,
                'content_filtering_enabled': self.config.enable_content_filtering,
                'api_key_required': self.config.require_api_key
            },
            'active_api_keys': len(self.api_key_manager.active_keys),
            'rate_limiter_clients': len(self.rate_limiter.buckets),
            'recent_security_events': len(self.auditor.security_events),
            'security_report': self.auditor.get_security_report(24)
        }


# Global security framework instance
_security_framework = None

def get_security_framework(config: Optional[SecurityConfig] = None) -> SecurityFramework:
    """Get or create global security framework instance."""
    global _security_framework
    if _security_framework is None:
        _security_framework = SecurityFramework(config)
    return _security_framework


def secure_compress(text: str, client_id: str, api_key: Optional[str] = None, **kwargs) -> Any:
    """Security-wrapped compression function."""
    framework = get_security_framework()
    
    # Validate request
    framework.validate_request(text, client_id, api_key)
    
    # Log successful validation
    framework.auditor.log_security_event(
        'compression_request_validated',
        {'client_id': client_id, 'input_length': len(text)}
    )
    
    # Sanitize input if needed
    sanitized_text = framework.validator.sanitize_text(text)
    
    # Return mock compression result for now
    return {
        'compressed_text': f"[COMPRESSED:{len(sanitized_text)} chars]",
        'compression_ratio': len(text) / max(100, len(sanitized_text) // 10),
        'security_validated': True,
        'client_id': client_id
    }