"""Security utilities and safeguards."""

import os
import sys
import logging
import hashlib
import tempfile
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SecurityScan:
    """Result of security scanning."""
    
    passed: bool
    vulnerabilities: List[Dict[str, Any]]
    warnings: List[str]
    scan_time: float
    scan_id: str


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
        
        # Check for potential security issues
        security_patterns = [
            (r'eval\s*\(', 'Use of eval() function'),
            (r'exec\s*\(', 'Use of exec() function'),
            (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'Shell command injection risk'),
            (r'os\.system\s*\(', 'Use of os.system()'),
            (r'pickle\.loads?\s*\(', 'Use of pickle (deserialization risk)'),
        ]
        
        import re
        for pattern, description in security_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                vulnerabilities.append({
                    'type': 'pattern_match',
                    'severity': 'medium',
                    'description': f"{description} in {file_path}",
                    'file': file_path,
                    'recommendation': 'Review and validate usage'
                })
    
    except Exception as e:
        warnings.append(f"Could not scan {file_path}: {e}")