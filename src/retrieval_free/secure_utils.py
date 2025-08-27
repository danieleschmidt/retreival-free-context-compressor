"""
Secure utilities for the retrieval-free compression system.

This module provides secure alternatives to potentially dangerous operations.
"""

import hashlib
import secrets
import subprocess
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


class SecureSerializer:
    """Secure serialization utilities."""
    
    @staticmethod
    def safe_json_loads(data: str) -> Any:
        """Safely load JSON data with size limits."""
        if len(data) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("JSON data too large")
        
        try:
            return json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON data: {e}")
    
    @staticmethod
    def safe_json_dumps(data: Any) -> str:
        """Safely dump data to JSON."""
        try:
            return json.dumps(data, ensure_ascii=True, separators=(',', ':'))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot serialize to JSON: {e}")


class SecureRandom:
    """Secure random number generation."""
    
    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_bytes(length: int = 32) -> bytes:
        """Generate secure random bytes."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def choose_secure(choices: List[Any]) -> Any:
        """Securely choose from a list."""
        if not choices:
            raise ValueError("Cannot choose from empty list")
        return secrets.choice(choices)


class SecureHash:
    """Secure hashing utilities."""
    
    @staticmethod
    def hash_data(data: Union[str, bytes], algorithm: str = "sha256") -> str:
        """Securely hash data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hasher = hashlib.new(algorithm)
        hasher.update(data)
        return hasher.hexdigest()
    
    @staticmethod
    def verify_hash(data: Union[str, bytes], expected_hash: str, algorithm: str = "sha256") -> bool:
        """Verify data against expected hash."""
        actual_hash = SecureHash.hash_data(data, algorithm)
        return secrets.compare_digest(actual_hash, expected_hash)


class SecureSubprocess:
    """Secure subprocess execution."""
    
    @staticmethod
    def run_command(command: List[str], timeout: int = 30, cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run command securely with proper argument handling."""
        if not command or not isinstance(command, list):
            raise ValueError("Command must be a non-empty list")
        
        # Validate command components
        for arg in command:
            if not isinstance(arg, str):
                raise ValueError("All command arguments must be strings")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                shell=False,  # Never use shell=True
                check=False
            )
            
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Command timed out after {timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Command execution failed: {e}")


class InputValidator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_string(value: str, max_length: int = 1000, allow_empty: bool = False) -> str:
        """Validate string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        if not allow_empty and not value.strip():
            raise ValueError("Input cannot be empty")
        
        if len(value) > max_length:
            raise ValueError(f"Input too long (max {max_length} characters)")
        
        return value.strip()
    
    @staticmethod
    def validate_path(path: Union[str, Path], must_exist: bool = False) -> Path:
        """Validate file path for security."""
        path_obj = Path(path)
        
        # Check for path traversal
        if ".." in str(path_obj):
            raise ValueError("Path traversal not allowed")
        
        # Check for absolute paths outside project
        if path_obj.is_absolute():
            # Allow only paths within reasonable system directories
            allowed_prefixes = ["/tmp", "/var/tmp", "/home", "/root/repo"]
            if not any(str(path_obj).startswith(prefix) for prefix in allowed_prefixes):
                raise ValueError("Absolute path not in allowed directories")
        
        if must_exist and not path_obj.exists():
            raise ValueError(f"Path does not exist: {path_obj}")
        
        return path_obj
