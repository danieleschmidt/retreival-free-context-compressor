"""
Secure Configuration Management for Retrieval-Free Compression System

This module provides secure configuration management with environment variable
support and validation.
"""

import os
import json
import secrets
from pathlib import Path
from typing import Any, Dict, Optional


class SecureConfig:
    """Secure configuration manager with environment variable support."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path.cwd() / ".env"
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from environment variables."""
        # Default secure settings
        self._config = {
            "COMPRESSION_MAX_MEMORY": os.getenv("COMPRESSION_MAX_MEMORY", "8GB"),
            "COMPRESSION_TIMEOUT": int(os.getenv("COMPRESSION_TIMEOUT", "300")),
            "ENABLE_TELEMETRY": os.getenv("ENABLE_TELEMETRY", "false").lower() == "true",
            "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
            "SECURE_MODE": os.getenv("SECURE_MODE", "true").lower() == "true",
        }
        
        # Generate secure session key if not provided
        if "SESSION_KEY" not in os.environ:
            self._config["SESSION_KEY"] = secrets.token_hex(32)
        else:
            self._config["SESSION_KEY"] = os.getenv("SESSION_KEY")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value safely."""
        return self._config.get(key, default)
    
    def validate_config(self) -> bool:
        """Validate configuration for security compliance."""
        if not self._config.get("SECURE_MODE", True):
            raise ValueError("Secure mode must be enabled in production")
        
        if len(self._config.get("SESSION_KEY", "")) < 32:
            raise ValueError("Session key must be at least 32 characters")
        
        return True


# Global secure config instance
secure_config = SecureConfig()
