"""Custom exceptions for the retrieval-free context compressor."""

from typing import Any, Dict, List, Optional


class RetrievalFreeError(Exception):
    """Base exception for all retrieval-free compression errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context
        }


class ValidationError(RetrievalFreeError):
    """Raised when input validation fails."""
    
    def __init__(
        self, 
        message: str, 
        validation_errors: List[str] = None,
        field: Optional[str] = None
    ):
        super().__init__(message, "VALIDATION_ERROR")
        self.validation_errors = validation_errors or []
        self.field = field
        self.context.update({
            "validation_errors": self.validation_errors,
            "field": field
        })


class SecurityError(RetrievalFreeError):
    """Raised when security validation fails."""
    
    def __init__(self, message: str, threat_type: Optional[str] = None):
        super().__init__(message, "SECURITY_ERROR")
        self.threat_type = threat_type
        self.context.update({"threat_type": threat_type})


class CompressionError(RetrievalFreeError):
    """Raised when compression operation fails."""
    
    def __init__(
        self, 
        message: str, 
        stage: Optional[str] = None,
        original_length: Optional[int] = None
    ):
        super().__init__(message, "COMPRESSION_ERROR")
        self.stage = stage
        self.original_length = original_length
        self.context.update({
            "stage": stage,
            "original_length": original_length
        })


class ModelError(RetrievalFreeError):
    """Raised when model loading or inference fails."""
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        model_type: Optional[str] = None
    ):
        super().__init__(message, "MODEL_ERROR")
        self.model_name = model_name
        self.model_type = model_type
        self.context.update({
            "model_name": model_name,
            "model_type": model_type
        })


class ConfigurationError(RetrievalFreeError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, "CONFIGURATION_ERROR")
        self.config_key = config_key
        self.context.update({"config_key": config_key})


class ResourceError(RetrievalFreeError):
    """Raised when system resources are insufficient."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        required: Optional[Any] = None,
        available: Optional[Any] = None
    ):
        super().__init__(message, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.required = required
        self.available = available
        self.context.update({
            "resource_type": resource_type,
            "required": required,
            "available": available
        })


class StreamingError(RetrievalFreeError):
    """Raised during streaming compression operations."""
    
    def __init__(
        self, 
        message: str, 
        window_id: Optional[int] = None,
        stream_position: Optional[int] = None
    ):
        super().__init__(message, "STREAMING_ERROR")
        self.window_id = window_id
        self.stream_position = stream_position
        self.context.update({
            "window_id": window_id,
            "stream_position": stream_position
        })


class RateLimitError(RetrievalFreeError):
    """Raised when rate limits are exceeded."""
    
    def __init__(
        self, 
        message: str, 
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        retry_after: Optional[int] = None
    ):
        super().__init__(message, "RATE_LIMIT_ERROR")
        self.limit = limit
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        self.context.update({
            "limit": limit,
            "window_seconds": window_seconds,
            "retry_after": retry_after
        })


class PluginError(RetrievalFreeError):
    """Raised when plugin integration fails."""
    
    def __init__(
        self, 
        message: str, 
        plugin_name: Optional[str] = None,
        framework: Optional[str] = None
    ):
        super().__init__(message, "PLUGIN_ERROR")
        self.plugin_name = plugin_name
        self.framework = framework
        self.context.update({
            "plugin_name": plugin_name,
            "framework": framework
        })


# Exception hierarchy for better error handling
RECOVERABLE_ERRORS = {
    RateLimitError,
    ResourceError,  # Sometimes recoverable
}

CRITICAL_ERRORS = {
    SecurityError,
    ModelError,
    ConfigurationError,
}

USER_ERRORS = {
    ValidationError,
    ConfigurationError,
}


def is_recoverable_error(error: Exception) -> bool:
    """Check if an error is potentially recoverable."""
    return type(error) in RECOVERABLE_ERRORS


def is_critical_error(error: Exception) -> bool:
    """Check if an error is critical and requires immediate attention."""
    return type(error) in CRITICAL_ERRORS


def is_user_error(error: Exception) -> bool:
    """Check if an error is caused by user input/configuration."""
    return type(error) in USER_ERRORS


def create_error_response(error: Exception) -> Dict[str, Any]:
    """Create a standardized error response."""
    if isinstance(error, RetrievalFreeError):
        return {
            "success": False,
            "error": error.to_dict(),
            "is_recoverable": is_recoverable_error(error),
            "is_critical": is_critical_error(error),
            "is_user_error": is_user_error(error)
        }
    else:
        return {
            "success": False,
            "error": {
                "error_type": type(error).__name__,
                "message": str(error),
                "error_code": "UNKNOWN_ERROR"
            },
            "is_recoverable": False,
            "is_critical": True,
            "is_user_error": False
        }