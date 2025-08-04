"""Custom exception classes for retrieval-free compression."""

from typing import Optional, Dict, Any, List


class RetrievalFreeError(Exception):
    """Base exception for retrieval-free compression errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize base exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }


class CompressionError(RetrievalFreeError):
    """Error during compression operation."""
    
    def __init__(
        self, 
        message: str, 
        input_length: Optional[int] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize compression error.
        
        Args:
            message: Error message
            input_length: Length of input that failed to compress
            model_name: Name of model that failed
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if input_length is not None:
            details['input_length'] = input_length
        if model_name is not None:
            details['model_name'] = model_name
        
        super().__init__(message, error_code="COMPRESSION_FAILED", details=details, **kwargs)


class ModelLoadError(RetrievalFreeError):
    """Error loading compression model."""
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize model load error.
        
        Args:
            message: Error message
            model_name: Name of model that failed to load
            model_path: Path to model that failed to load
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if model_name is not None:
            details['model_name'] = model_name
        if model_path is not None:
            details['model_path'] = model_path
        
        super().__init__(message, error_code="MODEL_LOAD_FAILED", details=details, **kwargs)


class ValidationError(RetrievalFreeError):
    """Error in input validation."""
    
    def __init__(
        self, 
        message: str, 
        validation_errors: Optional[List[str]] = None,
        field_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_errors: List of specific validation errors
            field_name: Name of field that failed validation
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if validation_errors is not None:
            details['validation_errors'] = validation_errors
        if field_name is not None:
            details['field_name'] = field_name
        
        super().__init__(message, error_code="VALIDATION_FAILED", details=details, **kwargs)


class SecurityError(RetrievalFreeError):
    """Security-related error."""
    
    def __init__(
        self, 
        message: str, 
        security_check: Optional[str] = None,
        risk_score: Optional[float] = None,
        **kwargs
    ):
        """Initialize security error.
        
        Args:
            message: Error message
            security_check: Name of security check that failed
            risk_score: Risk score (0.0 to 1.0)
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if security_check is not None:
            details['security_check'] = security_check
        if risk_score is not None:
            details['risk_score'] = risk_score
        
        super().__init__(message, error_code="SECURITY_VIOLATION", details=details, **kwargs)


class ResourceError(RetrievalFreeError):
    """Error related to resource limits or availability."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        requested: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs
    ):
        """Initialize resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, disk, gpu)
            requested: Amount requested
            available: Amount available
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if resource_type is not None:
            details['resource_type'] = resource_type
        if requested is not None:
            details['requested'] = requested
        if available is not None:
            details['available'] = available
        
        super().__init__(message, error_code="RESOURCE_EXHAUSTED", details=details, **kwargs)


class ConfigurationError(RetrievalFreeError):
    """Error in configuration or setup."""
    
    def __init__(
        self, 
        message: str, 
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that has issue
            config_value: Problematic configuration value
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if config_key is not None:
            details['config_key'] = config_key
        if config_value is not None:
            details['config_value'] = str(config_value)
        
        super().__init__(message, error_code="CONFIGURATION_ERROR", details=details, **kwargs)


class NetworkError(RetrievalFreeError):
    """Network-related error (downloading models, etc.)."""
    
    def __init__(
        self, 
        message: str, 
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        """Initialize network error.
        
        Args:
            message: Error message
            url: URL that failed
            status_code: HTTP status code
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if url is not None:
            details['url'] = url
        if status_code is not None:
            details['status_code'] = status_code
        
        super().__init__(message, error_code="NETWORK_ERROR", details=details, **kwargs)


class IncompatibilityError(RetrievalFreeError):
    """Error due to version or compatibility issues."""
    
    def __init__(
        self, 
        message: str, 
        component: Optional[str] = None,
        required_version: Optional[str] = None,
        current_version: Optional[str] = None,
        **kwargs
    ):
        """Initialize incompatibility error.
        
        Args:
            message: Error message
            component: Component with compatibility issue
            required_version: Required version
            current_version: Current version
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if component is not None:
            details['component'] = component
        if required_version is not None:
            details['required_version'] = required_version
        if current_version is not None:
            details['current_version'] = current_version
        
        super().__init__(message, error_code="INCOMPATIBILITY", details=details, **kwargs)


class TimeoutError(RetrievalFreeError):
    """Operation timeout error."""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs
    ):
        """Initialize timeout error.
        
        Args:
            message: Error message
            operation: Operation that timed out
            timeout_seconds: Timeout duration in seconds
            **kwargs: Additional arguments
        """
        details = kwargs.pop('details', {})
        if operation is not None:
            details['operation'] = operation
        if timeout_seconds is not None:
            details['timeout_seconds'] = timeout_seconds
        
        super().__init__(message, error_code="TIMEOUT", details=details, **kwargs)


def handle_exception(
    exc: Exception, 
    context: str = "unknown",
    reraise: bool = True
) -> Optional[RetrievalFreeError]:
    """Handle and convert exceptions to retrieval-free exceptions.
    
    Args:
        exc: Original exception
        context: Context where exception occurred
        reraise: Whether to reraise the converted exception
        
    Returns:
        Converted exception if not reraising
        
    Raises:
        RetrievalFreeError: Converted exception if reraise=True
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # If already a retrieval-free exception, just pass through
    if isinstance(exc, RetrievalFreeError):
        if reraise:
            raise exc
        return exc
    
    # Convert common exception types
    converted_exc = None
    
    if isinstance(exc, FileNotFoundError):
        converted_exc = ModelLoadError(
            f"File not found in {context}: {str(exc)}",
            details={'original_error': str(exc), 'context': context}
        )
    elif isinstance(exc, MemoryError):
        converted_exc = ResourceError(
            f"Out of memory in {context}: {str(exc)}",
            resource_type="memory",
            details={'original_error': str(exc), 'context': context}
        )
    elif isinstance(exc, ValueError):
        converted_exc = ValidationError(
            f"Invalid value in {context}: {str(exc)}",
            details={'original_error': str(exc), 'context': context}
        )
    elif isinstance(exc, ImportError):
        converted_exc = IncompatibilityError(
            f"Missing dependency in {context}: {str(exc)}",
            details={'original_error': str(exc), 'context': context}
        )
    elif "timeout" in str(exc).lower():
        converted_exc = TimeoutError(
            f"Timeout in {context}: {str(exc)}",
            operation=context,
            details={'original_error': str(exc)}
        )
    else:
        # Generic conversion
        converted_exc = RetrievalFreeError(
            f"Error in {context}: {str(exc)}",
            error_code="GENERIC_ERROR",
            details={'original_error': str(exc), 'context': context, 'exception_type': type(exc).__name__}
        )
    
    # Log the conversion
    logger.error(f"Converted {type(exc).__name__} to {type(converted_exc).__name__} in {context}")
    
    if reraise:
        raise converted_exc
    
    return converted_exc


def create_error_response(exc: Exception) -> Dict[str, Any]:
    """Create standardized error response from exception.
    
    Args:
        exc: Exception to convert
        
    Returns:
        Standardized error response dictionary
    """
    if isinstance(exc, RetrievalFreeError):
        return {
            'success': False,
            'error': exc.to_dict(),
            'timestamp': __import__('time').time()
        }
    else:
        # Convert to retrieval-free exception first
        converted = handle_exception(exc, reraise=False)
        return {
            'success': False,
            'error': converted.to_dict(),
            'timestamp': __import__('time').time()
        }