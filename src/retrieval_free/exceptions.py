"""Custom exceptions for the retrieval-free context compressor."""

import logging
from functools import wraps
from typing import Any


logger = logging.getLogger(__name__)


class RetrievalFreeError(Exception):
    """Base exception class for retrieval-free context compressor."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
    ):
        """Initialize base exception.

        Args:
            message: Error message
            details: Additional error details
            error_code: Optional error code for categorization
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.error_code = error_code

    def __str__(self) -> str:
        """String representation of the error."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class ModelLoadError(RetrievalFreeError):
    """Exception raised when a model fails to load."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "MODEL_LOAD_ERROR")
        self.model_name = model_name
        if model_name:
            self.details["model_name"] = model_name


class CompressionError(RetrievalFreeError):
    """Exception raised during text compression operations."""

    def __init__(
        self,
        message: str,
        input_length: int | None = None,
        model_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "COMPRESSION_ERROR")
        self.input_length = input_length
        self.model_name = model_name

        if input_length is not None:
            self.details["input_length"] = input_length
        if model_name:
            self.details["model_name"] = model_name


class ValidationError(RetrievalFreeError):
    """Exception raised when input validation fails."""

    def __init__(
        self,
        message: str,
        validation_errors: list[str] | None = None,
        field_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "VALIDATION_ERROR")
        self.validation_errors = validation_errors or []
        self.field_name = field_name

        if validation_errors:
            self.details["validation_errors"] = validation_errors
        if field_name:
            self.details["field_name"] = field_name


class ConfigurationError(RetrievalFreeError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        config_value: Any | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "CONFIG_ERROR")
        self.config_key = config_key
        self.config_value = config_value

        if config_key:
            self.details["config_key"] = config_key
        if config_value is not None:
            self.details["config_value"] = str(config_value)


class ResourceError(RetrievalFreeError):
    """Exception raised when system resources are insufficient."""

    def __init__(
        self,
        message: str,
        resource_type: str | None = None,
        required_amount: str | None = None,
        available_amount: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "RESOURCE_ERROR")
        self.resource_type = resource_type
        self.required_amount = required_amount
        self.available_amount = available_amount

        if resource_type:
            self.details["resource_type"] = resource_type
        if required_amount:
            self.details["required_amount"] = required_amount
        if available_amount:
            self.details["available_amount"] = available_amount


class NetworkError(RetrievalFreeError):
    """Exception raised for network-related errors."""

    def __init__(
        self,
        message: str,
        endpoint: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "NETWORK_ERROR")
        self.endpoint = endpoint
        self.status_code = status_code

        if endpoint:
            self.details["endpoint"] = endpoint
        if status_code:
            self.details["status_code"] = status_code


class CacheError(RetrievalFreeError):
    """Exception raised for caching-related errors."""

    def __init__(
        self,
        message: str,
        cache_key: str | None = None,
        cache_operation: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "CACHE_ERROR")
        self.cache_key = cache_key
        self.cache_operation = cache_operation

        if cache_key:
            self.details["cache_key"] = cache_key
        if cache_operation:
            self.details["cache_operation"] = cache_operation


class PluginError(RetrievalFreeError):
    """Exception raised for plugin-related errors."""

    def __init__(
        self,
        message: str,
        plugin_name: str | None = None,
        plugin_version: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details, "PLUGIN_ERROR")
        self.plugin_name = plugin_name
        self.plugin_version = plugin_version

        if plugin_name:
            self.details["plugin_name"] = plugin_name
        if plugin_version:
            self.details["plugin_version"] = plugin_version


def handle_exception(func):
    """Decorator to handle and convert exceptions to our custom types."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RetrievalFreeError:
            # Re-raise our custom exceptions as-is
            raise
        except ImportError as e:
            raise ModelLoadError(
                f"Failed to import required dependency: {str(e)}",
                details={"original_error": str(e), "error_type": "ImportError"},
            ) from e
        except FileNotFoundError as e:
            raise ModelLoadError(
                f"Model file not found: {str(e)}",
                details={"original_error": str(e), "error_type": "FileNotFoundError"},
            ) from e
        except MemoryError as e:
            raise ResourceError(
                f"Insufficient memory: {str(e)}",
                resource_type="memory",
                details={"original_error": str(e), "error_type": "MemoryError"},
            ) from e
        except ValueError as e:
            # Convert ValueError to appropriate custom exception based on context
            if "validation" in str(e).lower():
                raise ValidationError(
                    f"Input validation failed: {str(e)}",
                    details={"original_error": str(e), "error_type": "ValueError"},
                ) from e
            else:
                raise CompressionError(
                    f"Operation failed: {str(e)}",
                    details={"original_error": str(e), "error_type": "ValueError"},
                ) from e
        except Exception as e:
            # Convert other exceptions to generic compression error
            logger.warning(
                f"Converting {type(e).__name__} to CompressionError: {str(e)}"
            )
            raise CompressionError(
                f"Unexpected error during operation: {str(e)}",
                details={"original_error": str(e), "error_type": type(e).__name__},
            ) from e

    return wrapper


def log_exception(exception: Exception, context: dict[str, Any] | None = None) -> None:
    """Log exception with additional context information.

    Args:
        exception: Exception to log
        context: Additional context information
    """
    context = context or {}

    if isinstance(exception, RetrievalFreeError):
        logger.error(
            f"RetrievalFreeError: {exception.to_dict()}", extra={"context": context}
        )
    else:
        logger.error(
            f"{type(exception).__name__}: {str(exception)}",
            extra={"context": context},
            exc_info=True,
        )


# Exception registry for programmatic access
EXCEPTION_REGISTRY = {
    "MODEL_LOAD_ERROR": ModelLoadError,
    "COMPRESSION_ERROR": CompressionError,
    "VALIDATION_ERROR": ValidationError,
    "CONFIG_ERROR": ConfigurationError,
    "RESOURCE_ERROR": ResourceError,
    "NETWORK_ERROR": NetworkError,
    "CACHE_ERROR": CacheError,
    "PLUGIN_ERROR": PluginError,
}


def create_exception(error_code: str, message: str, **kwargs) -> RetrievalFreeError:
    """Create an exception by error code.

    Args:
        error_code: Error code from EXCEPTION_REGISTRY
        message: Error message
        **kwargs: Additional arguments for the exception

    Returns:
        Appropriate exception instance

    Raises:
        ValueError: If error_code is not recognized
    """
    if error_code not in EXCEPTION_REGISTRY:
        raise ValueError(f"Unknown error code: {error_code}")

    exception_class = EXCEPTION_REGISTRY[error_code]
    return exception_class(message, **kwargs)
