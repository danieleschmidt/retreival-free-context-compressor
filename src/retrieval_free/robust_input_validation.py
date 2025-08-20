"""
Robust Input Validation Framework - Generation 2
Comprehensive input validation and sanitization for all user inputs.
"""

import re
import json
import html
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

from .exceptions import ValidationError


logger = logging.getLogger(__name__)


class InputType(Enum):
    """Enumeration of supported input types."""
    TEXT = "text"
    JSON = "json"
    FILE_PATH = "file_path" 
    URL = "url"
    EMAIL = "email"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


@dataclass
class ValidationRule:
    """Defines a validation rule for input fields."""
    
    field_name: str
    input_type: InputType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Compile regex pattern if provided."""
        if self.pattern:
            self.compiled_pattern = re.compile(self.pattern)
        else:
            self.compiled_pattern = None


class InputValidator:
    """Validates and sanitizes user inputs based on defined rules."""
    
    def __init__(self):
        self.validation_rules: Dict[str, ValidationRule] = {}
        self.global_sanitizers: List[Callable[[str], str]] = []
        
        # Add default global sanitizers
        self.add_global_sanitizer(self._strip_html_tags)
        self.add_global_sanitizer(self._normalize_whitespace)
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.validation_rules[rule.field_name] = rule
        logger.debug(f"Added validation rule for field: {rule.field_name}")
    
    def add_global_sanitizer(self, sanitizer: Callable[[str], str]) -> None:
        """Add a global sanitizer that applies to all text inputs."""
        self.global_sanitizers.append(sanitizer)
    
    def validate_field(self, field_name: str, value: Any) -> Any:
        """Validate a single field value."""
        if field_name not in self.validation_rules:
            logger.warning(f"No validation rule found for field: {field_name}")
            return value
        
        rule = self.validation_rules[field_name]
        
        # Check required
        if rule.required and (value is None or value == ""):
            raise ValidationError(
                f"Field '{field_name}' is required",
                field_name=field_name,
                validation_errors=[f"Field is required but got: {value}"]
            )
        
        # Skip validation if value is None and field is not required
        if value is None and not rule.required:
            return None
        
        # Apply sanitization first
        sanitized_value = self._sanitize_value(value, rule)
        
        # Validate type
        validated_value = self._validate_type(field_name, sanitized_value, rule)
        
        # Apply specific validations
        self._apply_validation_rules(field_name, validated_value, rule)
        
        return validated_value
    
    def validate_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a dictionary of values."""
        validated_data = {}
        validation_errors = []
        
        # Validate known fields
        for field_name, rule in self.validation_rules.items():
            try:
                value = data.get(field_name)
                validated_data[field_name] = self.validate_field(field_name, value)
            except ValidationError as e:
                validation_errors.extend(e.validation_errors)
        
        # Check for unknown fields
        unknown_fields = set(data.keys()) - set(self.validation_rules.keys())
        if unknown_fields:
            logger.warning(f"Unknown fields in input: {unknown_fields}")
            # Include unknown fields in output but log warning
            for field in unknown_fields:
                validated_data[field] = data[field]
        
        if validation_errors:
            raise ValidationError(
                f"Validation failed for {len(validation_errors)} fields",
                validation_errors=validation_errors
            )
        
        return validated_data
    
    def _sanitize_value(self, value: Any, rule: ValidationRule) -> Any:
        """Apply sanitization to a value."""
        # Apply custom sanitizer if provided
        if rule.sanitizer:
            value = rule.sanitizer(value)
        
        # Apply global sanitizers for text
        if isinstance(value, str):
            for sanitizer in self.global_sanitizers:
                value = sanitizer(value)
        
        return value
    
    def _validate_type(self, field_name: str, value: Any, rule: ValidationRule) -> Any:
        """Validate and convert value type."""
        try:
            if rule.input_type == InputType.TEXT:
                return str(value)
            elif rule.input_type == InputType.NUMBER:
                if isinstance(value, str):
                    # Try to convert string to number
                    if '.' in value:
                        return float(value)
                    else:
                        return int(value)
                return value
            elif rule.input_type == InputType.BOOLEAN:
                if isinstance(value, str):
                    return value.lower() in ('true', '1', 'yes', 'on')
                return bool(value)
            elif rule.input_type == InputType.JSON:
                if isinstance(value, str):
                    return json.loads(value)
                return value
            elif rule.input_type == InputType.LIST:
                if not isinstance(value, list):
                    raise ValueError(f"Expected list, got {type(value)}")
                return value
            elif rule.input_type == InputType.DICT:
                if not isinstance(value, dict):
                    raise ValueError(f"Expected dict, got {type(value)}")
                return value
            else:
                return value
                
        except (ValueError, json.JSONDecodeError) as e:
            raise ValidationError(
                f"Type validation failed for field '{field_name}': {e}",
                field_name=field_name,
                field_value=value,
                expected_type=rule.input_type.value
            )
    
    def _apply_validation_rules(self, field_name: str, value: Any, rule: ValidationRule) -> None:
        """Apply validation rules to a value."""
        errors = []
        
        # Length validation
        if hasattr(value, '__len__'):
            length = len(value)
            if rule.min_length is not None and length < rule.min_length:
                errors.append(f"Length {length} is less than minimum {rule.min_length}")
            if rule.max_length is not None and length > rule.max_length:
                errors.append(f"Length {length} exceeds maximum {rule.max_length}")
        
        # Value range validation
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                errors.append(f"Value {value} is less than minimum {rule.min_value}")
            if rule.max_value is not None and value > rule.max_value:
                errors.append(f"Value {value} exceeds maximum {rule.max_value}")
        
        # Pattern validation
        if rule.compiled_pattern and isinstance(value, str):
            if not rule.compiled_pattern.match(value):
                errors.append(f"Value does not match required pattern: {rule.pattern}")
        
        # Allowed values validation
        if rule.allowed_values is not None and value not in rule.allowed_values:
            errors.append(f"Value must be one of: {rule.allowed_values}")
        
        # Custom validation
        if rule.custom_validator and not rule.custom_validator(value):
            errors.append("Failed custom validation")
        
        if errors:
            error_message = rule.error_message or f"Validation failed for field '{field_name}'"
            raise ValidationError(
                error_message,
                field_name=field_name,
                field_value=value,
                validation_errors=errors
            )
    
    def _strip_html_tags(self, text: str) -> str:
        """Strip HTML tags from text."""
        # Simple HTML tag removal - use proper HTML parser for production
        tag_pattern = re.compile(r'<[^>]+>')
        return tag_pattern.sub('', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple whitespace with single space and strip
        return re.sub(r'\s+', ' ', text).strip()


class CompressionInputValidator(InputValidator):
    """Specialized validator for compression operations."""
    
    def __init__(self):
        super().__init__()
        
        # Add compression-specific validation rules
        self.add_rule(ValidationRule(
            field_name="text",
            input_type=InputType.TEXT,
            required=True,
            min_length=1,
            max_length=1000000,  # 1M characters
            error_message="Text must be between 1 and 1,000,000 characters"
        ))
        
        self.add_rule(ValidationRule(
            field_name="model_name",
            input_type=InputType.TEXT,
            required=False,
            max_length=100,
            pattern=r'^[a-zA-Z0-9_-]+$',
            error_message="Model name must contain only alphanumeric characters, hyphens, and underscores"
        ))
        
        self.add_rule(ValidationRule(
            field_name="compression_ratio",
            input_type=InputType.NUMBER,
            required=False,
            min_value=1.0,
            max_value=100.0,
            error_message="Compression ratio must be between 1.0 and 100.0"
        ))
        
        self.add_rule(ValidationRule(
            field_name="max_tokens",
            input_type=InputType.NUMBER,
            required=False,
            min_value=1,
            max_value=1000000,
            error_message="Max tokens must be between 1 and 1,000,000"
        ))
        
        self.add_rule(ValidationRule(
            field_name="enable_streaming",
            input_type=InputType.BOOLEAN,
            required=False
        ))
        
        self.add_rule(ValidationRule(
            field_name="metadata",
            input_type=InputType.DICT,
            required=False,
            custom_validator=self._validate_metadata
        ))
        
        # Add compression-specific sanitizers
        self.add_global_sanitizer(self._remove_suspicious_patterns)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate metadata dictionary."""
        if not isinstance(metadata, dict):
            return False
        
        # Check metadata size
        if len(json.dumps(metadata)) > 10000:  # 10KB limit
            return False
        
        # Check for allowed keys
        allowed_keys = {'source', 'author', 'timestamp', 'tags', 'category'}
        if not all(key in allowed_keys for key in metadata.keys()):
            return False
        
        return True
    
    def _remove_suspicious_patterns(self, text: str) -> str:
        """Remove potentially suspicious patterns from text."""
        # Remove potential script injections
        script_pattern = re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL)
        text = script_pattern.sub('', text)
        
        # Remove potential SQL injection attempts
        sql_pattern = re.compile(r'(union|select|insert|update|delete|drop)\s+', re.IGNORECASE)
        text = sql_pattern.sub('', text)
        
        return text


class BatchValidator:
    """Validates batch operations with multiple inputs."""
    
    def __init__(self, input_validator: InputValidator):
        self.input_validator = input_validator
        self.max_batch_size = 100
        self.max_total_text_length = 10000000  # 10M characters total
    
    def validate_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a batch of inputs."""
        if not isinstance(batch_data, list):
            raise ValidationError("Batch data must be a list")
        
        if len(batch_data) > self.max_batch_size:
            raise ValidationError(
                f"Batch size {len(batch_data)} exceeds maximum {self.max_batch_size}",
                details={"batch_size": len(batch_data), "max_size": self.max_batch_size}
            )
        
        validated_batch = []
        batch_errors = []
        total_text_length = 0
        
        for i, item in enumerate(batch_data):
            try:
                validated_item = self.input_validator.validate_dict(item)
                validated_batch.append(validated_item)
                
                # Track total text length
                if 'text' in validated_item:
                    total_text_length += len(validated_item['text'])
                
            except ValidationError as e:
                batch_errors.append({
                    'index': i,
                    'errors': e.validation_errors,
                    'item': item
                })
        
        # Check total text length
        if total_text_length > self.max_total_text_length:
            raise ValidationError(
                f"Total text length {total_text_length} exceeds maximum {self.max_total_text_length}",
                details={"total_length": total_text_length, "max_length": self.max_total_text_length}
            )
        
        if batch_errors:
            raise ValidationError(
                f"Validation failed for {len(batch_errors)} items in batch",
                validation_errors=[f"Item {err['index']}: {err['errors']}" for err in batch_errors],
                details={"failed_items": batch_errors}
            )
        
        return validated_batch


# Pre-configured validators for common use cases
def get_compression_validator() -> CompressionInputValidator:
    """Get pre-configured validator for compression operations."""
    return CompressionInputValidator()


def get_api_validator() -> InputValidator:
    """Get pre-configured validator for API operations."""
    validator = InputValidator()
    
    # Common API field validations
    validator.add_rule(ValidationRule(
        field_name="api_key",
        input_type=InputType.TEXT,
        required=True,
        min_length=32,
        max_length=128,
        pattern=r'^[a-fA-F0-9]+$',
        error_message="API key must be a valid hexadecimal string"
    ))
    
    validator.add_rule(ValidationRule(
        field_name="client_id",
        input_type=InputType.TEXT,
        required=True,
        max_length=50,
        pattern=r'^[a-zA-Z0-9_-]+$',
        error_message="Client ID must contain only alphanumeric characters, hyphens, and underscores"
    ))
    
    validator.add_rule(ValidationRule(
        field_name="timeout",
        input_type=InputType.NUMBER,
        required=False,
        min_value=1,
        max_value=300,  # 5 minutes max
        error_message="Timeout must be between 1 and 300 seconds"
    ))
    
    return validator


# Global validator instances
_compression_validator = None
_api_validator = None

def validate_compression_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input for compression operations."""
    global _compression_validator
    if _compression_validator is None:
        _compression_validator = get_compression_validator()
    
    return _compression_validator.validate_dict(data)


def validate_api_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input for API operations."""
    global _api_validator
    if _api_validator is None:
        _api_validator = get_api_validator()
    
    return _api_validator.validate_dict(data)