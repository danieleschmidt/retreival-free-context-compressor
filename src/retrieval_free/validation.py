"""Input validation and parameter validation for the retrieval-free context compressor."""

import logging
import re
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation operation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    sanitized_input: dict[str, Any]

    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0


class InputValidator:
    """Validates and sanitizes input data."""

    def __init__(
        self,
        max_text_length: int = 1_000_000,
        min_text_length: int = 1,
        allowed_encodings: list[str] | None = None,
        enable_sanitization: bool = True,
    ):
        """Initialize input validator.

        Args:
            max_text_length: Maximum allowed text length in characters
            min_text_length: Minimum allowed text length in characters
            allowed_encodings: List of allowed text encodings
            enable_sanitization: Whether to enable text sanitization
        """
        self.max_text_length = max_text_length
        self.min_text_length = min_text_length
        self.allowed_encodings = allowed_encodings or ["utf-8", "ascii"]
        self.enable_sanitization = enable_sanitization

        # Patterns for potentially problematic content
        self.suspicious_patterns = [
            r"<script.*?</script>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"data:.*base64",  # Base64 data URLs
            r"<\!--.*?-->",  # HTML comments (might hide malicious content)
        ]

    def validate_text_input(self, text: str) -> ValidationResult:
        """Validate text input.

        Args:
            text: Input text to validate

        Returns:
            ValidationResult with validation status and sanitized text
        """
        errors = []
        warnings = []
        sanitized_text = text

        # Check if text is string
        if not isinstance(text, str):
            errors.append(f"Text input must be string, got {type(text).__name__}")

        # Check text length
        if len(text) < self.min_text_length:
            errors.append(f"Text too short: {len(text)} < {self.min_text_length}")
        elif len(text) > self.max_text_length:
            errors.append(f"Text too long: {len(text)} > {self.max_text_length}")

        # Check for empty or whitespace-only text
        if not text.strip():
            errors.append("Text cannot be empty or whitespace-only")

        # Check encoding
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as e:
            errors.append(f"Text contains invalid Unicode characters: {e}")

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                warnings.append(f"Potentially suspicious content detected: {pattern}")

                # Sanitize if enabled
                if self.enable_sanitization:
                    sanitized_text = re.sub(
                        pattern,
                        "[REMOVED]",
                        sanitized_text,
                        flags=re.IGNORECASE | re.DOTALL,
                    )

        # Check for very long lines (might indicate structured data)
        lines = text.split("\n")
        long_lines = [i for i, line in enumerate(lines) if len(line) > 10000]
        if long_lines:
            warnings.append(
                f"Found {len(long_lines)} very long lines (>10k chars), might be structured data"
            )

        # Check for high proportion of non-ASCII characters
        if text:
            non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
            if non_ascii_ratio > 0.5:
                warnings.append(
                    f"High proportion of non-ASCII characters: {non_ascii_ratio:.1%}"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"text": sanitized_text},
        )

    def validate_list_input(self, texts: list[str]) -> ValidationResult:
        """Validate list of text inputs.

        Args:
            texts: List of text strings to validate

        Returns:
            ValidationResult with validation status and sanitized texts
        """
        errors = []
        warnings = []
        sanitized_texts = []

        # Check if input is list
        if not isinstance(texts, list):
            errors.append(f"Input must be list, got {type(texts).__name__}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                sanitized_input={"texts": []},
            )

        # Check list length
        if not texts:
            errors.append("Text list cannot be empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                sanitized_input={"texts": []},
            )

        # Validate each text individually
        for i, text in enumerate(texts):
            result = self.validate_text_input(text)

            # Collect errors with index information
            for error in result.errors:
                errors.append(f"Text {i}: {error}")

            # Collect warnings with index information
            for warning in result.warnings:
                warnings.append(f"Text {i}: {warning}")

            # Add sanitized text
            sanitized_texts.append(result.sanitized_input["text"])

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"texts": sanitized_texts},
        )


class ParameterValidator:
    """Validates compression parameters."""

    def __init__(self):
        """Initialize parameter validator."""
        self.valid_models = [
            "rfcc-base-8x",
            "rfcc-large-16x",
            "rfcc-small-4x",
            "context-compressor-base",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

    def validate_compression_ratio(self, ratio: float) -> ValidationResult:
        """Validate compression ratio parameter.

        Args:
            ratio: Compression ratio to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check type
        if not isinstance(ratio, (int, float)):
            errors.append(
                f"Compression ratio must be numeric, got {type(ratio).__name__}"
            )
        else:
            # Check range
            if ratio <= 0:
                errors.append(f"Compression ratio must be positive, got {ratio}")
            elif ratio < 1:
                warnings.append(
                    f"Compression ratio < 1 ({ratio}) means expansion, not compression"
                )
            elif ratio > 100:
                warnings.append(
                    f"Very high compression ratio ({ratio}) may result in significant information loss"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"compression_ratio": ratio},
        )

    def validate_model_name(self, model_name: str) -> ValidationResult:
        """Validate model name parameter.

        Args:
            model_name: Model name to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check type
        if not isinstance(model_name, str):
            errors.append(f"Model name must be string, got {type(model_name).__name__}")
        else:
            # Check if empty
            if not model_name.strip():
                errors.append("Model name cannot be empty")
            elif model_name not in self.valid_models:
                warnings.append(
                    f"Unknown model name '{model_name}', might not be supported"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"model_name": model_name.strip()},
        )

    def validate_chunk_size(self, chunk_size: int) -> ValidationResult:
        """Validate chunk size parameter.

        Args:
            chunk_size: Chunk size to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check type
        if not isinstance(chunk_size, int):
            errors.append(
                f"Chunk size must be integer, got {type(chunk_size).__name__}"
            )
        else:
            # Check range
            if chunk_size <= 0:
                errors.append(f"Chunk size must be positive, got {chunk_size}")
            elif chunk_size < 32:
                warnings.append(
                    f"Very small chunk size ({chunk_size}) may result in poor compression"
                )
            elif chunk_size > 8192:
                warnings.append(
                    f"Very large chunk size ({chunk_size}) may cause memory issues"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"chunk_size": chunk_size},
        )

    def validate_overlap(
        self, overlap: int, chunk_size: int | None = None
    ) -> ValidationResult:
        """Validate overlap parameter.

        Args:
            overlap: Overlap size to validate
            chunk_size: Optional chunk size for relative validation

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check type
        if not isinstance(overlap, int):
            errors.append(f"Overlap must be integer, got {type(overlap).__name__}")
        else:
            # Check range
            if overlap < 0:
                errors.append(f"Overlap cannot be negative, got {overlap}")
            elif chunk_size and overlap >= chunk_size:
                errors.append(
                    f"Overlap ({overlap}) must be less than chunk size ({chunk_size})"
                )
            elif chunk_size and overlap > chunk_size * 0.5:
                warnings.append(
                    f"Large overlap ({overlap}/{chunk_size}) may cause inefficiency"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"overlap": overlap},
        )

    def validate_device(self, device: str) -> ValidationResult:
        """Validate device parameter.

        Args:
            device: Device string to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check type
        if not isinstance(device, str):
            errors.append(f"Device must be string, got {type(device).__name__}")
        else:
            device = device.lower().strip()

            # Check valid device formats
            valid_devices = ["cpu", "cuda", "mps", "auto"]
            cuda_pattern = r"^cuda:\d+$"

            if device not in valid_devices and not re.match(cuda_pattern, device):
                warnings.append(
                    f"Unknown device '{device}', supported: {valid_devices} or 'cuda:N'"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_input={"device": device},
        )


def validate_compression_request(
    text: str | list[str], parameters: dict[str, Any]
) -> ValidationResult:
    """Validate a complete compression request.

    Args:
        text: Input text or list of texts
        parameters: Compression parameters

    Returns:
        ValidationResult with overall validation status
    """
    input_validator = InputValidator()
    param_validator = ParameterValidator()

    all_errors = []
    all_warnings = []
    sanitized_input = {}

    # Validate text input
    if isinstance(text, str):
        text_result = input_validator.validate_text_input(text)
        sanitized_input.update(text_result.sanitized_input)
    elif isinstance(text, list):
        text_result = input_validator.validate_list_input(text)
        sanitized_input.update(text_result.sanitized_input)
    else:
        text_result = ValidationResult(
            is_valid=False,
            errors=[
                f"Text must be string or list of strings, got {type(text).__name__}"
            ],
            warnings=[],
            sanitized_input={},
        )

    all_errors.extend(text_result.errors)
    all_warnings.extend(text_result.warnings)

    # Validate parameters
    param_validations = [
        ("compression_ratio", param_validator.validate_compression_ratio),
        ("model_name", param_validator.validate_model_name),
        ("chunk_size", param_validator.validate_chunk_size),
        ("device", param_validator.validate_device),
    ]

    chunk_size = parameters.get("chunk_size")

    for param_name, validator_func in param_validations:
        if param_name in parameters:
            if param_name == "overlap" and chunk_size:
                result = param_validator.validate_overlap(
                    parameters[param_name], chunk_size
                )
            else:
                result = validator_func(parameters[param_name])

            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)
            sanitized_input.update(result.sanitized_input)

    # Special validation for overlap with chunk_size context
    if "overlap" in parameters:
        overlap_result = param_validator.validate_overlap(
            parameters["overlap"], parameters.get("chunk_size")
        )
        all_errors.extend(overlap_result.errors)
        all_warnings.extend(overlap_result.warnings)
        sanitized_input.update(overlap_result.sanitized_input)

    return ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        sanitized_input=sanitized_input,
    )


def validate_mega_token_structure(mega_token: Any) -> ValidationResult:
    """Validate mega-token structure.

    Args:
        mega_token: Mega-token object to validate

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []

    # Check if it has required attributes
    required_attrs = ["embedding", "metadata", "source_range", "compression_ratio"]

    for attr in required_attrs:
        if not hasattr(mega_token, attr):
            errors.append(f"Missing required attribute: {attr}")

    # Validate embedding if present
    if hasattr(mega_token, "embedding"):
        try:
            import torch

            if not isinstance(mega_token.embedding, torch.Tensor):
                errors.append(
                    f"Embedding must be torch.Tensor, got {type(mega_token.embedding)}"
                )
            elif mega_token.embedding.dim() != 1:
                errors.append(
                    f"Embedding must be 1D tensor, got {mega_token.embedding.dim()}D"
                )
        except ImportError:
            warnings.append("Cannot validate embedding type - torch not available")

    # Validate compression ratio if present
    if hasattr(mega_token, "compression_ratio"):
        if mega_token.compression_ratio <= 0:
            errors.append(
                f"Compression ratio must be positive, got {mega_token.compression_ratio}"
            )

    # Validate source range if present
    if hasattr(mega_token, "source_range"):
        if (
            not isinstance(mega_token.source_range, (tuple, list))
            or len(mega_token.source_range) != 2
        ):
            errors.append("Source range must be tuple/list of 2 elements")
        elif mega_token.source_range[0] > mega_token.source_range[1]:
            errors.append("Source range start must be <= end")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        sanitized_input={"mega_token": mega_token},
    )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for filesystem operations
    """
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', "_", filename)
    filename = re.sub(r"\.{2,}", ".", filename)  # Replace multiple dots
    filename = filename.strip(". ")  # Remove leading/trailing dots and spaces

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[: 250 - len(ext)] + ("." + ext if ext else "")

    return filename or "unnamed_file"
