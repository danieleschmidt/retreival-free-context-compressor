"""Advanced configuration management with runtime updates, feature flags, and A/B testing."""

import hashlib
import json
import logging
import os
import random
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError


logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class FeatureToggleStrategy(Enum):
    """Feature toggle strategies."""
    ON = "on"                      # Always on
    OFF = "off"                    # Always off
    PERCENTAGE = "percentage"      # Percentage of users
    USER_LIST = "user_list"        # Specific users
    A_B_TEST = "a_b_test"          # A/B testing


@dataclass
class FeatureToggle:
    """Feature toggle configuration."""
    name: str
    strategy: FeatureToggleStrategy
    enabled: bool = True
    percentage: float = 100.0      # For percentage strategy
    user_list: set[str] = field(default_factory=set)  # For user_list strategy
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    name: str
    control_percentage: float = 50.0  # Percentage for control group
    treatment_percentage: float = 50.0  # Percentage for treatment group
    user_hash_salt: str = ""          # Salt for consistent user hashing
    start_date: datetime | None = None
    end_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigSchema:
    """Schema definition for configuration validation."""
    required_fields: set[str] = field(default_factory=set)
    optional_fields: set[str] = field(default_factory=set)
    field_types: dict[str, type] = field(default_factory=dict)
    field_validators: dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    nested_schemas: dict[str, 'ConfigSchema'] = field(default_factory=dict)


class ConfigValidator:
    """Validates configuration against schemas."""

    def __init__(self):
        """Initialize configuration validator."""
        self.schemas: dict[str, ConfigSchema] = {}

    def register_schema(self, name: str, schema: ConfigSchema) -> None:
        """Register a configuration schema.
        
        Args:
            name: Schema name
            schema: Schema definition
        """
        self.schemas[name] = schema
        logger.info(f"Registered configuration schema: {name}")

    def validate(self, config: dict[str, Any], schema_name: str) -> list[str]:
        """Validate configuration against a schema.
        
        Args:
            config: Configuration to validate
            schema_name: Name of schema to use
            
        Returns:
            List of validation errors (empty if valid)
        """
        if schema_name not in self.schemas:
            return [f"Unknown schema: {schema_name}"]

        schema = self.schemas[schema_name]
        errors = []

        # Check required fields
        for field in schema.required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")

        # Check field types and validators
        for field, value in config.items():
            if field in schema.field_types:
                expected_type = schema.field_types[field]
                if not isinstance(value, expected_type):
                    errors.append(f"Field '{field}' should be of type {expected_type.__name__}, got {type(value).__name__}")

            if field in schema.field_validators:
                validator = schema.field_validators[field]
                try:
                    if not validator(value):
                        errors.append(f"Field '{field}' failed validation")
                except Exception as e:
                    errors.append(f"Validation error for field '{field}': {e}")

            # Validate nested objects
            if field in schema.nested_schemas and isinstance(value, dict):
                nested_errors = self.validate(value, field)
                for error in nested_errors:
                    errors.append(f"{field}.{error}")

        # Check for unknown fields
        allowed_fields = schema.required_fields | schema.optional_fields
        if allowed_fields:
            for field in config.keys():
                if field not in allowed_fields and field not in schema.nested_schemas:
                    errors.append(f"Unknown field: {field}")

        return errors


class ConfigurationManager:
    """Advanced configuration management with hot reloading and validation."""

    def __init__(self, config_dir: str | Path | None = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._config: dict[str, Any] = {}
        self._file_mtimes: dict[str, float] = {}
        self._change_callbacks: list[Callable[[str, Any, Any], None]] = []
        self._lock = threading.RLock()

        self.validator = ConfigValidator()
        self._setup_default_schemas()

        # Load initial configuration
        self.reload_configuration()

        logger.info(f"Configuration manager initialized with directory: {self.config_dir}")

    def _setup_default_schemas(self) -> None:
        """Set up default configuration schemas."""
        # Compression schema
        compression_schema = ConfigSchema(
            required_fields={"model_name", "chunk_size"},
            optional_fields={"compression_ratio", "overlap_ratio", "device", "batch_size"},
            field_types={
                "model_name": str,
                "chunk_size": int,
                "compression_ratio": (int, float),
                "overlap_ratio": (int, float),
                "device": str,
                "batch_size": int
            },
            field_validators={
                "chunk_size": lambda x: x > 0 and x <= 8192,
                "compression_ratio": lambda x: x > 0 and x <= 100,
                "overlap_ratio": lambda x: 0 <= x <= 0.5,
                "batch_size": lambda x: x > 0 and x <= 128
            }
        )
        self.validator.register_schema("compression", compression_schema)

        # Security schema
        security_schema = ConfigSchema(
            required_fields={"enable_authentication"},
            optional_fields={"rate_limit_requests", "rate_limit_window", "max_input_size"},
            field_types={
                "enable_authentication": bool,
                "rate_limit_requests": int,
                "rate_limit_window": int,
                "max_input_size": int
            },
            field_validators={
                "rate_limit_requests": lambda x: x > 0 and x <= 10000,
                "rate_limit_window": lambda x: x > 0 and x <= 3600,
                "max_input_size": lambda x: x > 0 and x <= 100_000_000
            }
        )
        self.validator.register_schema("security", security_schema)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            return self._get_nested_value(self._config, key, default)

    def set(self, key: str, value: Any, validate: bool = True) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            validate: Whether to validate the configuration
            
        Raises:
            ConfigurationError: If validation fails
        """
        with self._lock:
            old_value = self.get(key)

            # Set the value
            self._set_nested_value(self._config, key, value)

            # Validate if requested
            if validate:
                errors = self._validate_configuration()
                if errors:
                    # Rollback on validation failure
                    self._set_nested_value(self._config, key, old_value)
                    raise ConfigurationError(
                        f"Configuration validation failed: {'; '.join(errors)}",
                        config_key=key,
                        config_value=value
                    )

            # Notify callbacks
            for callback in self._change_callbacks:
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    logger.error(f"Configuration change callback failed: {e}")

            logger.info(f"Configuration updated: {key} = {value}")

    def update(self, updates: dict[str, Any], validate: bool = True) -> None:
        """Update multiple configuration values.
        
        Args:
            updates: Dictionary of key-value pairs to update
            validate: Whether to validate the configuration
            
        Raises:
            ConfigurationError: If validation fails
        """
        with self._lock:
            old_values = {}

            try:
                # Apply all updates
                for key, value in updates.items():
                    old_values[key] = self.get(key)
                    self._set_nested_value(self._config, key, value)

                # Validate if requested
                if validate:
                    errors = self._validate_configuration()
                    if errors:
                        raise ConfigurationError(
                            f"Configuration validation failed: {'; '.join(errors)}"
                        )

                # Notify callbacks
                for key, value in updates.items():
                    for callback in self._change_callbacks:
                        try:
                            callback(key, old_values[key], value)
                        except Exception as e:
                            logger.error(f"Configuration change callback failed: {e}")

                logger.info(f"Configuration updated: {len(updates)} values")

            except Exception:
                # Rollback all changes on failure
                for key, old_value in old_values.items():
                    self._set_nested_value(self._config, key, old_value)
                raise

    def reload_configuration(self) -> None:
        """Reload configuration from files."""
        with self._lock:
            self._load_configuration_files()
            logger.info("Configuration reloaded from files")

    def watch_for_changes(self) -> bool:
        """Check for configuration file changes and reload if needed.
        
        Returns:
            True if configuration was reloaded, False otherwise
        """
        with self._lock:
            needs_reload = False

            for config_file in self.config_dir.glob("*.{json,yaml,yml}"):
                current_mtime = config_file.stat().st_mtime
                if (str(config_file) not in self._file_mtimes or
                    current_mtime > self._file_mtimes[str(config_file)]):
                    needs_reload = True
                    break

            if needs_reload:
                self.reload_configuration()
                return True

            return False

    def register_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Register a callback for configuration changes.
        
        Args:
            callback: Function called with (key, old_value, new_value)
        """
        with self._lock:
            self._change_callbacks.append(callback)

    def export_configuration(self, format: str = "json") -> str:
        """Export current configuration.
        
        Args:
            format: Export format ("json" or "yaml")
            
        Returns:
            Serialized configuration
        """
        with self._lock:
            if format.lower() == "json":
                return json.dumps(self._config, indent=2, default=str)
            elif format.lower() in ["yaml", "yml"]:
                return yaml.dump(self._config, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

    def import_configuration(self, config_str: str, format: str = "json", validate: bool = True) -> None:
        """Import configuration from string.
        
        Args:
            config_str: Serialized configuration
            format: Configuration format ("json" or "yaml")
            validate: Whether to validate the configuration
            
        Raises:
            ConfigurationError: If parsing or validation fails
        """
        try:
            if format.lower() == "json":
                new_config = json.loads(config_str)
            elif format.lower() in ["yaml", "yml"]:
                new_config = yaml.safe_load(config_str)
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Backup current configuration
            old_config = self._config.copy()

            try:
                self._config.update(new_config)

                if validate:
                    errors = self._validate_configuration()
                    if errors:
                        raise ConfigurationError(
                            f"Configuration validation failed: {'; '.join(errors)}"
                        )

                logger.info(f"Configuration imported from {format}")

            except Exception:
                # Restore on failure
                self._config = old_config
                raise

        except Exception as e:
            raise ConfigurationError(f"Failed to import configuration: {e}")

    def _load_configuration_files(self) -> None:
        """Load configuration from files in the config directory."""
        new_config = {}

        # Load configuration files
        for config_file in sorted(self.config_dir.glob("*.{json,yaml,yml}")):
            try:
                self._file_mtimes[str(config_file)] = config_file.stat().st_mtime

                with open(config_file) as f:
                    if config_file.suffix.lower() == '.json':
                        file_config = json.load(f)
                    elif config_file.suffix.lower() in ['.yaml', '.yml']:
                        file_config = yaml.safe_load(f)
                    else:
                        continue

                # Merge configuration
                self._deep_merge(new_config, file_config)
                logger.debug(f"Loaded configuration from {config_file}")

            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")

        # Apply environment-specific overrides
        env = os.environ.get('RETRIEVAL_FREE_ENV', Environment.DEVELOPMENT.value)
        env_config_file = self.config_dir / f"{env}.json"
        if env_config_file.exists():
            try:
                with open(env_config_file) as f:
                    env_config = json.load(f)
                self._deep_merge(new_config, env_config)
                logger.info(f"Applied environment-specific configuration: {env}")
            except Exception as e:
                logger.error(f"Failed to load environment configuration: {e}")

        # Apply environment variable overrides
        self._apply_env_var_overrides(new_config)

        self._config = new_config

    def _apply_env_var_overrides(self, config: dict[str, Any]) -> None:
        """Apply environment variable overrides."""
        env_prefix = "RETRIEVAL_FREE_"

        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')

                # Try to parse as JSON, otherwise use as string
                try:
                    parsed_value = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    parsed_value = value

                self._set_nested_value(config, config_key, parsed_value)
                logger.debug(f"Applied environment override: {config_key} = {parsed_value}")

    def _get_nested_value(self, obj: dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = obj

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def _set_nested_value(self, obj: dict[str, Any], key: str, value: Any) -> None:
        """Set value in nested dictionary using dot notation."""
        keys = key.split('.')
        current = obj

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Deep merge source dictionary into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def _validate_configuration(self) -> list[str]:
        """Validate current configuration against all schemas."""
        all_errors = []

        for schema_name in self.validator.schemas:
            if schema_name in self._config:
                errors = self.validator.validate(self._config[schema_name], schema_name)
                for error in errors:
                    all_errors.append(f"{schema_name}: {error}")

        return all_errors


class FeatureToggleManager:
    """Manages feature toggles and A/B tests."""

    def __init__(self, config_manager: ConfigurationManager | None = None):
        """Initialize feature toggle manager.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._toggles: dict[str, FeatureToggle] = {}
        self._ab_tests: dict[str, ABTestConfig] = {}
        self._lock = threading.RLock()

        # Load toggles from configuration
        self._load_feature_toggles()

        logger.info("Feature toggle manager initialized")

    def register_toggle(
        self,
        name: str,
        strategy: FeatureToggleStrategy = FeatureToggleStrategy.ON,
        **kwargs
    ) -> None:
        """Register a feature toggle.
        
        Args:
            name: Toggle name
            strategy: Toggle strategy
            **kwargs: Additional toggle parameters
        """
        with self._lock:
            toggle = FeatureToggle(
                name=name,
                strategy=strategy,
                **kwargs
            )
            self._toggles[name] = toggle

            logger.info(f"Registered feature toggle: {name} ({strategy.value})")

    def is_enabled(self, toggle_name: str, user_id: str | None = None, context: dict[str, Any] | None = None) -> bool:
        """Check if a feature toggle is enabled.
        
        Args:
            toggle_name: Name of the toggle
            user_id: User identifier for personalized toggles
            context: Additional context for toggle evaluation
            
        Returns:
            True if feature is enabled, False otherwise
        """
        with self._lock:
            if toggle_name not in self._toggles:
                logger.warning(f"Unknown feature toggle: {toggle_name}")
                return False

            toggle = self._toggles[toggle_name]

            if not toggle.enabled:
                return False

            if toggle.strategy == FeatureToggleStrategy.ON:
                return True
            elif toggle.strategy == FeatureToggleStrategy.OFF:
                return False
            elif toggle.strategy == FeatureToggleStrategy.PERCENTAGE:
                if user_id:
                    # Consistent hashing for user
                    user_hash = self._hash_user(user_id, toggle.name)
                    return user_hash < toggle.percentage
                else:
                    # Random for anonymous users
                    return random.random() * 100 < toggle.percentage
            elif toggle.strategy == FeatureToggleStrategy.USER_LIST:
                return user_id in toggle.user_list if user_id else False
            elif toggle.strategy == FeatureToggleStrategy.A_B_TEST:
                return self._evaluate_ab_test(toggle_name, user_id, context)

            return False

    def register_ab_test(
        self,
        name: str,
        control_percentage: float = 50.0,
        treatment_percentage: float = 50.0,
        **kwargs
    ) -> None:
        """Register an A/B test.
        
        Args:
            name: Test name
            control_percentage: Percentage for control group
            treatment_percentage: Percentage for treatment group
            **kwargs: Additional test parameters
        """
        with self._lock:
            ab_test = ABTestConfig(
                name=name,
                control_percentage=control_percentage,
                treatment_percentage=treatment_percentage,
                **kwargs
            )
            self._ab_tests[name] = ab_test

            logger.info(f"Registered A/B test: {name} ({control_percentage}/{treatment_percentage})")

    def get_ab_test_variant(self, test_name: str, user_id: str) -> str | None:
        """Get A/B test variant for a user.
        
        Args:
            test_name: Test name
            user_id: User identifier
            
        Returns:
            "control", "treatment", or None if not in test
        """
        with self._lock:
            if test_name not in self._ab_tests:
                return None

            ab_test = self._ab_tests[test_name]

            # Check test date bounds
            now = datetime.now()
            if ab_test.start_date and now < ab_test.start_date:
                return None
            if ab_test.end_date and now > ab_test.end_date:
                return None

            # Hash user to determine variant
            user_hash = self._hash_user(user_id, ab_test.name + ab_test.user_hash_salt)

            if user_hash < ab_test.control_percentage:
                return "control"
            elif user_hash < ab_test.control_percentage + ab_test.treatment_percentage:
                return "treatment"
            else:
                return None  # Not in test

    def update_toggle(self, name: str, **updates) -> None:
        """Update a feature toggle.
        
        Args:
            name: Toggle name
            **updates: Fields to update
        """
        with self._lock:
            if name not in self._toggles:
                raise ValueError(f"Unknown feature toggle: {name}")

            toggle = self._toggles[name]

            for field, value in updates.items():
                if hasattr(toggle, field):
                    setattr(toggle, field, value)
                    logger.info(f"Updated toggle {name}.{field} = {value}")

            toggle.updated_at = datetime.now()

    def get_toggle_status(self, name: str) -> dict[str, Any] | None:
        """Get status of a feature toggle.
        
        Args:
            name: Toggle name
            
        Returns:
            Toggle status dictionary or None if not found
        """
        with self._lock:
            if name not in self._toggles:
                return None

            toggle = self._toggles[name]
            return asdict(toggle)

    def list_toggles(self) -> dict[str, dict[str, Any]]:
        """List all feature toggles.
        
        Returns:
            Dictionary of toggle names to their status
        """
        with self._lock:
            return {name: asdict(toggle) for name, toggle in self._toggles.items()}

    def list_ab_tests(self) -> dict[str, dict[str, Any]]:
        """List all A/B tests.
        
        Returns:
            Dictionary of test names to their configuration
        """
        with self._lock:
            return {name: asdict(test) for name, test in self._ab_tests.items()}

    def _load_feature_toggles(self) -> None:
        """Load feature toggles from configuration manager."""
        if not self.config_manager:
            return

        toggles_config = self.config_manager.get("feature_toggles", {})
        for name, config in toggles_config.items():
            try:
                strategy = FeatureToggleStrategy(config.get("strategy", "on"))
                self.register_toggle(name, strategy, **config)
            except Exception as e:
                logger.error(f"Failed to load feature toggle {name}: {e}")

    def _hash_user(self, user_id: str, salt: str = "") -> float:
        """Hash user ID to a percentage (0-100).
        
        Args:
            user_id: User identifier
            salt: Salt for hashing
            
        Returns:
            Percentage between 0 and 100
        """
        hash_input = f"{user_id}:{salt}".encode()
        hash_value = hashlib.md5(hash_input).hexdigest()
        # Convert first 8 hex chars to int, then to percentage
        return (int(hash_value[:8], 16) % 10000) / 100.0

    def _evaluate_ab_test(self, toggle_name: str, user_id: str | None, context: dict[str, Any] | None) -> bool:
        """Evaluate A/B test for feature toggle.
        
        Args:
            toggle_name: Toggle name
            user_id: User identifier
            context: Additional context
            
        Returns:
            True if user is in treatment group
        """
        if not user_id:
            return False

        variant = self.get_ab_test_variant(toggle_name, user_id)
        return variant == "treatment"


# Global instances
_config_manager: ConfigurationManager | None = None
_feature_toggle_manager: FeatureToggleManager | None = None


def get_config_manager(config_dir: str | Path | None = None) -> ConfigurationManager:
    """Get global configuration manager.
    
    Args:
        config_dir: Configuration directory
        
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager


def get_feature_toggle_manager() -> FeatureToggleManager:
    """Get global feature toggle manager.
    
    Returns:
        FeatureToggleManager instance
    """
    global _feature_toggle_manager
    if _feature_toggle_manager is None:
        _feature_toggle_manager = FeatureToggleManager(get_config_manager())
    return _feature_toggle_manager


def feature_flag(flag_name: str, default: bool = False):
    """Decorator to conditionally execute function based on feature flag.
    
    Args:
        flag_name: Name of the feature flag
        default: Default value if flag not found
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            toggle_manager = get_feature_toggle_manager()

            # Try to get user_id from kwargs
            user_id = kwargs.get('user_id') or kwargs.get('_api_key_info', {}).get('key_id')

            if toggle_manager.is_enabled(flag_name, user_id=user_id):
                return func(*args, **kwargs)
            else:
                logger.debug(f"Feature flag '{flag_name}' is disabled, skipping function")
                return None

        return wrapper
    return decorator


def config_value(key: str, default: Any = None):
    """Decorator to inject configuration value into function.
    
    Args:
        key: Configuration key
        default: Default value if key not found
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config_manager = get_config_manager()
            value = config_manager.get(key, default)
            kwargs[f'config_{key.replace(".", "_")}'] = value
            return func(*args, **kwargs)

        return wrapper
    return decorator
