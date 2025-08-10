"""Internationalization and localization support."""

import json
import logging
from functools import lru_cache
from pathlib import Path


logger = logging.getLogger(__name__)


class LocalizationManager:
    """Manager for internationalization and localization."""

    def __init__(
        self,
        locales_dir: str | None = None,
        default_locale: str = "en",
        supported_locales: list[str] | None = None,
    ):
        """Initialize localization manager.

        Args:
            locales_dir: Directory containing locale files
            default_locale: Default locale code
            supported_locales: List of supported locale codes
        """
        self.default_locale = default_locale
        self.current_locale = default_locale

        if locales_dir is None:
            self.locales_dir = Path(__file__).parent / "locales"
        else:
            self.locales_dir = Path(locales_dir)

        self.supported_locales = supported_locales or [
            "en",
            "es",
            "fr",
            "de",
            "ja",
            "zh",
            "pt",
            "ru",
            "it",
            "ko",
        ]

        # Translation cache
        self._translations: dict[str, dict[str, str]] = {}
        self._load_translations()

    def _load_translations(self) -> None:
        """Load all translation files."""
        for locale in self.supported_locales:
            self._load_locale_translations(locale)

    def _load_locale_translations(self, locale: str) -> None:
        """Load translations for a specific locale."""
        locale_file = self.locales_dir / f"{locale}.json"

        if locale_file.exists():
            try:
                with open(locale_file, encoding="utf-8") as f:
                    translations = json.load(f)
                    self._translations[locale] = translations
                    logger.debug(
                        f"Loaded {len(translations)} translations for {locale}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load translations for {locale}: {e}")
                self._translations[locale] = {}
        else:
            # Create empty translation dict
            self._translations[locale] = {}

            # Create default translations for English
            if locale == "en":
                self._create_default_translations(locale)

    def _create_default_translations(self, locale: str) -> None:
        """Create default English translations."""
        default_translations = {
            # General
            "loading": "Loading",
            "processing": "Processing",
            "completed": "Completed",
            "failed": "Failed",
            "error": "Error",
            "warning": "Warning",
            "info": "Information",
            # Compression
            "compressing_text": "Compressing text",
            "compression_complete": "Compression complete",
            "compression_failed": "Compression failed",
            "compression_ratio": "Compression ratio",
            "original_tokens": "Original tokens",
            "compressed_tokens": "Compressed tokens",
            "processing_time": "Processing time",
            # Validation
            "validation_failed": "Validation failed",
            "invalid_input": "Invalid input",
            "input_too_long": "Input text is too long",
            "malicious_content": "Potentially malicious content detected",
            "security_violation": "Security violation",
            # Models
            "loading_model": "Loading model",
            "model_loaded": "Model loaded successfully",
            "model_load_failed": "Failed to load model",
            "model_not_found": "Model not found",
            # Performance
            "cache_hit": "Cache hit",
            "cache_miss": "Cache miss",
            "memory_usage": "Memory usage",
            "gpu_memory": "GPU memory",
            "throughput": "Throughput",
            # File operations
            "file_not_found": "File not found",
            "permission_denied": "Permission denied",
            "invalid_file_format": "Invalid file format",
            # Network
            "connection_error": "Connection error",
            "timeout": "Operation timed out",
            "download_failed": "Download failed",
            # Configuration
            "config_error": "Configuration error",
            "invalid_parameter": "Invalid parameter",
            "missing_parameter": "Missing required parameter",
        }

        self._translations[locale] = default_translations

        # Save to file
        self.locales_dir.mkdir(parents=True, exist_ok=True)
        locale_file = self.locales_dir / f"{locale}.json"

        try:
            with open(locale_file, "w", encoding="utf-8") as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            logger.info(f"Created default translations file: {locale_file}")
        except Exception as e:
            logger.warning(f"Failed to save default translations: {e}")

    def set_locale(self, locale: str) -> bool:
        """Set current locale.

        Args:
            locale: Locale code to set

        Returns:
            True if locale was set successfully
        """
        if locale in self.supported_locales:
            self.current_locale = locale
            logger.info(f"Locale set to: {locale}")
            return True
        else:
            logger.warning(f"Unsupported locale: {locale}")
            return False

    def get_locale(self) -> str:
        """Get current locale.

        Returns:
            Current locale code
        """
        return self.current_locale

    def get_supported_locales(self) -> list[str]:
        """Get list of supported locales.

        Returns:
            List of supported locale codes
        """
        return self.supported_locales.copy()

    @lru_cache(maxsize=1000)
    def translate(
        self, key: str, locale: str | None = None, default: str | None = None, **kwargs
    ) -> str:
        """Translate a key to the current or specified locale.

        Args:
            key: Translation key
            locale: Specific locale (uses current if None)
            default: Default text if translation not found
            **kwargs: Variables for string formatting

        Returns:
            Translated text
        """
        target_locale = locale or self.current_locale

        # Try target locale
        if target_locale in self._translations:
            translation = self._translations[target_locale].get(key)
            if translation:
                try:
                    return translation.format(**kwargs) if kwargs else translation
                except KeyError as e:
                    logger.warning(f"Missing format key {e} in translation '{key}'")
                    return translation

        # Fallback to default locale
        if target_locale != self.default_locale:
            if self.default_locale in self._translations:
                translation = self._translations[self.default_locale].get(key)
                if translation:
                    try:
                        return translation.format(**kwargs) if kwargs else translation
                    except KeyError:
                        return translation

        # Use provided default or key itself
        fallback = default or key
        try:
            return fallback.format(**kwargs) if kwargs else fallback
        except KeyError:
            return fallback

    def add_translation(self, locale: str, key: str, translation: str) -> None:
        """Add a translation for a specific locale.

        Args:
            locale: Locale code
            key: Translation key
            translation: Translated text
        """
        if locale not in self._translations:
            self._translations[locale] = {}

        self._translations[locale][key] = translation

        # Clear cache
        self.translate.cache_clear()

    def add_translations(self, locale: str, translations: dict[str, str]) -> None:
        """Add multiple translations for a locale.

        Args:
            locale: Locale code
            translations: Dictionary of key-value translations
        """
        if locale not in self._translations:
            self._translations[locale] = {}

        self._translations[locale].update(translations)

        # Clear cache
        self.translate.cache_clear()

    def get_completion_status(self) -> dict[str, float]:
        """Get translation completion status for all locales.

        Returns:
            Dictionary mapping locale to completion percentage
        """
        if not self._translations.get(self.default_locale):
            return {}

        base_count = len(self._translations[self.default_locale])
        status = {}

        for locale in self.supported_locales:
            if locale in self._translations:
                locale_count = len(self._translations[locale])
                completion = (locale_count / base_count) * 100 if base_count > 0 else 0
                status[locale] = completion
            else:
                status[locale] = 0.0

        return status

    def export_for_translation(self, locale: str, output_file: str) -> None:
        """Export untranslated keys for a locale.

        Args:
            locale: Target locale
            output_file: Output file path
        """
        if self.default_locale not in self._translations:
            logger.error("No base translations found")
            return

        base_keys = set(self._translations[self.default_locale].keys())
        translated_keys = set(self._translations.get(locale, {}).keys())

        untranslated = base_keys - translated_keys

        export_data = {
            "locale": locale,
            "untranslated_count": len(untranslated),
            "total_count": len(base_keys),
            "completion": ((len(base_keys) - len(untranslated)) / len(base_keys)) * 100,
            "translations_needed": {
                key: self._translations[self.default_locale][key]
                for key in sorted(untranslated)
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(untranslated)} untranslated keys to {output_file}")


# Global localization manager instance
_localization_manager: LocalizationManager | None = None


def init_localization(
    locales_dir: str | None = None,
    default_locale: str = "en",
    supported_locales: list[str] | None = None,
) -> LocalizationManager:
    """Initialize global localization manager.

    Args:
        locales_dir: Directory containing locale files
        default_locale: Default locale code
        supported_locales: List of supported locale codes

    Returns:
        Initialized LocalizationManager instance
    """
    global _localization_manager
    _localization_manager = LocalizationManager(
        locales_dir=locales_dir,
        default_locale=default_locale,
        supported_locales=supported_locales,
    )
    return _localization_manager


def get_localization_manager() -> LocalizationManager:
    """Get global localization manager.

    Returns:
        LocalizationManager instance

    Raises:
        RuntimeError: If localization not initialized
    """
    global _localization_manager
    if _localization_manager is None:
        # Auto-initialize with defaults
        _localization_manager = LocalizationManager()
    return _localization_manager


def set_locale(locale: str) -> bool:
    """Set current locale.

    Args:
        locale: Locale code to set

    Returns:
        True if locale was set successfully
    """
    manager = get_localization_manager()
    return manager.set_locale(locale)


def get_locale() -> str:
    """Get current locale.

    Returns:
        Current locale code
    """
    manager = get_localization_manager()
    return manager.get_locale()


def translate(
    key: str, locale: str | None = None, default: str | None = None, **kwargs
) -> str:
    """Translate a key to the current or specified locale.

    Args:
        key: Translation key
        locale: Specific locale (uses current if None)
        default: Default text if translation not found
        **kwargs: Variables for string formatting

    Returns:
        Translated text
    """
    manager = get_localization_manager()
    return manager.translate(key, locale=locale, default=default, **kwargs)


# Convenience alias
_ = translate


def detect_system_locale() -> str:
    """Detect system locale.

    Returns:
        Detected locale code or 'en' as fallback
    """
    try:
        import locale

        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            # Extract language code (e.g., 'en_US' -> 'en')
            lang_code = system_locale.split("_")[0].lower()

            # Check if supported
            manager = get_localization_manager()
            if lang_code in manager.get_supported_locales():
                return lang_code
    except Exception as e:
        logger.debug(f"Failed to detect system locale: {e}")

    return "en"  # Fallback to English


def auto_set_locale() -> str:
    """Auto-detect and set system locale.

    Returns:
        Set locale code
    """
    detected_locale = detect_system_locale()
    set_locale(detected_locale)
    return detected_locale
