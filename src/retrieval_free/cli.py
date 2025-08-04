"""Command-line interface module."""

from .plugins import main

# Re-export main function for CLI entry point
__all__ = ["main"]