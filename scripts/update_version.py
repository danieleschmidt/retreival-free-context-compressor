#!/usr/bin/env python3
"""
Version update script for semantic-release automation.
Updates version in pyproject.toml and __init__.py files.
"""

import sys
import re
from pathlib import Path


def update_pyproject_version(version: str) -> None:
    """Update version in pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Update version line
    updated_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{version}"',
        content
    )
    
    pyproject_path.write_text(updated_content)
    print(f"Updated pyproject.toml version to {version}")


def update_init_version(version: str) -> None:
    """Update version in __init__.py."""
    init_path = Path("src/retrieval_free/__init__.py")
    
    if not init_path.exists():
        # Create __init__.py with version if it doesn't exist
        init_path.write_text(f'"""Retrieval-Free Context Compressor."""\n\n__version__ = "{version}"\n')
        print(f"Created __init__.py with version {version}")
        return
    
    content = init_path.read_text()
    
    # Check if __version__ exists, if not add it
    if "__version__" not in content:
        content += f'\n__version__ = "{version}"\n'
    else:
        # Update existing version
        content = re.sub(
            r'__version__ = "[^"]+"',
            f'__version__ = "{version}"',
            content
        )
    
    init_path.write_text(content)
    print(f"Updated __init__.py version to {version}")


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python update_version.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Remove 'v' prefix if present
    if version.startswith('v'):
        version = version[1:]
    
    try:
        update_pyproject_version(version)
        update_init_version(version)
        print(f"✅ Successfully updated version to {version}")
    except Exception as e:
        print(f"❌ Error updating version: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()