#!/usr/bin/env python3
"""Automated release management script for retrieval-free context compressor."""

import argparse
import subprocess
import sys
import re
import json
from pathlib import Path
from typing import Tuple, List, Optional
import tempfile
import shutil


class ReleaseManager:
    """Manages the release process including versioning, changelog, and publishing."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.repo_root = Path(__file__).parent.parent
        self.pyproject_path = self.repo_root / "pyproject.toml"
        self.changelog_path = self.repo_root / "CHANGELOG.md"
        
    def run_command(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command, respecting dry-run mode."""
        if self.dry_run:
            print(f"[DRY RUN] Would run: {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        else:
            print(f"Running: {' '.join(cmd)}")
            return subprocess.run(cmd, capture_output=True, text=True, check=check)
    
    def get_current_version(self) -> str:
        """Get the current version from pyproject.toml."""
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        match = re.search(r'version = "([^"]+)"', content)
        if not match:
            raise ValueError("Could not find version in pyproject.toml")
        
        return match.group(1)
    
    def bump_version(self, current_version: str, bump_type: str) -> str:
        """Bump version according to semantic versioning."""
        parts = current_version.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {current_version}")
        
        major, minor, patch = map(int, parts)
        
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")
        
        return f"{major}.{minor}.{patch}"
    
    def update_version_in_files(self, new_version: str) -> None:
        """Update version in all relevant files."""
        # Update pyproject.toml
        with open(self.pyproject_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
        
        if not self.dry_run:
            with open(self.pyproject_path, 'w') as f:
                f.write(content)
        
        # Update __init__.py
        init_path = self.repo_root / "src" / "retrieval_free" / "__init__.py"
        if init_path.exists():
            with open(init_path, 'r') as f:
                content = f.read()
            
            content = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{new_version}"', content)
            
            if not self.dry_run:
                with open(init_path, 'w') as f:
                    f.write(content)
        
        print(f"Updated version to {new_version}")
    
    def get_commits_since_last_tag(self) -> List[str]:
        """Get commit messages since the last tag."""
        try:
            # Get the latest tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                capture_output=True, text=True, check=True
            )
            last_tag = result.stdout.strip()
            
            # Get commits since last tag
            result = subprocess.run(
                ["git", "log", f"{last_tag}..HEAD", "--oneline"],
                capture_output=True, text=True, check=True
            )
            
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        except subprocess.CalledProcessError:
            # No previous tags, get all commits
            result = subprocess.run(
                ["git", "log", "--oneline"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split('\n')
    
    def categorize_commits(self, commits: List[str]) -> dict:
        """Categorize commits by type."""
        categories = {
            "Features": [],
            "Bug Fixes": [],
            "Documentation": [],
            "Performance": [],
            "Refactoring": [],
            "Testing": [],
            "Other": []
        }
        
        for commit in commits:
            if not commit.strip():
                continue
            
            # Extract commit message (remove hash)
            msg = ' '.join(commit.split()[1:])
            
            # Categorize based on conventional commit format
            if msg.startswith(('feat:', 'feat(')):
                categories["Features"].append(msg)
            elif msg.startswith(('fix:', 'fix(')):
                categories["Bug Fixes"].append(msg)
            elif msg.startswith(('docs:', 'docs(')):
                categories["Documentation"].append(msg)
            elif msg.startswith(('perf:', 'perf(')):
                categories["Performance"].append(msg)
            elif msg.startswith(('refactor:', 'refactor(')):
                categories["Refactoring"].append(msg)
            elif msg.startswith(('test:', 'test(')):
                categories["Testing"].append(msg)
            else:
                categories["Other"].append(msg)
        
        return categories
    
    def update_changelog(self, version: str, commits: List[str]) -> None:
        """Update CHANGELOG.md with new version information."""
        from datetime import datetime
        
        # Categorize commits
        categorized = self.categorize_commits(commits)
        
        # Generate changelog entry
        entry_lines = [
            f"## [{version}] - {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ]
        
        for category, commit_list in categorized.items():
            if commit_list:
                entry_lines.append(f"### {category}")
                entry_lines.append("")
                for commit in commit_list:
                    entry_lines.append(f"- {commit}")
                entry_lines.append("")
        
        # Read existing changelog
        if self.changelog_path.exists():
            with open(self.changelog_path, 'r') as f:
                existing_content = f.read()
        else:
            existing_content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
        
        # Insert new entry after the header
        lines = existing_content.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith('## [') or line.startswith('## Unreleased'):
                header_end = i
                break
        else:
            # No existing entries, add after header
            header_end = len(lines)
        
        # Insert new entry
        new_lines = lines[:header_end] + entry_lines + lines[header_end:]
        new_content = '\n'.join(new_lines)
        
        if not self.dry_run:
            with open(self.changelog_path, 'w') as f:
                f.write(new_content)
        
        print(f"Updated changelog with {len(commits)} commits")
    
    def run_tests(self) -> bool:
        """Run the test suite to ensure everything is working."""
        print("Running test suite...")
        
        try:
            result = self.run_command(["python", "-m", "pytest", "-v"], check=False)
            if result.returncode != 0:
                print("Tests failed!")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("All tests passed!")
            return True
        
        except FileNotFoundError:
            print("pytest not found, skipping tests")
            return True
    
    def run_linting(self) -> bool:
        """Run linting checks."""
        print("Running linting checks...")
        
        commands = [
            ["python", "-m", "ruff", "check", "src", "tests"],
            ["python", "-m", "black", "--check", "src", "tests"],
            ["python", "-m", "mypy", "src"]
        ]
        
        all_passed = True
        
        for cmd in commands:
            try:
                result = self.run_command(cmd, check=False)
                if result.returncode != 0:
                    print(f"Linting failed for: {' '.join(cmd)}")
                    print(result.stdout)
                    print(result.stderr)
                    all_passed = False
            except FileNotFoundError:
                print(f"Tool not found: {cmd[2]}, skipping")
        
        if all_passed:
            print("All linting checks passed!")
        
        return all_passed
    
    def build_package(self) -> bool:
        """Build the Python package."""
        print("Building package...")
        
        try:
            # Clean previous builds
            dist_dir = self.repo_root / "dist"
            if dist_dir.exists():
                shutil.rmtree(dist_dir)
            
            # Build package
            result = self.run_command(["python", "-m", "build"], check=False)
            if result.returncode != 0:
                print("Package build failed!")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print("Package built successfully!")
            return True
        
        except FileNotFoundError:
            print("build module not found, install with: pip install build")
            return False
    
    def create_git_tag(self, version: str) -> None:
        """Create a git tag for the release."""
        tag_name = f"v{version}"
        
        # Create annotated tag
        self.run_command([
            "git", "tag", "-a", tag_name, 
            "-m", f"Release version {version}"
        ])
        
        print(f"Created git tag: {tag_name}")
    
    def commit_changes(self, version: str) -> None:
        """Commit version changes."""
        # Add changed files
        self.run_command(["git", "add", str(self.pyproject_path)])
        self.run_command(["git", "add", str(self.changelog_path)])
        
        init_path = self.repo_root / "src" / "retrieval_free" / "__init__.py"
        if init_path.exists():
            self.run_command(["git", "add", str(init_path)])
        
        # Commit changes
        self.run_command([
            "git", "commit", "-m", f"chore: bump version to {version}"
        ])
        
        print(f"Committed version changes for {version}")
    
    def publish_package(self, test_pypi: bool = False) -> bool:
        """Publish package to PyPI."""
        repository = "testpypi" if test_pypi else "pypi"
        print(f"Publishing package to {repository}...")
        
        try:
            cmd = ["python", "-m", "twine", "upload"]
            if test_pypi:
                cmd.extend(["--repository", "testpypi"])
            cmd.append("dist/*")
            
            result = self.run_command(cmd, check=False)
            if result.returncode != 0:
                print(f"Package publishing to {repository} failed!")
                print(result.stdout)
                print(result.stderr)
                return False
            
            print(f"Package published to {repository} successfully!")
            return True
        
        except FileNotFoundError:
            print("twine not found, install with: pip install twine")
            return False
    
    def release(self, bump_type: str, skip_tests: bool = False, 
                skip_publish: bool = False, test_pypi: bool = False) -> None:
        """Execute the complete release process."""
        print(f"Starting release process (bump: {bump_type})")
        
        if self.dry_run:
            print("DRY RUN MODE - No changes will be made")
        
        # Get current version and calculate new version
        current_version = self.get_current_version()
        new_version = self.bump_version(current_version, bump_type)
        
        print(f"Current version: {current_version}")
        print(f"New version: {new_version}")
        
        # Run tests and linting
        if not skip_tests:
            if not self.run_tests():
                print("Tests failed, aborting release")
                return
            
            if not self.run_linting():
                print("Linting failed, aborting release")
                return
        
        # Get commits for changelog
        commits = self.get_commits_since_last_tag()
        print(f"Found {len(commits)} commits since last release")
        
        # Update version and changelog
        self.update_version_in_files(new_version)
        self.update_changelog(new_version, commits)
        
        # Build package
        if not self.build_package():
            print("Package build failed, aborting release")
            return
        
        # Commit changes and create tag
        if not self.dry_run:
            self.commit_changes(new_version)
            self.create_git_tag(new_version)
        
        # Publish package
        if not skip_publish:
            if not self.publish_package(test_pypi):
                print("Package publishing failed")
                return
        
        print(f"Release {new_version} completed successfully!")
        
        if not self.dry_run:
            print("Don't forget to push the changes and tag:")
            print(f"  git push origin main")
            print(f"  git push origin v{new_version}")


def main():
    """Main release script."""
    parser = argparse.ArgumentParser(description="Automated release management")
    parser.add_argument("bump_type", choices=["major", "minor", "patch"],
                       help="Type of version bump")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests and linting")
    parser.add_argument("--skip-publish", action="store_true",
                       help="Skip publishing to PyPI")
    parser.add_argument("--test-pypi", action="store_true",
                       help="Publish to Test PyPI instead of PyPI")
    
    args = parser.parse_args()
    
    # Create release manager
    release_manager = ReleaseManager(dry_run=args.dry_run)
    
    try:
        release_manager.release(
            bump_type=args.bump_type,
            skip_tests=args.skip_tests,
            skip_publish=args.skip_publish,
            test_pypi=args.test_pypi
        )
    except KeyboardInterrupt:
        print("\nRelease cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Release failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()