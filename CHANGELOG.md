# Changelog

All notable changes to the Retrieval-Free Context Compressor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of retrieval-free context compression
- Support for 8x compression with improved F1 scores
- Streaming compression for infinite contexts
- Multi-document compression capabilities
- Selective compression based on content type
- Integration plugins for popular frameworks
- Comprehensive testing and evaluation suite

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-12-XX

### Added
- Initial project structure and foundational components
- Core compression interfaces and base classes
- Basic documentation and development environment setup
- Pre-commit hooks and code quality tools
- Docker containerization support

---

## Template for Future Releases

Copy the template below for new releases:

```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security improvements
```

## Release Process

1. Update version in `pyproject.toml` and `src/retrieval_free/__init__.py`
2. Update `CHANGELOG.md` with new version and changes
3. Create a new git tag: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. Push tags: `git push origin --tags`
5. GitHub Actions will automatically create a release