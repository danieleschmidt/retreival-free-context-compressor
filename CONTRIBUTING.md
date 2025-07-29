# Contributing to Retrieval-Free Context Compressor

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Quick Start

1. **Fork the repository** and clone your fork
2. **Set up development environment**:
   ```bash
   cd retrieval-free-context-compressor
   make install-dev
   ```
3. **Create a feature branch**: `git checkout -b feature-name`
4. **Make your changes** following our guidelines below
5. **Run tests and checks**: `make ci`
6. **Submit a pull request**

## Development Setup

### Prerequisites
- Python 3.10+
- Git
- Make (optional, for convenience commands)

### Installation
```bash
# Clone your fork
git clone https://github.com/yourusername/retrieval-free-context-compressor.git
cd retrieval-free-context-compressor

# Install in development mode
make install-dev
# OR manually:
pip install -e ".[dev]"
pre-commit install
```

## Development Guidelines

### Code Style
- **Black** for code formatting (line length: 88)
- **Ruff** for linting and import sorting
- **Type hints** required for all public APIs
- **Docstrings** required for all public functions/classes

### Testing
- Write tests for all new features
- Maintain >80% test coverage
- Use descriptive test names
- Test both success and error cases

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new public APIs
- Include examples in docstrings when helpful

## Pull Request Process

1. **Create focused PRs** - one feature/fix per PR
2. **Write clear descriptions** - explain what and why
3. **Include tests** - ensure new code is tested
4. **Update documentation** - keep docs current
5. **Follow commit conventions**:
   ```
   type(scope): description
   
   feat(compression): add streaming compression support
   fix(core): handle empty document edge case
   docs(readme): update installation instructions
   ```

### PR Checklist
- [ ] Tests pass locally (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions

## Types of Contributions

### 🐛 Bug Reports
- Use GitHub Issues with bug template
- Include minimal reproduction case
- Provide environment details (Python version, OS, etc.)

### ✨ Feature Requests
- Use GitHub Issues with feature template
- Explain the use case and proposed solution
- Consider starting with a discussion first

### 🔧 Code Contributions
Priority areas for contributions:
- **New compression algorithms**
- **Performance optimizations**
- **Integration examples** (LangChain, HuggingFace, etc.)
- **Multilingual support**
- **Documentation improvements**

### 📚 Documentation
- README improvements
- API documentation
- Tutorials and examples
- Performance guides

## Development Commands

```bash
# Install dependencies
make install-dev

# Run all checks (CI pipeline)
make ci

# Individual commands
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Check code style
make format        # Format code
make typecheck     # Type checking
make clean         # Clean build artifacts
```

## Code Organization

```
src/retrieval_free/
├── core/           # Core compression algorithms
├── streaming/      # Streaming compression
├── training/       # Model training utilities
├── evaluation/     # Evaluation metrics
├── plugins/        # Framework integrations
└── utils/          # Shared utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── fixtures/       # Test data and fixtures

docs/
├── tutorials/      # User tutorials
├── api/           # API documentation
└── workflows/     # CI/CD documentation
```

## Testing Guidelines

### Test Structure
```python
def test_feature_name():
    """Test description of what is being tested."""
    # Arrange
    input_data = setup_test_data()
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    assert result.expected_property == expected_value
```

### Test Categories
- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical paths
- **GPU tests**: Test CUDA functionality (marked with `@pytest.mark.gpu`)

## Security Guidelines

- **No hardcoded secrets** in code or tests
- **Validate all inputs** in public APIs
- **Use secure defaults** for cryptographic operations
- **Report security issues** privately via email

## Getting Help

- **GitHub Discussions** for questions and ideas
- **GitHub Issues** for bugs and feature requests
- **Discord** for real-time chat (link in README)
- **Documentation** at https://retrieval-free.readthedocs.io

## Recognition

Contributors are recognized through:
- **GitHub contributors page**
- **Release notes** for significant contributions
- **Hall of fame** in documentation

Thank you for contributing to making long-context processing more efficient! 🚀