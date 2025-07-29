# Development Guide

This guide helps you set up a development environment for the Retrieval-Free Context Compressor.

## Quick Setup

```bash
git clone https://github.com/yourusername/retrieval-free-context-compressor.git
cd retrieval-free-context-compressor
make install-dev
```

## Prerequisites

- **Python 3.10+** (3.11+ recommended)
- **Git** for version control
- **Make** for build automation (optional)
- **CUDA 11.8+** for GPU acceleration (optional)

## Environment Setup

### Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

### Development Installation
```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Optional Dependencies
```bash
# For GPU support
pip install -e ".[gpu]"

# For optimization features  
pip install -e ".[optimization]"

# For evaluation tools
pip install -e ".[evaluation]"

# Install everything
pip install -e ".[all]"
```

## Development Workflow

### 1. Code Changes
- Create feature branch: `git checkout -b feature-name`
- Make changes following code style guidelines
- Add tests for new functionality

### 2. Quality Checks
```bash
# Run all checks
make ci

# Individual checks
make lint       # Linting with ruff
make format     # Code formatting with black  
make typecheck  # Type checking with mypy
make test       # Run tests with pytest
```

### 3. Testing
```bash
# Basic test run
pytest

# With coverage
pytest --cov=retrieval_free

# Specific test file
pytest tests/test_basic.py

# Run only fast tests
pytest -m "not slow"

# Run GPU tests (requires CUDA)
pytest -m gpu
```

## IDE Configuration

### VS Code (Recommended)
The repository includes VS Code configuration:
- Python interpreter setup
- Formatting and linting integration
- Test discovery
- Recommended extensions

Install recommended extensions when prompted.

### PyCharm
1. Open project in PyCharm
2. Configure Python interpreter to use project venv
3. Enable Black formatting in Settings → Tools → External Tools
4. Configure pytest as test runner

## Project Structure

```
├── src/retrieval_free/     # Main package
│   ├── core/              # Core compression algorithms
│   ├── streaming/         # Streaming compression
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics
│   ├── plugins/           # Framework integrations
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── fixtures/         # Test data
├── docs/                 # Documentation
├── scripts/              # Utility scripts
└── examples/             # Usage examples
```

## Common Development Tasks

### Adding New Features
1. Create module in appropriate `src/retrieval_free/` subdirectory
2. Add comprehensive tests in `tests/`
3. Update `__init__.py` to export public APIs
4. Add documentation and examples

### Running Benchmarks
```bash
# Basic compression benchmark
python scripts/benchmark_compression.py

# Full evaluation suite
python scripts/evaluate_models.py --models base-8x --datasets nq
```

### Building Documentation
```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

### Release Process
1. Update version in `pyproject.toml` and `__init__.py`
2. Update CHANGELOG.md
3. Create PR with version bump
4. Tag release: `git tag v0.1.1`
5. Push tag: `git push origin v0.1.1`

## Performance Development

### Profiling
```python
# Profile compression performance
python -m cProfile -o profile.stats scripts/profile_compression.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

### Memory Profiling
```bash
pip install memory-profiler
python -m memory_profiler scripts/memory_test.py
```

### GPU Profiling
```bash
# Install NVIDIA profiling tools
pip install nvidia-ml-py3

# Profile GPU usage
python scripts/gpu_benchmark.py
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Reinstall in editable mode
pip install -e .
```

**CUDA Issues**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

**Test Failures**
```bash
# Run tests in verbose mode
pytest -v

# Run single failing test
pytest tests/test_module.py::test_function -v
```

### Getting Help
- Check existing GitHub issues
- Join Discord community (link in README)
- Start GitHub Discussion for questions

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

Happy coding! 🚀