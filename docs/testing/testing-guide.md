# Testing Guide

This guide covers the comprehensive testing strategy for the Retrieval-Free Context Compressor.

## Testing Philosophy

Our testing approach follows the **Testing Pyramid** principle:

1. **Unit Tests (70%)**: Fast, focused tests for individual components
2. **Integration Tests (20%)**: Tests for component interactions
3. **End-to-End Tests (10%)**: Full system validation tests

## Test Categories

### Unit Tests
Located in `tests/unit/`, these test individual functions and classes in isolation.

```bash
# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=retrieval_free --cov-report=html
```

### Integration Tests  
Located in `tests/integration/`, these test interactions between components.

```bash
# Run integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_end_to_end.py -v
```

### Performance Tests
Located in `tests/performance/`, these benchmark system performance.

```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Generate performance report
pytest tests/performance/ --benchmark-json=results/performance.json
```

### Property-Based Tests
Located in `tests/property/`, these use hypothesis for property-based testing.

```bash
# Run property-based tests
pytest tests/property/ -v

# Run with more examples
pytest tests/property/ --hypothesis-max-examples=1000
```

## Test Configuration

### Pytest Configuration

The main test configuration is in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "--strict-markers",
    "--strict-config", 
    "--cov=retrieval_free",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80"
]
```

### Test Markers

Use pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests that take >5 seconds
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance benchmarks
- `@pytest.mark.memory`: Memory usage tests

```python
@pytest.mark.slow
@pytest.mark.gpu
def test_large_model_compression():
    # Test implementation
    pass
```

### Test Data and Fixtures

#### Shared Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def sample_documents():
    """Sample documents for testing."""
    return TestDataLoader.load_sample_documents()

@pytest.fixture
def temp_model_dir():
    """Temporary directory for model files."""
    # Implementation
```

#### Test Data

Test data is organized in `tests/fixtures/`:

- `test_data.py`: Programmatically generated test data
- `sample_documents/`: Real document samples
- `benchmarks/`: Benchmark datasets

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_compression_core.py

# Run specific test function
pytest tests/unit/test_compression_core.py::test_token_encoding

# Run tests matching pattern
pytest -k "compression"
```

### Test Selection

```bash
# Run only fast tests
pytest -m "not slow"

# Run only GPU tests
pytest -m "gpu"

# Run integration tests
pytest -m "integration"

# Run performance benchmarks
pytest -m "performance" --benchmark-only
```

### Parallel Testing

```bash
# Run tests in parallel
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

### Coverage Testing

```bash
# Run with coverage
pytest --cov=retrieval_free

# Generate HTML coverage report
pytest --cov=retrieval_free --cov-report=html

# Fail if coverage below threshold
pytest --cov=retrieval_free --cov-fail-under=80
```

## Test Environment Setup

### Local Development

```bash
# Install test dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run test suite
make test
```

### CI/CD Environment

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly builds

Environment variables for CI:
```bash
export PYTHONPATH=src
export PYTEST_CURRENT_TEST=true
export TEST_OUTPUT_DIR=tests/output
```

### GPU Testing

For GPU-enabled tests:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Run GPU tests
pytest -m "gpu" -v

# Skip GPU tests if not available
pytest -m "not gpu"
```

## Writing Effective Tests

### Test Structure

Follow the **Arrange-Act-Assert** pattern:

```python
def test_compression_ratio():
    # Arrange
    original_text = "Sample document text"
    expected_ratio = 8.0
    
    # Act
    compressed = compressor.compress(original_text)
    actual_ratio = len(original_text) / len(compressed)
    
    # Assert
    assert abs(actual_ratio - expected_ratio) < 0.1
```

### Test Naming

Use descriptive test names that explain what is being tested:

```python
def test_compression_preserves_semantic_meaning():
    """Test that compression maintains semantic content for QA tasks."""
    
def test_streaming_compression_handles_window_overlap():
    """Test that streaming compression properly manages overlapping windows."""
```

### Mocking and Fixtures

Use mocks for external dependencies:

```python
@patch('retrieval_free.models.load_pretrained_model')
def test_compressor_initialization(mock_load_model):
    mock_load_model.return_value = Mock()
    compressor = AutoCompressor.from_pretrained("test-model")
    assert compressor is not None
```

### Parameterized Tests

Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("compression_ratio,expected_quality", [
    (4.0, 0.95),
    (8.0, 0.90),
    (16.0, 0.85),
])
def test_compression_quality_tradeoff(compression_ratio, expected_quality):
    # Test implementation
    pass
```

## Performance Testing

### Benchmark Configuration

Performance tests use `pytest-benchmark`:

```python
def test_compression_speed(benchmark):
    result = benchmark(compressor.compress, large_document)
    assert result is not None
```

### Memory Profiling

Use `memory-profiler` for memory tests:

```python
@pytest.mark.memory
def test_memory_usage():
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    compressor.compress(large_document)
    final_memory = process.memory_info().rss
    
    memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
    assert memory_increase < 500  # Less than 500MB
```

### GPU Performance

Test GPU utilization and memory:

```python
@pytest.mark.gpu
def test_gpu_memory_efficiency():
    import torch
    torch.cuda.empty_cache()
    
    initial_memory = torch.cuda.memory_allocated()
    result = gpu_compressor.compress(document)
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_used = (peak_memory - initial_memory) / 1024**2  # MB
    assert memory_used < 1000  # Less than 1GB
```

## Quality Assurance

### Test Quality Metrics

Monitor these metrics:
- **Test Coverage**: >80% line coverage
- **Test Speed**: Unit tests <100ms each
- **Test Reliability**: <1% flaky test rate
- **Documentation Coverage**: All public APIs tested

### Continuous Quality

```bash
# Run quality checks
make ci

# Individual quality checks
make lint          # Code linting
make typecheck     # Type checking
make test          # Run test suite
make security      # Security scanning
```

### Test Data Quality

Ensure test data is:
- **Representative**: Covers real-world scenarios
- **Diverse**: Multiple domains and content types
- **Maintained**: Regularly updated and validated
- **Realistic**: Appropriate size and complexity

## Troubleshooting

### Common Issues

**Tests fail on CI but pass locally**
- Check environment differences
- Verify dependency versions
- Check for race conditions

**GPU tests fail**
- Verify CUDA installation
- Check GPU memory availability
- Ensure proper device selection

**Performance tests unstable**
- Run multiple iterations
- Check system load
- Use consistent hardware

**Memory tests fail**
- Clear caches before testing
- Monitor background processes
- Use memory profiling tools

### Debug Strategies

```bash
# Run single test with debugging
pytest -s -vv tests/test_specific.py::test_function

# Drop into debugger on failure
pytest --pdb

# Run with detailed output
pytest --tb=long -v

# Profile test execution
pytest --profile
```

## Best Practices

### Test Development

1. **Write tests first** (TDD approach when possible)
2. **Test behavior, not implementation**
3. **Use meaningful assertions**
4. **Keep tests simple and focused**
5. **Avoid test interdependencies**

### Test Maintenance

1. **Regularly review and update tests**
2. **Remove obsolete tests**
3. **Refactor test code like production code**
4. **Monitor test execution time**
5. **Keep test data current**

### Performance Testing

1. **Establish baseline metrics**
2. **Test on representative hardware**
3. **Monitor performance trends**
4. **Set realistic performance goals**
5. **Profile before optimizing**

---

For more information, see:
- [Contributing Guide](../../CONTRIBUTING.md)
- [Development Setup](../../DEVELOPMENT.md)
- [Performance Guidelines](performance-guidelines.md)