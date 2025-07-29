# CI/CD Workflow Documentation

This document describes the required GitHub Actions workflows for the retrieval-free-context-compressor project.

## Required Workflows

### 1. Continuous Integration (ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: ruff check src tests
    
    - name: Format check with black
      run: black --check src tests
    
    - name: Type check with mypy
      run: mypy src
    
    - name: Test with pytest
      run: pytest --cov=retrieval_free --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning (security.yml)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Mondays

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        directory: src
    
    - name: Run Safety Check
      run: |
        pip install safety
        safety check
    
    - name: Dependency Review
      uses: actions/dependency-review-action@v3
      if: github.event_name == 'pull_request'
```

### 3. Release Management (release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Test package
      run: |
        pip install dist/*.whl
        python -c "import retrieval_free; print(retrieval_free.__version__)"
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        draft: false
        prerelease: false
```

## Setup Instructions

1. **Create `.github/workflows/` directory** in your repository
2. **Add the three workflow files** above with appropriate names
3. **Configure repository secrets** for:
   - `CODECOV_TOKEN` (for coverage reporting)
   - `PYPI_API_TOKEN` (for package publishing)
4. **Enable security features** in repository settings:
   - Dependency graph
   - Dependabot alerts
   - Security advisories

## Branch Protection Rules

Configure the following branch protection rules for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging:
  - `test (3.10)`
  - `test (3.11)` 
  - `test (3.12)`
  - `security`
- Require branches to be up to date before merging
- Restrict pushes that create files that match a pattern: `**/*.key`, `**/*.pem`

## Manual Setup Required

**Important**: GitHub Actions workflows cannot be automatically created by automation tools. 
After implementing the code changes, manually:

1. Copy the workflow YAML content above into `.github/workflows/` files
2. Configure repository settings for security scanning
3. Add required secrets to repository settings
4. Set up branch protection rules

This ensures proper CI/CD pipeline functionality and security compliance.