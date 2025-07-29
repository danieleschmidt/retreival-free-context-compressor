# Required GitHub Actions Workflows

This document outlines the essential CI/CD workflows that need to be implemented for this repository.

## Core Workflows Required

### 1. Continuous Integration (`ci.yml`)

```yaml
# Location: .github/workflows/ci.yml
# Purpose: Run tests, linting, and type checking on every push and PR

name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run pre-commit hooks
      run: pre-commit run --all-files
    
    - name: Run tests
      run: |
        pytest --cov=retrieval_free --cov-report=xml
        
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### 2. Security Scanning (`security.yml`)

```yaml
# Location: .github/workflows/security.yml
# Purpose: Security scanning, dependency check, SBOM generation

name: Security
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6AM UTC

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json
        
    - name: Upload SBOM artifact
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.json
```

### 3. Release Automation (`release.yml`)

```yaml
# Location: .github/workflows/release.yml
# Purpose: Automated releases with semantic versioning

name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### 4. Performance Benchmarks (`benchmarks.yml`)

```yaml
# Location: .github/workflows/benchmarks.yml
# Purpose: Performance regression detection

name: Benchmarks
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        
    - name: Run benchmarks
      run: |
        python benchmarks/compression_benchmark.py --output=benchmark_results.json
        
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'customSmallerIsBetter'
        output-file-path: benchmark_results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        comment-on-alert: true
        alert-threshold: '150%'
```

### 5. Docker Build (`docker.yml`)

```yaml
# Location: .github/workflows/docker.yml
# Purpose: Build and publish Docker images

name: Docker
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Login to Docker Hub
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          yourusername/retrieval-free:latest
          yourusername/retrieval-free:${{ github.sha }}
```

## Implementation Instructions

1. **Create Workflow Files**: Copy each workflow template to `.github/workflows/` with the specified filename
2. **Configure Secrets**: Add required secrets in GitHub repository settings:
   - `PYPI_API_TOKEN`: For PyPI publishing
   - `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`: For Docker Hub
   - `CODECOV_TOKEN`: For code coverage reporting

3. **Branch Protection**: Configure branch protection rules for `main`:
   - Require status checks: CI, Security
   - Require up-to-date branches
   - Restrict pushes to maintainers

4. **Repository Settings**:
   - Enable Dependabot alerts
   - Enable code scanning alerts
   - Configure merge button options (squash merge recommended)

## Workflow Dependencies

### Required GitHub Actions Marketplace Actions:
- `actions/checkout@v4`
- `actions/setup-python@v4`
- `github/codeql-action@v2`
- `aquasecurity/trivy-action@master`
- `codecov/codecov-action@v3`

### Required Repository Secrets:
- `PYPI_API_TOKEN` (for releases)
- `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` (for Docker)
- `CODECOV_TOKEN` (for coverage)

### Optional Enhancements:
- Slack/Discord notifications for failed builds
- Matrix testing across multiple OS (Linux, macOS, Windows)
- GPU runner for ML-specific tests
- Integration with external ML experiment tracking

## Monitoring and Maintenance

- **Weekly Review**: Check workflow run history for failures
- **Monthly Updates**: Update action versions to latest
- **Quarterly Audit**: Review workflow efficiency and add improvements
- **Security Updates**: Monitor for security advisories on used actions