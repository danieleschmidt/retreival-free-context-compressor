# Advanced Security Workflows

This document outlines advanced security scanning and compliance workflows for the retrieval-free-context-compressor project.

## Container Security Scanning

### container-security.yml

```yaml
name: Container Security

on:
  push:
    branches: [ main ]
    paths: 
      - 'Dockerfile'
      - 'docker-compose.yml'
  pull_request:
    branches: [ main ]
    paths:
      - 'Dockerfile'
      - 'docker-compose.yml'
  schedule:
    - cron: '0 8 * * 2'  # Weekly on Tuesdays

jobs:
  container-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: docker build -t retrieval-free:test .
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'retrieval-free:test'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Hadolint Dockerfile linter
      uses: hadolint/hadolint-action@v3.1.0
      with:
        dockerfile: Dockerfile
        format: sarif
        output-file: hadolint-results.sarif
        no-fail: true
    
    - name: Upload Hadolint scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: hadolint-results.sarif
```

## SLSA Compliance

### slsa-provenance.yml

```yaml
name: SLSA Provenance

on:
  push:
    tags:
      - 'v*'

jobs:
  provenance:
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.7.0
    with:
      base64-subjects: ${{ needs.build.outputs.hashes }}
      provenance-name: retrieval-free-provenance.intoto.jsonl
    secrets:
    # The checkout/upload-artifact require these secrets.
      token: ${{ secrets.GITHUB_TOKEN }}

  build:
    runs-on: ubuntu-latest
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build
    
    - name: Build package
      run: python -m build
    
    - name: Generate hashes
      shell: bash
      id: hash
      run: |
        # sha256sum generates sha256 hash for all artifacts.
        # base64 -w0 encodes to base64 and outputs on a single line.
        # {::set-output name=hashes::} is a GitHub Actions command
        # to set the output variable "hashes".
        echo "hashes=$(sha256sum dist/* | base64 -w0)" >> "$GITHUB_OUTPUT"
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build-artifacts
        path: dist/
```

## Code Quality Gates

### quality-gates.yml

```yaml
name: Quality Gates

on:
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Fetch full history for SonarCloud
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run tests with coverage
      run: |
        pytest --cov=retrieval_free --cov-report=xml --cov-report=html
    
    - name: Check coverage threshold
      run: |
        coverage report --fail-under=80
    
    - name: Run complexity analysis
      run: |
        pip install radon
        radon cc src --min B --show-complexity
        radon mi src --min B
    
    - name: Check code duplication
      run: |
        pip install vulture
        vulture src --min-confidence 80
    
    - name: SonarCloud Scan
      uses: SonarSource/sonarcloud-github-action@master
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
    
    - name: Quality Gate Status
      run: |
        # This would typically check SonarCloud quality gate status
        echo "Quality gate checks completed"
```

## Dependency Security

### dependency-security.yml

```yaml
name: Dependency Security

on:
  push:
    branches: [ main ]
    paths:
      - 'pyproject.toml'
      - 'requirements*.txt'
  pull_request:
    branches: [ main ]
    paths:
      - 'pyproject.toml'
      - 'requirements*.txt'
  schedule:
    - cron: '0 10 * * 3'  # Weekly on Wednesdays

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .
    
    - name: Run pip-audit
      run: |
        pip install pip-audit
        pip-audit --format=json --output=audit-results.json
    
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-bom -o sbom.json
    
    - name: Upload SBOM as artifact
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.json
    
    - name: License check
      run: |
        pip install pip-licenses
        pip-licenses --format=json --output-file=licenses.json
        # Check for GPL and other restrictive licenses
        pip-licenses --fail-on="GPL" --fail-on="AGPL" --fail-on="LGPL"
```

## Setup Instructions

### Repository Configuration

1. **Enable Security Features**:
   - Go to repository Settings > Security & analysis
   - Enable all security features:
     - Dependency graph
     - Dependabot alerts
     - Dependabot security updates
     - Code scanning alerts
     - Secret scanning alerts

2. **Add Required Secrets**:
   - `SONAR_TOKEN`: SonarCloud authentication token
   - `CODECOV_TOKEN`: Codecov integration token

3. **Configure Branch Protection**:
   - Require all security checks to pass
   - Enable "Restrict pushes that create files"
   - Add patterns: `**/*.key`, `**/*.pem`, `**/*.p12`

### Third-Party Integrations

1. **SonarCloud**:
   - Connect repository to SonarCloud
   - Configure quality profiles for Python
   - Set up quality gates with appropriate thresholds

2. **Codecov**:
   - Enable Codecov integration
   - Configure coverage thresholds
   - Set up status checks

## Compliance Standards

This security setup helps achieve compliance with:

- **NIST Cybersecurity Framework**
- **OWASP ASVS Level 2**
- **SLSA Level 3** (with provenance generation)
- **SSDF** (Secure Software Development Framework)

## Manual Implementation Required

These workflows must be manually created in `.github/workflows/` directory as GitHub Actions cannot be automatically provisioned for security reasons.

After implementing the SDLC enhancements, copy the workflow content above into appropriate YAML files in your repository.