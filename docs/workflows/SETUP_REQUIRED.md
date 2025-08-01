# GitHub Workflows Setup Required

This document outlines the manual steps required to set up GitHub Actions workflows for the Retrieval-Free Context Compressor project.

## Overview

Due to GitHub App permission limitations, the following workflow files must be manually created in the `.github/workflows/` directory. All templates are provided in `docs/workflows/implementations/`.

## Required Workflows

### 1. Continuous Integration (ci.yml)
**Priority: CRITICAL**
- **Template**: `docs/workflows/implementations/ci.yml`
- **Purpose**: Code quality, testing, and build validation
- **Features**:
  - Multi-version Python testing (3.10, 3.11, 3.12)
  - Pre-commit hooks execution
  - Type checking with mypy
  - Code coverage reporting
  - Build artifact generation

**Setup Steps:**
1. Copy `docs/workflows/implementations/ci.yml` to `.github/workflows/ci.yml`
2. Configure Codecov token in repository secrets as `CODECOV_TOKEN`
3. Verify branch protection rules are configured for `main` branch

### 2. Security Scanning (security.yml)
**Priority: HIGH**
- **Template**: `docs/workflows/implementations/security.yml`
- **Purpose**: Automated security vulnerability scanning
- **Features**:
  - SAST scanning with CodeQL
  - Dependency vulnerability scanning
  - Secret detection
  - License compliance checking
  - SBOM generation

**Setup Steps:**
1. Copy `docs/workflows/implementations/security.yml` to `.github/workflows/security.yml`
2. Enable GitHub Advanced Security features
3. Configure security policy in `SECURITY.md`

### 3. Dependency Review (dependency-review.yml)
**Priority: HIGH**
- **Template**: `docs/workflows/implementations/dependency-review.yml`
- **Purpose**: Automated dependency security and license review
- **Features**:
  - License policy enforcement
  - Vulnerability threshold enforcement
  - Dependency change impact analysis

**Setup Steps:**
1. Copy `docs/workflows/implementations/dependency-review.yml` to `.github/workflows/dependency-review.yml`
2. Configure dependency review policies in repository settings

### 4. Docker Build and Push (docker.yml)
**Priority: MEDIUM**
- **Template**: `docs/workflows/implementations/docker.yml`
- **Purpose**: Multi-architecture container builds
- **Features**:
  - Multi-platform builds (linux/amd64, linux/arm64)
  - Security scanning with Trivy
  - SBOM generation
  - Image signing with Cosign

**Setup Steps:**
1. Copy `docs/workflows/implementations/docker.yml` to `.github/workflows/docker.yml`
2. Configure container registry credentials:
   - `DOCKER_USERNAME` - Registry username
   - `DOCKER_PASSWORD` - Registry password/token
3. Set up image signing (optional):
   - `COSIGN_PRIVATE_KEY` - Cosign private key
   - `COSIGN_PASSWORD` - Key password

### 5. Release Automation (release.yml)
**Priority: MEDIUM**
- **Template**: `docs/workflows/implementations/release.yml`
- **Purpose**: Automated semantic releases
- **Features**:
  - Semantic version calculation
  - Release notes generation
  - Multi-format publishing (PyPI, GitHub, Docker)
  - Asset generation and upload

**Setup Steps:**
1. Copy `docs/workflows/implementations/release.yml` to `.github/workflows/release.yml`
2. Configure publishing secrets:
   - `PYPI_API_TOKEN` - PyPI publishing token
   - `GITHUB_TOKEN` - Automatically provided
3. Install semantic-release locally: `npm install -g semantic-release`

### 6. Performance Benchmarking (performance.yml)
**Priority: LOW**
- **Template**: `docs/workflows/implementations/performance.yml`
- **Purpose**: Automated performance regression testing
- **Features**:
  - Benchmark execution
  - Performance comparison
  - Regression detection
  - Results archival

**Setup Steps:**
1. Copy `docs/workflows/implementations/performance.yml` to `.github/workflows/performance.yml`
2. Configure performance thresholds in workflow file

## Security Configuration

### Required Secrets

Configure the following secrets in GitHub repository settings:

#### CI/CD Secrets
- `CODECOV_TOKEN` - Code coverage reporting
- `PYPI_API_TOKEN` - PyPI package publishing
- `DOCKER_USERNAME` - Container registry username
- `DOCKER_PASSWORD` - Container registry password

#### Security Secrets
- `COSIGN_PRIVATE_KEY` - Container image signing (optional)
- `COSIGN_PASSWORD` - Cosign key password (optional)

#### Monitoring Secrets (if using)
- `SENTRY_DSN` - Error tracking
- `SLACK_WEBHOOK_URL` - Slack notifications

### Branch Protection Rules

Configure branch protection for `main` branch:

1. **Required status checks**:
   - `lint`
   - `test (3.10)`
   - `test (3.11)`  
   - `test (3.12)`
   - `build`
   - `dependency-review`

2. **Additional rules**:
   - Require pull request reviews (1 reviewer minimum)
   - Dismiss stale reviews when new commits are pushed
   - Require status checks to pass before merging
   - Require branches to be up to date before merging
   - Include administrators in restrictions

### Repository Settings

#### Security & Analysis
- [x] Dependency graph
- [x] Dependabot alerts
- [x] Dependabot security updates
- [x] Code scanning alerts
- [x] Secret scanning alerts

#### Actions Permissions
- [x] Allow all actions and reusable workflows
- [x] Allow actions created by GitHub
- [x] Allow actions by Marketplace verified creators

## Manual Installation Steps

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Files
```bash
# Copy all workflow templates
cp docs/workflows/implementations/*.yml .github/workflows/

# Verify files are copied
ls -la .github/workflows/
```

### Step 3: Configure Repository Secrets
1. Navigate to repository Settings → Secrets and variables → Actions
2. Add required secrets listed above
3. Verify secret names match workflow file references

### Step 4: Configure Branch Protection
1. Navigate to repository Settings → Branches
2. Add branch protection rule for `main`
3. Configure required status checks
4. Enable additional protection rules

### Step 5: Test Workflows
1. Create a test pull request
2. Verify all workflows execute successfully
3. Check that branch protection prevents merging on failures

## Verification Checklist

After setup, verify the following:

- [ ] CI workflow runs on pull requests
- [ ] All status checks appear in PR
- [ ] Security scanning completes without errors
- [ ] Docker builds succeed for multi-platform
- [ ] Release workflow triggers on main branch pushes
- [ ] Branch protection prevents merging failed checks
- [ ] Dependency review blocks vulnerable dependencies
- [ ] Code coverage reports are generated
- [ ] Performance benchmarks execute (if applicable)

## Troubleshooting

### Common Issues

#### Workflow Not Triggering
- Check workflow file syntax with GitHub Actions validator
- Verify trigger conditions (branches, paths, events)
- Check repository permissions and secrets

#### Permission Errors
- Verify GITHUB_TOKEN has required permissions
- Check if repository requires specific permissions
- Review branch protection rule configuration

#### Build Failures
- Check dependency versions in `pyproject.toml`
- Verify test compatibility across Python versions
- Review error logs in Actions tab

#### Docker Build Issues
- Verify multi-platform builder setup
- Check Dockerfile syntax and build context
- Review registry authentication configuration

### Getting Help

1. **GitHub Actions Documentation**: https://docs.github.com/en/actions
2. **Workflow Templates**: Browse `docs/workflows/implementations/`
3. **Security Configuration**: Review `docs/workflows/advanced-security.md`
4. **Performance Setup**: See `docs/workflows/READY_FOR_DEPLOYMENT.md`

## Maintenance

### Regular Updates

1. **Monthly**: Update action versions to latest
2. **Quarterly**: Review and update Python version matrix
3. **Annually**: Audit security configurations and policies

### Monitoring

1. **Workflow Health**: Monitor workflow success rates
2. **Performance**: Track build times and resource usage
3. **Security**: Review security scan results regularly
4. **Dependencies**: Monitor Dependabot alerts and updates

---

**Next Steps**: After completing the setup, proceed to configure advanced security features as described in `docs/workflows/advanced-security.md`.