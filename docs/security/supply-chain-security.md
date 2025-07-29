# Supply Chain Security Guidelines

This document outlines the supply chain security measures implemented for the Retrieval-Free Context Compressor project.

## Overview

Supply chain security is critical for AI/ML projects due to the complex dependency chains and potential for malicious model distribution. This project implements comprehensive measures to ensure the integrity of all components.

## Dependency Management

### Python Dependencies

**Pinned Versions**: All production dependencies are pinned to specific versions in `pyproject.toml` to prevent unexpected updates.

**Vulnerability Scanning**: Dependencies are continuously monitored for known vulnerabilities using:
- GitHub's Dependabot security alerts
- `pip-audit` in CI/CD pipeline
- Renovate bot for automated security updates

**Trusted Sources**: Dependencies are sourced only from:
- PyPI (Python Package Index) - official packages
- Conda-forge (for ML-specific packages)
- Direct GitHub repositories (with SHA pinning)

### Model Dependencies

**Model Verification**: All pre-trained models must include:
- SHA256 checksums for integrity verification
- Digital signatures from trusted sources
- Model cards documenting training data and methodology

**Staging Process**: Models follow a staged deployment:
1. **Development**: Experimental models in dev environment
2. **Staging**: Verified models in staging environment  
3. **Production**: Fully audited models in production

### Container Security

**Base Images**: Use official, minimal base images:
```dockerfile
FROM python:3.11-slim  # Official Python images
FROM nvidia/cuda:11.8-devel-ubuntu22.04  # Official NVIDIA images
```

**Multi-stage Builds**: Separate build and runtime environments to minimize attack surface.

**Non-root Execution**: Containers run as non-root user (`appuser:1000`).

## SBOM (Software Bill of Materials)

### Generation

SBOM is automatically generated using `syft`:

```bash
# Generate SBOM for Python dependencies
syft packages dir:. -o spdx-json > sbom.spdx.json

# Generate SBOM for container images
syft packages retrieval-free:latest -o spdx-json > container-sbom.spdx.json
```

### Contents

The SBOM includes:
- All direct and transitive Python dependencies
- Container base image components
- Build tools and their versions
- License information for all components

### Verification

SBOM integrity is verified using digital signatures:

```bash
# Sign SBOM
cosign sign-blob --key cosign.key sbom.spdx.json > sbom.sig

# Verify SBOM signature
cosign verify-blob --key cosign.pub --signature sbom.sig sbom.spdx.json
```

## Vulnerability Management

### Scanning Schedule

- **Daily**: Automated vulnerability scans of dependencies
- **Weekly**: Full container image security scans
- **Monthly**: Manual security review of critical dependencies

### Response Process

1. **Detection**: Vulnerability identified by automated tools
2. **Assessment**: Impact analysis within 24 hours
3. **Patching**: Security patches applied within 7 days for high/critical
4. **Verification**: Patch effectiveness validated in staging
5. **Deployment**: Emergency deployment for critical vulnerabilities

### Tools Integration

```yaml
# Example CI integration
- name: Run pip-audit
  run: pip-audit --desc --format=json --output=audit-report.json

- name: Scan container with Trivy
  run: trivy image --format json retrieval-free:latest > trivy-report.json
```

## Code Signing and Verification

### Package Signing

Production packages are signed using `sigstore`:

```bash
# Sign package
python -m sigstore sign retrieval_free-*.whl

# Verify signature
python -m sigstore verify retrieval_free-*.whl
```

### Commit Signing

All commits must be signed with GPG keys:

```bash
# Configure Git signing
git config --global user.signingkey YOUR_GPG_KEY
git config --global commit.gpgsign true
```

## Access Controls

### Repository Access

- **Admin**: Project maintainers only
- **Write**: Core contributors with 2FA enabled
- **Read**: Public repository for transparency

### Secrets Management

- API keys stored in GitHub Secrets
- Model signing keys in secure key management service
- Rotation schedule for all credentials (90 days)

### Branch Protection

```yaml
# Branch protection rules
required_status_checks:
  strict: true
  contexts:
    - "security-scan"
    - "dependency-audit"
    - "sbom-generation"
enforce_admins: true
required_pull_request_reviews:
  required_approving_review_count: 2
  dismiss_stale_reviews: true
```

## Incident Response

### Security Incident Process

1. **Detection**: Automated alerts or manual discovery
2. **Containment**: Immediate isolation of affected components
3. **Investigation**: Root cause analysis within 48 hours
4. **Remediation**: Patch deployment and verification
5. **Recovery**: Service restoration with monitoring
6. **Lessons Learned**: Post-incident review and improvements

### Communication

- **Internal**: Security team notification within 1 hour
- **External**: Public disclosure after remediation (coordinated disclosure)
- **Users**: Security advisories through GitHub Security Advisories

## Compliance and Auditing

### Standards Alignment

- **NIST Cybersecurity Framework**: Risk management alignment
- **OWASP Top 10**: Web application security best practices
- **SLSA Level 3**: Supply chain security framework compliance

### Audit Trail

All security-relevant activities are logged:
- Dependency updates and approvals
- Code signing operations
- Vulnerability scan results
- Access control changes

### Regular Reviews

- **Quarterly**: Security posture review
- **Annually**: Third-party security audit
- **As needed**: Post-incident security assessment

## Tools and Resources

### Security Tools
- **Dependabot**: Automated dependency updates
- **CodeQL**: Static code analysis
- **Trivy**: Container vulnerability scanning
- **pip-audit**: Python dependency vulnerability scanning
- **Bandit**: Python security linting

### Documentation
- [NIST Secure Software Development Framework](https://csrc.nist.gov/Projects/ssdf)
- [SLSA Supply Chain Security Framework](https://slsa.dev/)
- [OWASP Dependency Check](https://owasp.org/www-project-dependency-check/)

## Contact

For security concerns or vulnerability reports:
- **Email**: security@retrieval-free.com
- **GPG Key**: `security-gpg-key.asc`
- **Response Time**: 24 hours for critical issues

---

*This document is reviewed quarterly and updated as security practices evolve.*