# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Create Public Issues
Please do not report security vulnerabilities through public GitHub issues.

### 2. Private Reporting
Send details to **retrieval-free-security@yourdomain.com** including:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if available)

### 3. Response Timeline
- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Critical issues within 30 days

### 4. Disclosure Policy
- We will acknowledge receipt of your vulnerability report
- We will confirm the vulnerability and determine affected versions
- We will release fixes and coordinate disclosure timing with you
- We will credit you in the security advisory (unless you prefer anonymity)

## Security Best Practices

### For Users
- Keep dependencies updated to latest secure versions
- Use virtual environments to isolate package dependencies
- Validate and sanitize all input data before processing
- Monitor for security advisories on dependencies

### For Contributors
- Follow secure coding practices
- Never commit secrets, API keys, or credentials
- Use dependency scanning tools
- Run security tests before submitting PRs

## Security Measures

### Automated Security
- **Dependency scanning** with Dependabot
- **Code scanning** with CodeQL
- **Secret scanning** for accidental commits
- **SBOM generation** for transparency

### Manual Security Reviews
- Security review for all major releases
- Penetration testing for critical components
- Third-party security audits (planned for v1.0)

## Vulnerability Disclosure Examples

### What to Report
- Code injection vulnerabilities
- Authentication/authorization bypasses
- Data exposure issues
- Dependency vulnerabilities with exploits
- Cryptographic implementation flaws

### What Not to Report
- Issues requiring physical access to user devices
- Social engineering attacks
- Denial of Service attacks requiring excessive resources
- Issues in third-party dependencies without proof of exploitability

## Contact

- **Security Email**: retrieval-free-security@yourdomain.com
- **General Contact**: retrieval-free@yourdomain.com
- **GPG Key**: Available upon request

Thank you for helping keep our project and users secure!