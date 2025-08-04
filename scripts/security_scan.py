"""Security scanning script for the codebase."""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class SecurityScanner:
    """Security scanner for code analysis."""
    
    def __init__(self, project_root: str):
        """Initialize security scanner.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.results = {
            'vulnerabilities': [],
            'warnings': [],
            'info': [],
            'summary': {}
        }
        
    def run_all_scans(self) -> Dict[str, Any]:
        """Run all security scans.
        
        Returns:
            Dictionary with scan results
        """
        logger.info("Starting security scan...")
        
        # 1. Pattern-based security scan
        self._pattern_based_scan()
        
        # 2. Dependency vulnerability scan
        self._dependency_scan()
        
        # 3. Configuration security scan
        self._config_scan()
        
        # 4. File permission scan
        self._permission_scan()
        
        # 5. Secret detection
        self._secret_detection()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def _pattern_based_scan(self) -> None:
        """Scan for security patterns in source code."""
        logger.info("Running pattern-based security scan...")
        
        # Security patterns to detect
        patterns = {
            'eval_usage': {
                'pattern': r'\beval\s*\(',
                'severity': 'high',
                'description': 'Use of eval() function can lead to code injection'
            },
            'exec_usage': {
                'pattern': r'\bexec\s*\(',
                'severity': 'high', 
                'description': 'Use of exec() function can lead to code injection'
            },
            'shell_injection': {
                'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                'severity': 'high',
                'description': 'Shell command injection vulnerability'
            },
            'os_system': {
                'pattern': r'\bos\.system\s*\(',
                'severity': 'medium',
                'description': 'Use of os.system() can be unsafe'
            },
            'pickle_usage': {
                'pattern': r'\bpickle\.(loads?|dumps?)\s*\(',
                'severity': 'medium',
                'description': 'Pickle deserialization can be unsafe with untrusted data'
            },
            'hardcoded_secrets': {
                'pattern': r'(password|secret|key|token)\s*=\s*["\'][^"\']+["\']',
                'severity': 'high',
                'description': 'Potential hardcoded secret'
            },
            'sql_injection': {
                'pattern': r'(SELECT|INSERT|UPDATE|DELETE).*\+.*%s',
                'severity': 'high',
                'description': 'Potential SQL injection vulnerability'
            }
        }
        
        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern_name, pattern_info in patterns.items():
                    matches = re.finditer(pattern_info['pattern'], content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Find line number
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vulnerability = {
                            'type': 'pattern_match',
                            'pattern': pattern_name,
                            'severity': pattern_info['severity'],
                            'description': pattern_info['description'],
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'code': match.group().strip()
                        }
                        
                        if pattern_info['severity'] == 'high':
                            self.results['vulnerabilities'].append(vulnerability)
                        else:
                            self.results['warnings'].append(vulnerability)
                            
            except Exception as e:
                logger.warning(f"Failed to scan {file_path}: {e}")
    
    def _dependency_scan(self) -> None:
        """Scan dependencies for known vulnerabilities."""
        logger.info("Running dependency vulnerability scan...")
        
        # Check for requirements files
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py']
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                self._scan_dependency_file(req_path)
    
    def _scan_dependency_file(self, file_path: Path) -> None:
        """Scan a dependency file for vulnerabilities."""
        try:
            # Try using safety tool if available
            result = subprocess.run([
                'safety', 'check', '--file', str(file_path), '--json'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse safety results
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerability = {
                            'type': 'dependency_vulnerability',
                            'severity': 'high',
                            'description': f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                            'file': str(file_path.name),
                            'details': vuln
                        }
                        self.results['vulnerabilities'].append(vulnerability)
                except json.JSONDecodeError:
                    pass
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Safety tool not available or timeout
            self.results['info'].append({
                'type': 'info',
                'message': f'Could not scan {file_path.name} for vulnerabilities (safety tool not available)'
            })
    
    def _config_scan(self) -> None:
        """Scan configuration files for security issues."""
        logger.info("Running configuration security scan...")
        
        # Configuration files to check
        config_patterns = {
            '*.yml': ['docker-compose.yml', 'github/workflows/*.yml'],
            '*.yaml': ['*.yaml'],
            '*.json': ['package.json', 'tsconfig.json'],
            '*.toml': ['pyproject.toml'],
            '.env*': ['.env', '.env.local', '.env.prod']
        }
        
        for pattern, files in config_patterns.items():
            for file_pattern in files:
                config_files = list(self.project_root.rglob(file_pattern))
                
                for config_file in config_files:
                    self._scan_config_file(config_file)
    
    def _scan_config_file(self, file_path: Path) -> None:
        """Scan individual configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for sensitive patterns in config
            sensitive_patterns = [
                r'password\s*[:=]\s*[\'"][^\'"]*[\'"]',
                r'secret\s*[:=]\s*[\'"][^\'"]*[\'"]',
                r'api_key\s*[:=]\s*[\'"][^\'"]*[\'"]',
                r'private_key\s*[:=]\s*[\'"][^\'"]*[\'"]',
            ]
            
            for pattern in sensitive_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    warning = {
                        'type': 'config_security',
                        'severity': 'medium',
                        'description': 'Potential sensitive data in configuration file',
                        'file': str(file_path.relative_to(self.project_root)),
                        'line': line_num,
                        'pattern': pattern
                    }
                    
                    self.results['warnings'].append(warning)
                    
        except Exception as e:
            logger.warning(f"Failed to scan config file {file_path}: {e}")
    
    def _permission_scan(self) -> None:
        """Scan file permissions for security issues."""
        logger.info("Running file permission scan...")
        
        # Check for overly permissive files
        sensitive_files = [
            'private_key*', '*.key', '*.pem', '.env*', 'config/*.yml'
        ]
        
        for pattern in sensitive_files:
            files = list(self.project_root.rglob(pattern))
            
            for file_path in files:
                try:
                    stat_info = file_path.stat()
                    mode = stat_info.st_mode
                    
                    # Check if file is world-readable or world-writable
                    if mode & 0o044:  # World readable
                        warning = {
                            'type': 'file_permissions',
                            'severity': 'medium',
                            'description': 'Sensitive file is world-readable',
                            'file': str(file_path.relative_to(self.project_root)),
                            'permissions': oct(mode)[-3:]
                        }
                        self.results['warnings'].append(warning)
                    
                    if mode & 0o022:  # World writable
                        vulnerability = {
                            'type': 'file_permissions',
                            'severity': 'high',
                            'description': 'Sensitive file is world-writable',
                            'file': str(file_path.relative_to(self.project_root)),
                            'permissions': oct(mode)[-3:]
                        }
                        self.results['vulnerabilities'].append(vulnerability)
                        
                except Exception as e:
                    logger.debug(f"Could not check permissions for {file_path}: {e}")
    
    def _secret_detection(self) -> None:
        """Detect potential secrets in code."""
        logger.info("Running secret detection...")
        
        # Secret patterns
        secret_patterns = {
            'aws_access_key': r'AKIA[0-9A-Z]{16}',
            'aws_secret_key': r'aws_secret_access_key\s*=\s*[\'"][A-Za-z0-9/\+=]{40}[\'"]',
            'github_token': r'ghp_[A-Za-z0-9]{36}',
            'slack_token': r'xox[bpoa]-[0-9]{12}-[0-9]{12}-[A-Za-z0-9]{24}',
            'generic_api_key': r'api_key\s*[:=]\s*[\'"][A-Za-z0-9]{20,}[\'"]',
            'jwt_token': r'eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*',
        }
        
        # Scan all text files
        text_files = []
        for ext in ['*.py', '*.js', '*.ts', '*.json', '*.yml', '*.yaml', '*.md']:
            text_files.extend(self.project_root.rglob(ext))
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for secret_type, pattern in secret_patterns.items():
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        vulnerability = {
                            'type': 'secret_detection',
                            'secret_type': secret_type,
                            'severity': 'high',
                            'description': f'Potential {secret_type} found',
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num
                        }
                        
                        self.results['vulnerabilities'].append(vulnerability)
                        
            except Exception as e:
                logger.debug(f"Could not scan {file_path} for secrets: {e}")
    
    def _generate_summary(self) -> None:
        """Generate summary of scan results."""
        self.results['summary'] = {
            'vulnerabilities_count': len(self.results['vulnerabilities']),
            'warnings_count': len(self.results['warnings']),
            'info_count': len(self.results['info']),
            'files_scanned': self._count_scanned_files(),
            'scan_date': __import__('datetime').datetime.now().isoformat(),
        }
        
        # Categorize by severity
        severity_counts = {}
        for vuln in self.results['vulnerabilities']:
            severity = vuln.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        self.results['summary']['severity_breakdown'] = severity_counts
    
    def _count_scanned_files(self) -> int:
        """Count number of files scanned."""
        extensions = ['*.py', '*.js', '*.ts', '*.json', '*.yml', '*.yaml', '*.md']
        count = 0
        
        for ext in extensions:
            count += len(list(self.project_root.rglob(ext)))
        
        return count
    
    def save_results(self, output_file: str) -> None:
        """Save scan results to file.
        
        Args:
            output_file: Path to output file
        """
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Security scan results saved to: {output_file}")
    
    def generate_report(self) -> str:
        """Generate human-readable security report.
        
        Returns:
            Formatted security report
        """
        lines = [
            "=== SECURITY SCAN REPORT ===",
            f"Scan Date: {self.results['summary'].get('scan_date', 'Unknown')}",
            f"Files Scanned: {self.results['summary'].get('files_scanned', 0)}",
            "",
            "SUMMARY:",
            f"  🔴 Vulnerabilities: {self.results['summary']['vulnerabilities_count']}",
            f"  🟡 Warnings: {self.results['summary']['warnings_count']}",
            f"  🔵 Info: {self.results['summary']['info_count']}",
            "",
        ]
        
        # Severity breakdown
        if self.results['summary'].get('severity_breakdown'):
            lines.append("SEVERITY BREAKDOWN:")
            for severity, count in self.results['summary']['severity_breakdown'].items():
                lines.append(f"  {severity.upper()}: {count}")
            lines.append("")
        
        # Vulnerabilities
        if self.results['vulnerabilities']:
            lines.append("🔴 VULNERABILITIES:")
            lines.append("-" * 50)
            
            for i, vuln in enumerate(self.results['vulnerabilities'][:10], 1):  # Limit to first 10
                lines.extend([
                    f"{i}. {vuln.get('description', 'Unknown vulnerability')}",
                    f"   File: {vuln.get('file', 'Unknown')}:{vuln.get('line', 'Unknown')}",
                    f"   Severity: {vuln.get('severity', 'Unknown').upper()}",
                    ""
                ])
            
            if len(self.results['vulnerabilities']) > 10:
                lines.append(f"... and {len(self.results['vulnerabilities']) - 10} more")
                lines.append("")
        
        # Warnings (top 5)
        if self.results['warnings']:
            lines.append("🟡 WARNINGS (Top 5):")
            lines.append("-" * 30)
            
            for i, warning in enumerate(self.results['warnings'][:5], 1):
                lines.extend([
                    f"{i}. {warning.get('description', 'Unknown warning')}",
                    f"   File: {warning.get('file', 'Unknown')}:{warning.get('line', 'Unknown')}",
                    ""
                ])
        
        # Recommendations
        lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20,
        ])
        
        if self.results['vulnerabilities']:
            lines.append("• Review and fix all identified vulnerabilities")
        if any(v.get('type') == 'secret_detection' for v in self.results['vulnerabilities']):
            lines.append("• Remove hardcoded secrets and use environment variables")
        if any(v.get('type') == 'pattern_match' for v in self.results['vulnerabilities']):
            lines.append("• Replace unsafe functions with secure alternatives")
        if self.results['warnings']:
            lines.append("• Address security warnings to improve overall security posture")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def main():
    """Main security scan execution."""
    print("Starting Security Scan...")
    print("=" * 40)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Run scanner
    scanner = SecurityScanner(str(project_root))
    results = scanner.run_all_scans()
    
    # Save results
    output_file = project_root / "security_scan_results.json"
    scanner.save_results(str(output_file))
    
    # Generate and display report
    report = scanner.generate_report()
    print(report)
    
    # Exit code based on results
    if results['summary']['vulnerabilities_count'] > 0:
        print("\n❌ Security scan found vulnerabilities")
        return 1
    elif results['summary']['warnings_count'] > 0:
        print("\n⚠️  Security scan found warnings")
        return 0
    else:
        print("\n✅ Security scan passed - no issues found")
        return 0


if __name__ == "__main__":
    exit(main())