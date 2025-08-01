#!/usr/bin/env python3
"""
Automated metrics collection script for Retrieval-Free Context Compressor.

This script collects various project metrics from different sources and updates
the project-metrics.json file.
"""

import json
import os
import sys
import requests
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and updates project metrics from various sources."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_name = self._extract_repo_name()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the metrics configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            sys.exit(1)
    
    def _extract_repo_name(self) -> str:
        """Extract repository name from config or environment."""
        repo_url = self.config.get('project', {}).get('repository', '')
        if 'github.com' in repo_url:
            return repo_url.split('github.com/')[-1]
        
        # Fallback to environment or git remote
        if 'GITHUB_REPOSITORY' in os.environ:
            return os.environ['GITHUB_REPOSITORY']
        
        try:
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                capture_output=True, text=True, check=True
            )
            url = result.stdout.strip()
            if 'github.com' in url:
                return url.split('github.com/')[-1].replace('.git', '')
        except subprocess.CalledProcessError:
            pass
        
        logger.warning("Could not determine repository name")
        return "unknown/unknown"
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect metrics from GitHub API."""
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not set, skipping GitHub metrics")
            return {}
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        metrics = {}
        
        try:
            # Repository stats
            repo_url = f"https://api.github.com/repos/{self.repo_name}"
            repo_response = requests.get(repo_url, headers=headers)
            repo_response.raise_for_status()
            repo_data = repo_response.json()
            
            metrics['github_stars'] = repo_data.get('stargazers_count', 0)
            metrics['forks'] = repo_data.get('forks_count', 0)
            metrics['open_issues'] = repo_data.get('open_issues_count', 0)
            
            # Contributors
            contributors_url = f"https://api.github.com/repos/{self.repo_name}/contributors"
            contributors_response = requests.get(contributors_url, headers=headers)
            contributors_response.raise_for_status()
            contributors_data = contributors_response.json()
            
            metrics['contributors'] = len(contributors_data)
            
            # Recent issues/PRs
            issues_url = f"https://api.github.com/repos/{self.repo_name}/issues"
            issues_params = {'state': 'closed', 'since': '2025-07-01T00:00:00Z'}
            issues_response = requests.get(issues_url, headers=headers, params=issues_params)
            issues_response.raise_for_status()
            closed_issues = issues_response.json()
            
            total_issues_url = f"https://api.github.com/repos/{self.repo_name}/issues"
            total_params = {'state': 'all', 'since': '2025-07-01T00:00:00Z'}
            total_response = requests.get(total_issues_url, headers=headers, params=total_params)
            total_response.raise_for_status()
            total_issues = total_response.json()
            
            if total_issues:
                metrics['issues_resolved'] = (len(closed_issues) / len(total_issues)) * 100
            else:
                metrics['issues_resolved'] = 100
            
        except requests.RequestException as e:
            logger.error(f"Error collecting GitHub metrics: {e}")
        
        return metrics
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        try:
            # Test coverage (from coverage report)
            coverage_file = Path('coverage.xml')
            if coverage_file.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                coverage_elem = root.find('.//coverage')
                if coverage_elem is not None:
                    line_rate = float(coverage_elem.get('line-rate', 0))
                    metrics['test_coverage'] = round(line_rate * 100, 2)
            
            # Code complexity (using radon)
            try:
                result = subprocess.run(
                    ['radon', 'cc', 'src/', '-a', '--total-average'],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    output = result.stdout.strip()
                    # Parse average complexity from radon output
                    for line in output.split('\n'):
                        if 'Average complexity:' in line:
                            complexity = float(line.split(':')[-1].strip())
                            metrics['code_complexity'] = complexity
                            break
            except FileNotFoundError:
                logger.warning("radon not installed, skipping complexity metrics")
            
            # Technical debt (placeholder - could integrate with SonarQube)
            metrics['technical_debt'] = 0  # Hours of estimated tech debt
            
        except Exception as e:
            logger.error(f"Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from benchmarks."""
        metrics = {}
        
        try:
            # Look for benchmark results
            benchmark_file = Path('benchmarks/results.json')
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                metrics['compression_ratio'] = benchmark_data.get('compression_ratio', 0)
                metrics['compression_latency'] = benchmark_data.get('latency_ms', 0)
                metrics['f1_score'] = benchmark_data.get('f1_score', 0)
                metrics['memory_usage'] = benchmark_data.get('memory_gb', 0)
            
            # Prometheus metrics (if available)
            prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
            try:
                # Query current compression ratio
                query_url = f"{prometheus_url}/api/v1/query"
                params = {'query': 'compression_ratio_current'}
                response = requests.get(query_url, params=params, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'success' and data['data']['result']:
                    metrics['compression_ratio'] = float(data['data']['result'][0]['value'][1])
                    
            except requests.RequestException:
                logger.debug("Prometheus not available, using benchmark data")
        
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        metrics = {}
        
        try:
            # Check for vulnerability scan results
            security_file = Path('security-report.json')
            if security_file.exists():
                with open(security_file, 'r') as f:
                    security_data = json.load(f)
                
                # Count vulnerabilities by severity
                vulnerabilities = security_data.get('vulnerabilities', [])
                metrics['vulnerability_count'] = len(vulnerabilities)
                
                high_critical = sum(1 for v in vulnerabilities 
                                  if v.get('severity') in ['high', 'critical'])
                metrics['high_critical_vulnerabilities'] = high_critical
            
            # Dependency freshness (check for outdated packages)
            try:
                result = subprocess.run(
                    ['pip', 'list', '--outdated', '--format=json'],
                    capture_output=True, text=True, check=False
                )
                if result.returncode == 0:
                    outdated = json.loads(result.stdout)
                    metrics['outdated_dependencies'] = len(outdated)
                    
                    # Calculate average days behind (simplified)
                    if outdated:
                        metrics['dependency_freshness'] = 15  # Placeholder
                    else:
                        metrics['dependency_freshness'] = 0
            except Exception:
                metrics['dependency_freshness'] = 0
        
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
        
        return metrics
    
    def collect_pypi_metrics(self) -> Dict[str, Any]:
        """Collect PyPI download statistics."""
        metrics = {}
        
        try:
            package_name = self.config.get('project', {}).get('name', '').lower().replace(' ', '-')
            if package_name:
                # PyPI download stats (using pypistats or similar)
                # Note: This would require the pypistats package
                # pip install pypistats
                try:
                    result = subprocess.run(
                        ['pypistats', 'recent', package_name],
                        capture_output=True, text=True, check=False
                    )
                    if result.returncode == 0:
                        # Parse download stats
                        metrics['monthly_downloads'] = 0  # Placeholder
                except FileNotFoundError:
                    logger.debug("pypistats not available")
        
        except Exception as e:
            logger.error(f"Error collecting PyPI metrics: {e}")
        
        return metrics
    
    def update_metrics_config(self, collected_metrics: Dict[str, Any]) -> None:
        """Update the metrics configuration with collected data."""
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Update metrics in config
        for category, metrics in collected_metrics.items():
            if category in self.config.get('metrics', {}):
                for metric_name, value in metrics.items():
                    if metric_name in self.config['metrics'][category]:
                        self.config['metrics'][category][metric_name]['current'] = value
        
        # Update tracking information
        self.config['tracking']['last_collection'] = current_time
        
        # Save updated configuration
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2, sort_keys=True)
        
        logger.info(f"Metrics updated and saved to {self.config_path}")
    
    def check_thresholds(self) -> Dict[str, Any]:
        """Check if any metrics exceed defined thresholds."""
        alerts = []
        thresholds = self.config.get('thresholds', {})
        
        for severity, threshold_config in thresholds.items():
            for metric_path, threshold_value in threshold_config.items():
                # Navigate to metric value
                current_value = self._get_metric_value(metric_path)
                if current_value is not None:
                    if self._threshold_exceeded(current_value, threshold_value, metric_path):
                        alerts.append({
                            'metric': metric_path,
                            'current_value': current_value,
                            'threshold': threshold_value,
                            'severity': severity,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
        
        return {'alerts': alerts}
    
    def _get_metric_value(self, metric_path: str) -> Optional[float]:
        """Get current value for a metric path."""
        try:
            # Find the metric in the config structure
            for category, metrics in self.config.get('metrics', {}).items():
                if metric_path in metrics:
                    return metrics[metric_path].get('current')
        except Exception:
            pass
        return None
    
    def _threshold_exceeded(self, current: float, threshold: float, metric_name: str) -> bool:
        """Check if threshold is exceeded based on metric type."""
        # Define metrics where lower values are better
        lower_is_better = [
            'vulnerability_count', 'dependency_freshness', 'compression_latency',
            'memory_usage', 'technical_debt'
        ]
        
        if any(name in metric_name for name in lower_is_better):
            return current > threshold
        else:
            return current < threshold
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        logger.info("Starting metrics collection...")
        
        all_metrics = {}
        
        # Collect from different sources
        collectors = [
            ('code_quality', self.collect_code_quality_metrics),
            ('performance', self.collect_performance_metrics),
            ('security', self.collect_security_metrics),
            ('community', self.collect_github_metrics),
            ('business', self.collect_pypi_metrics),
        ]
        
        for category, collector_func in collectors:
            try:
                logger.info(f"Collecting {category} metrics...")
                metrics = collector_func()
                if metrics:
                    all_metrics[category] = metrics
                    logger.info(f"Collected {len(metrics)} {category} metrics")
            except Exception as e:
                logger.error(f"Error collecting {category} metrics: {e}")
        
        return all_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--output', help='Output file for collected metrics')
    parser.add_argument('--check-thresholds', action='store_true',
                       help='Check if metrics exceed thresholds')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize collector
    collector = MetricsCollector(args.config)
    
    # Collect metrics
    collected_metrics = collector.collect_all_metrics()
    
    # Update configuration
    collector.update_metrics_config(collected_metrics)
    
    # Check thresholds if requested
    if args.check_thresholds:
        alerts = collector.check_thresholds()
        if alerts['alerts']:
            logger.warning(f"Found {len(alerts['alerts'])} threshold violations")
            for alert in alerts['alerts']:
                logger.warning(f"Alert: {alert['metric']} = {alert['current_value']} "
                             f"({alert['severity']} threshold: {alert['threshold']})")
        else:
            logger.info("All metrics within thresholds")
    
    # Save output if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(collected_metrics, f, indent=2)
        logger.info(f"Metrics saved to {args.output}")
    
    logger.info("Metrics collection completed successfully")


if __name__ == '__main__':
    main()