#!/usr/bin/env python3
"""Health check script for production deployments."""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import torch
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


console = Console()


class HealthChecker:
    """Comprehensive health check for the compression service."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.checks = []
        
    def add_check(self, name: str, func, critical: bool = True):
        """Add a health check."""
        self.checks.append({
            'name': name,
            'func': func,
            'critical': critical,
            'status': 'pending',
            'message': '',
            'duration': 0
        })
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        console.print("[bold blue]Running health checks...")
        
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': [],
            'summary': {}
        }
        
        total_checks = len(self.checks)
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0
        
        for check in self.checks:
            console.print(f"[yellow]Running check: {check['name']}")
            
            start_time = time.time()
            try:
                success, message = check['func']()
                check['duration'] = time.time() - start_time
                
                if success:
                    check['status'] = 'passed'
                    check['message'] = message or 'OK'
                    passed_checks += 1
                    console.print(f"[green]✓ {check['name']}: {check['message']}")
                else:
                    if check['critical']:
                        check['status'] = 'failed'
                        failed_checks += 1
                        console.print(f"[red]✗ {check['name']}: {message}")
                    else:
                        check['status'] = 'warning'
                        warning_checks += 1
                        console.print(f"[yellow]⚠ {check['name']}: {message}")
                    check['message'] = message
                    
            except Exception as e:
                check['duration'] = time.time() - start_time
                check['status'] = 'error'
                check['message'] = f"Check failed with error: {str(e)}"
                
                if check['critical']:
                    failed_checks += 1
                    console.print(f"[red]✗ {check['name']}: {check['message']}")
                else:
                    warning_checks += 1
                    console.print(f"[yellow]⚠ {check['name']}: {check['message']}")
            
            results['checks'].append({
                'name': check['name'],
                'status': check['status'],
                'message': check['message'],
                'duration': check['duration'],
                'critical': check['critical']
            })
        
        # Determine overall status
        if failed_checks > 0:
            results['overall_status'] = 'unhealthy'
        elif warning_checks > 0:
            results['overall_status'] = 'degraded'
        else:
            results['overall_status'] = 'healthy'
        
        results['summary'] = {
            'total': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'warnings': warning_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0
        }
        
        return results
    
    def check_api_endpoint(self) -> tuple[bool, str]:
        """Check if API endpoint is responsive."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                return True, f"API responsive (HTTP {response.status_code})"
            else:
                return False, f"API returned HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"API unreachable: {str(e)}"
    
    def check_model_loading(self) -> tuple[bool, str]:
        """Check if models can be loaded."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=self.timeout)
            if response.status_code == 200:
                models = response.json()
                if models and len(models) > 0:
                    return True, f"Models loaded: {', '.join(models.keys())}"
                else:
                    return False, "No models available"
            else:
                return False, f"Model endpoint returned HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Model check failed: {str(e)}"
    
    def check_compression_basic(self) -> tuple[bool, str]:
        """Test basic compression functionality."""
        try:
            test_document = "This is a test document for compression health check. " * 20
            
            payload = {
                "document": test_document,
                "compression_ratio": 4.0
            }
            
            response = requests.post(
                f"{self.base_url}/compress",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'compressed_tokens' in result and 'compression_ratio' in result:
                    ratio = result['compression_ratio']
                    return True, f"Compression working (ratio: {ratio:.1f}x)"
                else:
                    return False, "Invalid compression response format"
            else:
                return False, f"Compression failed with HTTP {response.status_code}"
                
        except requests.exceptions.RequestException as e:
            return False, f"Compression test failed: {str(e)}"
    
    def check_gpu_availability(self) -> tuple[bool, str]:
        """Check GPU availability and memory."""
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        try:
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            # Check GPU memory
            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_reserved = torch.cuda.memory_reserved(current_device)
            
            memory_allocated_mb = memory_allocated / 1024 / 1024
            memory_reserved_mb = memory_reserved / 1024 / 1024
            
            return True, f"GPU available: {device_name} ({gpu_count} GPUs, {memory_allocated_mb:.0f}MB allocated)"
            
        except Exception as e:
            return False, f"GPU check failed: {str(e)}"
    
    def check_system_resources(self) -> tuple[bool, str]:
        """Check system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                load_1min = load_avg[0]
            except (AttributeError, OSError):
                load_1min = 0
            
            # Check thresholds
            warnings = []
            if cpu_percent > 90:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory_percent > 90:
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
            if disk_percent > 90:
                warnings.append(f"High disk usage: {disk_percent:.1f}%")
            
            if warnings:
                return False, "; ".join(warnings)
            else:
                return True, f"Resources OK (CPU: {cpu_percent:.1f}%, RAM: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%)"
                
        except Exception as e:
            return False, f"Resource check failed: {str(e)}"
    
    def check_dependencies(self) -> tuple[bool, str]:
        """Check critical dependencies."""
        try:
            # Check Python packages
            import torch
            import transformers
            import einops
            
            torch_version = torch.__version__
            transformers_version = transformers.__version__
            
            # Check versions
            version_info = f"torch={torch_version}, transformers={transformers_version}"
            
            return True, f"Dependencies OK ({version_info})"
            
        except ImportError as e:
            return False, f"Missing dependency: {str(e)}"
        except Exception as e:
            return False, f"Dependency check failed: {str(e)}"
    
    def check_cache_system(self) -> tuple[bool, str]:
        """Check cache system availability."""
        try:
            response = requests.get(f"{self.base_url}/cache/status", timeout=self.timeout)
            if response.status_code == 200:
                cache_info = response.json()
                hit_rate = cache_info.get('hit_rate', 0)
                size = cache_info.get('size', 0)
                return True, f"Cache OK (hit rate: {hit_rate:.1%}, size: {size})"
            else:
                return False, f"Cache endpoint returned HTTP {response.status_code}"
        except requests.exceptions.RequestException as e:
            # Cache might not be implemented yet, so this is a warning
            return False, f"Cache not available: {str(e)}"
    
    def display_results(self, results: Dict[str, Any]):
        """Display health check results."""
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red'
        }
        
        overall_color = status_color.get(results['overall_status'], 'white')
        
        # Overall status panel
        console.print(Panel(
            f"[bold {overall_color}]{results['overall_status'].upper()}[/bold {overall_color}]",
            title="Overall Health Status",
            border_style=overall_color
        ))
        
        # Summary table
        summary_table = Table(title="Health Check Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary = results['summary']
        summary_table.add_row("Total Checks", str(summary['total']))
        summary_table.add_row("Passed", f"[green]{summary['passed']}[/green]")
        summary_table.add_row("Failed", f"[red]{summary['failed']}[/red]")
        summary_table.add_row("Warnings", f"[yellow]{summary['warnings']}[/yellow]")
        summary_table.add_row("Success Rate", f"{summary['success_rate']:.1%}")
        
        console.print(summary_table)
        
        # Detailed results table
        details_table = Table(title="Detailed Results")
        details_table.add_column("Check", style="cyan")
        details_table.add_column("Status", style="bold")
        details_table.add_column("Message", style="white")
        details_table.add_column("Duration", style="blue")
        
        for check in results['checks']:
            status_style = {
                'passed': '[green]PASS[/green]',
                'failed': '[red]FAIL[/red]',
                'warning': '[yellow]WARN[/yellow]',
                'error': '[red]ERROR[/red]'
            }.get(check['status'], check['status'])
            
            details_table.add_row(
                check['name'],
                status_style,
                check['message'][:80] + "..." if len(check['message']) > 80 else check['message'],
                f"{check['duration']:.2f}s"
            )
        
        console.print(details_table)


def setup_standard_checks(checker: HealthChecker):
    """Setup standard health checks."""
    checker.add_check("API Endpoint", checker.check_api_endpoint, critical=True)
    checker.add_check("Model Loading", checker.check_model_loading, critical=True)
    checker.add_check("Basic Compression", checker.check_compression_basic, critical=True)
    checker.add_check("GPU Availability", checker.check_gpu_availability, critical=False)
    checker.add_check("System Resources", checker.check_system_resources, critical=False)
    checker.add_check("Dependencies", checker.check_dependencies, critical=True)
    checker.add_check("Cache System", checker.check_cache_system, critical=False)


def main():
    """Main health check script."""
    parser = argparse.ArgumentParser(description="Health check for compression service")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the service")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", type=Path, help="Output results to JSON file")
    parser.add_argument("--exit-code", action="store_true", help="Exit with non-zero code on failure")
    parser.add_argument("--quiet", action="store_true", help="Suppress output (for scripting)")
    
    args = parser.parse_args()
    
    # Initialize health checker
    checker = HealthChecker(args.url, args.timeout)
    setup_standard_checks(checker)
    
    # Run health checks
    results = checker.run_all_checks()
    
    # Display results
    if not args.quiet:
        checker.display_results(results)
    
    # Save results to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            console.print(f"[blue]Results saved to {args.output}")
    
    # Exit with appropriate code
    if args.exit_code:
        if results['overall_status'] == 'unhealthy':
            sys.exit(1)  # Critical failure
        elif results['overall_status'] == 'degraded':
            sys.exit(2)  # Warning state
        else:
            sys.exit(0)  # Healthy
    
    if not args.quiet:
        console.print(f"\n[bold]Health check completed: {results['overall_status'].upper()}")


if __name__ == "__main__":
    main()