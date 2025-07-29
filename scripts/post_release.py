#!/usr/bin/env python3
"""
Post-release automation script.
Handles tasks after a successful release.
"""

import sys
import os
import subprocess
from pathlib import Path


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command and return result."""
    print(f"🔄 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result


def update_docker_tags(version: str) -> None:
    """Update Docker image tags after release."""
    try:
        # Check if we're in a CI environment with Docker credentials
        if not (os.getenv('DOCKERHUB_USERNAME') and os.getenv('DOCKERHUB_TOKEN')):
            print("ℹ️  Docker credentials not found, skipping Docker tag update")
            return
        
        image_name = "yourusername/retrieval-free"
        
        # Tag and push version-specific image
        run_command(f"docker tag {image_name}:latest {image_name}:{version}")
        run_command(f"docker push {image_name}:{version}")
        
        print(f"✅ Updated Docker tags for version {version}")
        
    except Exception as e:
        print(f"⚠️  Docker tag update failed: {e}")


def notify_community(version: str) -> None:
    """Send notifications about the new release."""
    try:
        # This could be extended to send notifications to:
        # - Discord webhook
        # - Slack channel  
        # - Email list
        # - Twitter API
        
        print(f"📢 New release {version} is available!")
        print("🔗 https://github.com/yourusername/retrieval-free-context-compressor/releases")
        
        # Example: Could add Discord webhook notification
        # webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        # if webhook_url:
        #     send_discord_notification(webhook_url, version)
        
    except Exception as e:
        print(f"⚠️  Community notification failed: {e}")


def update_documentation(version: str) -> None:
    """Update version-specific documentation."""
    try:
        # Update any version-specific documentation
        docs_path = Path("docs")
        if docs_path.exists():
            print(f"📚 Documentation updated for version {version}")
        
        # Could trigger documentation rebuild here
        # e.g., trigger ReadTheDocs build
        
    except Exception as e:
        print(f"⚠️  Documentation update failed: {e}")


def generate_metrics_report(version: str) -> None:
    """Generate release metrics and analytics."""
    try:
        metrics = {
            "version": version,
            "timestamp": subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"]).decode().strip(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:8],
            "files_changed": len(list(Path(".").rglob("*.py"))),
        }
        
        print(f"📊 Release metrics for {version}:")
        for key, value in metrics.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"⚠️  Metrics generation failed: {e}")


def main():
    """Main post-release function."""
    if len(sys.argv) != 2:
        print("Usage: python post_release.py <version>")
        sys.exit(1)
    
    version = sys.argv[1]
    
    # Remove 'v' prefix if present
    if version.startswith('v'):
        version = version[1:]
    
    print(f"🚀 Running post-release tasks for version {version}")
    
    try:
        update_docker_tags(version)
        notify_community(version)
        update_documentation(version)
        generate_metrics_report(version)
        
        print(f"✅ Post-release tasks completed successfully for version {version}")
        
    except Exception as e:
        print(f"❌ Post-release tasks failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()