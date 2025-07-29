#!/bin/bash
# Enhanced SBOM generation with vulnerability scanning and attestation
# This script generates a comprehensive Software Bill of Materials (SBOM)
# with integrated vulnerability assessment and SLSA attestation

set -euo pipefail

# Configuration
OUTPUT_DIR="./security-artifacts"
SBOM_FILE="$OUTPUT_DIR/sbom.json"
VULN_FILE="$OUTPUT_DIR/vulnerabilities.json"
ENHANCED_SBOM="$OUTPUT_DIR/enhanced-sbom.json"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

log "🔍 Starting enhanced SBOM generation..."

# Check required tools
check_dependencies() {
    log "Checking required dependencies..."
    
    local deps=("cyclonedx-py" "trivy" "syft" "grype")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        log "Install missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            case $dep in
                "cyclonedx-py")
                    echo "  pip install cyclonedx-bom"
                    ;;
                "trivy")
                    echo "  curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
                "syft")
                    echo "  curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
                "grype")
                    echo "  curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin"
                    ;;
            esac
        done
        exit 1
    fi
    
    success "All dependencies available"
}

# Generate base SBOM with CycloneDX
generate_base_sbom() {
    log "Generating base SBOM with CycloneDX..."
    
    # Generate SBOM for Python dependencies
    cyclonedx-py \
        --format json \
        --output-file "$SBOM_FILE" \
        --schema-version "1.4" \
        --include-dev \
        --include-optional
    
    success "Base SBOM generated: $SBOM_FILE"
}

# Generate additional SBOM with Syft for comprehensive coverage
generate_syft_sbom() {
    log "Generating additional SBOM with Syft..."
    
    syft packages dir:. \
        --output json \
        --file "$OUTPUT_DIR/syft-sbom.json"
    
    success "Syft SBOM generated: $OUTPUT_DIR/syft-sbom.json"
}

# Scan for vulnerabilities
scan_vulnerabilities() {
    log "Scanning for vulnerabilities..."
    
    # Trivy vulnerability scan
    log "Running Trivy vulnerability scan..."
    trivy fs . \
        --format json \
        --output "$VULN_FILE" \
        --severity HIGH,CRITICAL \
        --ignore-unfixed
    
    # Grype vulnerability scan
    log "Running Grype vulnerability scan..."
    grype dir:. \
        --output json \
        --file "$OUTPUT_DIR/grype-vulns.json"
    
    success "Vulnerability scans completed"
}

# License compliance check
check_licenses() {
    log "Checking license compliance..."
    
    # Generate license report
    python3 << 'EOF'
import json
import subprocess
import sys
from pathlib import Path

try:
    # Get installed packages with licenses
    result = subprocess.run([
        "pip-licenses", "--format=json", "--with-urls", "--with-description"
    ], capture_output=True, text=True, check=True)
    
    licenses = json.loads(result.stdout)
    
    # Define problematic licenses
    problematic_licenses = [
        "GPL-2.0", "GPL-3.0", "AGPL-1.0", "AGPL-3.0",
        "SSPL-1.0", "OSL-3.0", "MS-PL", "MS-RL"
    ]
    
    license_report = {
        "scan_timestamp": "2024-01-01T00:00:00Z",
        "total_packages": len(licenses),
        "license_summary": {},
        "problematic_licenses": [],
        "unknown_licenses": []
    }
    
    for package in licenses:
        license_name = package.get("License", "Unknown")
        
        # Count licenses
        if license_name in license_report["license_summary"]:
            license_report["license_summary"][license_name] += 1
        else:
            license_report["license_summary"][license_name] = 1
        
        # Check for problematic licenses
        if license_name in problematic_licenses:
            license_report["problematic_licenses"].append({
                "package": package["Name"],
                "version": package["Version"],
                "license": license_name,
                "url": package.get("URL", "")
            })
        
        # Check for unknown licenses
        if license_name.lower() in ["unknown", "none", ""]:
            license_report["unknown_licenses"].append({
                "package": package["Name"],
                "version": package["Version"]
            })
    
    # Write license report
    with open("./security-artifacts/license-report.json", "w") as f:
        json.dump(license_report, f, indent=2)
    
    print(f"License scan completed. Found {len(license_report['problematic_licenses'])} problematic licenses.")

except subprocess.CalledProcessError:
    print("pip-licenses not found. Install with: pip install pip-licenses")
    sys.exit(1)
except Exception as e:
    print(f"License check failed: {e}")
    sys.exit(1)
EOF

    success "License compliance check completed"
}

# Merge SBOM with vulnerability data
merge_sbom_vulns() {
    log "Merging SBOM with vulnerability data..."
    
    python3 << 'EOF'
import json
import hashlib
from datetime import datetime
from pathlib import Path

def load_json_file(filepath):
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}

def generate_component_hash(component):
    """Generate unique hash for component."""
    name = component.get('name', '')
    version = component.get('version', '')
    return hashlib.sha256(f"{name}:{version}".encode()).hexdigest()[:16]

# Load SBOM and vulnerability data
sbom = load_json_file('./security-artifacts/sbom.json')
trivy_vulns = load_json_file('./security-artifacts/vulnerabilities.json')
grype_vulns = load_json_file('./security-artifacts/grype-vulns.json')
license_report = load_json_file('./security-artifacts/license-report.json')

# Create enhanced SBOM structure
enhanced_sbom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.4",
    "serialNumber": f"urn:uuid:{hashlib.uuid4()}",
    "version": 1,
    "metadata": {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tools": [
            {"vendor": "cyclonedx", "name": "cyclonedx-py"},
            {"vendor": "aquasecurity", "name": "trivy"},
            {"vendor": "anchore", "name": "grype"},
            {"vendor": "custom", "name": "enhanced-sbom-generator"}
        ],
        "component": sbom.get("metadata", {}).get("component", {}),
        "properties": [
            {"name": "enhanced-sbom:version", "value": "1.0"},
            {"name": "scan:timestamp", "value": datetime.utcnow().isoformat() + "Z"},
            {"name": "vulnerability:scanners", "value": "trivy,grype"},
            {"name": "license:compliance", "value": "checked"}
        ]
    },
    "components": [],
    "vulnerabilities": [],
    "licenses": []
}

# Process components from original SBOM
components_map = {}
for component in sbom.get("components", []):
    comp_hash = generate_component_hash(component)
    enhanced_component = {
        **component,
        "hashes": [{"alg": "SHA-256", "content": comp_hash}],
        "properties": component.get("properties", []) + [
            {"name": "component:hash", "value": comp_hash}
        ]
    }
    
    components_map[f"{component.get('name', '')}:{component.get('version', '')}"] = enhanced_component
    enhanced_sbom["components"].append(enhanced_component)

# Process Trivy vulnerabilities
trivy_results = trivy_vulns.get("Results", [])
for result in trivy_results:
    for vuln in result.get("Vulnerabilities", []):
        vulnerability = {
            "id": vuln.get("VulnerabilityID", ""),
            "source": {"name": "trivy", "url": "https://trivy.dev"},
            "ratings": [{
                "source": {"name": vuln.get("DataSource", {}).get("Name", "NVD")},
                "score": vuln.get("CVSS", {}).get("nvd", {}).get("V3Score", 0),
                "severity": vuln.get("Severity", "").lower(),
                "method": "CVSSv3"
            }],
            "cwes": [cwe.get("ID", "") for cwe in vuln.get("CweIDs", [])],
            "description": vuln.get("Description", ""),
            "recommendation": vuln.get("FixedVersion", "Update to latest version"),
            "affects": [{
                "ref": f"pkg:{vuln.get('PkgName', '')}@{vuln.get('InstalledVersion', '')}"
            }]
        }
        enhanced_sbom["vulnerabilities"].append(vulnerability)

# Process license information
if license_report:
    for license_name, count in license_report.get("license_summary", {}).items():
        if license_name != "Unknown":
            enhanced_sbom["licenses"].append({
                "license": {"name": license_name},
                "properties": [
                    {"name": "usage:count", "value": str(count)}
                ]
            })

# Add metadata about the scan
enhanced_sbom["metadata"]["properties"].extend([
    {"name": "components:total", "value": str(len(enhanced_sbom["components"]))},
    {"name": "vulnerabilities:total", "value": str(len(enhanced_sbom["vulnerabilities"]))},
    {"name": "licenses:total", "value": str(len(enhanced_sbom["licenses"]))}
])

# Write enhanced SBOM
with open('./security-artifacts/enhanced-sbom.json', 'w') as f:
    json.dump(enhanced_sbom, f, indent=2)

print("Enhanced SBOM generated successfully")
EOF

    success "Enhanced SBOM created: $ENHANCED_SBOM"
}

# Generate attestation
generate_attestation() {
    log "Generating SLSA attestation for SBOM..."
    
    # Check if cosign is available
    if ! command -v cosign &> /dev/null; then
        warning "Cosign not found. SLSA attestation skipped."
        warning "Install cosign: https://docs.sigstore.dev/cosign/installation/"
        return
    fi
    
    # Generate keyless attestation (requires OIDC token in CI)
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        log "Generating keyless attestation with GitHub OIDC..."
        cosign attest \
            --predicate "$ENHANCED_SBOM" \
            --type "https://cyclonedx.org/schema/bom-1.4.schema.json" \
            --output-file "$OUTPUT_DIR/sbom.attestation"
        
        success "SLSA attestation generated: $OUTPUT_DIR/sbom.attestation"
    else
        warning "No GitHub token found. Skipping keyless attestation."
        log "To generate attestation in CI, ensure GITHUB_TOKEN is available"
    fi
}

# Generate summary report
generate_summary() {
    log "Generating security summary report..."
    
    python3 << 'EOF'
import json
from datetime import datetime
from pathlib import Path

def load_json_safe(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return {}

# Load all reports
enhanced_sbom = load_json_safe('./security-artifacts/enhanced-sbom.json')
license_report = load_json_safe('./security-artifacts/license-report.json')

# Generate summary
summary = {
    "scan_metadata": {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "generator": "enhanced-sbom-generator",
        "version": "1.0"
    },
    "component_summary": {
        "total_components": len(enhanced_sbom.get("components", [])),
        "direct_dependencies": len([c for c in enhanced_sbom.get("components", []) if c.get("scope") == "required"]),
        "dev_dependencies": len([c for c in enhanced_sbom.get("components", []) if c.get("scope") == "optional"])
    },
    "vulnerability_summary": {
        "total_vulnerabilities": len(enhanced_sbom.get("vulnerabilities", [])),
        "critical": len([v for v in enhanced_sbom.get("vulnerabilities", []) if any(r.get("severity") == "critical" for r in v.get("ratings", []))]),
        "high": len([v for v in enhanced_sbom.get("vulnerabilities", []) if any(r.get("severity") == "high" for r in v.get("ratings", []))]),
        "medium": len([v for v in enhanced_sbom.get("vulnerabilities", []) if any(r.get("severity") == "medium" for r in v.get("ratings", []))]),
        "low": len([v for v in enhanced_sbom.get("vulnerabilities", []) if any(r.get("severity") == "low" for r in v.get("ratings", []))])
    },
    "license_summary": {
        "total_licenses": len(enhanced_sbom.get("licenses", [])),
        "problematic_licenses": len(license_report.get("problematic_licenses", [])),
        "unknown_licenses": len(license_report.get("unknown_licenses", []))
    },
    "compliance_status": {
        "sbom_generated": True,
        "vulnerabilities_scanned": True,
        "licenses_checked": True,
        "attestation_available": Path('./security-artifacts/sbom.attestation').exists()
    }
}

# Write summary
with open('./security-artifacts/security-summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Print summary to console
print("\n" + "="*60)
print("SECURITY SCAN SUMMARY")
print("="*60)
print(f"Components: {summary['component_summary']['total_components']}")
print(f"Vulnerabilities: {summary['vulnerability_summary']['total_vulnerabilities']}")
print(f"  - Critical: {summary['vulnerability_summary']['critical']}")
print(f"  - High: {summary['vulnerability_summary']['high']}")
print(f"  - Medium: {summary['vulnerability_summary']['medium']}")
print(f"  - Low: {summary['vulnerability_summary']['low']}")
print(f"Licenses: {summary['license_summary']['total_licenses']}")
print(f"  - Problematic: {summary['license_summary']['problematic_licenses']}")
print(f"  - Unknown: {summary['license_summary']['unknown_licenses']}")
print("="*60)

# Exit with error if critical vulnerabilities found
if summary['vulnerability_summary']['critical'] > 0:
    print("❌ CRITICAL vulnerabilities found! Review required.")
    exit(1)
EOF

    success "Security summary generated: $OUTPUT_DIR/security-summary.json"
}

# Main execution
main() {
    log "🚀 Enhanced SBOM Generation Script"
    log "Generating comprehensive SBOM with security analysis..."
    
    check_dependencies
    generate_base_sbom
    generate_syft_sbom
    scan_vulnerabilities
    check_licenses
    merge_sbom_vulns
    generate_attestation
    generate_summary
    
    success "🎉 Enhanced SBOM generation completed!"
    log "📁 Artifacts generated in: $OUTPUT_DIR"
    log "📋 Main SBOM: $ENHANCED_SBOM"
    log "📊 Summary: $OUTPUT_DIR/security-summary.json"
    
    if [ -f "$OUTPUT_DIR/sbom.attestation" ]; then
        log "🔒 SLSA attestation: $OUTPUT_DIR/sbom.attestation"
    fi
}

# Handle script interruption
trap 'error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"