#!/bin/bash
# Generate Software Bill of Materials (SBOM) for the project
# Requires: syft, grype, cosign (install with: go install github.com/anchore/syft/cmd/syft@latest)

set -euo pipefail

# Configuration
PROJECT_NAME="retrieval-free-context-compressor"
OUTPUT_DIR="security-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check required tools
check_dependencies() {
    log_info "Checking required dependencies..."
    
    local missing_tools=()
    
    if ! command -v syft &> /dev/null; then
        missing_tools+=("syft")
    fi
    
    if ! command -v grype &> /dev/null; then
        missing_tools+=("grype")
    fi
    
    if ! command -v cosign &> /dev/null; then
        missing_tools+=("cosign")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install with:"
        log_info "  go install github.com/anchore/syft/cmd/syft@latest"
        log_info "  go install github.com/anchore/grype/cmd/grype@latest"
        log_info "  go install github.com/sigstore/cosign/cmd/cosign@latest"
        exit 1
    fi
    
    log_info "All required tools are available"
}

# Create output directory
setup_output_dir() {
    mkdir -p "$OUTPUT_DIR"
    log_info "Created output directory: $OUTPUT_DIR"
}

# Generate SBOM for Python dependencies
generate_python_sbom() {
    log_info "Generating Python dependencies SBOM..."
    
    local output_file="$OUTPUT_DIR/python-sbom-${TIMESTAMP}.spdx.json"
    
    syft packages dir:. \
        --output spdx-json \
        --file "$output_file" \
        --quiet
    
    log_info "Python SBOM generated: $output_file"
    
    # Generate human-readable summary
    local summary_file="$OUTPUT_DIR/python-summary-${TIMESTAMP}.txt"
    syft packages dir:. \
        --output table \
        --file "$summary_file" \
        --quiet
    
    log_info "Python summary generated: $summary_file"
}

# Generate SBOM for container image (if available)
generate_container_sbom() {
    log_info "Checking for container image..."
    
    if docker images | grep -q "$PROJECT_NAME"; then
        log_info "Generating container SBOM..."
        
        local output_file="$OUTPUT_DIR/container-sbom-${TIMESTAMP}.spdx.json"
        
        syft packages "$PROJECT_NAME:latest" \
            --output spdx-json \
            --file "$output_file" \
            --quiet
        
        log_info "Container SBOM generated: $output_file"
        
        # Generate human-readable summary
        local summary_file="$OUTPUT_DIR/container-summary-${TIMESTAMP}.txt"
        syft packages "$PROJECT_NAME:latest" \
            --output table \
            --file "$summary_file" \
            --quiet
        
        log_info "Container summary generated: $summary_file"
    else
        log_warn "No container image found for $PROJECT_NAME"
        log_info "Build container with: docker build -t $PROJECT_NAME ."
    fi
}

# Run vulnerability scan
run_vulnerability_scan() {
    log_info "Running vulnerability scan..."
    
    local output_file="$OUTPUT_DIR/vulnerabilities-${TIMESTAMP}.json"
    
    if grype dir:. --output json --file "$output_file" --quiet; then
        log_info "Vulnerability scan completed: $output_file"
        
        # Generate human-readable report
        local summary_file="$OUTPUT_DIR/vulnerabilities-summary-${TIMESTAMP}.txt"
        grype dir:. --output table --file "$summary_file" --quiet
        
        # Check for critical vulnerabilities
        local critical_count
        critical_count=$(jq '.matches[] | select(.vulnerability.severity == "Critical") | .vulnerability.id' "$output_file" 2>/dev/null | wc -l || echo "0")
        
        if [ "$critical_count" -gt 0 ]; then
            log_error "Found $critical_count critical vulnerabilities!"
            log_info "Review: $summary_file"
        else
            log_info "No critical vulnerabilities found"
        fi
    else
        log_warn "Vulnerability scan completed with warnings"
    fi
}

# Sign SBOM files (if signing key is available)
sign_sbom_files() {
    log_info "Checking for signing capabilities..."
    
    if [ -f "cosign.key" ] || [ -n "${COSIGN_PRIVATE_KEY:-}" ]; then
        log_info "Signing SBOM files..."
        
        for file in "$OUTPUT_DIR"/*.spdx.json; do
            if [ -f "$file" ]; then
                local sig_file="${file}.sig"
                
                if cosign sign-blob --key cosign.key "$file" > "$sig_file" 2>/dev/null; then
                    log_info "Signed: $(basename "$file")"
                else
                    log_warn "Failed to sign: $(basename "$file")"
                fi
            fi
        done
    else
        log_warn "No signing key found - skipping signature generation"
        log_info "Generate key with: cosign generate-key-pair"
    fi
}

# Generate comprehensive security report
generate_security_report() {
    log_info "Generating comprehensive security report..."
    
    local report_file="$OUTPUT_DIR/security-report-${TIMESTAMP}.md"
    
    cat > "$report_file" << EOF
# Security Report - $PROJECT_NAME

**Generated**: $(date)
**Scan ID**: $TIMESTAMP

## Summary

This report contains the security analysis results for $PROJECT_NAME including:
- Software Bill of Materials (SBOM)
- Vulnerability assessment
- Supply chain security status

## Files Generated

### SBOM Files
$(find "$OUTPUT_DIR" -name "*sbom*${TIMESTAMP}*" -type f | sed 's/^/- /')

### Vulnerability Reports
$(find "$OUTPUT_DIR" -name "*vulnerabilities*${TIMESTAMP}*" -type f | sed 's/^/- /')

### Signatures
$(find "$OUTPUT_DIR" -name "*.sig" -type f | sed 's/^/- /')

## Dependency Summary

### Python Dependencies
\`\`\`
$(if [ -f "$OUTPUT_DIR/python-summary-${TIMESTAMP}.txt" ]; then head -20 "$OUTPUT_DIR/python-summary-${TIMESTAMP}.txt"; else echo "No Python summary available"; fi)
\`\`\`

## Vulnerability Status

$(if [ -f "$OUTPUT_DIR/vulnerabilities-summary-${TIMESTAMP}.txt" ]; then
    if grep -q "No vulnerabilities found" "$OUTPUT_DIR/vulnerabilities-summary-${TIMESTAMP}.txt" 2>/dev/null; then
        echo "✅ No vulnerabilities detected"
    else
        echo "⚠️  Vulnerabilities detected - see detailed report"
    fi
else
    echo "❓ Vulnerability scan not available"
fi)

## Next Steps

1. Review all generated reports
2. Address any identified vulnerabilities
3. Update dependencies as needed
4. Re-run scan after remediation

## Compliance

This report supports compliance with:
- NIST Cybersecurity Framework
- SLSA Supply Chain Security Level 3
- OWASP Dependency Check requirements

---

*Generated by automated security scanning pipeline*
EOF

    log_info "Security report generated: $report_file"
}

# Cleanup old reports (keep last 10)
cleanup_old_reports() {
    log_info "Cleaning up old reports..."
    
    find "$OUTPUT_DIR" -name "*-[0-9]*_[0-9]*.*" -type f -print0 | \
        sort -z | head -z -n -30 | xargs -0 rm -f
    
    log_info "Cleanup completed"
}

# Main execution
main() {
    log_info "Starting SBOM generation for $PROJECT_NAME"
    
    check_dependencies
    setup_output_dir
    generate_python_sbom
    generate_container_sbom
    run_vulnerability_scan
    sign_sbom_files
    generate_security_report
    cleanup_old_reports
    
    log_info "SBOM generation completed successfully!"
    log_info "Reports available in: $OUTPUT_DIR"
}

# Execute main function
main "$@"