#!/bin/bash

# Retrieval-Free Context Compressor Build Script
# This script automates the build process for different deployment targets

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Configuration
PROJECT_NAME="retrieval-free-context-compressor"
IMAGE_NAME="retrieval-free"
VERSION=${VERSION:-$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])")}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=${GITHUB_SHA:-$(git rev-parse --short HEAD)}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Retrieval-Free Context Compressor Build Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
  dev         Build development image
  prod        Build production image
  gpu         Build GPU-enabled image
  all         Build all images
  test        Build and test images
  push        Build and push to registry
  clean       Clean build artifacts

Options:
  -h, --help     Show this help message
  -v, --version  Show version
  -t, --tag      Specify image tag (default: $VERSION)
  -r, --registry Registry prefix (default: none)
  --no-cache     Build without Docker cache
  --platform     Target platform (default: linux/amd64)

Examples:
  $0 prod                           # Build production image
  $0 all -t latest                  # Build all images with 'latest' tag
  $0 push -r myregistry.com/        # Build and push to registry
  $0 test --no-cache                # Build and test without cache

EOF
}

# Parse command line arguments
COMMAND=""
TAG="$VERSION"
REGISTRY=""
NO_CACHE=""
PLATFORM="linux/amd64"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            echo "$VERSION"
            exit 0
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        dev|prod|gpu|all|test|push|clean)
            COMMAND="$1"
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate command
if [[ -z "$COMMAND" ]]; then
    log_error "No command specified"
    show_help
    exit 1
fi

# Build arguments
BUILD_ARGS=(
    --build-arg VERSION="$VERSION"
    --build-arg BUILD_DATE="$BUILD_DATE"
    --build-arg GIT_COMMIT="$GIT_COMMIT"
    --platform "$PLATFORM"
)

if [[ -n "$NO_CACHE" ]]; then
    BUILD_ARGS+=("$NO_CACHE")
fi

# Helper functions
build_image() {
    local target=$1
    local image_tag="${REGISTRY}${IMAGE_NAME}:${target}-${TAG}"
    
    log_info "Building $target image: $image_tag"
    
    if [[ "$target" == "prod" ]]; then
        docker build "${BUILD_ARGS[@]}" --target production -t "$image_tag" .
    elif [[ "$target" == "dev" ]]; then
        docker build "${BUILD_ARGS[@]}" --target builder -t "$image_tag" .
    elif [[ "$target" == "gpu" ]]; then
        docker build "${BUILD_ARGS[@]}" --target production -t "$image_tag" -f Dockerfile.gpu .
    else
        docker build "${BUILD_ARGS[@]}" -t "$image_tag" .
    fi
    
    # Also tag as latest for convenience
    if [[ "$TAG" != "latest" ]]; then
        docker tag "$image_tag" "${REGISTRY}${IMAGE_NAME}:${target}-latest"
    fi
    
    log_success "Built $image_tag"
}

test_image() {
    local target=$1
    local image_tag="${REGISTRY}${IMAGE_NAME}:${target}-${TAG}"
    
    log_info "Testing $target image: $image_tag"
    
    # Basic functionality test
    docker run --rm "$image_tag" python -c "import retrieval_free; print('Import successful')"
    
    # Health check test
    if [[ "$target" != "dev" ]]; then
        container_id=$(docker run -d -p 8000:8000 "$image_tag")
        sleep 10
        
        if docker exec "$container_id" python -c "import retrieval_free; print('Health check passed')"; then
            log_success "Health check passed for $image_tag"
        else
            log_error "Health check failed for $image_tag"
            docker logs "$container_id"
            docker stop "$container_id"
            exit 1
        fi
        
        docker stop "$container_id"
    fi
    
    log_success "Tests passed for $image_tag"
}

push_image() {
    local target=$1
    local image_tag="${REGISTRY}${IMAGE_NAME}:${target}-${TAG}"
    local latest_tag="${REGISTRY}${IMAGE_NAME}:${target}-latest"
    
    log_info "Pushing $target image: $image_tag"
    
    docker push "$image_tag"
    
    if [[ "$TAG" != "latest" ]]; then
        docker push "$latest_tag"
    fi
    
    log_success "Pushed $image_tag"
}

clean_images() {
    log_info "Cleaning build artifacts"
    
    # Remove dangling images
    docker image prune -f
    
    # Remove old images (keep last 5 versions)
    docker images "${REGISTRY}${IMAGE_NAME}" --format "table {{.Repository}}:{{.Tag}}\t{{.CreatedAt}}" | \
        tail -n +6 | \
        awk '{print $1}' | \
        xargs -r docker rmi
    
    log_success "Cleaned build artifacts"
}

# Security scan
security_scan() {
    local image_tag=$1
    
    log_info "Running security scan on $image_tag"
    
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL "$image_tag"
    else
        log_warning "Trivy not found, skipping security scan"
        log_info "Install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"
    fi
}

# Validate Docker environment
validate_environment() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if buildx is available for multi-platform builds
    if [[ "$PLATFORM" != "linux/amd64" ]] && ! docker buildx version &> /dev/null; then
        log_warning "Docker buildx not available, falling back to native build"
        PLATFORM="linux/amd64"
    fi
}

# Pre-build checks
pre_build_checks() {
    log_info "Running pre-build checks"
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Are you in the project root?"
        exit 1
    fi
    
    # Check if Dockerfile exists
    if [[ ! -f "Dockerfile" ]]; then
        log_error "Dockerfile not found"
        exit 1
    fi
    
    # Validate Python syntax
    if command -v python &> /dev/null; then
        python -m py_compile src/retrieval_free/__init__.py || {
            log_error "Python syntax errors found"
            exit 1
        }
    fi
    
    log_success "Pre-build checks passed"
}

# Main execution
main() {
    log_info "Starting build process for $PROJECT_NAME"
    log_info "Version: $VERSION"
    log_info "Git commit: $GIT_COMMIT"
    log_info "Build date: $BUILD_DATE"
    log_info "Platform: $PLATFORM"
    
    validate_environment
    pre_build_checks
    
    case $COMMAND in
        dev)
            build_image "dev"
            security_scan "${REGISTRY}${IMAGE_NAME}:dev-${TAG}"
            ;;
        prod)
            build_image "prod"
            security_scan "${REGISTRY}${IMAGE_NAME}:prod-${TAG}"
            ;;
        gpu)
            if [[ ! -f "Dockerfile.gpu" ]]; then
                log_warning "Dockerfile.gpu not found, using regular Dockerfile"
            fi
            build_image "gpu"
            security_scan "${REGISTRY}${IMAGE_NAME}:gpu-${TAG}"
            ;;
        all)
            build_image "dev"
            build_image "prod"
            if [[ -f "Dockerfile.gpu" ]]; then
                build_image "gpu"
            fi
            ;;
        test)
            build_image "prod"
            test_image "prod"
            build_image "dev"
            test_image "dev"
            ;;
        push)
            if [[ -z "$REGISTRY" ]]; then
                log_error "Registry not specified. Use -r flag to set registry"
                exit 1
            fi
            build_image "prod"
            test_image "prod"
            push_image "prod"
            ;;
        clean)
            clean_images
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
    
    log_success "Build process completed successfully"
}

# Run main function
main "$@"