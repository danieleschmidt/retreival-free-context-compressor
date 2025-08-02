#!/bin/bash

# Post-create script for development container setup
echo "🚀 Setting up Retrieval-Free Context Compressor development environment..."

# Update pip and install development dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,evaluation]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create common development directories
echo "📁 Creating development directories..."
mkdir -p {data,logs,models,notebooks,experiments}

# Set up git configuration if not already set
if ! git config user.name > /dev/null 2>&1; then
    echo "⚙️ Setting up git configuration..."
    git config --global init.defaultBranch main
    git config --global pull.rebase false
    git config --global core.autocrlf input
    git config --global core.editor "code --wait"
    
    # Set placeholder values - user should update these
    git config --global user.name "Developer"
    git config --global user.email "developer@example.com"
    
    echo "⚠️  Please update your git configuration:"
    echo "   git config --global user.name 'Your Name'"
    echo "   git config --global user.email 'your.email@example.com'"
fi

# Download sample data and models (if available)
echo "💾 Setting up development data..."
if [ -f "scripts/setup-dev.sh" ]; then
    bash scripts/setup-dev.sh
fi

# Set up monitoring tools (lightweight versions for development)
echo "📊 Setting up development monitoring..."
if [ -f "monitoring/docker-compose.monitoring.yml" ]; then
    echo "Monitoring stack available at monitoring/docker-compose.monitoring.yml"
    echo "Run: docker-compose -f monitoring/docker-compose.monitoring.yml up -d"
fi

# Verify installation
echo "✅ Running installation verification..."
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('PyTorch not installed - will be installed with project dependencies')

try:
    from retrieval_free import __version__
    print(f'retrieval-free version: {__version__}')
except ImportError:
    print('retrieval-free package not yet installed - run pip install -e .')
"

# Run tests to verify everything works
echo "🧪 Running quick test verification..."
if [ -d "tests" ]; then
    python -m pytest tests/ -x -v --tb=short || echo "⚠️ Some tests failed - this is expected in initial setup"
fi

# Set up shell enhancements
echo "🐚 Setting up shell enhancements..."
cat >> ~/.bashrc << 'EOF'

# Retrieval-Free Context Compressor development shortcuts
alias rf-test="python -m pytest tests/ -v"
alias rf-test-fast="python -m pytest tests/unit/ -v"
alias rf-lint="black . && ruff check . && mypy src/"
alias rf-clean="find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true"
alias rf-benchmark="python -m pytest tests/performance/ --benchmark-only"
alias rf-coverage="python -m pytest tests/ --cov=retrieval_free --cov-report=html"

# Useful development functions
rf-profile() {
    python -m cProfile -o profile.out "$@" && python -c "import pstats; pstats.Stats('profile.out').sort_stats('cumulative').print_stats(20)"
}

rf-memory() {
    python -m memory_profiler "$@"
}

# Show useful info on login
echo "🔬 Retrieval-Free Context Compressor Dev Environment"
echo "📁 Working directory: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "📦 Installed packages: pip list | wc -l packages"
echo ""
echo "🚀 Quick commands:"
echo "  rf-test       - Run all tests"
echo "  rf-test-fast  - Run unit tests only"
echo "  rf-lint       - Run code quality checks"
echo "  rf-benchmark  - Run performance benchmarks"
echo "  rf-coverage   - Generate coverage report"
echo ""
EOF

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Next steps:"
echo "  1. Update your git configuration with your details"
echo "  2. Run 'rf-test' to verify everything works"
echo "  3. Start developing with full IDE support!"
echo ""
echo "📖 Documentation: docs/"
echo "🧪 Run tests: rf-test"
echo "🔧 Code quality: rf-lint"
echo "📊 Performance: rf-benchmark"
echo ""