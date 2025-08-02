#!/bin/bash

# Post-create script for devcontainer setup
set -e

echo "🚀 Setting up Retrieval-Free Context Compressor development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    unzip \
    htop \
    tree \
    jq \
    tmux \
    vim

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install the package in development mode
pip install -e ".[dev,gpu,evaluation,all]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data models cache logs
mkdir -p examples notebooks

# Set up git configuration (if not already set)
if [ -z "$(git config --global user.name)" ]; then
    echo "⚙️  Setting up git configuration..."
    git config --global user.name "Dev Container User"
    git config --global user.email "dev@example.com"
fi

# Install additional development tools
echo "🛠️  Installing additional development tools..."
pip install \
    jupyter \
    jupyterlab \
    ipywidgets \
    tensorboard \
    wandb \
    mlflow

# Check GPU availability
echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || echo "PyTorch not yet installed"

# Create example environment file
echo "📝 Creating .env.example if it doesn't exist..."
if [ ! -f .env.example ]; then
    cat > .env.example << 'EOF'
# Development Environment Variables
# Copy this file to .env and customize as needed

# Python Configuration
PYTHONPATH=src
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Model Configuration
HUGGINGFACE_HUB_CACHE=./cache/huggingface
TRANSFORMERS_CACHE=./cache/transformers
MODEL_CACHE_DIR=./cache/models

# Training Configuration
WANDB_PROJECT=retrieval-free-compression
WANDB_ENTITY=your-username
# WANDB_API_KEY=your-api-key

# Evaluation Configuration
EVAL_OUTPUT_DIR=./results
BENCHMARK_DATA_DIR=./data/benchmarks

# Security and API Keys (uncomment and set as needed)
# OPENAI_API_KEY=your-openai-key
# ANTHROPIC_API_KEY=your-anthropic-key
# COHERE_API_KEY=your-cohere-key

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
PYTEST_CURRENT_TEST=true
EOF
fi

# Initialize git hooks
echo "🔗 Initializing git hooks..."
git config core.hooksPath .githooks
mkdir -p .githooks

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name retrieval-free --display-name "Retrieval-Free Context Compressor"

# Create useful aliases
echo "⚡ Setting up development aliases..."
cat >> ~/.bashrc << 'EOF'

# Retrieval-Free Context Compressor aliases
alias ll='ls -la'
alias rf-test='pytest -v'
alias rf-lint='ruff check src/ tests/'
alias rf-format='black src/ tests/ && ruff check --fix src/ tests/'
alias rf-typecheck='mypy src/'
alias rf-ci='make ci'
alias rf-install='pip install -e ".[dev,gpu,evaluation,all]"'
alias rf-benchmark='python scripts/benchmark_compression.py'
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  rf-test       - Run tests"
echo "  rf-lint       - Check code quality"
echo "  rf-format     - Format code"
echo "  rf-ci         - Run full CI pipeline"
echo "  rf-benchmark  - Run performance benchmarks"
echo ""
echo "📚 Documentation: https://retrieval-free.readthedocs.io"
echo "💬 Community: Join our Discord for support and discussions"
echo ""
echo "Happy coding! 🚀"