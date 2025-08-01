#!/bin/bash
set -e

echo "🚀 Setting up development environment for Retrieval-Free Context Compressor"

echo "📦 Installing package in development mode..."
pip install -e ".[dev,evaluation]"

echo "🔧 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

echo "🧪 Running initial code quality checks..."
echo "  - Black formatting check..."
black --check . || (echo "❌ Code formatting issues found. Run 'black .' to fix." && exit 1)

echo "  - Ruff linting check..."
ruff check . || (echo "❌ Linting issues found. Run 'ruff check . --fix' to fix." && exit 1)

echo "  - MyPy type checking..."
mypy src/ || (echo "❌ Type checking issues found." && exit 1)

echo "  - Security scan..."
bandit -r src/ || (echo "❌ Security issues found." && exit 1)

echo "  - Dependency security check..."
safety check || (echo "❌ Vulnerable dependencies found." && exit 1)

echo "🧪 Running tests..."
pytest tests/ --cov=retrieval_free --cov-report=term-missing

echo "✅ Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Run tests: pytest tests/"
echo "  2. Run benchmarks: pytest tests/performance/ --benchmark-only"
echo "  3. Start developing!"
echo ""
echo "🔍 Useful commands:"
echo "  - Format code: black ."
echo "  - Lint code: ruff check . --fix"
echo "  - Type check: mypy src/"
echo "  - Security scan: bandit -r src/"
echo "  - Run pre-commit on all files: pre-commit run --all-files"