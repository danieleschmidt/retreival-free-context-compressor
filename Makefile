.PHONY: help install install-dev test lint format typecheck clean build docs docker-build docker-build-dev docker-build-gpu docker-build-all docker-test docker-clean compose-up compose-down compose-dev compose-gpu compose-logs release-build release-test security security-docker ci
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	pip install -e .

install-dev: ## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest

test-cov: ## Run tests with coverage
	pytest --cov=retrieval_free --cov-report=html --cov-report=term

lint: ## Run linting
	ruff check src tests
	black --check src tests

format: ## Format code
	black src tests
	ruff check --fix src tests

typecheck: ## Run type checking
	mypy src

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build package
	python -m build

docs: ## Build documentation
	@echo "Documentation build command to be implemented"

# Docker commands
docker-build: ## Build Docker image
	./scripts/build.sh prod

docker-build-dev: ## Build development Docker image
	./scripts/build.sh dev

docker-build-gpu: ## Build GPU Docker image
	./scripts/build.sh gpu

docker-build-all: ## Build all Docker images
	./scripts/build.sh all

docker-test: ## Build and test Docker images
	./scripts/build.sh test

docker-clean: ## Clean Docker artifacts
	./scripts/build.sh clean

# Docker Compose commands
compose-up: ## Start services with docker-compose
	docker-compose up -d

compose-down: ## Stop services with docker-compose
	docker-compose down

compose-dev: ## Start development services
	docker-compose --profile dev up -d

compose-gpu: ## Start GPU services
	docker-compose --profile gpu up -d

compose-logs: ## Show docker-compose logs
	docker-compose logs -f

# Release commands
release-build: clean lint typecheck test docker-build ## Build release artifacts
	python -m build

release-test: ## Test release artifacts
	python -m twine check dist/*

# Security commands
security: ## Run security checks
	bandit -r src/
	safety check

security-docker: ## Run Docker security scan
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image retrieval-free:prod-latest; \
	else \
		echo "Trivy not installed. Install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"; \
	fi

ci: lint typecheck test security ## Run all CI checks