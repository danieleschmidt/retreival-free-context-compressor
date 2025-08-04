# Multi-stage Dockerfile for Retrieval-Free Context Compressor
# Optimized for production deployment

# Build stage
FROM python:3.11-slim-bullseye AS builder

# Set build arguments
ARG BUILD_VERSION=0.1.0
ARG BUILD_DATE
ARG VCS_REF

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .[all]

# Production stage
FROM python:3.11-slim-bullseye AS production

# Set build metadata
LABEL org.opencontainers.image.title="Retrieval-Free Context Compressor" \
      org.opencontainers.image.description="A transformer plug-in that compresses long documents into dense mega-tokens" \
      org.opencontainers.image.version="${BUILD_VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/photonic-mlir-synth-bridge" \
      org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH" \
    RETRIEVAL_FREE_CONFIG_DIR="/app/config" \
    RETRIEVAL_FREE_DATA_DIR="/app/data" \
    RETRIEVAL_FREE_CACHE_DIR="/app/cache"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directories
RUN mkdir -p /app/config /app/data /app/cache /app/logs \
    && chown -R appuser:appuser /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser pyproject.toml README.md /app/

# Switch to non-root user
USER appuser
WORKDIR /app

# Install application
RUN pip install --no-cache-dir --user -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.retrieval_free.monitoring import HealthChecker; hc = HealthChecker(); result = hc.get_overall_health(); exit(0 if result['healthy'] else 1)"

# Expose port (if running as API server)
EXPOSE 8000

# Default command
CMD ["python", "-m", "retrieval_free.cli", "--help"]

# Development stage
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Install development Python packages
RUN pip install --no-cache-dir --user \
    pytest \
    pytest-cov \
    black \
    ruff \
    mypy \
    jupyter \
    ipython

# Override entrypoint for development
CMD ["bash"]