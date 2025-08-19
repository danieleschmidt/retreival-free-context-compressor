# Production Dockerfile for Retrieval-Free Context Compression
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r compression && useradd -r -g compression compression

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY *.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp && \
    chown -R compression:compression /app

# Switch to non-root user
USER compression

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app/src
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# Start application
CMD ["python", "-m", "uvicorn", "src.retrieval_free.async_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
