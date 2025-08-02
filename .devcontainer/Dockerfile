# Use Python base image with development tools
FROM mcr.microsoft.com/vscode/devcontainers/python:1-3.11-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    curl \
    wget \
    git \
    # GPU support (if needed)
    nvidia-cuda-toolkit \
    # Development tools
    vim \
    nano \
    htop \
    tree \
    jq \
    # Security tools
    gpg \
    # Monitoring tools
    netcat \
    telnet \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python development tools
RUN pip install --no-cache-dir \
    # Core development
    pip-tools \
    pipx \
    # Code quality
    black \
    ruff \
    mypy \
    pre-commit \
    # Testing
    pytest \
    pytest-cov \
    pytest-benchmark \
    pytest-xdist \
    hypothesis \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    mkdocs \
    mkdocs-material \
    # Security
    bandit \
    safety \
    pip-audit \
    # Debugging
    ipdb \
    ipython \
    # Jupyter
    jupyterlab \
    notebook

# Install Node.js for additional tooling
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Create workspace directory
RUN mkdir -p /workspace \
    && chown -R vscode:vscode /workspace

# Set up git configuration for the container
RUN git config --system --add safe.directory /workspace

# Switch to vscode user
USER vscode

# Set up Python environment
RUN python -m pip install --user --upgrade pip setuptools wheel

# Install additional user tools
RUN pipx install \
    cookiecutter \
    poetry \
    pre-commit

# Ensure pipx is in PATH
ENV PATH="/home/vscode/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Default command
CMD ["bash"]