# Docker Deployment Guide

This guide covers deploying the Retrieval-Free Context Compressor using Docker.

## Quick Start

### Basic Usage

```bash
# Build the image
docker build -t retrieval-free:latest .

# Run the container
docker run -p 8000:8000 retrieval-free:latest
```

### Using Docker Compose

```bash
# Start production service
docker-compose up -d

# Start development service
docker-compose --profile dev up -d

# Start GPU-enabled service
docker-compose --profile gpu up -d
```

## Container Variants

### Production Container

Optimized for production deployment with minimal attack surface:

```dockerfile
FROM python:3.11-slim AS production
```

Features:
- Multi-stage build for smaller image size
- Non-root user for security
- Health checks enabled
- Minimal runtime dependencies

```bash
# Build production image
docker build --target production -t retrieval-free:prod .

# Run with resource limits
docker run \
  --memory=2g \
  --cpus=2 \
  -p 8000:8000 \
  retrieval-free:prod
```

### Development Container

Includes development tools and hot reload:

```bash
# Build development image
docker build --target builder -t retrieval-free:dev .

# Run with volume mounts for development
docker run \
  -v $(pwd):/app \
  -p 8001:8001 \
  retrieval-free:dev
```

### GPU Container

Optimized for GPU workloads:

```bash
# Build GPU image
docker build -t retrieval-free:gpu .

# Run with GPU support
docker run \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  retrieval-free:gpu
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python module path | `/app/src` |
| `RETRIEVAL_FREE_LOG_LEVEL` | Logging level | `INFO` |
| `RETRIEVAL_FREE_USE_GPU` | Enable GPU acceleration | `false` |
| `RETRIEVAL_FREE_MODEL_CACHE` | Model cache directory | `/app/models` |
| `RETRIEVAL_FREE_DATA_DIR` | Data directory | `/app/data` |

### Volume Mounts

| Host Path | Container Path | Purpose | Mode |
|-----------|----------------|---------|------|
| `./data` | `/app/data` | Input data | `ro` |
| `./models` | `/app/models` | Model storage | `rw` |
| `./logs` | `/app/logs` | Application logs | `rw` |
| `./cache` | `/app/.cache` | Cache storage | `rw` |

## Deployment Scenarios

### Single Container Deployment

For simple deployments:

```bash
# Create necessary directories
mkdir -p data models logs

# Run with proper mounts
docker run -d \
  --name retrieval-free \
  --restart unless-stopped \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models:rw \
  -v $(pwd)/logs:/app/logs:rw \
  retrieval-free:latest
```

### Docker Compose Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  retrieval-free:
    image: retrieval-free:latest
    container_name: retrieval-free-prod
    restart: always
    ports:
      - "8000:8000"
    environment:
      - RETRIEVAL_FREE_LOG_LEVEL=WARNING
      - RETRIEVAL_FREE_USE_GPU=false
    volumes:
      - /opt/retrieval-free/data:/app/data:ro
      - /opt/retrieval-free/models:/app/models:rw
      - /opt/retrieval-free/logs:/app/logs:rw
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - production
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

networks:
  production:
    driver: bridge
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retrieval-free
  labels:
    app: retrieval-free
spec:
  replicas: 3
  selector:
    matchLabels:
      app: retrieval-free
  template:
    metadata:
      labels:
        app: retrieval-free
    spec:
      containers:
      - name: retrieval-free
        image: retrieval-free:latest
        ports:
        - containerPort: 8000
        env:
        - name: RETRIEVAL_FREE_LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: data
          mountPath: /app/data
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: retrieval-free-models
      - name: data
        configMap:
          name: retrieval-free-data
---
apiVersion: v1
kind: Service
metadata:
  name: retrieval-free-service
spec:
  selector:
    app: retrieval-free
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Security Considerations

### Container Security

1. **Non-root user**: Container runs as non-privileged user
2. **Minimal base image**: Uses slim Python image
3. **No secrets in image**: Secrets passed via environment/volumes
4. **Read-only filesystem**: Where possible, mount volumes as read-only

### Image Scanning

```bash
# Scan image for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image retrieval-free:latest

# Check for high/critical vulnerabilities only
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image --severity HIGH,CRITICAL retrieval-free:latest
```

### Runtime Security

```bash
# Run with security options
docker run \
  --security-opt=no-new-privileges:true \
  --cap-drop=ALL \
  --read-only \
  --tmpfs /tmp \
  retrieval-free:latest
```

## Performance Optimization

### Build Optimization

1. **Multi-stage builds**: Reduce final image size
2. **Layer caching**: Order Dockerfile commands by change frequency
3. **Minimize layers**: Combine RUN commands where possible

### Runtime Optimization

```bash
# Set memory limits based on workload
docker run \
  --memory=4g \
  --memory-swap=4g \
  --oom-kill-disable=false \
  retrieval-free:latest

# CPU optimization
docker run \
  --cpus=2 \
  --cpu-shares=1024 \
  retrieval-free:latest
```

### GPU Optimization

```bash
# Limit GPU memory
docker run \
  --gpus all \
  -e NVIDIA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
  -e NVIDIA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
  retrieval-free:gpu
```

## Monitoring and Logging

### Health Checks

Built-in health check endpoint:

```bash
# Check container health
curl http://localhost:8000/health

# Docker health status
docker inspect --format='{{.State.Health.Status}}' retrieval-free
```

### Logging Configuration

```bash
# Configure log driver
docker run \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  retrieval-free:latest

# Send logs to syslog
docker run \
  --log-driver=syslog \
  --log-opt syslog-address=tcp://logserver:514 \
  retrieval-free:latest
```

### Metrics Collection

```yaml
# Add Prometheus metrics sidecar
prometheus:
  image: prom/prometheus:latest
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
  ports:
    - "9090:9090"
  command:
    - '--config.file=/etc/prometheus/prometheus.yml'
    - '--storage.tsdb.path=/prometheus'
```

## Troubleshooting

### Common Issues

**Container fails to start**
```bash
# Check logs
docker logs retrieval-free

# Check container status
docker inspect retrieval-free
```

**Out of memory errors**
```bash
# Check memory usage
docker stats retrieval-free

# Increase memory limit
docker update --memory=4g retrieval-free
```

**GPU not detected**
```bash
# Verify NVIDIA runtime
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi

# Check NVIDIA Docker setup
docker run --rm --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 \
  bash -c 'echo "GPU Count: $(nvidia-smi --list-gpus | wc -l)"'
```

### Debug Mode

```bash
# Run in debug mode
docker run -it --entrypoint=/bin/bash retrieval-free:latest

# Check Python environment
docker exec -it retrieval-free python -c "import retrieval_free; print('OK')"

# Inspect filesystem
docker exec -it retrieval-free ls -la /app
```

### Performance Profiling

```bash
# Profile container resource usage
docker run --rm -it \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /sys:/sys:ro \
  google/cadvisor:latest

# Memory profiling
docker run --rm -it \
  -v $(pwd):/workspace \
  python:3.11-slim \
  python -m memory_profiler /workspace/scripts/profile_memory.py
```

## Best Practices

### Development

1. Use `.dockerignore` to exclude unnecessary files
2. Pin dependency versions for reproducibility
3. Use multi-stage builds for cleaner separation
4. Test images on multiple architectures

### Production

1. Use specific image tags, not `latest`
2. Implement proper health checks
3. Set resource limits and requests
4. Use secrets management for sensitive data
5. Regularly update base images for security patches

### CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Build Docker image
  run: |
    docker build -t retrieval-free:${{ github.sha }} .
    docker tag retrieval-free:${{ github.sha }} retrieval-free:latest

- name: Security scan
  run: |
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
      aquasec/trivy image retrieval-free:${{ github.sha }}

- name: Push to registry
  run: |
    echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
    docker push retrieval-free:${{ github.sha }}
    docker push retrieval-free:latest
```

---

For more deployment options, see:
- [Kubernetes Guide](kubernetes-guide.md)
- [Cloud Deployment](cloud-deployment.md)
- [Production Checklist](production-checklist.md)