# Deployment Guide

This guide covers deployment strategies for the Retrieval-Free Context Compressor across different environments.

## Overview

The system supports multiple deployment modes:
- **Local Development**: Single-machine setup for development and testing
- **Docker Containers**: Containerized deployment for consistency
- **Cloud Native**: Kubernetes deployment for scalability
- **Edge Deployment**: Lightweight deployment for edge environments
- **API Service**: REST API deployment for service integration

## Local Development Deployment

### Quick Start
```bash
# Clone and setup
git clone <repository-url>
cd retrieval-free-context-compressor
make install-dev

# Run local server
python -m retrieval_free.server --host 0.0.0.0 --port 8000
```

### Development Server Configuration
```yaml
# config/development.yaml
server:
  host: "127.0.0.1"
  port: 8000
  debug: true
  reload: true

model:
  name: "rfcc-base-8x"
  device: "auto"
  batch_size: 4
  max_length: 4096

cache:
  enabled: true
  backend: "memory"
  ttl: 3600

logging:
  level: "DEBUG"
  format: "detailed"
```

## Docker Deployment

### Basic Docker Setup
```dockerfile
# Dockerfile (production)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install -e ".[all]"

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "retrieval_free.server"]
```

### Docker Compose Setup
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  compressor:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME=rfcc-base-8x
      - DEVICE=cuda
      - BATCH_SIZE=8
      - REDIS_URL=redis://redis:6379
    volumes:
      - model_cache:/app/models
      - ./config:/app/config:ro
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - compressor

volumes:
  model_cache:
  redis_data:
```

### Multi-Stage Docker Build
```dockerfile
# Dockerfile.multi-stage
FROM python:3.11-slim as builder

WORKDIR /build
COPY pyproject.toml .
RUN pip install build && python -m build

FROM python:3.11-slim as runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built package
COPY --from=builder /build/dist/*.whl .
RUN pip install *.whl && rm *.whl

# Copy runtime files
COPY config/ ./config/
COPY scripts/entrypoint.sh ./

RUN chmod +x entrypoint.sh
USER 1000

ENTRYPOINT ["./entrypoint.sh"]
```

## Kubernetes Deployment

### Namespace Setup
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: retrieval-free
  labels:
    name: retrieval-free
    purpose: compression-service
```

### ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: compressor-config
  namespace: retrieval-free
data:
  config.yaml: |
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4
    
    model:
      name: "rfcc-base-8x"
      device: "cuda"
      batch_size: 16
      max_length: 8192
    
    cache:
      enabled: true
      backend: "redis"
      url: "redis://redis-service:6379"
      ttl: 7200
    
    monitoring:
      enabled: true
      prometheus_port: 9090
      health_check_interval: 30
```

### Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: compressor-deployment
  namespace: retrieval-free
  labels:
    app: compressor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: compressor
  template:
    metadata:
      labels:
        app: compressor
    spec:
      containers:
      - name: compressor
        image: retrieval-free/compressor:latest
        ports:
        - containerPort: 8000
        - containerPort: 9090
        env:
        - name: CONFIG_PATH
          value: "/etc/config/config.yaml"
        - name: MODEL_CACHE_DIR
          value: "/app/models"
        volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: model-cache
          mountPath: /app/models
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config-volume
        configMap:
          name: compressor-config
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      nodeSelector:
        nvidia.com/gpu.present: "true"
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

### Service
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: compressor-service
  namespace: retrieval-free
  labels:
    app: compressor
spec:
  selector:
    app: compressor
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: ClusterIP
```

### Horizontal Pod Autoscaler
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: compressor-hpa
  namespace: retrieval-free
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: compressor-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: gpu_utilization
      target:
        type: AverageValue
        averageValue: "75"
```

## Cloud Provider Deployments

### AWS EKS
```yaml
# aws/cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: retrieval-free-cluster
  region: us-west-2
  version: "1.28"

nodeGroups:
  - name: gpu-nodes
    instanceType: p3.2xlarge
    desiredCapacity: 2
    minSize: 1
    maxSize: 10
    volumeSize: 100
    ssh:
      allow: true
    iam:
      withAddonPolicies:
        imageBuilder: true
        autoScaler: true
        certManager: true
        efs: true
    labels:
      node-type: gpu
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule

managedNodeGroups:
  - name: cpu-nodes
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 50
    ssh:
      allow: true
    labels:
      node-type: cpu

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver
```

### Google GKE
```yaml
# gcp/cluster.yaml
apiVersion: container.v1
kind: Cluster
metadata:
  name: retrieval-free-cluster
spec:
  location: us-central1
  initialNodeCount: 1
  
  nodePools:
  - name: gpu-pool
    config:
      machineType: n1-standard-4
      accelerators:
      - acceleratorCount: 1
        acceleratorType: nvidia-tesla-t4
      diskSizeGb: 100
      oauthScopes:
      - https://www.googleapis.com/auth/compute
      - https://www.googleapis.com/auth/devstorage.read_only
      - https://www.googleapis.com/auth/logging.write
      - https://www.googleapis.com/auth/monitoring
    autoscaling:
      enabled: true
      minNodeCount: 0
      maxNodeCount: 10
    initialNodeCount: 1
```

### Azure AKS
```bash
# Azure CLI deployment
az aks create \
    --resource-group retrieval-free-rg \
    --name retrieval-free-cluster \
    --node-count 3 \
    --node-vm-size Standard_NC6s_v3 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 10 \
    --generate-ssh-keys \
    --enable-gpu
```

## Production Configuration

### Load Balancer Setup (NGINX)
```nginx
# nginx.conf
upstream compressor_backend {
    least_conn;
    server compressor-1:8000 max_fails=3 fail_timeout=30s;
    server compressor-2:8000 max_fails=3 fail_timeout=30s;
    server compressor-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen 443 ssl http2;
    server_name api.retrieval-free.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Compression
    gzip on;
    gzip_types application/json text/plain;

    location / {
        proxy_pass http://compressor_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts for large document processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Large request bodies
        client_max_body_size 50M;
        proxy_request_buffering off;
    }

    location /health {
        access_log off;
        proxy_pass http://compressor_backend/health;
    }

    location /metrics {
        access_log off;
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://compressor_backend/metrics;
    }
}
```

### SSL Configuration
```bash
# Generate SSL certificates (Let's Encrypt)
certbot certonly --nginx \
    -d api.retrieval-free.com \
    --email admin@retrieval-free.com \
    --agree-tos \
    --non-interactive
```

### Environment Configuration
```bash
# Production environment variables
export MODEL_NAME=rfcc-large-8x
export DEVICE=cuda
export BATCH_SIZE=16
export MAX_WORKERS=4
export REDIS_URL=redis://redis-cluster:6379
export LOG_LEVEL=INFO
export PROMETHEUS_PORT=9090
export HEALTH_CHECK_INTERVAL=30
export MODEL_CACHE_DIR=/app/models
export TEMP_DIR=/tmp/compressor
export MAX_MEMORY_MB=8192
export GPU_MEMORY_FRACTION=0.8
```

## Monitoring and Observability

### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'compressor'
    static_configs:
      - targets: ['compressor-service:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Compression Service Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Compression Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, compression_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{gpu}}"
          }
        ]
      }
    ]
  }
}
```

## Backup and Disaster Recovery

### Model Backup Strategy
```bash
#!/bin/bash
# backup-models.sh
MODEL_DIR="/app/models"
BACKUP_DIR="/backup/models"
S3_BUCKET="s3://retrieval-free-backups"

# Create backup
tar -czf "${BACKUP_DIR}/models-$(date +%Y%m%d-%H%M%S).tar.gz" -C "${MODEL_DIR}" .

# Upload to S3
aws s3 cp "${BACKUP_DIR}/" "${S3_BUCKET}/models/" --recursive

# Cleanup old backups (keep last 7 days)
find "${BACKUP_DIR}" -name "models-*.tar.gz" -mtime +7 -delete
```

### Database Backup (Redis)
```bash
#!/bin/bash
# backup-redis.sh
REDIS_HOST="redis-service"
REDIS_PORT="6379"
BACKUP_DIR="/backup/redis"

# Create backup
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" --rdb "${BACKUP_DIR}/dump-$(date +%Y%m%d-%H%M%S).rdb"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/" "s3://retrieval-free-backups/redis/" --recursive
```

## Performance Tuning

### GPU Optimization
```python
# Performance tuning configuration
CUDA_OPTIMIZATION = {
    'memory_fraction': 0.8,
    'allow_growth': True,
    'multi_gpu_strategy': 'mirrored',
    'mixed_precision': True,
    'cudnn_benchmark': True,
}

INFERENCE_OPTIMIZATION = {
    'batch_size': 16,
    'max_sequence_length': 8192,
    'use_cache': True,
    'torch_compile': True,
    'quantization': 'int8',
}
```

### Redis Configuration
```redis
# redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
tcp-keepalive 300
timeout 300
save 900 1
save 300 10
save 60 10000
```

This deployment guide provides comprehensive coverage of deployment scenarios from development to large-scale production environments with proper monitoring, security, and performance considerations.