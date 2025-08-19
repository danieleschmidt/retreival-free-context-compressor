# Production Deployment Guide

## Overview

This guide covers the complete production deployment of the Generation 4 Retrieval-Free Context Compression service.

## Prerequisites

- Kubernetes cluster (1.20+)
- Docker registry access
- kubectl configured
- Monitoring stack (Prometheus/Grafana)
- TLS certificates

## Quick Start

```bash
# Deploy to production
./scripts/deploy.sh production v1.0.0

# Scale service
./scripts/scale.sh 10 scale

# Enable autoscaling
./scripts/scale.sh 3 50 70  # min=3, max=50, cpu=70%

# Rollback if needed
./scripts/rollback.sh
```

## Configuration

### Environment Variables

- `ENVIRONMENT`: production/staging/development
- `LOG_LEVEL`: INFO/DEBUG/WARNING/ERROR
- `REDIS_URL`: Redis connection string
- `PROMETHEUS_ENABLED`: true/false

### Resource Requirements

- **CPU**: 500m request, 2000m limit
- **Memory**: 1Gi request, 4Gi limit
- **Storage**: 10Gi for logs and cache

### Scaling Configuration

- **Min Replicas**: 3
- **Max Replicas**: 100
- **CPU Target**: 70%
- **Scale Out Cooldown**: 5 minutes
- **Scale In Cooldown**: 10 minutes

## Security

### Network Policies

- Ingress limited to load balancer and monitoring
- Egress allowed for DNS, HTTPS, and Redis
- Inter-pod communication restricted

### RBAC

- Service account with minimal permissions
- Role-based access to ConfigMaps and Secrets only
- No cluster-wide permissions

### Pod Security

- Non-root user execution
- Read-only root filesystem
- No privilege escalation
- All capabilities dropped

## Monitoring

### Key Metrics

- Request rate and response time
- Compression ratios achieved
- Error rates and types
- Resource utilization
- Cache hit rates

### Alerts

- High error rate (>10% for 2min)
- High response time (>1s P95 for 5min)
- Low compression ratio (<4x for 5min)
- Service down (>1min)
- High memory usage (>3GB for 5min)

### Dashboards

Access monitoring at `http://grafana.your-domain.com`

Key dashboards:
- Service Overview
- Performance Metrics
- Resource Usage
- Error Analysis

## Maintenance

### Updates

1. Build new image: `docker build -t terragon/retrieval-free-compression:v1.1.0 .`
2. Push to registry: `docker push terragon/retrieval-free-compression:v1.1.0`
3. Update deployment: `kubectl set image deployment/compression-service compression-api=terragon/retrieval-free-compression:v1.1.0 -n compression-service`
4. Monitor rollout: `kubectl rollout status deployment/compression-service -n compression-service`

### Backup

Configuration and secrets are backed up through:
- ConfigMap version control
- Secret encryption at rest
- Persistent volume snapshots
- Database backups (if applicable)

### Disaster Recovery

1. Multi-region deployment for high availability
2. Automated failover with health checks
3. Data replication across availability zones
4. Backup restoration procedures documented

## Troubleshooting

### Common Issues

1. **Pod not starting**: Check resource limits and image availability
2. **High memory usage**: Review compression algorithm settings
3. **Slow response time**: Check CPU allocation and scaling settings
4. **Authentication failures**: Verify JWT configuration and secrets

### Log Analysis

```bash
# View service logs
kubectl logs -f -l app=compression-service -n compression-service

# Check specific pod
kubectl logs pod-name -n compression-service

# View events
kubectl get events -n compression-service --sort-by='.lastTimestamp'
```

### Performance Tuning

1. Adjust worker processes based on CPU cores
2. Tune JVM heap size for Java components
3. Optimize compression algorithm parameters
4. Configure connection pooling settings

## Support

For production issues:
- **Slack**: #compression-service-alerts
- **Email**: ops@terragon-labs.com
- **PagerDuty**: Critical alerts auto-page on-call engineer

## Architecture

```
Internet → Load Balancer → Ingress Controller → Service → Pods
                                                      ↓
                                              ConfigMap/Secrets
                                                      ↓
                                              Redis Cache
                                                      ↓
                                              Monitoring Stack
```

## Compliance

This deployment meets the following compliance requirements:
- SOC 2 Type II
- GDPR data protection
- CCPA privacy regulations
- ISO 27001 security standards
