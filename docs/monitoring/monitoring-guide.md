# Monitoring and Observability Guide

This guide covers the comprehensive monitoring and observability setup for the Retrieval-Free Context Compressor.

## Overview

Our monitoring stack provides:
- **Metrics Collection**: Prometheus for application and infrastructure metrics
- **Visualization**: Grafana dashboards for real-time insights
- **Log Aggregation**: Loki and Promtail for centralized logging
- **Distributed Tracing**: Jaeger for request tracing
- **Alerting**: AlertManager for proactive notifications

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│   Prometheus    │───▶│    Grafana      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Jaeger      │    │  AlertManager   │    │      Loki       │
│   (Tracing)     │    │   (Alerting)    │    │   (Logging)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        ▲
                                                        │
                                               ┌─────────────────┐
                                               │    Promtail     │
                                               │ (Log Collection)│
                                               └─────────────────┘
```

## Quick Start

### Start Monitoring Stack

```bash
# Start full monitoring stack
cd monitoring
docker-compose -f docker-compose.monitoring.yml up -d

# Start with GPU monitoring
docker-compose -f docker-compose.monitoring.yml --profile gpu up -d

# View services
docker-compose -f docker-compose.monitoring.yml ps
```

### Access Web Interfaces

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/admin123 |
| Prometheus | http://localhost:9090 | - |
| AlertManager | http://localhost:9093 | - |
| Jaeger | http://localhost:16686 | - |

## Metrics Collection

### Application Metrics

The application exposes metrics at `/metrics` endpoint:

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Request metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

# Compression metrics
COMPRESSION_TIME = Histogram('compression_duration_seconds', 'Time spent compressing documents')
COMPRESSION_RATIO = Gauge('compression_ratio', 'Current compression ratio achieved')
COMPRESSION_QUALITY = Gauge('compression_quality_score', 'Quality score of compression')

# Model metrics
MODEL_LOAD_TIME = Histogram('model_load_duration_seconds', 'Time to load models')
MODEL_MEMORY_USAGE = Gauge('model_memory_bytes', 'Memory used by models')

# GPU metrics (if available)
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU utilization percentage')
GPU_MEMORY_USED = Gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
```

### Infrastructure Metrics

- **Node Exporter**: System metrics (CPU, memory, disk, network)
- **cAdvisor**: Container metrics and resource usage
- **NVIDIA Exporter**: GPU metrics (if available)

### Custom Metrics

Add custom metrics for specific use cases:

```python
# Business metrics
DOCUMENTS_PROCESSED = Counter('documents_processed_total', 'Total documents processed')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate percentage')
USER_SESSIONS = Gauge('active_user_sessions', 'Number of active user sessions')

# Quality metrics
F1_SCORE = Gauge('compression_f1_score', 'F1 score of compressed vs original')
INFORMATION_RETENTION = Gauge('information_retention_rate', 'Information retention percentage')
```

## Logging

### Log Aggregation

Promtail collects logs from multiple sources:
- Application logs from `/var/log/retrieval-free/`
- Docker container logs
- System logs
- NGINX access logs (if applicable)

### Log Format

Use structured JSON logging:

```python
import structlog

logger = structlog.get_logger()

logger.info("Document compressed", 
    document_id="doc123",
    original_tokens=25600, 
    compressed_tokens=3200,
    compression_ratio=8.0,
    duration_ms=487
)
```

### Log Levels and Routing

| Level | Usage | Retention |
|-------|-------|-----------|
| ERROR | Critical errors requiring immediate attention | 90 days |
| WARN | Warnings that may need investigation | 30 days |
| INFO | General operational information | 14 days |
| DEBUG | Detailed debugging information | 7 days |

## Distributed Tracing

### Jaeger Integration

Enable tracing in your application:

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure Jaeger
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Use in application
tracer = trace.get_tracer(__name__)

def compress_document(document):
    with tracer.start_as_current_span("compress_document") as span:
        span.set_attribute("document.length", len(document))
        # Compression logic
        span.set_attribute("compression.ratio", ratio)
        return compressed_doc
```

### Trace Correlation

Correlate traces with logs and metrics:

```python
from opentelemetry.trace import get_current_span

def log_with_trace_context(message, **kwargs):
    span = get_current_span()
    if span:
        kwargs.update({
            "trace_id": format(span.get_span_context().trace_id, "032x"),
            "span_id": format(span.get_span_context().span_id, "016x")
        })
    logger.info(message, **kwargs)
```

## Alerting

### Alert Rules

Critical alerts configured in `prometheus/rules/`:

#### Service Availability
- Service down for > 1 minute
- High error rate (> 10% for 5 minutes)
- High latency (95th percentile > 1s for 5 minutes)

#### Performance
- Compression time > 5s (95th percentile for 10 minutes)
- Memory usage > 2GB for 5 minutes
- CPU usage > 80% for 10 minutes

#### Resources
- Disk space < 10%
- GPU memory > 90% (if applicable)
- Container killed/restarted

### Alert Routing

Configure AlertManager for different notification channels:

```yaml
# Critical alerts: immediate notification
- match:
    severity: critical
  receiver: 'critical-alerts'
  group_wait: 5s
  repeat_interval: 30m

# Warning alerts: batched notifications  
- match:
    severity: warning
  receiver: 'warning-alerts'
  repeat_interval: 2h
```

### Notification Channels

- **Email**: For critical and warning alerts
- **Slack**: Real-time team notifications
- **PagerDuty**: On-call escalation for critical issues
- **Webhook**: Custom integrations

## Dashboards

### Pre-built Dashboards

1. **Application Overview**: Key metrics and health status
2. **Performance Monitoring**: Latency, throughput, and error rates
3. **Resource Utilization**: CPU, memory, disk, and GPU usage
4. **Compression Analytics**: Compression ratios, quality metrics
5. **Infrastructure Health**: System and container metrics

### Custom Dashboard Creation

Create dashboards for specific use cases:

```json
{
  "dashboard": {
    "title": "Compression Performance",
    "panels": [
      {
        "title": "Compression Ratio",
        "type": "stat",
        "targets": [
          {
            "expr": "compression_ratio",
            "legendFormat": "Current Ratio"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, compression_duration_seconds_bucket)",
            "legendFormat": "95th Percentile"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Compression Ratio | 8:1 | < 6:1 |
| Processing Latency (95th) | < 500ms | > 1s |
| F1 Score Retention | > 95% | < 90% |
| Memory Usage | < 2GB | > 3GB |
| Error Rate | < 1% | > 5% |

### Performance Queries

Common Prometheus queries:

```promql
# Average compression ratio over time
avg_over_time(compression_ratio[1h])

# Request rate by endpoint
sum(rate(http_requests_total[5m])) by (endpoint)

# Error percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100

# Memory usage trend
process_resident_memory_bytes / 1024 / 1024

# GPU utilization
nvidia_ml_py_utilization_gpu
```

## Troubleshooting

### Common Issues

**High Memory Usage**
```promql
# Check memory usage by service
container_memory_usage_bytes{name=~"retrieval-free.*"} / 1024 / 1024

# Identify memory leaks
increase(process_resident_memory_bytes[1h])
```

**Performance Degradation**
```promql
# Check latency trends
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Compare with baseline
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) / 
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m] offset 24h))
```

**Service Connectivity**
```bash
# Check service health
curl http://localhost:8000/health

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets
```

### Debug Procedures

1. **Check service logs**:
   ```bash
   docker logs retrieval-free-app
   ```

2. **Verify metrics collection**:
   ```bash
   curl http://localhost:9090/api/v1/query?query=up
   ```

3. **Test alert rules**:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

4. **Validate configuration**:
   ```bash
   promtool check config prometheus.yml
   ```

## Best Practices

### Monitoring Strategy

1. **Golden Signals**: Focus on latency, traffic, errors, and saturation
2. **SLI/SLO Definition**: Define Service Level Indicators and Objectives
3. **Alert Fatigue**: Minimize false positives and alert noise
4. **Observability**: Combine metrics, logs, and traces for full visibility

### Metric Design

1. **Cardinality**: Keep metric cardinality low to avoid performance issues
2. **Labels**: Use consistent and meaningful label names
3. **Naming**: Follow Prometheus naming conventions
4. **Documentation**: Document custom metrics and their purpose

### Dashboard Design

1. **User-Centric**: Design dashboards for specific user roles
2. **Hierarchy**: Start with high-level overview, drill down to details
3. **Context**: Provide sufficient context for decision making
4. **Performance**: Optimize query performance for real-time updates

### Alert Design

1. **Actionable**: Every alert should have a clear action
2. **Contextual**: Provide enough context to diagnose issues
3. **Escalation**: Define clear escalation paths
4. **Testing**: Regularly test alert rules and notification channels

## Advanced Topics

### Multi-Cluster Monitoring

For distributed deployments:

```yaml
# Prometheus federation
- job_name: 'federate'
  scrape_interval: 15s
  honor_labels: true
  metrics_path: '/federate'
  params:
    'match[]':
      - '{job="retrieval-free-app"}'
      - '{__name__=~"job:.*"}'
  static_configs:
    - targets:
      - 'prometheus-cluster1:9090'
      - 'prometheus-cluster2:9090'
```

### Long-term Storage

Configure long-term metrics storage:

```yaml
# Thanos sidecar for long-term storage
thanos:
  image: thanosio/thanos:v0.32.0
  command:
    - 'sidecar'
    - '--tsdb.path=/prometheus'
    - '--prometheus.url=http://prometheus:9090'
    - '--objstore.config-file=/etc/thanos/config.yaml'
```

### Custom Exporters

Create custom exporters for specific metrics:

```python
from prometheus_client import start_http_server, Gauge
import time

class CustomExporter:
    def __init__(self):
        self.model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
        
    def collect_metrics(self):
        # Custom metric collection logic
        accuracy = self.calculate_model_accuracy()
        self.model_accuracy.set(accuracy)

if __name__ == '__main__':
    exporter = CustomExporter()
    start_http_server(8000)
    
    while True:
        exporter.collect_metrics()
        time.sleep(60)
```

---

For more information, see:
- [Grafana Dashboard Gallery](https://grafana.com/grafana/dashboards)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [AlertManager Configuration](https://prometheus.io/docs/alerting/latest/alertmanager/)