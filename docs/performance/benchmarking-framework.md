# Performance Benchmarking Framework

This document outlines the performance benchmarking and regression detection framework for the repository.

## Benchmarking Infrastructure

### 1. Core Benchmark Suite

**Location**: `/benchmarks/`

**Required Benchmark Categories**:

1. **Compression Performance** (`compression_benchmark.py`):
   ```python
   def benchmark_compression_speed():
       """Benchmark compression speed across different document sizes."""
       
   def benchmark_compression_ratio():
       """Benchmark achieved compression ratios."""
       
   def benchmark_memory_usage():
       """Benchmark peak memory consumption during compression."""
   ```

2. **Model Loading** (`model_benchmark.py`):
   ```python
   def benchmark_model_loading():
       """Benchmark model initialization time."""
       
   def benchmark_tokenizer_speed():
       """Benchmark tokenization performance."""
   ```

3. **Inference Performance** (`inference_benchmark.py`):
   ```python
   def benchmark_inference_latency():
       """Benchmark end-to-end inference time."""
       
   def benchmark_batch_processing():
       """Benchmark batch processing efficiency."""
   ```

### 2. Benchmark Configuration

**Required Configuration** (`benchmarks/config.yaml`):
```yaml
# Benchmark configuration
benchmarks:
  compression:
    document_sizes: [1000, 5000, 10000, 50000, 100000]  # tokens
    compression_ratios: [2, 4, 8, 16, 32]
    repeat_count: 5
    warmup_runs: 2
    
  models:
    test_models: ["rfcc-base-2x", "rfcc-base-4x", "rfcc-base-8x"]
    batch_sizes: [1, 4, 8, 16]
    
  hardware:
    cpu_only: true
    gpu_memory_limit: "8GB"  # Optional GPU testing
    
performance_thresholds:
  compression_speed: 
    min_tokens_per_second: 1000
    max_memory_gb: 4.0
  
  model_loading:
    max_load_time_seconds: 30
    
  inference:
    max_latency_ms: 500
    min_throughput_tokens_per_second: 100

reporting:
  output_format: ["json", "html", "csv"]
  include_system_info: true
  generate_plots: true
```

### 3. Automated Regression Detection

**GitHub Workflow** (`.github/workflows/performance.yml`):
```yaml
name: Performance Benchmarks
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2AM UTC

jobs:
  benchmark:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for comparison
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install pytest-benchmark matplotlib seaborn
    
    - name: Run benchmarks
      run: |
        python benchmarks/run_all_benchmarks.py --output=benchmark_results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        name: 'Compression Performance'
        tool: 'customSmallerIsBetter'
        output-file-path: benchmark_results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '120%'  # Alert if 20% slower
        fail-on-alert: false
        
    - name: Upload benchmark artifacts
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: |
          benchmark_results.json
          benchmark_report.html
          benchmark_plots/
```

## Performance Monitoring

### 1. Continuous Performance Tracking

**Required Metrics Collection**:

- **Throughput Metrics**:
  - Tokens processed per second
  - Documents processed per minute
  - Batch processing efficiency

- **Latency Metrics**:
  - End-to-end compression time
  - Model loading time
  - First token latency

- **Resource Utilization**:
  - Peak memory usage
  - CPU utilization patterns
  - GPU memory usage (if applicable)

- **Quality Metrics**:
  - Compression ratio achieved
  - Information retention score
  - Reconstruction quality

### 2. Performance Dashboard

**Required Dashboard Components**:

1. **Real-time Metrics**: Current performance indicators
2. **Historical Trends**: Performance over time
3. **Regression Alerts**: Automatic performance degradation detection
4. **Comparison Views**: Compare different model versions
5. **Resource Usage**: Memory and CPU utilization tracking

### 3. Alert Configuration

**Performance Alert Thresholds**:
```yaml
alerts:
  compression_speed:
    warning_threshold: 15%  # 15% slower than baseline
    critical_threshold: 25%  # 25% slower than baseline
    
  memory_usage:
    warning_threshold: 20%  # 20% more memory than baseline
    critical_threshold: 50%  # 50% more memory than baseline
    
  model_accuracy:
    warning_threshold: 2%   # 2% drop in F1 score
    critical_threshold: 5%  # 5% drop in F1 score

notification:
  channels: ["github", "slack", "email"]
  escalation_time: "24h"
```

## Benchmark Execution

### 1. Local Development

**Running Benchmarks Locally**:
```bash
# Run all benchmarks
python benchmarks/run_all_benchmarks.py

# Run specific benchmark category
python benchmarks/run_all_benchmarks.py --suite=compression

# Compare with baseline
python benchmarks/run_all_benchmarks.py --compare-with=main

# Generate detailed report
python benchmarks/run_all_benchmarks.py --detailed-report
```

### 2. CI/CD Integration

**Pre-commit Performance Check**:
```yaml
# Add to .pre-commit-config.yaml
- repo: local
  hooks:
    - id: performance-check
      name: Quick performance check
      entry: python benchmarks/quick_check.py
      language: system
      files: ^(src/.*\.py|benchmarks/.*)$
      pass_filenames: false
```

### 3. Release Performance Validation

**Release Checklist Performance Requirements**:
- [ ] All benchmarks pass without regression
- [ ] Memory usage within acceptable limits
- [ ] Compression quality maintained or improved
- [ ] Inference latency meets SLA requirements
- [ ] Performance report generated and reviewed

## Profiling and Optimization

### 1. Code Profiling

**Required Profiling Tools Setup**:
```bash
# Install profiling dependencies
pip install py-spy memory-profiler line-profiler

# Profile compression function
python -m memory_profiler scripts/profile_compression.py

# Line-by-line profiling
kernprof -l -v scripts/profile_detailed.py

# System-wide profiling
py-spy record -o profile.svg -- python your_script.py
```

### 2. Performance Analysis Scripts

**Required Analysis Tools** (`scripts/performance/`):

1. **Memory Profiler** (`memory_analysis.py`):
   - Track memory usage patterns
   - Identify memory leaks
   - Optimize memory allocation

2. **CPU Profiler** (`cpu_analysis.py`):
   - Identify computational bottlenecks
   - Analyze function call patterns
   - Optimize hot code paths

3. **GPU Profiler** (`gpu_analysis.py`):
   - Monitor GPU utilization
   - Analyze memory transfers
   - Optimize CUDA kernels

### 3. Optimization Guidelines

**Performance Optimization Priorities**:

1. **Critical Path Optimization**:
   - Focus on compression algorithm core
   - Optimize tokenization pipeline
   - Improve batch processing efficiency

2. **Memory Optimization**:
   - Implement memory pooling
   - Optimize tensor operations
   - Add garbage collection hints

3. **I/O Optimization**:
   - Implement async file operations
   - Optimize model loading
   - Add caching layers

## Results and Reporting

### 1. Benchmark Report Format

**Required Report Sections**:
- **Executive Summary**: Key performance metrics
- **Detailed Results**: Comprehensive benchmark data
- **Regression Analysis**: Comparison with previous versions
- **System Information**: Hardware and software configuration
- **Recommendations**: Performance improvement suggestions

### 2. Historical Data Management

**Performance Data Storage**:
- **Format**: JSON with timestamp metadata
- **Retention**: 12 months of historical data
- **Backup**: Weekly backup to external storage
- **Access**: Public read access to benchmark history

### 3. Performance Documentation

**Required Documentation Updates**:
- Update README with latest performance numbers
- Maintain performance changelog
- Document optimization techniques
- Provide performance tuning guide

## Implementation Checklist

- [ ] Set up benchmark infrastructure
- [ ] Implement core benchmark suites
- [ ] Configure automated regression detection
- [ ] Set up performance monitoring dashboard
- [ ] Create profiling and analysis tools
- [ ] Establish performance alert system
- [ ] Document performance optimization guidelines
- [ ] Integrate with CI/CD pipeline
- [ ] Set up historical data tracking
- [ ] Create performance reporting system

## Success Metrics

- **Benchmark Coverage**: >90% of critical code paths benchmarked
- **Regression Detection**: <24 hours to detect performance issues
- **Alert Accuracy**: <5% false positive rate on performance alerts
- **Documentation Currency**: Performance docs updated within 48 hours
- **Developer Adoption**: >80% of developers run benchmarks before commits