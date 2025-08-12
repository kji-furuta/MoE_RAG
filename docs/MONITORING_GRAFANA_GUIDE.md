# Grafana監視ダッシュボード実装ガイド

## アーキテクチャ構成

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   RAG App   │────▶│  Prometheus   │────▶│   Grafana   │
│  (Metrics)  │      │  (Storage)   │      │ (Visualize) │
└─────────────┘      └──────────────┘      └─────────────┘
       │                    │                     │
       ▼                    ▼                     ▼
  Export Metrics      Scrape & Store         Dashboard
```

## 1. Prometheusメトリクス実装

### メトリクスエクスポーター（RAGアプリ側）

```python
# app/monitoring/metrics_exporter.py
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi import Response
from functools import wraps
import time
import psutil
import torch

# ===== システムメトリクス =====
system_info = Info('system_info', 'System information')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
memory_percent = Gauge('memory_usage_percent', 'Memory usage percentage')
disk_usage = Gauge('disk_usage_percent', 'Disk usage percentage')

# GPU メトリクス
gpu_count = Gauge('gpu_count', 'Number of available GPUs')
gpu_memory_used = Gauge('gpu_memory_used_bytes', 'GPU memory used', ['device'])
gpu_memory_total = Gauge('gpu_memory_total_bytes', 'GPU memory total', ['device'])
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization', ['device'])

# ===== アプリケーションメトリクス =====
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ===== RAG固有メトリクス =====
document_processing_duration = Histogram(
    'document_processing_duration_seconds',
    'Document processing duration',
    ['document_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300)
)

vector_search_duration = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
)

embedding_generation_duration = Histogram(
    'embedding_generation_duration_seconds',
    'Embedding generation duration',
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
)

query_cache_hits = Counter('query_cache_hits_total', 'Query cache hits')
query_cache_misses = Counter('query_cache_misses_total', 'Query cache misses')

active_documents = Gauge('active_documents_count', 'Number of active documents')
total_embeddings = Gauge('total_embeddings_count', 'Total number of embeddings')

# ===== ヘルスチェックメトリクス =====
component_health_status = Gauge(
    'component_health_status',
    'Component health status (1=healthy, 0.5=degraded, 0=unhealthy)',
    ['component']
)

# ===== メトリクス収集関数 =====
def collect_system_metrics():
    """システムメトリクスを収集"""
    # CPU & Memory
    cpu_usage.set(psutil.cpu_percent())
    memory = psutil.virtual_memory()
    memory_usage.set(memory.used)
    memory_percent.set(memory.percent)
    
    # Disk
    disk = psutil.disk_usage('/')
    disk_usage.set(disk.percent)
    
    # GPU (if available)
    if torch.cuda.is_available():
        gpu_count.set(torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            mem_info = torch.cuda.mem_get_info(i)
            gpu_memory_used.labels(device=f'cuda:{i}').set(mem_info[1] - mem_info[0])
            gpu_memory_total.labels(device=f'cuda:{i}').set(mem_info[1])
            
            # GPU利用率（NVIDIA SMIが利用可能な場合）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization.labels(device=f'cuda:{i}').set(util.gpu)
            except:
                pass

def track_request_metrics(method: str, endpoint: str, status: int, duration: float):
    """HTTPリクエストメトリクスを記録"""
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)

def track_rag_metrics(metric_type: str, duration: float, **labels):
    """RAG固有のメトリクスを記録"""
    if metric_type == 'document_processing':
        document_processing_duration.labels(**labels).observe(duration)
    elif metric_type == 'vector_search':
        vector_search_duration.observe(duration)
    elif metric_type == 'embedding_generation':
        embedding_generation_duration.observe(duration)

# FastAPIミドルウェア
def metrics_middleware(app):
    @app.middleware("http")
    async def track_metrics(request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        track_request_metrics(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
        
        return response

# メトリクスエンドポイント
@app.get("/metrics")
async def get_metrics():
    """Prometheus用メトリクスエンドポイント"""
    collect_system_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
```

## 2. Docker Compose設定

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: rag-prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    networks:
      - rag-network

  grafana:
    image: grafana/grafana:latest
    container_name: rag-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    networks:
      - rag-network

  node-exporter:
    image: prom/node-exporter:latest
    container_name: rag-node-exporter
    restart: unless-stopped
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    networks:
      - rag-network

volumes:
  prometheus-data:
  grafana-data:

networks:
  rag-network:
    external: true
```

## 3. Prometheus設定

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # RAGアプリケーション
  - job_name: 'rag-app'
    static_configs:
      - targets: ['ai-ft-container:8050']
    metrics_path: /metrics
    scrape_interval: 10s

  # Node Exporter（システムメトリクス）
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  # Docker コンテナメトリクス
  - job_name: 'docker'
    static_configs:
      - targets: ['docker-host:9323']

  # Prometheusセルフモニタリング
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

## 4. Grafanaダッシュボード定義

```json
{
  "dashboard": {
    "title": "RAG System Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "title": "Response Time (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "P95"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "title": "Vector Search Performance",
        "targets": [
          {
            "expr": "rate(vector_search_duration_seconds_sum[5m]) / rate(vector_search_duration_seconds_count[5m])",
            "legendFormat": "Avg Search Time"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "rate(query_cache_hits_total[5m]) / (rate(query_cache_hits_total[5m]) + rate(query_cache_misses_total[5m]))",
            "legendFormat": "Hit Rate"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "gpu_memory_used_bytes / gpu_memory_total_bytes * 100",
            "legendFormat": "{{device}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16}
      },
      {
        "title": "System Health Status",
        "targets": [
          {
            "expr": "component_health_status",
            "legendFormat": "{{component}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16}
      }
    ]
  }
}
```

## 5. アラート設定

```yaml
# monitoring/alerts.yml
groups:
  - name: rag_alerts
    interval: 30s
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "P95 response time is above 2 seconds"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate"
          description: "Error rate is above 5%"

      - alert: LowCacheHitRate
        expr: rate(query_cache_hits_total[5m]) / (rate(query_cache_hits_total[5m]) + rate(query_cache_misses_total[5m])) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low cache hit rate"
          description: "Cache hit rate is below 50%"

      - alert: HighGPUMemoryUsage
        expr: gpu_memory_used_bytes / gpu_memory_total_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High GPU memory usage"
          description: "GPU memory usage is above 90%"
```

## 実装による効果

1. **問題の早期発見**: リアルタイムでシステムの異常を検知
2. **パフォーマンス分析**: ボトルネックの特定と最適化ポイントの発見
3. **キャパシティプランニング**: リソース使用傾向から将来の必要リソースを予測
4. **SLA管理**: サービスレベルの可視化と保証
