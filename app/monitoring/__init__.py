"""
Prometheusメトリクス収集モジュール
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import CollectorRegistry
from fastapi import Response
import time
import psutil
import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# カスタムレジストリ
registry = CollectorRegistry()

# システムメトリクス
system_info = Info('ai_ft_system', 'AI-FT System Information', registry=registry)
cpu_usage = Gauge('ai_ft_cpu_usage_percent', 'CPU Usage Percentage', registry=registry)
memory_usage = Gauge('ai_ft_memory_usage_percent', 'Memory Usage Percentage', registry=registry)
gpu_available = Gauge('ai_ft_gpu_available', 'GPU Available (1=Yes, 0=No)', registry=registry)
gpu_count = Gauge('ai_ft_gpu_count', 'Number of GPUs', registry=registry)
gpu_memory_used = Gauge('ai_ft_gpu_memory_used_mb', 'GPU Memory Used (MB)', ['gpu_id'], registry=registry)
gpu_memory_total = Gauge('ai_ft_gpu_memory_total_mb', 'GPU Memory Total (MB)', ['gpu_id'], registry=registry)

# HTTPリクエストメトリクス
http_requests_total = Counter(
    'ai_ft_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

http_request_duration = Histogram(
    'ai_ft_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

# RAGメトリクス
rag_queries_total = Counter(
    'ai_ft_rag_queries_total',
    'Total RAG queries',
    ['status'],
    registry=registry
)

rag_query_duration = Histogram(
    'ai_ft_rag_query_duration_seconds',
    'RAG query duration in seconds',
    registry=registry
)

rag_documents_total = Gauge(
    'ai_ft_rag_documents_total',
    'Total number of RAG documents',
    registry=registry
)

# ファインチューニングメトリクス
training_tasks_total = Counter(
    'ai_ft_training_tasks_total',
    'Total training tasks',
    ['status'],
    registry=registry
)

training_duration = Histogram(
    'ai_ft_training_duration_seconds',
    'Training task duration in seconds',
    ['model_type', 'method'],
    registry=registry
)

active_training_tasks = Gauge(
    'ai_ft_active_training_tasks',
    'Number of active training tasks',
    registry=registry
)

# キャッシュメトリクス
cache_hits = Counter(
    'ai_ft_cache_hits_total',
    'Total cache hits',
    ['namespace'],
    registry=registry
)

cache_misses = Counter(
    'ai_ft_cache_misses_total',
    'Total cache misses',
    ['namespace'],
    registry=registry
)

cache_operations = Counter(
    'ai_ft_cache_operations_total',
    'Total cache operations',
    ['operation', 'namespace'],
    registry=registry
)

redis_connected = Gauge(
    'ai_ft_redis_connected',
    'Redis connection status (1=connected, 0=disconnected)',
    registry=registry
)


class MetricsCollector:
    """メトリクス収集クラス"""
    
    def __init__(self):
        self.start_time = time.time()
        self.update_system_metrics()
    
    def update_system_metrics(self):
        """システムメトリクスを更新"""
        try:
            # CPU & メモリ
            cpu_usage.set(psutil.cpu_percent(interval=1))
            memory_usage.set(psutil.virtual_memory().percent)
            
            # GPU情報
            if torch.cuda.is_available():
                gpu_available.set(1)
                gpu_count.set(torch.cuda.device_count())
                
                for i in range(torch.cuda.device_count()):
                    if torch.cuda.is_available():
                        # GPU メモリ使用量
                        mem_info = torch.cuda.mem_get_info(i)
                        used = (mem_info[1] - mem_info[0]) / (1024 * 1024)  # MB
                        total = mem_info[1] / (1024 * 1024)  # MB
                        gpu_memory_used.labels(gpu_id=str(i)).set(used)
                        gpu_memory_total.labels(gpu_id=str(i)).set(total)
            else:
                gpu_available.set(0)
                gpu_count.set(0)
            
            # システム情報
            system_info.info({
                'version': '1.0.0',
                'python_version': '3.11',
                'pytorch_version': torch.__version__,
                'cuda_available': str(torch.cuda.is_available())
            })
            
        except Exception as e:
            logger.error(f"システムメトリクス更新エラー: {e}")
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """HTTPリクエストを記録"""
        http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        http_request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_rag_query(self, success: bool, duration: float):
        """RAGクエリを記録"""
        status = "success" if success else "failed"
        rag_queries_total.labels(status=status).inc()
        rag_query_duration.observe(duration)
    
    def record_training_task(self, model_type: str, method: str, status: str, duration: Optional[float] = None):
        """トレーニングタスクを記録"""
        training_tasks_total.labels(status=status).inc()
        if duration:
            training_duration.labels(model_type=model_type, method=method).observe(duration)
    
    def record_cache_operation(self, operation: str, namespace: str, hit: Optional[bool] = None):
        """キャッシュ操作を記録"""
        cache_operations.labels(operation=operation, namespace=namespace).inc()
        if hit is not None:
            if hit:
                cache_hits.labels(namespace=namespace).inc()
            else:
                cache_misses.labels(namespace=namespace).inc()
    
    def set_redis_status(self, connected: bool):
        """Redis接続状態を設定"""
        redis_connected.set(1 if connected else 0)
    
    def set_rag_documents_count(self, count: int):
        """RAG文書数を設定"""
        rag_documents_total.set(count)
    
    def set_active_training_tasks(self, count: int):
        """アクティブなトレーニングタスク数を設定"""
        active_training_tasks.set(count)
    
    def get_metrics(self) -> bytes:
        """Prometheus形式のメトリクスを取得"""
        self.update_system_metrics()
        return generate_latest(registry)


# グローバルメトリクスコレクター
metrics_collector = MetricsCollector()


def get_prometheus_metrics() -> Response:
    """FastAPIレスポンスとしてメトリクスを返す"""
    metrics = metrics_collector.get_metrics()
    return Response(content=metrics, media_type=CONTENT_TYPE_LATEST)
